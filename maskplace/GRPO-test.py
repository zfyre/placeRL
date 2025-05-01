import argparse
import pickle
from collections import namedtuple
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nnet import MyCNN, MyCNNCoarse
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import place_env
import torchvision
from place_db import PlaceDB
import time
from tqdm import tqdm
import random
from comp_res import comp_res
import wandb

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# Parameters
parser = argparse.ArgumentParser(description='Solve the Chip Placement with GRPO')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--pnm', type=int, default=128)
parser.add_argument('--benchmark', type=str, default='macro_tiles_10x10')
parser.add_argument('--soft_coefficient', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--save_fig', action='store_true', default=True)
parser.add_argument('--wandb_project', type=str, default='placerl')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--num_groups', type=int, default=4, help='Number of groups for GRPO')
parser.add_argument('--group_size', type=int, default=16, help='Size of each group')
parser.add_argument('--relative_scale', type=float, default=0.1, help='Scale factor for relative policy optimization')
args = parser.parse_args()

# Initialize wandb
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config={
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "placed_num_macro": args.pnm,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "num_groups": args.num_groups,
        "group_size": args.group_size,
        "relative_scale": args.relative_scale
    },
    mode="online"
)

benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = 224
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb=placedb, placed_num_macro=placed_num_macro, grid=grid).unwrapped

num_emb_state = 64 + 2 + 1
num_state = 1 + grid*grid*5 + 2

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    try:
        env.seed(seed)
    except:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_action = env.action_space.shape
seed_torch(args.seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic', 'group_id'])
TrainingRecord = namedtuple('TrainRecord', ['episode', 'reward'])

class GroupEmbedding(nn.Module):
    def __init__(self, num_groups, embedding_dim):
        super(GroupEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_groups, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, group_id):
        emb = self.embedding(group_id)
        return self.fc(emb)

class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_emb_state, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, grid * grid)
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.gcn = None
        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)
        
        # Group-specific components
        self.group_emb = GroupEmbedding(args.num_groups, 64)
        self.group_attention = nn.MultiheadAttention(64, 4)
        self.group_norm = nn.LayerNorm(64)

    def forward(self, x, graph=None, cnn_res=None, gcn_res=None, graph_node=None, group_id=None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            mask = x[:, 1+grid*grid*2: 1+grid*grid*3].reshape(-1, grid, grid)
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                    x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                    ), dim=1).reshape(-1, 3, grid, grid)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))

        net_img = x[:, 1+grid*grid: 1+grid*grid*2]
        net_img = net_img + x[:, 1+grid*grid*2: 1+grid*grid*3] * 10
        net_img_min = net_img.min() + args.soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, grid * grid)
        
        # Group-specific processing
        if group_id is not None:
            group_features = self.group_emb(group_id)
            # Apply attention mechanism
            group_features = group_features.unsqueeze(0)  # Add sequence dimension
            attn_output, _ = self.group_attention(group_features, group_features, group_features)
            group_features = attn_output.squeeze(0)
            group_features = self.group_norm(group_features)
            x = x + group_features.view(-1, 1)

        x = torch.where(mask + mask2 >= 1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x, cnn_res, gcn_res

class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn
        
        # Group-specific components
        self.group_emb = GroupEmbedding(args.num_groups, 64)
        self.group_attention = nn.MultiheadAttention(64, 4)
        self.group_norm = nn.LayerNorm(64)

    def forward(self, x, graph=None, cnn_res=None, gcn_res=None, graph_node=None, group_id=None):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        
        # Group-specific processing
        if group_id is not None:
            group_features = self.group_emb(group_id)
            # Apply attention mechanism
            group_features = group_features.unsqueeze(0)
            attn_output, _ = self.group_attention(group_features, group_features, group_features)
            group_features = attn_output.squeeze(0)
            group_features = self.group_norm(group_features)
            x1 = x1 + group_features

        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value

class GRPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    if placed_num_macro:
        buffer_capacity = 10 * (placed_num_macro)
    else:
        buffer_capacity = 5120
    batch_size = args.batch_size

    def __init__(self):
        super(GRPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        self.actor_net = Actor(cnn=self.cnn, gcn=self.gcn, cnn_coarse=self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn=self.cnn, gcn=self.gcn, cnn_coarse=None, res_net=self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)
        self.num_groups = args.num_groups
        self.group_size = args.group_size

    def select_action(self, state, group_id=None):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        if group_id is not None:
            group_id = torch.tensor([group_id], dtype=torch.long).to(device)
        with torch.no_grad():
            action_prob, _, _ = self.actor_net(state, group_id=group_id)
            action = torch.multinomial(action_prob, 1)
            a_log_prob = torch.log(action_prob.gather(1, action))
            # Ensure we're getting scalar values
            return action.squeeze().item(), a_log_prob.squeeze().item()

    def get_value(self, state, group_id=None):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        if group_id is not None:
            group_id = torch.tensor([group_id], dtype=torch.long).to(device)
        with torch.no_grad():
            value = self.critic_net(state, group_id=group_id)
            return value.squeeze().item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def compute_relative_advantages(self, rewards, values, group_ids):
        advantages = torch.zeros_like(rewards)
        for g in range(self.num_groups):
            group_mask = (group_ids == g)
            if group_mask.sum() > 0:
                group_rewards = rewards[group_mask]
                group_values = values[group_mask]
                
                # Compute group-specific baseline
                group_baseline = group_values.mean()
                
                # Compute relative advantages
                group_advantages = group_rewards - group_baseline
                
                # Scale relative advantages
                group_advantages = group_advantages * args.relative_scale
                
                advantages[group_mask] = group_advantages
        return advantages

    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).to(device).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).to(device)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).to(device).view(-1, 1)
        group_id = torch.tensor([t.group_id for t in self.buffer], dtype=torch.long).to(device)

        # Get current values
        with torch.no_grad():
            values = torch.tensor([self.get_value(s, g) for s, g in zip(state, group_id)], dtype=torch.float).to(device)

        # Compute relative advantages
        advantage = self.compute_relative_advantages(reward, values, group_id)
        
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # Get group-specific data
                group_id_batch = group_id[index]
                
                # Update actor
                action_prob, _, _ = self.actor_net(state[index], group_id=group_id_batch)
                new_action_log_prob = torch.log(action_prob.gather(1, action[index]))
                ratio = torch.exp(new_action_log_prob - old_action_log_prob[index])
                surr1 = ratio * advantage[index].view(-1, 1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage[index].view(-1, 1)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Update critic
                value_pred = self.critic_net(state[index], group_id=group_id_batch)
                value_target = advantage[index].view(-1, 1) + value_pred.detach()
                critic_loss = F.mse_loss(value_pred, value_target)

                # Optimize
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_net_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]

def main():
    agent = GRPO()
    running_reward = 0
    log_interval = args.log_interval
    running_rewards = []
    training_records = []
    
    for i_episode in range(1000):
        state = env.reset()
        episode_reward = 0
        group_id = random.randint(0, args.num_groups - 1)  # Randomly assign group for this episode
        
        for t in range(10000):
            action, action_log_prob = agent.select_action(state, group_id)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Store transition with group information
            transition = Transition(state, action, reward, action_log_prob, next_state, 0, group_id)
            if agent.store_transition(transition):
                agent.update()
            
            if done:
                break
            state = next_state
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        running_rewards.append(running_reward)
        training_records.append(TrainingRecord(i_episode, running_reward))
        
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, episode_reward, running_reward))
            wandb.log({
                "episode": i_episode,
                "reward": episode_reward,
                "running_reward": running_reward
            })
        
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(
                running_reward, t))
            break

if __name__ == '__main__':
    main()
