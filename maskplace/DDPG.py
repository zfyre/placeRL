import argparse
import pickle
from collections import namedtuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import place_env
import torchvision
from place_db import PlaceDB
import time
from tqdm import tqdm
import random
from comp_res import comp_res
from torch.utils.tensorboard import SummaryWriter   

from maskplace.nnet import MyCNN, MyCNNCoarse

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# Parameters
parser = argparse.ArgumentParser(description='Solve placement with DDPG')
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--buffer_size', type=int, default=1000000)
parser.add_argument('--pnm', type=int, default=128)
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--disable_tqdm', type=int, default=1)
args = parser.parse_args()

# Environment setup
benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = 224
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb=placedb, placed_num_macro=placed_num_macro, grid=grid).unwrapped

class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Actor, self).__init__()
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.merge = nn.Conv2d(2, 1, 1)
        self.fc1 = nn.Linear(grid * grid, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, grid * grid)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, graph=None, cnn_res=None, gcn_res=None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                    x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                    ), dim=1).reshape(-1, 3, grid, grid)
            cnn_res = self.cnn(cnn_input)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
            
        x = cnn_res.reshape(-1, grid * grid)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x, cnn_res, gcn_res

class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Critic, self).__init__()
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.merge = nn.Conv2d(2, 1, 1)
        self.fc1 = nn.Linear(grid * grid * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action, cnn_res=None):
        if not cnn_res:
            cnn_input = state[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            coarse_input = torch.cat((state[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                    state[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                    ), dim=1).reshape(-1, 3, grid, grid)
            cnn_res = self.cnn(cnn_input)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
            
        state = cnn_res.reshape(-1, grid * grid)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self):
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        
        self.actor = Actor(self.cnn, self.gcn, self.cnn_coarse).to(device)
        self.actor_target = Actor(self.cnn, self.gcn, self.cnn_coarse).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.cnn, self.gcn, self.cnn_coarse).to(device)
        self.critic_target = Critic(self.cnn, self.gcn, self.cnn_coarse).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        self.memory = ReplayBuffer(args.buffer_size)
        self.writer = SummaryWriter('./tb_log')
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _, _ = self.actor(state)
        # Get the index of highest probability as discrete action
        action = torch.argmax(action_probs, dim=1)
        return action.item()
        
    def update(self):
        if len(self.memory) < args.batch_size:
            return
            
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.memory.sample(args.batch_size)
            
        # Convert to torch tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        
        # Create one-hot encoding of actions
        action_onehot = F.one_hot(action_batch, num_classes=grid*grid).float()
        
        # Compute target Q value 
        next_action_probs, _, _ = self.actor_target(next_state_batch)
        target_Q = self.critic_target(next_state_batch, next_action_probs)
        target_Q = reward_batch + (1.0 - done_batch) * args.gamma * target_Q
        
        # Update critic
        current_Q = self.critic(state_batch, action_onehot)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_action_probs, _, _ = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actor_action_probs).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()

def main():
    agent = DDPG()
    
    for i_episode in range(1000):
        state = env.reset()
        episode_reward = 0
        
        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward/200.0, next_state, done)
            
            if len(agent.memory) > args.batch_size:
                actor_loss, critic_loss = agent.update()
                agent.writer.add_scalar('Loss/actor', actor_loss, i_episode)
                agent.writer.add_scalar('Loss/critic', critic_loss, i_episode)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        print(f"Episode {i_episode}, Reward: {episode_reward}")
        agent.writer.add_scalar('Reward/episode', episode_reward, i_episode)

if __name__ == '__main__':
    main()
