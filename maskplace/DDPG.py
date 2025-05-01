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
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from nnet import MyCNN, MyCNNCoarse
from PPO2 import Actor, Critic  # Reuse PPO2's network architectures
import place_env
import torchvision
from place_db import PlaceDB
import time
from tqdm import tqdm
import random
from comp_res import comp_res
from torch.utils.tensorboard import SummaryWriter
import wandb

# set device to cpu or cuda
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# Parameters
parser = argparse.ArgumentParser(description='Solve placement with DDPG')
parser.add_argument('--tau', type=float, default=0.001, help='Target network update rate')
# ...rest of parameters from PPO2.py...
parser.add_argument(
    '--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
parser.add_argument('--pnm', type=int, default=128)
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--soft_coefficient', type=float, default = 1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--save_fig', action='store_true', default=True)
parser.add_argument('--wandb_project', type=str, default='placerl', help='Weights & Biases project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity name')
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
        "tau": args.tau
    },
    mode="online"
)

# Environment setup
benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = 224
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb=placedb, placed_num_macro=placed_num_macro, grid=grid).unwrapped

# Define state dimensions
num_emb_state = 64 + 2 + 1  # Position embedding + additional features
num_state = 1 + grid*grid*5 + 2  # Total state dimension

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord', ['episode', 'reward'])

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

    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            mask = x[:, 1+grid*grid*2: 1+grid*grid*3].reshape(-1, grid, grid)
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                        x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                        ),dim= 1).reshape(-1, 3, grid, grid)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
        net_img = x[:, 1+grid*grid: 1+grid*grid*2]
        net_img = net_img + x[:, 1+grid*grid*2: 1+grid*grid*3] * 10
        net_img_min = net_img.min() + args.soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, grid * grid)
        x = torch.where(mask + mask2 >=1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x, cnn_res, gcn_res

class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64 + grid * grid, 512)  # Added action dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn

    def forward(self, x, action, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        x1 = F.relu(self.fc1(torch.cat([self.pos_emb(x[:, 0].long()), action], dim=1)))
        x2 = F.relu(self.fc2(x1))
        value = self.fc3(x2)
        return value

class DDPG():
    buffer_capacity = 10000  # Larger buffer for off-policy learning
    batch_size = args.batch_size
    max_grad_norm = 0.5
    update_frequency = 10  # Update networks more frequently

    def __init__(self):
        super(DDPG, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')  # Updated to use weights instead of pretrained
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        
        # Main networks
        self.actor_net = Actor(cnn=self.cnn, gcn=self.gcn, cnn_coarse=self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn=self.cnn, gcn=self.gcn, cnn_coarse=None, res_net=self.resnet).float().to(device)
        
        # Target networks
        self.actor_target = Actor(cnn=self.cnn, gcn=self.gcn, cnn_coarse=self.cnn_coarse).float().to(device)
        self.critic_target = Critic(cnn=self.cnn, gcn=self.gcn, cnn_coarse=None, res_net=self.resnet).float().to(device)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.critic_target.load_state_dict(self.critic_net.state_dict())
        
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)

    def select_action(self, state, epsilon=0.1):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = torch.randint(0, grid * grid, (1,))
        else:
            action = torch.argmax(action_probs, dim=1)
        
        return action.item(), 0.0  # No log prob needed for deterministic policy

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        next_state = torch.tensor(np.array([t.next_state for t in self.buffer]), dtype=torch.float)
        
        # Sample mini-batch
        for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, True):
            self.training_step += 1
            
            # Get batch data
            state_batch = state[index].to(device)
            next_state_batch = next_state[index].to(device)
            reward_batch = reward[index]
            action_batch = action[index]
            
            # Compute target Q value using target networks
            with torch.no_grad():
                next_action_probs, _, _ = self.actor_target(next_state_batch)
                next_action = torch.argmax(next_action_probs, dim=1).unsqueeze(1)
                target_Q = self.critic_target(next_state_batch, next_action)
                target_Q = reward_batch + args.gamma * target_Q
            
            # Update critic
            current_Q = self.critic_net(state_batch, action_batch)
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # Update actor using deterministic policy gradient
            action_probs, _, _ = self.actor_net(state_batch)
            actor_loss = -self.critic_net(state_batch, action_probs).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Soft update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                
            for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
            
            # Enhanced logging
            wandb.log({
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'training_step': self.training_step,
                'target_q_mean': target_Q.mean().item(),
                'current_q_mean': current_Q.mean().item(),
                'reward_mean': reward_batch.mean().item(),
                'actor_grad_norm': torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), float('inf')).item(),
                'critic_grad_norm': torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), float('inf')).item(),
                'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': self.critic_optimizer.param_groups[0]['lr']
            })
        
        del self.buffer[:]

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.critic_target.load_state_dict(self.critic_net.state_dict())
    
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, running_reward):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("save_models"):
            os.mkdir("save_models")
        torch.save({"actor_net_dict": self.actor_net.state_dict(),
                    "critic_net_dict": self.critic_net.state_dict()},
                    "./save_models/net_dict-{}-{}-".format(benchmark, placed_num_macro)+strftime+"{}".format(int(running_reward))+".pkl")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

def save_placement(file_path, node_pos, ratio):
    with open(file_path, "w") as f:
        for i in range(len(node_pos)):
            f.write(str(node_pos[i][0] * ratio) + " " + str(node_pos[i][1] * ratio) + "\n")

def calculate_metrics(env, canvas):
    # Calculate congestion (using RUDY)
    congestion = np.mean(np.abs(env.rudy))
    
    # Calculate density
    total_area = env.grid * env.grid
    occupied_area = np.sum(canvas > 0)
    density = occupied_area / total_area
    
    # Calculate overlap
    overlap_count = 0
    for node_name in env.node_pos:
        x, y, size_x, size_y = env.node_pos[node_name]
        overlap_count += np.sum(canvas[x:x+size_x, y:y+size_y] > 1.0)
    overlap_percentage = (overlap_count / total_area) * 100 if total_area > 0 else 0
    
    return {
        'congestion': congestion,
        'density': density,
        'overlap_percentage': overlap_percentage
    }

def main():
    agent = DDPG()
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    training_records = []
    running_reward = -1000000
    
    log_file_name = "logs/log_"+ benchmark + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
    if not os.path.exists("logs"):
        os.mkdir("logs")
    fwrite = open(log_file_name, "w")
    load_model_path = None

    if load_model_path:
        agent.load_param(load_model_path)
        
    best_reward = running_reward
    if args.is_test:
        torch.inference_mode()
        
    for i_epoch in range(100000):
        score = 0
        raw_score = 0
        start = time.time()
        state = env.reset()
        
        done = False
        while not done:
            # Decay epsilon over time
            epsilon = max(0.01, 0.1 * (1.0 - i_epoch / 1000))
            action, action_log_prob = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            
            if not args.is_test:
                # Scale reward less aggressively
                trans = Transition(state, action, reward/100.0, action_log_prob, next_state, 0)
                if agent.store_transition(trans):
                    agent.update()
                    
            score += reward
            raw_score += info["raw_reward"]
            state = next_state
            
        end = time.time()

        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print("score = {}, raw_score = {}".format(score, raw_score))

        # Saving every 100 epoch
        if i_epoch % 100 == 0:
            if args.save_fig:
                strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                if not os.path.exists("figures_ckpt"):
                    os.mkdir("figures_ckpt")
                fig_path = "./figures_ckpt/{}{}.png".format(strftime_now,int(raw_score))
                env.save_fig(fig_path)
                wandb.log({"placement_figure_ckpt": wandb.Image(fig_path)})
                print("save_figure: figures_ckpt/{}{}.png".format(strftime_now,int(raw_score)))
                if not os.path.exists("figures"):
                    os.mkdir("figures")
                fig_path = "./figures/{}{}.png".format(strftime_now,int(raw_score))
                env.save_fig(fig_path)
                wandb.log({"placement_figure": wandb.Image(fig_path)})
                print("save_figure: figures/{}{}.png".format(strftime_now,int(raw_score)))

        # Save checkpoint every 1000 epochs
        if i_epoch % 1000 == 0 and i_epoch != 0:
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")
            checkpoint_path = f"./checkpoints/checkpoint_epoch_{i_epoch}_{strftime_now}.pkl"
            agent.save_param(running_reward)
            print(f"Saved checkpoint at epoch {i_epoch} to {checkpoint_path}")

        if running_reward > best_reward * 0.975:
            best_reward = running_reward
            if i_epoch >= 10:
                agent.save_param(running_reward)
                if args.save_fig:
                    strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    if not os.path.exists("figures_ckpt"):
                        os.mkdir("figures_ckpt")
                    fig_path = "./figures_ckpt/{}{}.png".format(strftime_now,int(raw_score))
                    env.save_fig(fig_path)
                    wandb.log({"placement_figure_ckpt": wandb.Image(fig_path)})
                    print("save_figure: figures_ckpt/{}{}.png".format(strftime_now,int(raw_score)))
                    if not os.path.exists("figures"):
                        os.mkdir("figures")
                    fig_path = "./figures/{}{}.png".format(strftime_now,int(raw_score))
                    env.save_fig(fig_path)
                    wandb.log({"placement_figure": wandb.Image(fig_path)})
                    print("save_figure: figures/{}{}.png".format(strftime_now,int(raw_score)))
                try:
                    print("start try")
                    # cost is the routing estimation based on the MST algorithm
                    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
                    print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
                    
                    # Calculate additional metrics
                    canvas = env.state[1:1+env.grid*env.grid].reshape(env.grid, env.grid)
                    metrics = calculate_metrics(env, canvas)
                    
                    wandb.log({
                        'hpwl': hpwl,
                        'cost': cost,
                        'congestion': metrics['congestion'],
                        'density': metrics['density'],
                        'overlap_percentage': metrics['overlap_percentage'],
                        'epoch': i_epoch
                    })
                except:
                    assert False
        
        if args.is_test:
            print("save node_pos")
            hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
            
            # Calculate additional metrics
            canvas = env.state[1:1+env.grid*env.grid].reshape(env.grid, env.grid)
            metrics = calculate_metrics(env, canvas)
            
            wandb.log({
                'hpwl': hpwl,
                'cost': cost,
                'congestion': metrics['congestion'],
                'density': metrics['density'],
                'overlap_percentage': metrics['overlap_percentage'],
                'epoch': i_epoch
            })
            print("time = {}s".format(end-start))
            pl_file_path = "{}-{}-{}.pl".format(benchmark, int(hpwl), time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) ) 
            save_placement(pl_file_path, env.node_pos, env.ratio)
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            pl_path = 'gg_place_new/{}-{}-{}-{}.pl'.format(benchmark, strftime_now, int(hpwl), int(cost))
            fwrite_pl = open(pl_path, 'w')
            for node_name in env.node_pos:
                if node_name == "V":
                    continue
                x, y, size_x, size_y = env.node_pos[node_name]
                x = x * env.ratio + placedb.node_info[node_name]['x'] /2.0
                y = y * env.ratio + placedb.node_info[node_name]['y'] /2.0
                fwrite_pl.write("{}\t{:.4f}\t{:.4f}\n".format(node_name, x, y))
            fwrite_pl.close()
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fig_path = "./figures/{}-{}-{}-{}.png".format(benchmark, strftime_now, int(hpwl), int(cost))
            env.save_fig(fig_path)
            wandb.log({"final_placement_figure": wandb.Image(fig_path)})
        
        training_records.append(TrainingRecord(i_epoch, running_reward))

        if i_epoch % 1 ==0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
            fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
            fwrite.flush()
            hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            
            # Calculate additional metrics
            canvas = env.state[1:1+env.grid*env.grid].reshape(env.grid, env.grid)
            metrics = calculate_metrics(env, canvas)
            
            wandb.log({
                'reward': running_reward,
                'score': score,
                'epoch': i_epoch,
                'hpwl': hpwl,
                'cost': cost,
                'congestion': metrics['congestion'],
                'density': metrics['density'],
                'overlap_percentage': metrics['overlap_percentage']
            })

        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            wandb.finish()
            break

        if i_epoch % 100 == 0:
            if placed_num_macro is None:
                env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))

if __name__ == '__main__':
    main()
