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

# set device to cpu or cuda
device = torch.device('cuda')

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
parser.add_argument('--save_fig', action='store_true', default=False)
args = parser.parse_args()
writer = SummaryWriter('./tb_log')

# Environment setup
benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = 224
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb=placedb, placed_num_macro=placed_num_macro, grid=grid).unwrapped

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord', ['episode', 'reward'])

class DDPG():
    buffer_capacity = 10000  # Larger buffer for off-policy learning
    batch_size = args.batch_size
    max_grad_norm = 0.5

    def __init__(self):
        super(DDPG, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        
        # Main networks (same architecture as PPO)
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

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state)
        # Get discrete action using categorical distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def update(self):
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
            
            # Compute target Q value using target networks
            with torch.no_grad():
                next_action_probs, _, _ = self.actor_target(next_state_batch)
                target_Q = self.critic_target(next_state_batch)
                target_Q = reward_batch + args.gamma * target_Q
            
            # Update critic
            current_Q = self.critic_net(state_batch)
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update actor using deterministic policy gradient
            action_probs, _, _ = self.actor_net(state_batch)
            actor_loss = -self.critic_net(state_batch).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                
            for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
            
            writer.add_scalar('Loss/critic', critic_loss.item(), self.training_step)
            writer.add_scalar('Loss/actor', actor_loss.item(), self.training_step)
        
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
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            if not args.is_test:
                trans = Transition(state, action, reward/200.0, action_log_prob, next_state, 0)
                if agent.store_transition(trans):
                    agent.update()
                    
            score += reward
            raw_score += info["raw_reward"]
            state = next_state
            
        # ...existing epoch end code (saving, logging etc)...
        end = time.time()

        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print("score = {}, raw_score = {}".format(score, raw_score))

        if running_reward > best_reward * 0.975:
            best_reward = running_reward
            if i_epoch >= 10:
                agent.save_param(running_reward)
                if args.save_fig:
                    strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    if not os.path.exists("figures"):
                        os.mkdir("figures")
                    env.save_fig("./figures/{}{}.png".format(strftime_now,int(raw_score)))
                    print("save_figure: figures/{}{}.png".format(strftime_now,int(raw_score)))
                try:
                    print("start try")
                    # cost is the routing estimation based on the MST algorithm
                    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
                    print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
                except:
                    assert False
        
        if args.is_test:
            print("save node_pos")
            hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
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
            env.save_fig("./figures/{}-{}-{}-{}.png".format(benchmark, strftime_now, int(hpwl), int(cost)))
        
        training_records.append(TrainingRecord(i_epoch, running_reward))
        if i_epoch % 1 ==0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
            fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
            fwrite.flush()
        writer.add_scalar('reward', running_reward, i_epoch)
        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break
        if i_epoch % 100 == 0:
            if placed_num_macro is None:
                env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))

if __name__ == '__main__':
    main()
