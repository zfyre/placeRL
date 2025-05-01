import argparse
import pickle
from collections import namedtuple

import os
# Set OpenMP environment variable to avoid duplicate runtime error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nnet import MyCNN, MyCNNCoarse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import place_env
import torchvision
from place_db import PlaceDB
import time
from tqdm import tqdm
import random
from comp_res import comp_res
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
parser = argparse.ArgumentParser(description='Solve the Chip Placement with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.95)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 42)')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
parser.add_argument('--pnm', type=int, default=128)
parser.add_argument('--benchmark', type=str, default='macro_tiles_10x10', help='Choose the benchmark from adaptec1, macro_tiles_10x10, ariane')
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
        "seed": args.seed
    },
    mode="online"
)

benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = 224
placed_num_macro = args.pnm # Imporatant Parameter: Handles how many macros to be placed in grid before resetting the reward to 0.
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb = placedb, placed_num_macro = placed_num_macro, grid = grid).unwrapped

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

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord',['episode', 'reward'])
print("seed = {}".format(args.seed))
print("lr = {}".format(args.lr))
print("placed_num_macro = {}".format(args.pnm))

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
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn
    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    if placed_num_macro:
        buffer_capacity = 10 * (placed_num_macro)
    else:
        buffer_capacity = 5120
    batch_size = args.batch_size
    print("batch_size = {}".format(batch_size))

    def __init__(self):
        super(PPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        self.actor_net = Actor(cnn = self.cnn, gcn = self.gcn, cnn_coarse = self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn = self.cnn, gcn = self.gcn,  cnn_coarse = None, res_net = self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

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

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + args.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
       
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = args.disable_tqdm):
                self.training_step +=1
                
                action_probs, _, _ = self.actor_net(state[index].to(device))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]                
                critic_net_output = self.critic_net(state[index].to(device))
                advantage = (target_v - critic_net_output).detach()

                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                wandb.log({
                    'action_loss': action_loss.item(),
                    'value_loss': value_loss.item(),
                    'training_step': self.training_step
                })


def save_placement(file_path, node_pos, ratio):
    fwrite = open(file_path, 'w')
    node_place = {}
    for node_name in node_pos:

        x, y,_ , _ = node_pos[node_name]
        x = round(x * ratio + ratio) 
        y = round(y * ratio + ratio)
        node_place[node_name] = (x, y)
    print("len node_place", len(node_place))
    for node_name in placedb.node_info:
        if node_name not in node_place:
            continue
        x, y = node_place[node_name]
        fwrite.write('{}\t{}\t{}\t:\tN /FIXED\n'.format(node_name, x, y))
    print(".pl has been saved to {}.".format(file_path))


def main():

    agent = PPO()
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    
    training_records = []
    running_reward = -1000000
    

    log_file_name = "logs/log_"+ benchmark + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
    if not os.path.exists("logs"):
        os.mkdir("logs")
    fwrite = open(log_file_name, "w")
    load_model_path = None #TODO I can save the ckpt and load it later
   
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
        while done is False:
            state_tmp = state.copy()
            action, action_log_prob = agent.select_action(state)
        
            next_state, reward, done, info = env.step(action)
            assert next_state.shape == (num_state, )
            reward_intrinsic = 0
            if not args.is_test:
                trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
            if not args.is_test and agent.store_transition(trans):                
                assert done == True
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
        if i_epoch % 1000 == 0 and i_epoch != 0 :
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
                    wandb.log({
                        'hpwl': hpwl,
                        'cost': cost,
                        'epoch': i_epoch
                    })
                except:
                    assert False
        
        if args.is_test:
            print("save node_pos")
            hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
            wandb.log({
                'hpwl': hpwl,
                'cost': cost,
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
            wandb.log({
                'reward': running_reward,
                'score': score,
                'epoch': i_epoch,
                'hpwl': hpwl,
                'cost': cost
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
