#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import random
import argparse
import numpy as np
import math
from collections import namedtuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

import gym
from gym import spaces
from gym.spaces import Box

import ppaquette_gym_super_mario


# In[ ]:


class NoopEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        num_actions = env.action_space.shape[0]
        self.noop_action = np.zeros(num_actions)
        
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1) # [1, noop_max+1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)


# In[ ]:


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# In[ ]:


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# In[ ]:


class TransformsFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.width = 128
        self.height = 128
        self.observation_space = Box(low=0, high=1, shape=(self.height, self.width, 1), dtype=np.float64)
        
    def observation(self, frame):
        frame = frame.transpose(2, 0, 1) # H, W, C -> C, H, W
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
        frame_tarns = transforms.functional.resize(frame, (self.width, self.height))
        frame_tarns = transforms.functional.rgb_to_grayscale(frame_tarns)
        return (frame_tarns, frame)


# In[ ]:


class SelfAttention(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        
        # Pointwise Convolution
        self.query_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(input_nc, input_nc, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        proj_query = self.query_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3]).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        s = torch.bmm(proj_query, proj_key) # バッチ毎の行列乗算
        attention_map_T = self.softmax(s)
        
        proj_value = self.value_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        o = torch.bmm(proj_value, attention_map_T)
        
        o = o.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        out = x + self.gamma * o
        
        return out#, attention_map_T.permute(0, 2, 1)


# In[ ]:


class LambdaLayer(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_depth=16, heads=4, dim_u=1, dim_recept=23):
        super().__init__()
        self.heads = heads
        
        dim_out = dim_in if dim_out is None else dim_out
        
        self.dim_depth = dim_depth
        self.dim_u = dim_u
        assert (dim_out % heads) == 0, 'must divide by heads for multi-head query'
        dim_v = dim_out // heads
        self.dim_v = dim_v
        
        self.queries = nn.Sequential(
            nn.Conv2d(dim_in, dim_depth * heads, 1, bias=False),
            nn.BatchNorm2d(dim_depth * heads)
        )
        self.keys = nn.Conv2d(dim_in, dim_depth * dim_u, 1, bias=False)
        self.values = nn.Sequential(
            nn.Conv2d(dim_in, dim_v * dim_u, 1, bias=False),
            nn.BatchNorm2d(dim_v * dim_u)
        )
        self.softmax = nn.Softmax(dim=-1)
        
        self.local_context = True if dim_recept > 0 else False
        if self.local_context:
            assert (dim_recept % 2) == 1, 'receptive kernel size must be odd'
            r = dim_recept
            self.embedding = nn.Parameter(torch.randn([dim_depth, dim_u, 1, r, r]))
            self.padding = (r - 1) // 2
        else:
            self.embedding = nn.Parameter(torch.randn([dim_depth, dim_u]))
        
        self.gamma = nn.Parameter(torch.randn(1))

    def forward(self, x):
        b, c, h, w = x.size()

        queries = self.queries(x).view(b, self.heads, self.dim_depth, h * w)
        softmax = self.softmax(self.keys(x).view(b, self.dim_depth, self.dim_u, h * w))
        values = self.values(x).view(b, self.dim_v, self.dim_u, h * w)

        lambda_c = torch.einsum('bkun,bvun->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(b, self.dim_u, -1, h, w)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(b, self.dim_depth, self.dim_v, h * w)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.reshape(b, -1, h, w)
        
        out = out + self.gamma * out
        
        return out


# In[ ]:


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# In[ ]:


class Net(nn.Module):
    def __init__(self, n_out, num_channels=1, conv_dim=64, n_repeat=1):
        super().__init__()
        
        model = [
            nn.Conv2d(num_channels, conv_dim, kernel_size=3, stride=2, padding=1), #128x128 -> 64x64
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        ]

        for _ in range(n_repeat): # -> 64x64 /= 2 ** n -> 8x8
            model += [
                nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(conv_dim),
                nn.ReLU(inplace=True)
            ]
            conv_dim *= 2
            
        self.net_base = nn.Sequential(*model)
        #self.self_attention = SelfAttention(conv_dim)
        self.self_attention = LambdaLayer(conv_dim)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(conv_dim, n_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, x):
        x = self.net_base(x)
        x = self.self_attention(x)
        x = self.conv_out(x)
        x = F.adaptive_max_pool2d(x, 1) # Global Average Pooling
        x = x.view(x.shape[0], -1)
        return x
    
    def act(self, x):
        output = self(x)
        probs = F.softmax(output, dim=1)
        action = probs.multinomial(num_samples=1)
        return action


# In[ ]:


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# In[ ]:


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0
        
    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# In[ ]:


class Brain:
    def __init__(self, use_cpu, lr, gamma, batch_size, mem_capacity, num_actions):
        use_cuda = torch.cuda.is_available() if not use_cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print(f'Use Device: {self.device}')

        self.net = Net(num_actions).to(self.device)
        self.net.apply(self.weights_init)
        self.memory = ReplayMemory(mem_capacity)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        
        self.load_state()
        
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
    
    def save_state(self, weight_dir, epoch):
        self.net.cpu()
        torch.save(self.net.state_dict(), os.path.join(weight_dir, f'weight.{epoch}.pth'))
        self.net.to(self.device)
    
    def load_state(self):
        if os.path.exists('weight.pth'):
            self.net.load_state_dict(torch.load('weight.pth', map_location=self.device))
            print('Loaded network state.')
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.net.train()
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        
        # 現在の状態
        state_action_values = self.net(state_batch).gather(1, action_batch)
        
        # 次の状態
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(self.device)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = self.net(non_final_next_states).max(dim=1)[0].detach()
        
        # 報酬から次のQ値を推定
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        expected_state_action_values = expected_state_action_values.unsqueeze(1).repeat(1, state_action_values.size(1))
        
        # Q値の誤差のL1ノルム
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # 誤差逆伝搬によるNNの勾配降下更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def array_action(self, x, num_actions):
        num_buttons = int(math.log2(num_actions))
        b = format(x, f'0{num_buttons}b') # 2進数
        return np.array([int(b[i]) for i in range(num_buttons)])
    
    def decide_action(self, state):
        self.net.eval()
        action = self.net.act(state.to(self.device))[0][0].cpu().item()
        return self.array_action(action, self.num_actions)


# In[ ]:


class Agent:
    def __init__(self, use_cpu, num_actions, lr, gamma, batch_size, mem_capacity):
        self.brain = Brain(use_cpu, lr, gamma, batch_size, mem_capacity, num_actions)
        
    def update_q_functions(self):
        return self.brain.replay()
        
    def get_action(self, state):
        return self.brain.decide_action(state)

    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)


# In[ ]:


class Environment:
    def __init__(self, args):
        self.args = args
        
        self.env = self.make_env(self.args.env_name)
        self.env_play = None
        
        self.num_buttons = self.env.action_space.shape[0]
        self.num_actions = 2 ** self.num_buttons
        self.agent = Agent(args.cpu, self.num_actions, args.lr, args.gamma, args.batch_size, args.mem_capacity)
        
    def close(self):
        self.env.close()
        if self.env_play:
            self.env_play.close()
    
    def make_env(self, env_name):
        env = gym.make(env_name)
        env = NoopEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = TransformsFrame(env)
        return env
    
    def make_env_play(self, env_name):
        env = gym.make(env_name)
        env = SkipEnv(env, skip=4)
        env = TransformsFrame(env)
        return env
    
    def train(self):
        hyper_params = {}
        hyper_params['Env Name'] = self.args.env_name
        hyper_params['Wieght Dir'] = self.args.weight_dir
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params['Gamma(Q-Learning)'] = self.args.gamma
        hyper_params['Batch Size'] = self.args.batch_size
        hyper_params['Memory Capacity'] = self.args.mem_capacity
        hyper_params['Num Updates'] = self.args.num_updates
        
        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        (state, _) = self.env.reset()
        done_num = 0
        episode_reward= 0
        rewards = []
        for i in tqdm(range(self.args.num_updates)):
            action = self.agent.get_action(state)
            (state_next, _), reward, done, _ = self.env.step(action)
            #print(f'action: {action} reward: {reward}')
            
            episode_reward += reward
            
            reward = torch.FloatTensor([reward])
            action = torch.from_numpy(action).unsqueeze(0)
            self.agent.memorize(state, action, state_next, reward)
            loss = self.agent.update_q_functions()
            state = state_next
            
            loss = loss.cpu().item() if loss is not None else None
            #experiment.log_metric('Loss', loss)
            
            if done:
                (state, _) = self.env.reset()
                rewards += [episode_reward]
                episode_reward = 0
                done_num += 1
                print(f'finished frames {i+1}, {done_num} times finished, reward {rewards[-1]:.1f}')
            
            if (i+1) % 1000 == 0:
                self.agent.brain.save_state(self.args.weight_dir, i+1)
    
    def save_movie(self):
        self.env_play = self.make_env_play(self.args.env_name)
        (state, frame) = self.env_play.reset()
        frames = [state[0]]
        
        while True:
            action = self.agent.get_action(state)
            (state, frame), _, done, _ = self.env_play.step(action)
            frames += [frame[0]]
            if done:
                break
        
        self.env_play.close()
        self.save_frames_as_gif(frames)
        
    def save_frames_as_gif(self, frames):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        
        ToPIL = transforms.ToPILImage()
        
        base = min(frames[0].shape[1], frames[0].shape[2])
        plt.figure(figsize=(frames[0].shape[1] / base * 6, frames[0].shape[2] / base * 6), dpi=72, tight_layout=True)
        patch = plt.imshow(ToPIL(frames[0]))
        plt.axis('off')

        def animate(i):
            patch.set_data(ToPIL(frames[i]))

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
        anim.save('output.gif', writer=animation.PillowWriter(fps=100))


# In[ ]:


def main(args):
    env = Environment(args)
    try:
        if args.savemovie:
            env.save_movie()
            return

        env.train()
    finally:
        env.close()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='ppaquette/SuperMarioBros-1-1-v0')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mem_capacity', type=int, default=10000)
    parser.add_argument('--num_updates', type=int, default=100000)
    parser.add_argument('--savemovie', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

