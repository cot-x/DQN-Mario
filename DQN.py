#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import random
import argparse
from collections import namedtuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym import spaces
from gym.spaces import Box

import cv2
cv2.ocl.setUseOpenCL(False)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[3]:


class NoopEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)


# In[4]:


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action):
        obs, reward,  done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


# In[5]:


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


# In[6]:


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


# In[7]:


class IncScoreEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += 0.1
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# In[8]:


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.width = 128
        self.height = 128
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


# In[9]:


class WrapForPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape # H, W, C -> C, H, W
        self.observation_space = Box(self.observation_space.low[0,0,0], self.observation_space.high[0,0,0],
                                    [obs_shape[2], obs_shape[0], obs_shape[1]], dtype=self.observation_space.dtype)
        
    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# In[10]:


class Util:
    @staticmethod
    def make_env(env_id, seed=None):
        def _thunk():
            env = gym.make(env_id)
            env = NoopEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            if seed is not None:
                env.seed(seed)
            env = EpisodicLifeEnv(env)
            #env = IncScoreEnv(env)
            env = WarpFrame(env)
            env = WrapForPyTorch(env)
            return env
        return _thunk

    @staticmethod
    def make_env_play(env_id, seed=None):
        def _thunk():
            env = gym.make(env_id)
            env = SkipEnv(env, skip=4)
            if seed is not None:
                env.seed(seed)
            return env
        return _thunk


# In[11]:


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.residual = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        return F.relu(self.residual(x) + shortcut, inplace=True)


# In[12]:


class SelfAttention(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        
        # Pointwise Convolution
        self.query_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(input_nc, input_nc, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        
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


# In[13]:


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# In[14]:


class Net(nn.Module):
    def __init__(self, n_out, num_channels=1, conv_dim=32, n_repeat=3):
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
        self.self_attention = SelfAttention(conv_dim)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(conv_dim, n_out, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, x):
        x /= 255.0
        x = self.net_base(x)
        x = self.self_attention(x)
        x = self.conv_out(x)
        x = F.adaptive_avg_pool2d(x, 1) # Global Average Pooling
        x = x.view(x.shape[0], -1)
        return x
    
    def act(self, x):
        output = self(x)
        probs = F.softmax(output, dim=1)
        action = probs.multinomial(num_samples=1)
        return action


# In[15]:


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# In[16]:


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


# In[17]:


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
        
        filename = 'weight.pth'
        if os.path.exists(filename):
            param = torch.load(filename, map_location=self.device)
            self.net.load_state_dict(param)
            print(f'loaded: {filename}')
        
    def weights_init(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight)
            module.bias.data.fill_(0)
        
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
        state_action_values = self.net(state_batch).gather(1, action_batch).squeeze(0)
        
        # 次の状態
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(self.device)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = self.net(non_final_next_states).max(1)[0].detach()
        
        # 報酬から次のQ値を推定
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        
        # Q値の誤差のL1ノルム
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # 誤差逆伝搬によるNNの勾配降下更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def decide_action(self, state):
        self.net.eval()
        return self.net.act(state.to(self.device))


# In[18]:


class Agent:
    def __init__(self, use_cpu, num_actions, lr, gamma, batch_size, mem_capacity):
        self.brain = Brain(use_cpu, lr, gamma, batch_size, mem_capacity, num_actions)
        
    def update_q_functions(self):
        return self.brain.replay()
        
    def get_action(self, state):
        return self.brain.decide_action(state)

    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)


# In[19]:


class Environment:
    def __init__(self, env_name, use_cpu, lr, gamma, batch_size, mem_capacity, num_updates):
        seed = 1
        if seed is not None:
            torch.manual_seed(seed)
            if not use_cpu:
                torch.cuda.manual_seed(seed)
        
        self.env = Util.make_env(env_name, seed)()
        self.env_play = Util.make_env_play(env_name, seed)()
        
        self.num_actions = self.env.action_space.n
        self.agent = Agent(use_cpu, self.num_actions, lr, gamma, batch_size, mem_capacity)
        
        self.num_updates = num_updates
        
    def train(self, weight_dir):
        state = self.env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        rewards = []
        frame_num = 0
        max_frame_num = 0
        done_num = 0
        for i in tqdm(range(self.num_updates)):
            action = self.agent.get_action(state)
            state_next, reward, done, _ = self.env.step(action)
            frame_num += 1
            
            rewards.append(reward)
            
            state_next = torch.from_numpy(state_next).float().unsqueeze(0)
            reward = torch.FloatTensor([reward])
            self.agent.memorize(state, action, state_next, reward)
            loss = self.agent.update_q_functions()
            state = state_next
            
            loss = loss.cpu().item() if loss is not None else None
            
            self.env.render()
            #print(f'action: {action.cpu().item()} reward: {reward.item()}')
            
            if done:
                state = self.env.reset()
                state = torch.from_numpy(state).float().unsqueeze(0)
                done_num += 1
                max_frame_num = max(max_frame_num, frame_num)
                frame_num = 0
                
            if (i+1) % 100 == 0:
                print(f'finished frames: {i+1}, max frames: {max_frame_num}, finished: {done_num}, sum rewards: {sum(rewards):.1f}, loss: {loss}')
                rewards = []
                done_num = 0
                max_frame_num = 0
                
            if (i+1) % 10000 == 0:
                torch.save(self.agent.brain.net.state_dict(), os.path.join(weight_dir, f'weight.{i+1}.pth'))
        torch.save(self.agent.brain.net.state_dict(), os.path.join(weight_dir, 'weight.latest.pth'))
    
    def save_movie(self):
        state = self.env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        frames = [self.env_play.reset()]
        frame_num = 0
        while True:
            action = self.agent.get_action(state)
            state_next, reward, done, _ = self.env.step(action)
            
            frame, _, _, _ = self.env_play.step(action)
            frames.append(frame)
            frame_num += 1
            
            state_next = torch.from_numpy(state_next).float().unsqueeze(0)
            reward = torch.FloatTensor([reward])
            self.agent.memorize(state, action, state_next, reward)
            state = state_next
            
            self.env.render()
            self.env_play.render()
            
            if done:
                #state = self.env.reset()
                #state = torch.from_numpy(state).float().unsqueeze(0)
                #frames = [self.env_play.reset()]
                #frame_num = 0
                break
                
        display_frames_as_gif(frames)


# In[20]:


def display_frames_as_gif(frames):
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    base = min(frames[0].shape[1], frames[0].shape[0])
    plt.figure(figsize=(frames[0].shape[1] / base * 6, frames[0].shape[0] / base * 6), dpi=72, tight_layout=True)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=20)
    anim.save('output.gif', writer=animation.PillowWriter())
    HTML(anim.to_jshtml())


# In[21]:


def main(args):
    hyper_params = {}
    hyper_params['Env Name'] = args.env_name
    hyper_params['Wieght Dir'] = args.weight_dir
    hyper_params['Learning Rate'] = args.lr
    hyper_params['Gamma(Q-Learning)'] = args.gamma
    hyper_params['Batch Size'] = args.batch_size
    hyper_params['Memory Capacity'] = args.mem_capacity
    hyper_params['Num Updates'] = args.num_updates
    
    for key in hyper_params.keys():
        print(f'{key}: {hyper_params[key]}')
    
    env = Environment(args.env_name, args.cpu, args.lr, args.gamma, args.batch_size, args.mem_capacity, args.num_updates)
    env.train(args.weight_dir)
    #env.save_movie()


# In[22]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    #parser.add_argument('--env_name', type=str, default='BeamRider-v0')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--lr', type=float, default=1e-10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mem_capacity', type=int, default=64)
    parser.add_argument('--num_updates', type=int, default=int(1e5))
    parser.add_argument('--cpu', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

