#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


# In[2]:


env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)


# In[4]:


state = env.reset()
for i in range(10000):
    env.render()
    action, _ = model.predict(state, deterministic=True)
    state, rewards, done, info = env.step(action)
    if done:
        break


# In[5]:


model.learn(total_timesteps=10000)


# In[7]:


state = env.reset()
for i in range(10000):
    env.render()
    action, _ = model.predict(state, deterministic=True)
    state, rewards, done, info = env.step(action)
    if done:
        break


# In[8]:


env.close()


# In[ ]:




