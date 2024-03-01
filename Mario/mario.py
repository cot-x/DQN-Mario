#!/usr/bin/env python

import gym
import ppaquette_gym_super_mario
import numpy as np


env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
observation = env.reset()

while True:
    #observation, reward, done, info = env.step(env.action_space.sample())
    action = np.random.randint(0, 1+1, 6)
    observation, reward, done, info = env.step(action)

    print("=" * 10)
    print("action=", action)
    print("observation=", observation)
    print("reward=", reward)
    print("done=", done)
    print("info=", info)

    if done:
        env.close()
        break