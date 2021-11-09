import gym
from stable_baselines3 import PPO
import os, sys



env = gym.make("Ant-v2")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
