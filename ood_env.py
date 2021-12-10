import gym
import random
import numpy as np
import wandb
from gym import spaces

# CartPole-v1: Box(4)
# Acrobot-v1: Box(6)
# MountainCar-v0: Box(2)
# Pendulum-v0: Box(3)

# Padded: Box(8)

PADDED_SIZE = 8

class OODEnv(gym.Env):
    '''
    A wrapper over a standard gym environment that provides OOD states
    upon executing step with a given probability.
    '''
    def __init__(self, base_env, ood_config):
        self.base_env = base_env
        self.ood_config = ood_config
        self.padding_size = PADDED_SIZE - self.base_env.observation_space.shape[0]
        low = np.array([-3.4e38 for x in range(PADDED_SIZE)])
        high = np.array([+3.4e38 for x in range(PADDED_SIZE)])
        self.observation_space = spaces.Box(low, high, dtype=self.base_env.observation_space.dtype)

        self.action_space = self.base_env.action_space
        self.reward_range = self.base_env.reward_range
        self.is_current_trajectory_ood = False

        self.trajectory_count = 0
        self.ood_trajectory_count = 0
        self.state_count = 0
        self.ood_state_count = 0

        # Task shift
        self.outlier_envs = []
        if self.ood_config.type == "task":
            self.outlier_envs = [gym.make(outlier_env_name) for outlier_env_name in ood_config.outlier_env_names]

    def step(self, action):
        observation, reward, done, info = self.base_env.step(action)
        return (self.observation(observation), reward, done, info,)

    def reset(self):
        self.trajectory_count += 1
        self.is_current_trajectory_ood = False
        if self.ood_config.use and random.random() < self.ood_config.prob:
            self.is_current_trajectory_ood = True
            self.ood_trajectory_count += 1

        wandb.log({
            "state_count": self.state_count,
            "trajectory_count": self.trajectory_count,
            "ood_trajectory_count": self.ood_trajectory_count,
            "ood_state_count": self.ood_state_count,
            "ood_state_percentage": 0 if self.state_count==0 else (self.ood_state_count / self.state_count),
            "ood_trajectory_percentage": 0 if self.trajectory_count==0 else (self.ood_trajectory_count / self.trajectory_count),
        })

        return self.observation(self.base_env.reset())

    def close(self):
        self.base_env.close()
        [env.close() for env in self.outlier_envs]

    def seed(self, seed=None):
        self.base_env.seed(seed)
        [env.seed(seed) for env in self.outlier_envs]

    def pad(self, observation):
        '''
        Pad observation to final state space size.
        Padding is sampled from Uniform(0,1).
        '''
        padding_size = PADDED_SIZE - observation.shape[0]
        padding = np.random.random(padding_size)
        return np.concatenate((observation, padding), axis=0)

    def observation(self, observation=None):
        self.state_count += 1
        observation = self.pad(observation)

        if self.is_current_trajectory_ood and random.random() < self.ood_config.ood_state_prob:
            self.ood_state_count += 1
            if self.ood_config.type == "background": # BG shift
                observation = self.generate_bg_shift_ood(observation)
            elif self.ood_config.type == "random":# random shift
                observation = self.generate_random_ood(self.ood_config.random_std, observation)
            elif self.ood_config.type == "task": # Task shift
                observation = self.generate_task_shift_ood(self.outlier_envs)
            elif self.ood_config.type == "blackout":
                observation = np.zeros(observation.shape).astype(float)

        return observation

    def generate_random_ood(self, random_std, state):
        return np.clip(state + np.random.normal(0, random_std, state.shape), a_min=0.0, a_max=1.0).astype(float)

    def generate_task_shift_ood(self, outlier_envs):
        observation = random.choice(outlier_envs).reset()
        return self.pad(observation)

    def generate_bg_shift_ood(self, observation):
        '''
        Generate background shift by changing the distribution of padded values from uniform to gaussian.
        '''
        observation[self.observation_space.shape[0]:] = np.random.randn(self.padding_size)
        return observation
