import gym
import random
import numpy as np
import wandb

from util import ImageInputWrapper


class OODEnv(gym.Env):
    '''
    A wrapper over a standard gym environment that provides OOD states
    upon executing step with a given probability.
    '''
    def __init__(self, base_env, ood_config, image_input=False, width=300, height=300, n_frames_stack=4):
        self.base_env = base_env
        self.ood_config = ood_config
        self.observation_space = self.base_env.observation_space
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

        # BG shift
        self.bg_shift_envs = []
        if self.ood_config.type == "background":
            self.bg_shift_envs = [gym.make(base_env.unwrapped.spec.id) for i in range(4)]
            print("Background shift environments initiated. Count: " + str(len(self.bg_shift_envs)))
            for i in range(4):
                m = self.bg_shift_envs[i].model
                if base_env.unwrapped.spec.id != 'Reacher-v2':
                    raise Exception("Background changes are currently implemented only for the environment 'Reacher-v2'")
                m.mat_texid[m.geom_matid[m.geom_name2id("ground")]] = i + 1 #0 stands for no texture according to our reacher.xml
        
        if image_input:
            if self.ood_config.type == "task":
                self.outlier_envs = [ImageInputWrapper(env, resize=True, height=height, width=width) for env in
                                 self.outlier_envs]
            if self.ood_config.type == "background":
                self.bg_shift_envs = [ImageInputWrapper(env, resize=True, height=height, width=width) for env in
                                      self.bg_shift_envs]

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
        [env.close() for env in self.bg_shift_envs]

    def seed(self, seed=None):
        self.base_env.seed(seed)
        [env.seed(seed) for env in self.outlier_envs]
        [env.seed(seed) for env in self.bg_shift_envs]

    def observation(self, observation=None):
        self.state_count += 1
        if self.is_current_trajectory_ood and random.random() < 0.6: # TODO: make this a config
            self.ood_state_count += 1
            if self.ood_config.type == "background": # BG shift
                if len(self.bg_shift_envs) > 0:
                    if random.random() < self.ood_config.prob:
                        #shift the .png file used as texture for the "ground" geom in reacher.xml of gym
                        i =  random.randrange(0, len(self.bg_shift_envs))
                        #self.bg_shift_envs[i].reset()
                        #CHANGE THIS LATER
                        #random action: env.action_space.sample()
                        #observation, reward, done, info = self.bg_shift_envs[i].step(self.bg_shift_envs[i].action_space.sample())
                        #observation = self.bg_shift_envs[i].observation()
                        self.bg_shift_envs[i].render(mode='rgb_array')
                else:
                    raise Exception("No background shift environments were initiated successfully.")
            elif self.ood_config.type == "random":# random shift
                observation = self.generate_random_ood(self.ood_config.random_std, observation)
            else: # Task shift
                observation = self.generate_task_shift_ood(self.outlier_envs)

        return observation

    def generate_random_ood(self, random_std, state):
        return np.clip(state + np.random.normal(0, random_std, state.shape), a_min=0.0, a_max=1.0).astype(float)

    def generate_task_shift_ood(self, outlier_envs):
        return random.choice(outlier_envs).reset()
