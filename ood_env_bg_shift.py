import gym
import random
import numpy as np
from gym.core import ObservationWrapper


def generate_ood(ood_config, outlier_envs, state=None):
    '''
    Generates OOD state based on the type and base state (optional)

    Returns: New state
    '''
    if ood_config.type == 'random':
        if state is None:
            raise Exception("Can't add random noise without initial state.")
        return np.clip(state + np.random.normal(0, ood_config.random_std, state.shape), a_min=0, a_max=255).astype(int)
    elif ood_config.type == 'task':
        return random.choice(outlier_envs).render(mode='rgb_array')
    else:  # TODO
        pass

class OODEnv(ObservationWrapper):
    '''
    A wrapper over a standar gym environment that provides OOD states
    upon executing step with a given probability.
    '''
    def __init__(self, base_env, ood_config):
        super().__init__(base_env)
        self.base_env = base_env
        self.ood_config = ood_config
        if self.ood_config.type != "background":
            self.outlier_envs = [gym.make(outlier_env_name) for outlier_env_name in ood_config.outlier_env_names] # TODO: Handle exception
        #self.training = True
        self.bg_shift_envs = []
        if self.ood_config.type == "background":
            self.bg_shift_envs = [gym.make(base_env.unwrapped.spec.id) for i in range(4)]
            print("Background shift environments initiated. Count: " + str(len(self.bg_shift_envs)))
            for i in range(4):
                m = self.bg_shift_envs[i].model
                #The following line works only for 'Reacher-v2'
                if base_env.unwrapped.spec.id != 'Reacher-v2':
                    raise Exception("Background changes are currently implemented only for the environment 'Reacher-v2'")
                m.mat_texid[m.geom_matid[m.geom_name2id("ground")]] = i + 1 #0 stands for no texture according to our reacher.xml

    def observation(self, observation=None):
        if self.ood_config.use:
            if self.ood_config.type == "background":
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
            else:#random noise
                if random.random() < self.ood_config.prob:
                    observation = generate_ood(self.ood_config, self.outlier_envs, observation)
        return observation

    # def eval(self):
    #     self.training = False
    #
    # def train(self):
    #     self.training = True

    def disable_ood(self):
        self.ood_config.use = False

    def enable_ood(self):
        self.ood_config.use = True
