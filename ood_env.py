import gym
import random
import numpy as np

def generate_ood(ood_config, state=None):
    '''
    Generates OOD state based on the type and base state (optional)

    Returns: New state
    '''
    if ood_config.type == 'random':
        if state is None:
            raise Exception("Can't add random noise without initial state")
        return state + np.random.normal(0, ood_config.random_std, state.shape)
    else: #TODO
        raise NotImplementedError

class OODEnv(gym.env):
    '''
    A wrapper over a standar gym environment that provides OOD states
    upon executing step with a given probability.
    '''
    def __init__(self, base_env, ood_config):
        self.base_env = base_env
        self.ood_config = ood_config
    
    def step(self, action):
        state, reward, done, info = self.base_env.step(action)
        if random.random() < self.ood_config.prob:
            state = generate_ood(self.ood_config, state)
        return state, reward, done, info
