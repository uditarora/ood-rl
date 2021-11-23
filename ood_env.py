import gym
import random
import numpy as np
from gym.core import ObservationWrapper


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

class OODEnv(ObservationWrapper):
    '''
    A wrapper over a standar gym environment that provides OOD states
    upon executing step with a given probability.
    '''
    def __init__(self, base_env, ood_config):
        super().__init__(base_env)
        self.base_env = base_env
        self.ood_config = ood_config

    def observation(self, observation=None):
        if random.random() < self.ood_config.prob:
            observation = generate_ood(self.ood_config, observation)
        return observation
