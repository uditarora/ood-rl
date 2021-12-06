import gym
import random
import numpy as np
from gym.core import ObservationWrapper
from util import ImageInputWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


def generate_ood(ood_config, outlier_envs, state=None):
    '''
    Generates OOD state based on the type and base state (optional)

    Returns: New state
    '''
    if ood_config.type == 'random':
        if state is None:
            raise Exception("Can't add random noise without initial state")
        return np.clip(state + np.random.normal(0, ood_config.random_std, state.shape), a_min=0.0, a_max=1.0).astype(int)
    elif ood_config.type == 'task':
        return random.choice(outlier_envs).reset()
    else:  # TODO: Implement background shift
        pass

class OODEnv(ObservationWrapper):
    '''
    A wrapper over a standard gym environment that provides OOD states
    upon executing step with a given probability.
    '''
    def __init__(self, base_env, ood_config, image_input=False, width=300, height=300, n_frames_stack=4):
        super().__init__(base_env)
        self.base_env = base_env
        self.ood_config = ood_config
        self.outlier_envs = [gym.make(outlier_env_name) for outlier_env_name in ood_config.outlier_env_names]
        if(image_input):
            self.outlier_envs = [ImageInputWrapper(env, resize=True, height=height, width=width) for env in self.outlier_envs]

    def observation(self, observation=None):
        if random.random() < self.ood_config.prob:
            observation = generate_ood(self.ood_config, self.outlier_envs, observation)
        return observation
