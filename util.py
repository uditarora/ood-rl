import gym
from gym.core import ObservationWrapper
import numpy as np
from collections import deque
from PIL import Image

class ImageInputWrapper(ObservationWrapper):
    resize = True
    _height = 128
    _width = 128

    def __init__(self, env, resize=True, height=128, width=128):
        super().__init__(env)
        self.resize = resize
        self._height = height
        self._width = width

        self._max_episode_steps = env._max_episode_steps

        env.reset()
        shp = self.observation().shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shp,
            dtype=np.uint8)

    def observation(self, observation=None):
        img = self.env.render(mode='rgb_array')
        if self.resize:
            img = self.resize_observation(img)
        return img

    def resize_observation(self, observation):
        obs_image = Image.fromarray(observation)
        resized_image = obs_image.resize(size=(self._width, self._height), resample=0)
        im = np.asarray(resized_image)
        return im
