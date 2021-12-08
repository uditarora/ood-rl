import gym
from gym.core import ObservationWrapper
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from PIL import Image
import torch
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

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
        img = img / 255.0  # Normalize observations
        return img

    def resize_observation(self, observation):
        obs_image = Image.fromarray(observation)
        resized_image = obs_image.resize(size=(self._width, self._height), resample=0)
        im = np.asarray(resized_image)
        return im


class CircularRolloutBuffer(RolloutBuffer):
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


class CircularDictRolloutBuffer(DictRolloutBuffer):
    def add(
            self,
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


class NoopCallback(BaseCallback):
    """
    Base class for callback.

    :param verbose:
    """

    def __init__(self, verbose: int = 0):
        super(BaseCallback, self).__init__()

    def init_callback(self, model) -> None:
        pass

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_) -> None:
        # Those are reference and will be updated automatically
        pass

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        return True

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        return True

    def update_locals(self, locals_) -> None:
        pass

    def update_child_locals(self, locals_) -> None:
        pass
