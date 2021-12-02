import torch
import wandb
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import numpy as np
import gym
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import time
from stable_baselines3.common.utils import safe_mean
from util import CircularRolloutBuffer, CircularDictRolloutBuffer

class OodDetectorModel:
    def __init__(self,
                 p,
                 k,
                 distance_threshold, # TODO: Set threshold based on chi-square distribution
                 obs_transform_function,
                 distance_metric, # "md" or "robust-md"
                 num_actions
    ):
        self.p = p # PCA input dimension
        self.k = k # PCA output dimension
        self.num_actions = num_actions
        self.distance_threshold = distance_threshold
        self.obs_transform_function = obs_transform_function # A function which returns the activations of the penultimate layer
        self.distance_metric = distance_metric
        self.pca_model = PCA(n_components=k)

        self.class_means = [np.zeros(k) for _ in range(num_actions)]
        self.class_covariances = [np.zeros((k, k)) for _ in range(num_actions)]

    def fit(self, buffer:DictRolloutBuffer):
        buffer_range = buffer.buffer_size if buffer.full else buffer.pos
        if buffer_range == 0:
            self.pca_model.fit(
                self.obs_transform_function(buffer.observations).detach().cpu().numpy()
            ) # Fit PCA to zeros to prevent errors
            return
        observations = buffer.observations[0:buffer_range]
        self.pca_model.fit(
            self.obs_transform_function(observations).detach().cpu().numpy()
        )
        observations = self.pca_model.transform(self.obs_transform_function(observations).detach().cpu().numpy())
        actions = buffer.actions
        observations_per_action = [[observations[i] for i in range(observations.shape[0]) if actions[i]==action] for action in range(self.num_actions)]

        for i, action_observations in enumerate(observations_per_action):
            if action_observations:
                self.class_means[i] = np.mean(action_observations, axis=0)
                self.class_covariances[i] = np.cov(action_observations, rowvar=False)

    def predict_outlier(self, obs):
        obs = self.obs_transform_function(obs).detach().cpu().numpy()
        obs = self.pca_model.transform(obs)
        distance = self.calculate_distance(obs)
        return distance > self.distance_threshold

    def calculate_distance(self, obs):
        if self.distance_metric == "md":
            distances = [mahalanobis(obs, class_mean, VI=class_cov) for class_mean, class_cov in zip(self.class_means, self.class_covariances)]
            min_distance = np.min(distances)
            return min_distance
        else:
            raise NotImplementedError

    def predict_inlier(self, obs):
        return not self.predict_outlier(obs)


class OodDetectorWrappedModel:
    def __init__(self, policy, pretrain_timesteps, fit_outlier_detectors_every_n, k, distance_threshold, distance_metric="md"):
        self.start_time = time.time()
        self.policy = policy
        self.pretrain_timesteps = pretrain_timesteps
        self.pretraining_done = False
        self.last_ood_train_step = 0
        self.fit_outlier_detectors_every_n = fit_outlier_detectors_every_n
        self.num_actions = self.policy.action_space.n
        self.p = self.policy.policy.mlp_extractor.latent_dim_pi
        self.k = k
        self.distance_threshold = distance_threshold
        self.obs_transform_function = (lambda x : self.policy.policy.mlp_extractor(
                self.policy.policy.extract_features(
                    torch.squeeze(torch.from_numpy(x), dim=1)
                )
            )[0]
        )
        self.distance_metric = distance_metric

        self._setup_model()

    def _setup_model(self) -> None:
        temp_buffer_cls = DictRolloutBuffer if isinstance(self.policy.observation_space, gym.spaces.Dict) else RolloutBuffer
        buffer_cls = CircularDictRolloutBuffer if isinstance(self.policy.observation_space, gym.spaces.Dict) else CircularRolloutBuffer

        self.inlier_buffer = buffer_cls(
            self.fit_outlier_detectors_every_n + self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        # This buffer is used to collect data. The data is then moved to the inlier/outlier buffer.
        self.temp_buffer = temp_buffer_cls(
            self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        self.policy.rollout_buffer = self.temp_buffer  # Set policy's buffer to temp buffer

        self.outlier_detector = OodDetectorModel(
            self.p,
            self.k,
            self.distance_threshold,
            self.obs_transform_function,
            self.distance_metric,
            self.num_actions
        )

    def learn(self,
            total_timesteps,
            callback = None,
            log_interval = 1,
            eval_env = None,
            eval_freq = -1,
            n_eval_episodes = 5,
            tb_log_name = "OnPolicyAlgorithm",
            eval_log_path = None,
            reset_num_timesteps = True,
    ):
        iteration = 0 # One iteration is basically one episode
        self.policy.policy.set_training_mode(True)

        total_timesteps, callback = self.policy._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.policy.num_timesteps < total_timesteps:

            continue_training = self.policy.collect_rollouts(self.policy.env, callback, self.temp_buffer, n_rollout_steps=self.policy.n_steps)
            if continue_training is False:
                break

            iteration += 1
            self.policy._update_current_progress_remaining(self.policy.num_timesteps, total_timesteps)

            if (self.policy.num_timesteps == 0) or (self.policy.num_timesteps - self.last_ood_train_step >= self.fit_outlier_detectors_every_n):
                self.policy.logger.info(f"{self.policy.num_timesteps}: Fitting OOD detectors.")
                self.outlier_detector.fit(self.inlier_buffer)
                self.last_ood_train_step = self.policy.num_timesteps

            # Figure out of current trajectory is id or ood
            buffer_range = self.temp_buffer.buffer_size if self.temp_buffer.full else self.temp_buffer.pos
            trajectory_begins = [i for i in range(buffer_range) if self.temp_buffer.episode_starts[i] != 0.0] + [buffer_range]
            trajectory_ranges = [(begin, end, ) for begin, end in zip(trajectory_begins, trajectory_begins[1:])]

            for trajectory_range in trajectory_ranges:
                trajectory_inlier = True
                if self.policy.num_timesteps > self.pretrain_timesteps and (not self.pretraining_done):
                    for i in range(trajectory_range[0], trajectory_range[1]):
                        obs = self.temp_buffer.observations[i]
                        if self.outlier_detector.predict_outlier(obs):
                            trajectory_inlier = False

                # Move the rollout to the proper buffer
                if trajectory_inlier:
                    for i in range(trajectory_range[0], trajectory_range[1]):
                        self.inlier_buffer.add(obs = self.temp_buffer.observations[i],
                                               action = self.temp_buffer.actions[i],
                                               reward = self.temp_buffer.rewards[i],
                                               episode_start = self.temp_buffer.episode_starts[i],
                                               value = torch.from_numpy(self.temp_buffer.values[i]),
                                               log_prob = torch.from_numpy(self.temp_buffer.log_probs[i]))

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.policy.num_timesteps / (time.time() - self.start_time))
                self.policy.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.policy.ep_info_buffer) > 0 and len(self.policy.ep_info_buffer[0]) > 0:
                    self.policy.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.policy.ep_info_buffer]))
                    self.policy.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.policy.ep_info_buffer]))
                self.policy.logger.record("time/fps", fps)
                self.policy.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.policy.logger.record("time/total_timesteps", self.policy.num_timesteps, exclude="tensorboard")
                self.policy.logger.dump(step=self.policy.num_timesteps)

            self.policy.rollout_buffer.full = True
            self.policy.train()
            self.temp_buffer.reset()
        callback.on_training_end()

        return self

    def eval(self, num_rollouts = 100, check_outlier=True):
        self.policy.env.reset()
        self.policy.policy.set_training_mode(False)

        rollout_returns = [0 for _ in range(num_rollouts)]
        for rollout_idx in range(num_rollouts):
            rollout_return = 0.0
            observation = self.policy.env.reset()
            action = self.policy.action_space.sample()
            for timestep in range(self.policy.n_steps):
                if (not check_outlier) or (not self.outlier_detector.predict_outlier(observation)):
                    action = self.policy.policy.forward(torch.from_numpy(observation))[0].detach().cpu().numpy()
                observation, reward, done, info = self.policy.env.step(action)
                rollout_return  += reward
                if done:
                    break

            self.policy.env.close()
            rollout_returns[rollout_idx] = rollout_return

        mean_return = np.mean(rollout_returns)
        wandb.log({
            "eval_mean_return": mean_return,
            "global_step": self.policy.num_timesteps
        })

    def save(self, model_save_filename):
        self.policy.save(model_save_filename)
