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
from util import NoopCallback

class OodDetectorModel:
    def __init__(self,
                 p,
                 k,
                 distance_threshold_percentile,
                 obs_transform_function,
                 method, # "md" or "msp"
                 num_actions,
                 probs_function=None,
                 prob_threshold=0.8
    ):
        self.p = p # PCA input dimension
        self.k = k # PCA output dimension
        self.num_actions = num_actions
        self.distance_threshold_percentile = distance_threshold_percentile
        self.distance_threshold = 10000000
        self.obs_transform_function = obs_transform_function # A function which returns the activations of the penultimate layer
        self.method = method
        self.pca_model = PCA(n_components=k)

        self.probs_function = probs_function # gets the action probability distribution from the policy
        self.prob_threshold = prob_threshold

        if self.num_actions == 1:
            self.class_means = np.zeros(k)
            self.class_covariances = np.zeros((k, k))
        else:
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

        if self.num_actions == 1:
            self.class_means = np.mean(observations, axis=0)
            self.class_covariances = np.cov(observations, rowvar=False)
        else:
            actions = buffer.actions
            observations_per_action = [[observations[i] for i in range(observations.shape[0]) if actions[i] == action]
                                       for action in range(self.num_actions)]
            for i, action_observations in enumerate(observations_per_action):
                if action_observations:
                    self.class_means[i] = np.mean(action_observations, axis=0)
                    self.class_covariances[i] = np.cov(action_observations, rowvar=False)

        if self.method != "msp":
            self.distance_threshold = np.percentile([np.min(self.calculate_distance(obs)) for obs in observations], self.distance_threshold_percentile)
            print(f"{self.distance_threshold_percentile} percentile distance: {self.distance_threshold}")

    def get_distance(self, obs):
        if self.method == "md":
            obs = self.obs_transform_function(obs).detach().cpu().numpy()
            obs = self.pca_model.transform(obs)
            min_distance = np.min(self.calculate_distance(obs))
            return min_distance
        elif self.method == "msp":
            if self.probs_function is None:
                raise ValueError("Probs_function that gives the probability over actions needs to be specified")
            probs = self.probs_function(obs).detach().cpu().numpy()
            return probs.max()

    def predict_outlier(self, obs):
        if self.method == "md":
            min_distance = self.get_distance(obs)
            return min_distance > self.distance_threshold
        elif self.method == "msp":
            if self.probs_function is None:
                raise ValueError("Probs_function that gives the probability over actions needs to be specified")
            max_prob = self.get_distance(obs)
            return max_prob < self.prob_threshold
        else:
            raise NotImplementedError("Method", self.method, "has not been implemented yet.")

    def calculate_distance(self, obs):
        if self.method == "md":
            if self.num_actions == 1:
                distances = [mahalanobis(obs, self.class_means, VI=self.class_covariances)]
            else:
                distances = [mahalanobis(obs, class_mean, VI=class_cov) for class_mean, class_cov in zip(self.class_means, self.class_covariances)]
            return distances
        else:
            raise NotImplementedError

    def predict_inlier(self, obs):
        return not self.predict_outlier(obs)


class OodDetectorWrappedModel:
    def __init__(self, policy, pretrain_timesteps, fit_outlier_detectors_every_n, k, distance_threshold_percentile, distance_metric="md"):
        self.start_time = time.time()
        self.policy = policy
        self.pretrain_timesteps = pretrain_timesteps
        self.pretraining_done = False
        self.last_ood_train_step = 0
        self.fit_outlier_detectors_every_n = fit_outlier_detectors_every_n
        self.num_actions = self.policy.action_space.n if isinstance(self.policy.action_space, gym.spaces.Discrete) else 1
        self.p = self.policy.policy.mlp_extractor.latent_dim_pi
        self.k = k
        self.distance_threshold_percentile = distance_threshold_percentile
        self.device = self.policy.device
        self.obs_transform_function = (lambda x : self.policy.policy.mlp_extractor(
                self.policy.policy.extract_features(
                    torch.squeeze(torch.from_numpy(x).to(self.device), dim=1)
                )
            )[0]
        )
        self.distance_metric = distance_metric

        self._setup_model()

    def _setup_model(self) -> None:
        rollout_buffer_cls = DictRolloutBuffer if isinstance(self.policy.observation_space, gym.spaces.Dict) else RolloutBuffer
        inlier_buffer_cls = CircularDictRolloutBuffer if isinstance(self.policy.observation_space, gym.spaces.Dict) else CircularRolloutBuffer

        self.inlier_buffer = inlier_buffer_cls(
            self.fit_outlier_detectors_every_n + self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        # This buffer is used to collect data. The data is then moved to the inlier/outlier buffer.
        self.rollout_buffer = rollout_buffer_cls(
            self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        self.policy_buffer = rollout_buffer_cls(
            self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        self.policy.rollout_buffer = self.policy_buffer # TODO: Change to policy_buffer

        self.eval_buffer = rollout_buffer_cls(
            5000,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        self.outlier_detector = OodDetectorModel(
            self.p,
            self.k,
            self.distance_threshold_percentile,
            self.obs_transform_function,
            self.distance_metric,
            self.num_actions
        )
        self.outlier_detector.fit(self.rollout_buffer)

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
            if (not self.pretraining_done) and self.policy.num_timesteps > self.pretrain_timesteps:
                self.pretraining_done = True

            continue_training = self.policy.collect_rollouts(self.policy.env, callback, self.rollout_buffer, n_rollout_steps=self.policy.n_steps)
            if continue_training is False:
                break

            iteration += 1
            self.policy._update_current_progress_remaining(self.policy.num_timesteps, total_timesteps)

            if (self.policy.num_timesteps == 0) or (self.policy.num_timesteps - self.last_ood_train_step >= self.fit_outlier_detectors_every_n):
                self.last_ood_train_step = self.policy.num_timesteps
                self.policy.logger.info(f"{self.policy.num_timesteps}: Fitting OOD detectors.")
                self.outlier_detector.fit(self.inlier_buffer)
                wandb.log({
                    "distance_threshold": self.outlier_detector.distance_threshold,
                    "global_step": self.policy.num_timesteps
                })


            # Figure out of current trajectory is id or ood
            buffer_range = self.rollout_buffer.buffer_size if self.rollout_buffer.full else self.rollout_buffer.pos
            trajectory_begins = [i for i in range(buffer_range) if self.rollout_buffer.episode_starts[i] != 0.0] + [buffer_range]
            trajectory_ranges = [(begin, end, ) for begin, end in zip(trajectory_begins, trajectory_begins[1:])]

            self.policy_buffer.reset()
            for trajectory_range in trajectory_ranges:
                trajectory_inlier = True
                if self.pretraining_done:
                    states_outlier_status = []
                    for i in range(trajectory_range[0], trajectory_range[1]):
                        obs = self.rollout_buffer.observations[i]
                        if self.outlier_detector.predict_outlier(obs): # TODO: Do this entire computation in 1 step
                            states_outlier_status.append(1.0)
                        else:
                            states_outlier_status.append(0.0)

                    if np.mean(states_outlier_status) >= 0.5: # Classify trajectory as outlier if half the states are outliers
                        trajectory_inlier = False

                # Move the rollout to the proper buffer
                if trajectory_inlier:
                    for i in range(trajectory_range[0], trajectory_range[1]):
                        self.policy_buffer.advantages[self.policy_buffer.pos] = self.rollout_buffer.advantages[i]
                        self.policy_buffer.returns[self.policy_buffer.pos] = self.rollout_buffer.returns[i]
                        self.policy_buffer.add(obs=self.rollout_buffer.observations[i],
                                               action=self.rollout_buffer.actions[i],
                                               reward=self.rollout_buffer.rewards[i],
                                               episode_start=self.rollout_buffer.episode_starts[i],
                                               value=torch.from_numpy(self.rollout_buffer.values[i]),
                                               log_prob=torch.from_numpy(self.rollout_buffer.log_probs[i]))
                        self.inlier_buffer.add(obs=self.rollout_buffer.observations[i],
                                               action=self.rollout_buffer.actions[i],
                                               reward=self.rollout_buffer.rewards[i],
                                               episode_start=self.rollout_buffer.episode_starts[i],
                                               value=torch.from_numpy(self.rollout_buffer.values[i]),
                                               log_prob=torch.from_numpy(self.rollout_buffer.log_probs[i]))

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
            if self.policy.rollout_buffer.pos > 0:
                self.policy.train()

        callback.on_training_end()
        return self

    def eval(self, num_rollouts = 100, check_outlier=True):
        self.policy.policy.set_training_mode(False)

        # Mean and covariance estimation
        self.eval_buffer.reset()
        self.policy.collect_rollouts(self.policy.env, NoopCallback(), self.eval_buffer, n_rollout_steps=3000)
        self.outlier_detector.fit(self.eval_buffer)

        # Detection
        self.policy.env.reset()
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
        print(f"Eval mean return: {mean_return}")
        wandb.log({
            "eval_mean_return": mean_return,
            "global_step": self.policy.num_timesteps
        })

    def save(self, model_save_filename):
        self.policy.save(model_save_filename)
