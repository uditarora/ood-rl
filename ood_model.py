import wandb
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import numpy as np
import gym
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis


class OodDetectorModel:
    def __init__(self,
                 p,
                 k,
                 distance_threshold,
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
        self.class_covariances = [np.zeros(k, k) for _ in range(num_actions)]

    def fit(self, buffer:DictRolloutBuffer):
        observations = self.pca_model.transform(self.obs_transform_function(buffer.observations))
        actions = buffer.observations

        self.class_means = [np.mean([observations[i] for i in range(observations.shape[0]) if actions[i]==action], axis=0) for action in range(self.num_actions)] # TODO: check
        self.class_covariances = [np.cov([observations[i] for i in range(observations.shape[0]) if actions[i] == action], axis=0) for action in range(self.num_actions)]

    def predict_outlier(self, obs):
        obs = self.obs_transform_function(obs)
        obs = self.pca_model.transform(obs)
        distance = self.calculate_distance(obs)
        return distance > self.distance_threshold

    def calculate_distance(self, obs):
        obs = self.obs_transform_function(obs)
        distances = [mahalanobis(obs, class_mean, VI=class_cov) for class_mean, class_cov in zip(self.class_means, self.class_covariances)]
        min_distance = np.min(distances)
        return min_distance

    def predict_inlier(self, obs):
        return not self.predict_outlier(obs)


class OodDetectorWrappedModel:
    def __init__(self, policy, pretrain_timesteps, fit_outlier_detectors_every_n, p, k, distance_threshold, distance_metric="md"):
        self.policy = policy
        self.pretrain_timesteps = pretrain_timesteps
        self.pretraining_done = False
        self.last_ood_train_step = 0
        self.fit_outlier_detectors_every_n = fit_outlier_detectors_every_n
        self.num_actions = self.policy.action_space.n # TODO: check
        self.p = p
        self.k = k
        self.distance_threshold = distance_threshold
        self.obs_transform_function = self.policy.policy.extract_features # TODO: Check
        self.distance_metric = distance_metric

        self._setup_model()

    def _setup_model(self) -> None:
        buffer_cls = DictRolloutBuffer if isinstance(self.policy.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.inlier_buffer = buffer_cls(
            self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        self.policy.rollout_buffer = self.inlier_buffer # Set policy's buffer to inlier buffer

        self.outlier_buffer = buffer_cls(
            self.policy.n_steps,
            self.policy.observation_space,
            self.policy.action_space,
            self.policy.device,
            gamma=self.policy.gamma,
            gae_lambda=self.policy.gae_lambda,
            n_envs=self.policy.n_envs,
        )

        # This buffer is used to initially collect data. The data is then moved to the inlier/outlier buffer.
        self.temp_buffer = buffer_cls(
            self.policy.n_steps,
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
            self.distance_threshold,
            self.obs_transform_function,
            self.distance_metric,
            self.num_actions
        )
        self.inlier_detector = OodDetectorModel(
            self.p,
            self.k,
            self.distance_threshold,
            self.obs_transform_function,
            self.distance_metric,
            self.num_actions
        )

    def learn(self, total_timesteps: int) -> "OodDetectorWrappedModel":
        iteration = 0 # One iteration is basically one episode

        while self.policy.num_timesteps < total_timesteps:
            continue_training = self.policy.collect_rollouts(self.policy.env, self.policy.callback, self.temp_buffer, n_rollout_steps=self.policy.n_steps)
            if continue_training is False:
                break

            iteration += 1

            # Figure out of current trajectory is id or ood
            # TODO: Simplify logic
            trajectory_inlier = True
            buffer_range = self.temp_buffer.buffer_size if self.temp_buffer.full else self.temp_buffer.pos
            if self.policy.num_timesteps > self.pretrain_timesteps and (not self.pretraining_done):
                for i in range(buffer_range):
                    obs = self.temp_buffer.observations[i]
                    if self.outlier_detector.predict_outlier(obs):
                        trajectory_inlier = False
                    else:
                        if self.inlier_detector.predict_inlier(obs):
                            trajectory_inlier = True
                        else:
                            trajectory_inlier = False

            # Move the rollout to the proper buffer
            destination_buffer = self.inlier_buffer if trajectory_inlier else self.outlier_buffer
            for i in range(buffer_range):
                destination_buffer.add(obs = self.temp_buffer.observations[i],
                                       action = self.temp_buffer.actions[i],
                                       reward = self.temp_buffer.rewards[i],
                                       episode_start = self.temp_buffer.episode_starts[i],
                                       value = self.temp_buffer.values[i],
                                       log_prob = self.temp_buffer.log_probs[i])

            self.temp_buffer.reset()
            self.policy.train()

            if self.policy.num_timesteps - self.last_ood_train_step >= self.fit_outlier_detectors_every_n:
                self.outlier_detector.fit(self.outlier_buffer)
                self.inlier_detector.fit(self.inlier_buffer)
                self.last_ood_train_step = self.policy.num_timesteps

        return self

    def eval(self, num_rollouts = 100, check_outlier=True):
        self.policy.env.reset()
        self.policy.policy.set_training_mode(True) # TODO: Check

        rollout_returns = [0 for _ in range(num_rollouts)]
        for rollout_idx in range(num_rollouts):
            rollout_return = 0.0
            observation = self.policy.env.reset()
            action = self.policy.action_space.sample()
            for timestep in range(self.policy.n_steps):
                if (not check_outlier) or (not self.outlier_detector.predict_outlier(self.obs_transform_function(observation))):
                    action = self.policy(observation)
                observation, reward, done, info = self.policy.env.step(action)
                rollout_return  += reward
                if done:
                    break

            self.policy.env.close()
            rollout_returns[rollout_idx] = rollout_return

        mean_return = np.mean(rollout_returns)
        wandb.log({
            "eval_mean_return": mean_return,
            "timestep": self.policy.num_timesteps
        })

        # TODO: train ood switch
        # Switch: Use OODWrappedModel
        # TODO: eval ood switch
        # Switch: OodDetectionDuringTrain
        # Switch: OodDetectionDuringEval
