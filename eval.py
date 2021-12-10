import gym
import hydra
from ood_env import OODEnv
from util import ImageInputWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.env_checker import check_env
from ood_model_wrapper import OodDetectorWrappedModel, OodDetectorModel
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import torch
import wandb
import os
from util import NoopCallback

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TQC, QRDQN, MaskablePPO

from wandb.integration.sb3 import WandbCallback

ALGO_DICT = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "TQC": TQC,
    "QRDQN": QRDQN,
    "PPO_MASK": MaskablePPO,
    "PPO_Mask": MaskablePPO,
    "MASKABLE_PPO": MaskablePPO
}


def eval(env, policy, cfg, num_actions, num_rollouts=100, check_outlier=True):
    # Mean and covariance estimation
    device = policy.device
    if check_outlier:
        obs_transform_function = (lambda x: policy.policy.mlp_extractor(
            policy.policy.extract_features(
                torch.squeeze(torch.from_numpy(x).to(device), dim=1)
            )
        )[0])
        outlier_detector = OodDetectorModel(
            policy.policy.mlp_extractor.latent_dim_pi,
            cfg.k,
            cfg.distance_threshold_percentile,
            obs_transform_function,
            "md",
            num_actions
        )
        temp_buffer_cls = DictRolloutBuffer if isinstance(policy.observation_space, gym.spaces.Dict) else RolloutBuffer
        eval_buffer = temp_buffer_cls(
            5000,
            policy.observation_space,
            policy.action_space,
            policy.device,
            gamma=policy.gamma,
            gae_lambda=policy.gae_lambda,
            n_envs=policy.n_envs,
        )
        eval_buffer.reset()
        policy.collect_rollouts(policy.env, NoopCallback(), eval_buffer, n_rollout_steps=3000)
        outlier_detector.fit(eval_buffer)

    # Detection
    rollout_returns = [0 for _ in range(num_rollouts)]
    for rollout_idx in range(num_rollouts):
        rollout_return = 0.0
        observation = env.reset()
        action = policy.action_space.sample()
        for timestep in range(policy.n_steps):
            if (not check_outlier) or (not outlier_detector.predict_outlier(observation)):
                action = policy.policy.forward(torch.from_numpy(observation))[0].detach().cpu().numpy()
            observation, reward, done, info = env.step(action)
            rollout_return += reward
            if done:
                break

        env.close()
        rollout_returns[rollout_idx] = rollout_return

    mean_return = np.mean(rollout_returns)
    print(f"Eval mean return: {mean_return}")
    return mean_return


@hydra.main(config_path='.', config_name='eval_config')
def main(cfg):

    # initialize wandb
    os.environ["WANDB_API_KEY"] = cfg.wandb.key

    model_name = cfg.model
    env_name = cfg.env
    env_key = env_name.replace("-", "_")
    model = ALGO_DICT[model_name].load(cfg.model_path, print_system_info=True)

    config = {
        "model": model_name,
        "env_name": env_name,
        "image_input": cfg.image_input,
        "n_frames_stack": cfg.n_frames_stack,
        "use_ood_detector_wrapped_model": cfg.use_ood_detector_wrapped_model,
        "eval_outlier_detection": cfg.eval_outlier_detection,
        "num_eval_rollouts": cfg.num_eval_rollouts,
        "ood_config.use": cfg.ood_config.use,
        "ood_config.prob": cfg.ood_config.prob,
        "ood_config.type": cfg.ood_config.type,
        "ood_config.random_std": cfg.ood_config.random_std,
        "ood_config.outlier_env_names": cfg.ood_config.outlier_env_names,
        "ood_config.ood_state_prob": cfg.ood_config.ood_state_prob,
    }

    try:
        config.update({
            "img_height": cfg.hyperparams[model_name][env_key].img_height,
            "img_width": cfg.hyperparams[model_name][env_key].img_width,
        })
    except:
        pass

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    wandb.config.update({"model_load_path": cfg.model_path})

    env = gym.make(cfg.env)

    if cfg.ood_config.use:
        env = OODEnv(env, cfg.ood_config)
        check_env(env)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    avg_eval_reward = eval(env, model, cfg, model.action_space.n, cfg.num_eval_rollouts, cfg.eval_outlier_detection)
    wandb.config.update({"avg_eval_reward": avg_eval_reward})

    run.finish()

if __name__ == '__main__':
    main()
