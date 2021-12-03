import gym
import hydra
from ood_env import OODEnv
from util import ImageInputWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env

from ood_model_wrapper import OodDetectorWrappedModel

import wandb

import os

from stable_baselines3 import PPO, SAC, DQN, A2C, DDPG, TD3
from sb3_contrib import TQC, QRDQN, MaskablePPO

from wandb.integration.sb3 import WandbCallback

ALGO_DICT = {
    "PPO": PPO,
    "SAC": SAC,
    "DQN": DQN,
    "A2C": A2C,
    "DDPG": DDPG,
    "TD3": TD3,
    "TQC": TQC,
    "QRDQN": QRDQN,
    "PPO_MASK": MaskablePPO,
    "PPO_Mask": MaskablePPO,
    "MASKABLE_PPO": MaskablePPO
}


@hydra.main(config_path='.', config_name='config')
def main(cfg):

    # initialize wandb
    os.environ["WANDB_API_KEY"] = cfg.wandb.key

    config = {
        "policy_type": cfg.stable_baselines.policy_class,
        "total_timesteps": cfg.total_timesteps,
        "env_name": cfg.env,
    }

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    env = gym.make(cfg.env)
    if cfg.image_input:
        env = ImageInputWrapper(env)
        check_env(env)

    if cfg.ood_config.use:
        env = OODEnv(env, cfg.ood_config)
        check_env(env)

    env = Monitor(env)
    env_name = env.spec.id
    env = DummyVecEnv([lambda: env])

    if cfg.image_input:
        env = VecFrameStack(env, 4)

    params = {
        "policy": cfg.stable_baselines.policy_class,
        "env": env,
        "verbose": cfg.stable_baselines.verbosity,
        "tensorboard_log": f"runs/{run.id}",
    }
    params = {**params, **cfg.hyperparams}

    # Initialize policy
    policy = ALGO_DICT[cfg.model](**params)
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )

    if cfg.use_ood_wrapped_model:
        model = OodDetectorWrappedModel(
            policy,
            cfg.ood_detector.pretrain_timesteps,
            cfg.ood_detector.fit_outlier_detectors_every_n,
            cfg.ood_detector.k,
            cfg.ood_detector.distance_threshold_percentile,
            cfg.ood_detector.distance_metric
        )
        model.learn(total_timesteps=cfg.total_timesteps, callback=wandb_callback)
        model.eval(cfg.num_eval_rollouts, check_outlier=cfg.eval_outlier_detection)
        model_save_filename = f"{cfg.model}_{env_name}_{run.id}"
        model.save(model_save_filename)
        print(f"Model saved to {os.getcwd()}/{model_save_filename}.zip")
    else:
        policy.learn(total_timesteps=cfg.total_timesteps, callback=wandb_callback)
        model_save_filename = f"{cfg.model}_{env_name}_{run.id}"
        policy.save(model_save_filename)
        print(f"Model saved to {os.getcwd()}/{model_save_filename}.zip")

    run.finish()

if __name__ == '__main__':
    main()
