import gym
import hydra
from ood_env import OODEnv
from util import ImageInputWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_checker import check_env

import wandb

import os

from stable_baselines3 import PPO, SAC, DQN, A2C, DDPG, TD3

from wandb.integration.sb3 import WandbCallback

ALGO_DICT = {
    "PPO": PPO,
    "SAC": SAC,
    "DQN": DQN,
    "A2C": A2C,
    "DDPG": DDPG,
    "TD3": TD3,
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
    env = DummyVecEnv([lambda: env])

    if cfg.image_input:
        env = VecFrameStack(env, 4)

    params = {
        "policy": cfg.stable_baselines.policy_class,
        "env": env,
        "verbose": cfg.stable_baselines.verbosity,
        "tensorboard_log": f"runs/{run.id}",
    }

    # Initialize model
    model = ALGO_DICT[cfg.model](**params)

    # Train model
    model.learn(total_timesteps=cfg.total_timesteps, callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2
    ))

    # Save final model
    model.save(f"{cfg.model}_{env.unwrapped.spec.id}_{run.id}")

    run.finish()


if __name__ == '__main__':
    main()
