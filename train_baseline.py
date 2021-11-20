import gym
import hydra
import wandb

import os, sys

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

def train_save_model(model, env, cfg):
    model.learn(total_timesteps=cfg.total_timesteps, callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2
    ))
    model.save(cfg.model + "_" + env.unwrapped.spec.id)


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
    params = {
        "policy": cfg.stable_baselines.policy_class,
        "env": env,
        "verbose": cfg.stable_baselines.verbosity,
        "tensorboard_log": f"runs/{run.id}",
    }
    model = ALGO_DICT[cfg.model](**params)
    train_save_model(model, env, cfg)
    run.finish()


if __name__ == '__main__':
    main()
