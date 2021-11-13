import gym
import hydra
from stable_baselines3 import PPO, SAC
from wandb.integration.sb3 import WandbCallback


def train_save_model(model, env, cfg):
    model.learn(total_timesteps=cfg.total_timesteps, callback=WandbCallback())
    model.save(cfg.model + "_" + env.unwrapped.spec.id)


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    env = gym.make(cfg.env)
    model = None
    if cfg.model == "PPO":
        model = PPO(cfg.stable_baselines_policy_class, env, verbose=cfg.stable_baselines_verbosity)
    elif cfg.model == "SAC":
        model = SAC(cfg.stable_baselines_policy_class, env, verbose=cfg.stable_baselines_verbosity)

    train_save_model(model, env, cfg)


if __name__ == '__main__':
    main()
