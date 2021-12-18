import gym
import hydra
from joblib import Parallel, delayed
from eval import eval, ALGO_DICT
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from ood_env import OODEnv
from copy import deepcopy
import seaborn as sns
import wandb
import sys, os
import pandas as pd


def generate_save_plot(eval_data):
    eval_data = pd.DataFrame(eval_data)
    # AUROC plot
    g = sns.catplot(
        data=eval_data, kind="bar",
        x="ood_prob", y="auroc", hue="ood_type",
        ci="sd", alpha=1.0, height=6
    )
    g.set_axis_labels("OOD %", "OOD Detector AU-ROC")
    g.legend.set_title("")
    g.fig.savefig("auroc_plot.png", dpi=300)

    # Reward plot
    g = sns.catplot(
        data=eval_data, kind="bar",
        x="ood_prob", y="mean_return", hue="ood_type",
        ci="sd", alpha=1.0, height=6
    )
    g.set_axis_labels("OOD %", "Reward")
    g.legend.set_title("")
    g.fig.savefig("reward_plot.png", dpi=300)


def run_expt(cfg, model, ood_prob, ood_type):
    env_name = cfg.env
    env = gym.make(env_name)

    ood_config = deepcopy(cfg.ood_config)
    ood_config.prob = ood_prob
    ood_config.type = ood_type

    if cfg.ood_config.use:
        env = OODEnv(env, ood_config)
        check_env(env)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    eval_metrics = eval(
        env=env,
        policy=model,
        cfg=cfg,
        num_actions=model.action_space.n,
        num_rollouts=cfg.num_eval_rollouts,
        check_outlier=True
    )

    return {
        "ood_prob": ood_config.prob,
        "ood_type": ood_config.type,
        "auroc": eval_metrics["auroc"],
        "mean_return": eval_metrics["mean_return"],
    }


@hydra.main(config_path='.', config_name='eval_config')
def main(cfg):
    # initialize wandb
    os.environ["WANDB_API_KEY"] = cfg.wandb.key

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config={},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    ood_probs = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
    shifts = ["task", "background", "random", "blackout"]

    model_name = cfg.model
    model = ALGO_DICT[model_name].load(cfg.model_path, print_system_info=True)

    expt_param = {
        "cfg": cfg,
        "model": model,
        "ood_prob": None,
        "ood_type": None,
    }

    expt_params = []
    for ood_prob in ood_probs:
        for shift in shifts:
            expt_param_ = expt_param.copy()
            expt_param_["ood_prob"] = ood_prob
            expt_param_["ood_type"] = shift
            expt_params.append(expt_param_)

    # Run experiments
    # eval_data = run_expt(**expt_params[0])
    eval_data = Parallel(n_jobs=10)(delayed(run_expt)(**expt_param) for expt_param in expt_params)

    # Plot
    generate_save_plot(eval_data)


if __name__ == '__main__':
    main()
