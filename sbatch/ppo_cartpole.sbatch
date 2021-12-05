#!/bin/bash
#SBATCH --job-name=ppo_cartpole
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./ppo_cartpole_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/public/ood-rl-overlay.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate /ext3/ood-rl-env

cd /scratch/$USER/ood-rl

source setup_env_vars.sh

xvfb-run python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
'~hyperparams.PPO.CartPole_v1.learning_rate' '~hyperparams.PPO.CartPole_v1.n_steps' \
'~hyperparams.PPO.CartPole_v1.batch_size' '~hyperparams.PPO.CartPole_v1.n_epochs' '~hyperparams.PPO.CartPole_v1.gamma' \
'~hyperparams.PPO.CartPole_v1.gae_lambda' '~hyperparams.PPO.CartPole_v1.clip_range' '~hyperparams.PPO.CartPole_v1.ent_coef' \
'~hyperparams.PPO.CartPole_v1.vf_coef' \
'~hyperparams.PPO.CartPole_v1.max_grad_norm' 'wandb.project=$wandb_project_name' &

xvfb-run python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
 'wandb.project=$wandb_project_name' &

xvfb-run python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
 'hyperparams.PPO.CartPole_v1.learning_rate=0.01' 'wandb.project=$wandb_project_name' &

xvfb-run python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
 'hyperparams.PPO.CartPole_v1.learning_rate=0.0001' 'wandb.project=$wandb_project_name' &

wait
"
