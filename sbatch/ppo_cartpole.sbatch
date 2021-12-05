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

python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
'~learning_rate' '~n_steps' '~batch_size' '~n_epochs' '~gamma' '~gae_lambda' '~clip_range' '~ent_coef' '~vf_coef' \
'~max_grad_norm' &

python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' &

python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
 'learning_rate=0.01' &

python -u train.py 'env=CartPole-v1' 'hyperparams.PPO.CartPole_v1.n_timesteps=5000000' 'model=PPO' 'image_input=True' \
 'learning_rate=0.0001' &

wait
"
