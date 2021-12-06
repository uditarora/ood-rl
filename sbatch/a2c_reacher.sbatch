#!/bin/bash
#SBATCH --job-name=a2c_reacher
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

singularity exec --nv --overlay $SCRATCH/public/ood-rl-overlay.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c '
source /ext3/miniconda3/bin/activate
conda activate /ext3/ood-rl-env

cd /scratch/$USER/ood-rl

source setup_env_vars.sh

set -x

wandb_project_name=a2c_reacher_baseline

python -u train.py "env=Reacher-v2" "hyperparams.A2C.Reacher_v2.n_timesteps=5000000" "model=A2C" "image_input=True" \
"~hyperparams.A2C.Reacher_v2.learning_rate" "wandb.project=$wandb_project_name" &

python -u train.py "env=Reacher-v2" "hyperparams.A2C.Reacher_v2.n_timesteps=5000000" "model=A2C" "image_input=True" \
"wandb.project=$wandb_project_name" &

python -u train.py "env=Reacher-v2" "hyperparams.A2C.Reacher_v2.n_timesteps=5000000" "model=A2C" "image_input=True" \
"wandb.project=$wandb_project_name" "hyperparams.A2C.Reacher_v2.learning_rate=0.001" &

python -u train.py "env=Reacher-v2" "hyperparams.A2C.Reacher_v2.n_timesteps=5000000" "model=A2C" "image_input=True" \
"wandb.project=$wandb_project_name" "hyperparams.A2C.Reacher_v2.learning_rate=0.0001" &

python -u train.py "env=Reacher-v2" "hyperparams.A2C.Reacher_v2.n_timesteps=5000000" "model=A2C" "image_input=True" \
"wandb.project=$wandb_project_name" "hyperparams.A2C.Reacher_v2.learning_rate=0.00001" &

wait
'
