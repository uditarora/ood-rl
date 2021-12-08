#!/bin/bash
#SBATCH --job-name=reacher_experiments
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

wandb_project_name=reacher_experiments

# Task ood (Train with ood, without ood_detection)

sleep 10; xvfb-run -a python -u train.py "env=Reacher-v2" "model=PPO" "image_input=True" \
"ood_config.use=True" "ood_config.prob=0.05" "ood_config.type=task" "use_ood_detector_wrapped_model=True" \
"ood_detector.pretrain_timesteps=70000" "ood_detector.fit_outlier_detectors_every_n=2000" \
"ood_config.outlier_env_names:[Ant-v2,Swimmer-v2,Hopper-v2]" \
"wandb.project=$wandb_project_name" &

sleep 10; xvfb-run -a python -u train.py "env=Reacher-v2" "model=PPO" "image_input=True" \
"ood_config.use=True" "ood_config.prob=0.1" "ood_config.type=task" "use_ood_detector_wrapped_model=True" \
"ood_detector.pretrain_timesteps=70000" "ood_detector.fit_outlier_detectors_every_n=2000" \
"ood_config.outlier_env_names:[Ant-v2,Swimmer-v2,Hopper-v2]" \
"wandb.project=$wandb_project_name" &

sleep 10; xvfb-run -a python -u train.py "env=Reacher-v2" "model=PPO" "image_input=True" \
"ood_config.use=True" "ood_config.prob=0.5" "ood_config.type=task" "use_ood_detector_wrapped_model=True" \
"ood_detector.pretrain_timesteps=70000" "ood_detector.fit_outlier_detectors_every_n=2000" \
"ood_config.outlier_env_names:[Ant-v2,Swimmer-v2,Hopper-v2]" \
"wandb.project=$wandb_project_name" &

sleep 10; xvfb-run -a python -u train.py "env=Reacher-v2" "model=PPO" "image_input=True" \
"ood_config.use=True" "ood_config.prob=0.9" "ood_config.type=task" "use_ood_detector_wrapped_model=True" \
"ood_detector.pretrain_timesteps=70000" "ood_detector.fit_outlier_detectors_every_n=2000" \
"ood_config.outlier_env_names:[Ant-v2,Swimmer-v2,Hopper-v2]" \
"wandb.project=$wandb_project_name" &

sleep 10; xvfb-run -a python -u train.py "env=Reacher-v2" "model=PPO" "image_input=True" \
"ood_config.use=True" "ood_config.prob=1.0" "ood_config.type=task" "use_ood_detector_wrapped_model=True" \
"ood_detector.pretrain_timesteps=70000" "ood_detector.fit_outlier_detectors_every_n=2000" \
"ood_config.outlier_env_names:[Ant-v2,Swimmer-v2,Hopper-v2]" \
"wandb.project=$wandb_project_name" &

wait
'
