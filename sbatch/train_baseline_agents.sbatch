#!/bin/bash
#SBATCH --job-name=train_baseline_agents
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./train_baseline_agents_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -c 16

singularity exec --nv --overlay $SCRATCH/public/ood-rl-overlay.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate /ext3/ood-rl-env
cd /scratch/$USER/ood-rl
source setup_env_vars.sh

wandb_project_name=train-baseline-agents
algos=(PPO A2C DQN)

env=CartPole-v1
for algo in "${algos[@]}"
do
    printf "Running for $algo, $env\n";
    CUDA_VISIBLE_DEVICES=0 python -u train.py "env=$env" "model=$algo" "image_input=True" "use_ood_detector_wrapped_model=False" "eval_outlier_detection=False" "ood_config.use=False" "wandb.project=$wandb_project_name" &
done


env=Reacher-v2
for algo in "${algos[@]}"
do
    printf "Running for $algo, $env\n";
    CUDA_VISIBLE_DEVICES=1 python -u train.py "env=$env" "model=$algo" "image_input=True" "use_ood_detector_wrapped_model=False" "eval_outlier_detection=False" "ood_config.use=False" "wandb.project=$wandb_project_name" &
done

wait
"
