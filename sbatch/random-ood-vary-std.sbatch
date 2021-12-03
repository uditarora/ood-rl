#!/bin/bash
#SBATCH --job-name=drl_expt_random_ood_very_std
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./drl_expt_random_ood_very_std_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -c 32

singularity exec --nv --overlay $SCRATCH/public/ood-rl-overlay.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate /ext3/ood-rl-env
cd /scratch/$USER/ood-rl
source setup_env_vars.sh

algos=(PPO A2C SAC DQN)
std_devs=(1 10 50 128 256)

for std_dev in "${std_devs[@]}"
do
    current_gpu=0
    for algo in "${algos[@]}"
    do
        printf "Running for $algo, $std_dev, $current_gpu\n";
        CUDA_VISIBLE_DEVICES=$current_gpu python -u train.py "env=Reacher-v2" "total_timesteps=5000000" "model=$algo" "image_input=True" "use_ood_wrapped_model=False" "eval_outlier_detection=False" "ood_config.use=True" "ood_config.prob=0.10" "ood_config.type=random" "ood_config.random_std=$std_dev" "wandb.project=vary-outlier-std" &
        current_gpu=$((current_gpu+1))
    done
done

wait
"
