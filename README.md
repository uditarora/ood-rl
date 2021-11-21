# ood-rl
OOD Detection in RL


- The overlayfs can be found at `/home/sj3549/public/ood-rl-overlay.ext3`. It contains the following-
  - A MuJoCo installation and a license key
  - A conda environment `/ext3/ood-rl-env` with the following
    - pytorch
    - mujoco-py
    - stable_baselines3

- To use:
```
srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=32GB --time=1:00:00 --gres=gpu:1 --pty /bin/bash
singularity exec --overlay /home/sj3549/public/ood-rl-overlay.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash
conda activate /ext3/ood-rl-env
```

- For mujoco-py to be able to detect your MuJoCo installation, you need to set the following env var:
```
MUJOCO_PY_MUJOCO_PATH=/ext3/mujoco210
MUJOCO_PY_MJKEY_PATH=/ext3/mujoco210/mjkey.txt
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ext3/mujoco210/bin
```
**Make sure you set these environment variables before running your code.**

Example training command: 
```bash
MUJOCO_PY_MUJOCO_PATH=/ext3/mujoco210 MUJOCO_PY_MJKEY_PATH=/ext3/mujoco210/mjkey.txt LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ext3/mujoco210/bin python train.py 'env=Reacher-v2'
```
