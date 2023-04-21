#!/bin/sh

module --force purge; module load modules/2.0-20220630
module load slurm gcc cmake/3.22.3 nccl cuda/11.4.4 cudnn/8.2.4.15-11.4 openmpi/4.0.7

source ~/miniconda3/bin/activate tf2
echo "Python in use:"
which python3
python3 --version

# GPU node
# srun -t 0-01:00:00 -N 1 -p gpu --constraint=a100 --gpus-per-task=4 --tasks-per-node=1 --exclusive --pty \
#     jupyter lab --no-browser --ip=0.0.0.0

# CPU node
srun -t 0-01:00:00 -N 1 -p genx --tasks-per-node=1 --exclusive --pty \
    jupyter lab --no-browser --ip=0.0.0.0
