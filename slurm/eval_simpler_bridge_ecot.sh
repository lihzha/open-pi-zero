#!/bin/bash

#SBATCH --job-name=eval-bridge
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh  # Load Conda environment script
conda activate openpi  # Activate the environment

# better to run jobs for each task
TASKS=(
    "widowx_carrot_on_plate"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_stack_cube"
)

N_EVAL_EPISODE=60   # octo simpler runs 3 seeds with 24 configs each, here we run 10 seeds

for TASK in ${TASKS[@]}; do

    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python \
        scripts/run.py \
        --config-name=bridge_ecot \
        --config-path=../config/eval \
        device=cuda:0 \
        seed=42 \
        n_eval_episode=$N_EVAL_EPISODE \
        n_video=0 \
        env.task=$TASK \
        pretrained_checkpoint="ecot-openvla-7b-bridge"
done
