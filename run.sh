#!/bin/bash
#SBATCH --partition=gpu_batch,vci_gpu_priority,gpu_high_mem,gpu_batch_high_mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# SBATCH --array=1-4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name=gears_baseline
#SBATCH --exclude=GPU115A

source $/home/rohankshah/miniconda.sh

conda activate pertsets

# Define test tasks for each fold - each fold holds out a different cell type
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    TEST_TASKS="[replogle:hepg2:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    TEST_TASKS="[replogle:jurkat:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    TEST_TASKS="[replogle:k562:fewshot]"
else
    TEST_TASKS="[replogle:rpe1:fewshot]"
fi

python -m train \
    data.kwargs.data_dir="/large_storage/ctc/userspace/aadduri/datasets/hvg/" \
    data.kwargs.train_task=replogle \
    data.kwargs.test_task="$TEST_TASKS" \
    data.kwargs.embed_key=X_scfound \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.should_yield_control_cells=True \
    data.kwargs.num_workers=16 \
    data.kwargs.batch_col=gem_group \
    data.kwargs.pert_col=gene \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.control_pert=non-targeting \
    data.kwargs.map_controls=True \
    training.max_steps=20000 \
    training.val_freq=5000 \
    training.ckpt_every_n_steps=4000 \
    training.batch_size=256 \
    model.kwargs.cell_set_len=32 \
    model.kwargs.softplus=True \
    wandb.tags="[replogle,replogle4,hvg,fold${SLURM_ARRAY_TASK_ID}]" \
    model=gears \
    output_dir=/home/rohankshah/outputs \
    name="fold${SLURM_ARRAY_TASK_ID}" \
    model.kwargs.hidden_dim=256 # model size 20M recommended for replogle