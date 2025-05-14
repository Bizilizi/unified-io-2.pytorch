#!/bin/bash
#SBATCH --job-name="jid:vla-fn48"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=96GB
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a-sophia.koepke@uni-tuebingen.de
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.out
 
nvidia-smi

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page 1"

# Set the appropriate prompt based on the modality
PROMPT="Do you see or hear \"{cl}\" class in this video? Answer only with yes or no."

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

srun $SRUN_ARGS bash -c " python process_vggsound.py \
    --tokenizer_path config/tokenizer.model \
    --dataset_path /leonardo_work/EUHPC_E03_068/akoepke/vs \
    --video_csv ../../data/test.csv \
    --output_csv csv/$modality/predictions.csv \
    --page \$SLURM_PROCID \
    --per_page 60 \
    --modality $modality \
    --prompt_mode multi \
    --prompt \"$PROMPT\" \
    --batch_size 256 \
    --temperature 0.9 \
    --top_p 0.9
    "