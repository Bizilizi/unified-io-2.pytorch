#!/bin/sh
#SBATCH --job-name="unified-io-2"
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4,mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --error=./logs/slurm-%A_%a.out
 
nvidia-smi

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page $SLURM_ARRAY_TASK_ID"

# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="Do you hear \"{cl}\" class in this audio? Answer only with yes or no."
else
    PROMPT="Do you see or hear \"{cl}\" class in this video? Answer only with yes or no."
fi

# Run the script on each node, assigning each task to a different GPU
srun --exclusive --ntasks=1 python process_vggsound.py \
    --tokenizer_path config/tokenizer.model \
    --dataset_path $MCMLSCRATCH/datasets/vggsound_test \
    --video_csv ../../data/test.csv \
    --output_csv csv/$modality/predictions.csv \
    --page $SLURM_ARRAY_TASK_ID \
    --per_page 100 \
    --modality $modality \
    --prompt_mode multi \
    --prompt "$PROMPT" \
    --batch_size 309 \
    --temperature 0.9 \
    --top_p 0.9
