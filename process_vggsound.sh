#!/bin/sh
#SBATCH --job-name="unified-io-2"
#SBATCH --array=0-16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4,mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --error=./logs/slurm-%A_%a.out
 
nvidia-smi

# Mount squashfs files
cleanup () {
    fusermount -u /tmp/zverev/$SLURM_JOB_ID/vggsound
    rmdir /tmp/zverev/$SLURM_JOB_ID/vggsound
}

trap cleanup EXIT

echo "Mounting VGGsound"
mkdir -p /tmp/zverev/$SLURM_JOB_ID/vggsound
/usr/bin/squashfuse /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/vggsound.squashfs /tmp/zverev/$SLURM_JOB_ID/vggsound

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page $SLURM_ARRAY_TASK_ID"

# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you hear in this audio? Answer using the exact names of the classes, separated by commas."
else
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you see or hear in this video? Answer using the exact names of the classes, separated by commas."
fi

# Run the script on each node, assigning each task to a different GPU
srun --exclusive --ntasks=1 python process_vggsound.py \
    --tokenizer_path config/tokenizer.model \
    --dataset_path /tmp/zverev/$SLURM_JOB_ID/vggsound \
    --video_csv ../../data/train.csv \
    --output_csv csv/$modality/predictions.csv \
    --page $SLURM_ARRAY_TASK_ID \
    --per_page 1000 \
    --modality $modality \
    --prompt_mode single \
    --prompt "$PROMPT"
