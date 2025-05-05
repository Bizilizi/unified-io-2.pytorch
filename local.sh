/usr/bin/squashfuse /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/vggsound.squashfs /tmp/vggsound

python process_vggsound.py \
    --tokenizer_path config/tokenizer.model \
    --dataset_path /tmp/zverev/$SLURM_ARRAY_TASK_ID/vggsound \
    --video_csv ../../data/train.csv \
    --output_csv csv/$modality/predictions.csv \
    --page $SLURM_ARRAY_TASK_ID \
    --per_page 1000 \
    --modality $modality \
    --prompt_mode multi \
    --prompt "Do you see {cl} class in this video? Answer only with yes or no."