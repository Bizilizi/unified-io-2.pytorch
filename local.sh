python process_vggsound.py \
    --tokenizer_path config/tokenizer.model \
    --dataset_path $MCMLSCRATCH/datasets/vggsound_test \
    --video_csv ../../data/test_sample.csv \
    --output_csv csv/av/predictions.csv \
    --page 1 \
    --per_page 100 \
    --modality av \
    --prompt_mode gpt \
    --prompt "What actions are being performed in this video, explain all sounds and actions in the video? Please provide a short answer."
