python process_vggsound.py \
    --tokenizer_path config/tokenizer.model \
    --dataset_path ~/vggsound \
    --video_csv ../../data/train.csv \
    --output_csv csv/av/predictions.csv \
    --page 1 \
    --per_page 10 \
    --modality av \
    --prompt_mode single \
    --prompt "Classes: {cl}. From the given list of classes, which ones do you see or hear in this video? Answer using the exact names of the classes, separated by commas."
#    --prompt_mode multi \
#    --prompt "Do you see {cl} class in this video? Answer only with yes or no."