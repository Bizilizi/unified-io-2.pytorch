#!/usr/bin/env python
import ast
import os
import sys
import traceback
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm, trange
from PIL import Image
import librosa
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uio2.model import UnifiedIOModel
from uio2.preprocessing import UnifiedIOPreprocessor, build_batch
from uio2.runner import TaskRunner
from uio2.config import get_tokenizer
from transformers import GenerationConfig

# Read audio classes from CSV
CLASSES = pd.read_csv("../../data/audio_classes.csv")["display_name"].tolist()


def load_model(args):
    """
    Load the Unified-IO 2 model
    """
    print(f"Loading Unified-IO 2 model from {args.model_path}")

    # Load model from local path or HuggingFace Hub
    if os.path.exists(args.model_path):
        model = UnifiedIOModel.from_pretrained(args.model_path)
    else:
        # Try to load from HuggingFace Hub
        os.makedirs(args.cache_dir, exist_ok=True)
        model = UnifiedIOModel.from_pretrained(
            args.model_path, cache_dir=args.cache_dir
        )

    # Load preprocessor from local path or HuggingFace Hub
    if os.path.exists(args.preprocessor_path):
        preprocessor = UnifiedIOPreprocessor.from_pretrained(
            args.preprocessor_path, tokenizer=args.tokenizer_path
        )
    else:
        # Try to load from HuggingFace Hub
        os.makedirs(args.cache_dir, exist_ok=True)
        preprocessor = UnifiedIOPreprocessor.from_pretrained(
            args.preprocessor_path,
            tokenizer=args.tokenizer_path,
            cache_dir=args.cache_dir,
        )

    # Convert to specified dtype for efficiency
    if args.dtype == "bfloat16":
        model.to_dtype(
            torch.bfloat16, vit_dtype=torch.float32, vqgan_dtype=torch.float32
        )
    model.to(args.device)
    model.eval()

    return model, preprocessor


def get_video_list(csv_path):
    """
    Reads video IDs from a CSV file.
    Assumes CSV with two columns: video_id and label. If the video_id does not
    end with '.mp4', it appends '.mp4'.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return []

    df = pd.read_csv(csv_path, names=["video_id", "label"], header=None)
    video_ids = df["video_id"].tolist()
    video_ids = [vid if vid.endswith(".mp4") else vid + ".mp4" for vid in video_ids]
    return video_ids


def write_predictions_csv(predictions, responses, output_csv):
    """
    Writes the predictions dictionary to a CSV file.
    The CSV will have columns: video_id, suggestions, and response.
    """
    df_table = {
        i: {
            "video_id": vid,
            "suggestions": list(predictions[vid]),
            "response": responses[vid],
        }
        for i, vid in enumerate(predictions.keys())
    }
    df = pd.DataFrame.from_dict(df_table, orient="index")
    df.to_csv(output_csv, index=False)
    print(f"Predictions CSV saved to {output_csv}")


@torch.inference_mode()
def process_video(
    model,
    preprocessor,
    dataset_path,
    video_id,
    temperature=0.2,
    top_p=0.9,
    max_gen_len=512,
    modality="av",
    prompt="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
    prompt_mode="single",
    batch_size=4,
):
    """
    Process a single video file and detect classes using Unified-IO 2 model.
    Returns a list of detected classes and the model's response.
    """
    # Set up paths based on modality
    video_path = os.path.join(dataset_path, "video", video_id)
    audio_path = os.path.join(dataset_path, "audio", video_id.replace(".mp4", ".wav"))

    detected = []
    response = ""

    try:
        # Process with either single prompt for all classes or individual prompts per class
        if prompt_mode in ["single", "gpt"]:
            # Format prompt with all classes
            prompt_text = prompt.format(cl=", ".join(CLASSES))

            # Create preprocessed inputs based on modality
            if modality == "av":
                preprocessed = preprocessor(
                    text_inputs=prompt_text,
                    video_inputs=video_path,
                    use_video_audio=True,
                    target_modality="text",
                )
            elif modality == "a":
                preprocessed = preprocessor(
                    text_inputs=prompt_text,
                    audio_inputs=audio_path,
                    target_modality="text",
                )
            elif modality == "v":
                preprocessed = preprocessor(
                    text_inputs=prompt_text,
                    video_inputs=video_path,
                    use_video_audio=False,
                    target_modality="text",
                )

            # Create batch and generate response
            batch = build_batch([preprocessed], device=model.device)

            # Configure generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=1,
            )

            # Generate response
            tokens = model.generate(
                batch=batch,
                generation_config=generation_config,
                modality="text",
            )

            # Decode response
            response = preprocessor.tokenizer.decode(tokens[0])

            # Check for detected classes
            for cl in CLASSES:
                if cl.lower() in response.lower():
                    detected.append(cl)

        elif prompt_mode == "multi":
            all_responses = []

            for i in trange(
                0, len(CLASSES), batch_size, desc="Processing classes", leave=False
            ):
                batch_classes = CLASSES[i : i + batch_size]
                batch_tokens = []

                for cl in batch_classes:
                    # Format prompt for this specific class
                    prompt_text = prompt.format(cl=cl)

                    # Create preprocessed inputs based on modality
                    if modality == "av":
                        preprocessed = preprocessor(
                            text_inputs=prompt_text,
                            video_inputs=video_path,
                            target_modality="text",
                        )
                    elif modality == "a":
                        preprocessed = preprocessor(
                            text_inputs=prompt_text,
                            audio_inputs=audio_path,
                            target_modality="text",
                        )
                    elif modality == "v":
                        preprocessed = preprocessor(
                            text_inputs=prompt_text,
                            video_inputs=video_path,
                            use_video_audio=False,
                            target_modality="text",
                        )
                    batch_tokens.append(preprocessed)

                # Create batch and generate response
                batch = build_batch(batch_tokens, device=model.device)

                # Configure generation parameters
                generation_config = GenerationConfig(
                    max_new_tokens=16,  # Shorter for yes/no answers
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=1,
                )

                # Generate response
                tokens = model.generate(
                    batch=batch,
                    generation_config=generation_config,
                    modality="text",
                )

                # Decode response
                for cl, class_response in zip(batch_classes, tokens):
                    class_response = preprocessor.tokenizer.decode(class_response)

                    if "yes" in class_response.lower():
                        detected.append(cl)

                    all_responses.append(f"{cl}: {class_response}")

            response = ",".join(all_responses)

        else:
            raise ValueError(
                f"Invalid prompt mode: {prompt_mode}. Supported modes: 'single', 'multi'"
            )

    except Exception as e:
        print(f"Error processing video {video_id}: {traceback.format_exc()}")
        return [], f"Error: {str(e)}"

    # Return the unique set of detected classes and response
    response = response.replace("\n", "\\n")
    return list(set(detected)), response


def main():
    parser = argparse.ArgumentParser(
        description="Process videos and generate predictions using Unified-IO 2 model"
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="allenai/uio2-xxl",
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--preprocessor_path",
        type=str,
        default="allenai/uio2-preprocessor",
        help="Path to preprocessor or HuggingFace preprocessor name",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Data type for model weights and inference",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for inference"
    )

    # Dataset and output configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/storage/slurm/zverev/datasets/vggsound",
        help="Path to the directory containing video files",
    )
    parser.add_argument(
        "--video_csv",
        type=str,
        default="../data/train.csv",
        help="CSV file that contains the list of video IDs",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="../data/unified_io2_predictions.csv",
        help="Output CSV file for writing predictions",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for generation"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top P for generation")
    parser.add_argument(
        "--max_output_tokens", type=int, default=512, help="Maximum output tokens"
    )

    # Processing configuration
    parser.add_argument("--page", type=int, default=0, help="Page number to process")
    parser.add_argument(
        "--per_page",
        type=int,
        default=1000,
        help="Number of videos to process per page",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="av",
        choices=["av", "a", "v"],
        help="Modality to use: 'av' for audio-visual, 'a' for audio-only, 'v' for video-only",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
        help="Prompt template for class detection",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="single",
        choices=["single", "multi", "gpt"],
        help="Prompt mode: 'single' for one prompt with all classes, 'multi' for individual prompts per class, 'gpt' for GPT assisted`",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for model and preprocessor",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./config/tokenizer.model",
        help="Path to tokenizer file (required for the UIO2 preprocessor)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing videos",
    )

    args = parser.parse_args()
    
    # Initialize model and preprocessor
    model, preprocessor = load_model(args)

    # Get list of videos to process
    video_list = get_video_list(args.video_csv)
    if not video_list:
        print("No videos found to process.")
        return

    # Process only a subset (page) of videos
    page_videos = video_list[
        args.page * args.per_page : (args.page + 1) * args.per_page
    ]

    # Update output CSV filename to include page and prompt mode
    args.output_csv = args.output_csv.replace(
        ".csv", f"_{args.prompt_mode}_page_{args.page}.csv"
    )
    
    if os.path.exists(args.output_csv):
        already_processed = pd.read_csv(args.output_csv)
        already_processed_ids = set(already_processed["video_id"].tolist())
        page_videos = [vid for vid in page_videos if vid not in already_processed_ids]
        print(f"Skipping {len(already_processed)} videos that were already processed.")
        
        predictions = {}
        responses = {}
        
        for _, row in already_processed.iterrows():
            predictions[row["video_id"]] = ast.literal_eval(row["suggestions"])
            responses[row["video_id"]] = row["response"]
    else:
        predictions = {}
        responses = {}
    
    print(f"Processing page {args.page} of {len(page_videos)} videos")
    # set model modalities
    if args.modality == "av":
        model.set_modalities(input_modalities=["text", "image_history", "audio"], target_modalities=["text"])
    elif args.modality == "a":
        model.set_modalities(input_modalities=["text", "audio"], target_modalities=["text"])
    elif args.modality == "v":
        model.set_modalities(input_modalities=["text", "image_history"], target_modalities=["text"])
        
    # Process each video
    for video_id in tqdm(page_videos, desc="Processing Videos"):
        detected_classes, response = process_video(
            model=model,
            preprocessor=preprocessor,
            dataset_path=args.dataset_path,
            video_id=video_id,
            temperature=args.temperature,
            top_p=args.top_p,
            max_gen_len=args.max_output_tokens,
            modality=args.modality,
            prompt=args.prompt,
            prompt_mode=args.prompt_mode,
            batch_size=args.batch_size,
        )

        predictions[video_id] = detected_classes
        responses[video_id] = response

        # Write predictions to CSV periodically
        write_predictions_csv(predictions, responses, args.output_csv)

    print(f"Completed processing {len(page_videos)} videos.")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
