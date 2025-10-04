#!/usr/bin/env python3
"""
Transcript generation pipeline using Whisper for MP4 videos.
Based on whisper_tests.ipynb notebook.
"""
import argparse
import os
import json
import torch
import whisper
import librosa
from pathlib import Path


def load_audio_from_mp4(filepath, sr=16000):
    """
    Load audio from MP4 file using librosa (which uses ffmpeg backend).
    Returns audio array compatible with Whisper.

    Args:
        filepath: Path to MP4 file
        sr: Sample rate (Whisper expects 16kHz)

    Returns:
        Audio array as numpy array
    """
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    return audio


def transcribe_single_file(input_file, output_file=None, model=whisper.load_model("tiny", device="cuda"), device="cuda"):
    """
    Transcribe a single MP4 file with shorter segments for better classification.

    Args:
        input_file: Path to MP4 file
        output_file: Optional path to save transcription JSON file (if None, doesn't save)
        model: Loaded Whisper model (or model name string)
        device: Device to use (cuda or cpu)

    Returns:
        tuple: (Transcription result from Whisper, audio array)
    """
    # Load model if string is passed
    if isinstance(model, str):
        print(f"Loading Whisper model: {model}")
        model = whisper.load_model(model, device=device)

    print(f"Transcribing {input_file}...")

    # Load audio from MP4 using librosa
    audio = load_audio_from_mp4(input_file)

    # Transcribe using Whisper with parameters for shorter, more precise segments
    result = model.transcribe(
        audio,
        word_timestamps=False,
        prepend_punctuations="\"'([{-",
        append_punctuations="\"'.。,!?:)]}、",
        temperature=0.0,               # Deterministic output
        compression_ratio_threshold=2.4,  # Reject segments with low compression
        logprob_threshold=-1.0,        # Reject low probability segments
        no_speech_threshold=0.6,       # Detect silence/no speech
    )

    # Optionally save transcription to JSON
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved to {output_file}")

    return result, audio


def transcribe_directory(input_dir, output_dir, model, device="cuda"):
    """
    Transcribe all MP4 files in a directory.

    Args:
        input_dir: Directory containing MP4 files
        output_dir: Directory to save transcription JSON files
        model: Loaded Whisper model
        device: Device to use (cuda or cpu)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    mp4_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return results

    print(f"Found {len(mp4_files)} MP4 files in {input_dir}")

    for idx, filename in enumerate(mp4_files, 1):
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")

        # Skip if already transcribed
        if os.path.exists(output_path):
            print(f"[{idx}/{len(mp4_files)}] Transcript for {filename} already exists, skipping.")
            continue

        filepath = os.path.join(input_dir, filename)
        print(f"[{idx}/{len(mp4_files)}] Transcribing {filename}...")

        try:
            # Load audio from MP4 using librosa
            audio = load_audio_from_mp4(filepath)

            # Transcribe using Whisper
            result = model.transcribe(audio)
            results[filename] = result

            # Save transcription to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"    ✓ Saved to {output_path}")

        except Exception as e:
            print(f"    ✗ Error processing {filename}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe MP4 videos using Whisper model"
    )

    # Single file or directory mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file",
        help="Single MP4 file to transcribe"
    )
    group.add_argument(
        "--input_dir",
        help="Directory containing MP4 files to transcribe"
    )

    parser.add_argument(
        "--output_file",
        help="Output JSON file (required for --input_file)"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save transcription JSON files (required for --input_dir)"
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple directories (expects input_dir to contain subdirectories)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input_file and not args.output_file:
        parser.error("--output_file is required when using --input_file")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load Whisper model
    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model, device=device)
    print("Model loaded successfully!")

    # Single file mode
    if args.input_file:
        print(f"\nProcessing single file: {args.input_file}")
        transcribe_single_file(args.input_file, args.output_file, model, device)
        print("\n✓ Transcription complete!")
        return

    # Directory mode
    if args.batch:
        # Batch mode: process multiple subdirectories
        print(f"\nBatch mode: processing subdirectories in {args.input_dir}")
        subdirs = [d for d in os.listdir(args.input_dir)
                   if os.path.isdir(os.path.join(args.input_dir, d))]

        for subdir in subdirs:
            input_path = os.path.join(args.input_dir, subdir)
            output_path = os.path.join(args.output_dir, subdir)
            print(f"\n{'='*60}")
            print(f"Processing: {subdir}")
            print(f"{'='*60}")
            transcribe_directory(input_path, output_path, model, device)
    else:
        # Single directory mode
        print(f"\nProcessing directory: {args.input_dir}")
        transcribe_directory(args.input_dir, args.output_dir, model, device)

    print("\n✓ All transcriptions complete!")


if __name__ == "__main__":
    main()
