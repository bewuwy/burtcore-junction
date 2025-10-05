#!/usr/bin/env python3
"""
Batch process videos in a folder and save results to JSON files.

This script processes all video and audio files in a specified directory using the evaluate()
function from extreme.py and saves each result to a corresponding JSON file.

Usage:
    python backend/batch_process_videos.py --input-dir /path/to/media --output-dir /path/to/results
    python backend/batch_process_videos.py -i ./media -o ./results
    python backend/batch_process_videos.py -i ./media  # Uses ./results by default
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import traceback
from datetime import datetime
import json

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# Import the evaluate function from extreme.py
from backend.extreme import evaluate


# Supported video and audio extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpeg', '.mpg'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.wma', '.opus'}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def transform_to_web_ui_format(result: dict, filename: str) -> dict:
    """
    Transform the evaluate() result to match the web-ui expected format.
    
    Args:
        result: The result dict from evaluate() function
        filename: The original filename
        
    Returns:
        Dict formatted for web-ui consumption
    """
    stats = result.get("statistics", {})
    segments = result.get("segments", [])
    
    # Calculate counts for summary
    total_count = stats.get("total_segments", 0)
    extremist_count = stats.get("extremist_segments", 0)
    toxic_count = stats.get("toxic_segments", 0)
    
    # Determine if content is extremist
    is_extremist_content = stats.get("is_extremist_content", False)
    
    # Build summary message similar to endpoints.py
    if is_extremist_content is not None:
        extremist_ratio = stats.get("extremist_ratio", 0)
        avg_extremist_prob = stats.get("avg_extremist_probability", 0)
        
        if is_extremist_content:
            summary = f"⚠️ EXTREMIST CONTENT DETECTED: {extremist_count}/{total_count} segments ({extremist_ratio*100:.1f}%). Avg probability: {avg_extremist_prob*100:.1f}%"
        else:
            summary = f"✓ Non-extremist content. {extremist_count}/{total_count} extremist segments detected ({extremist_ratio*100:.1f}%)."
    else:
        # Fallback to toxicity-based summary
        avg_toxicity = stats.get("avg_toxicity", 0)
        max_toxicity = stats.get("max_toxicity", 0)
        toxic_percentage = (toxic_count / total_count * 100) if total_count > 0 else 0
        
        if toxic_count == 0:
            summary = "No toxic content detected."
        elif toxic_count == 1:
            summary = f"1 toxic segment detected ({toxic_percentage:.1f}% of content). Max toxicity: {max_toxicity*100:.1f}%"
        else:
            summary = f"{toxic_count} toxic segments detected ({toxic_percentage:.1f}% of content). Average toxicity: {avg_toxicity*100:.1f}%, Max: {max_toxicity*100:.1f}%"
    
    # Return format matching web-ui expectations
    return {
        "success": True,
        "result": summary,
        "isExtremist": is_extremist_content,
        "segments": segments,
        "full_text_classification": result.get("classification"),
        "statistics": stats,
        "filename": filename
    }


def find_media_files(input_dir: str) -> List[Path]:
    """
    Find all video and audio files in the input directory.
    
    Args:
        input_dir: Path to the directory containing videos/audio files
        
    Returns:
        List of Path objects for media files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    media_files = []
    
    # Search for video and audio files
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            media_files.append(file_path)
    
    return sorted(media_files)


def process_videos_batch(input_dir: str, output_dir: str = "./results", skip_existing: bool = False):
    """
    Process all videos and audio files in a directory and save results to JSON files.
    
    Args:
        input_dir: Directory containing video/audio files
        output_dir: Directory to save JSON results (default: ./results)
        skip_existing: If True, skip files that already have result files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all media files
    print(f"Scanning for video and audio files in: {input_dir}")
    media_files = find_media_files(input_dir)
    
    if not media_files:
        print(f"No video or audio files found in {input_dir}")
        print(f"Supported video formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        print(f"Supported audio formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return
    
    print(f"Found {len(media_files)} file(s) to process")
    print("-" * 80)
    
    # Track statistics
    stats = {
        "total": len(media_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "start_time": datetime.now()
    }
    
    # Process each file
    for idx, media_path in enumerate(media_files, 1):
        # Generate output filename (same name as media file but with .json extension)
        output_filename = media_path.stem + ".json"
        output_file = output_path / output_filename
        
        # Determine file type for display
        file_type = "Audio" if media_path.suffix.lower() in AUDIO_EXTENSIONS else "Video"
        
        print(f"\n[{idx}/{len(media_files)}] Processing: {media_path.name}")
        
        # Check if result already exists
        if skip_existing and output_file.exists():
            print(f"  Skipping (result already exists): {output_file}")
            stats["skipped"] += 1
            continue
        
        try:
            # Process the file
            print(f"  {file_type}: {media_path}")
            print(f"  Output: {output_file}")
            
            # Call the evaluate function
            result = evaluate(str(media_path), output_file=str(output_file))
            
            # Transform result to match web-ui expected format
            web_ui_result = transform_to_web_ui_format(result, media_path.name)
            
            # Save in web-ui compatible format
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(web_ui_result, f, ensure_ascii=False, indent=2)
            
            print(f"  Success! Results saved to: {output_file}")
            stats["processed"] += 1
            
        except Exception as e:
            print(f"  Error processing {media_path.name}:")
            print(f"     {str(e)}")
            
            if "--verbose" in sys.argv or "-v" in sys.argv:
                print("\nFull traceback:")
                traceback.print_exc()
            
            stats["failed"] += 1
            
            # Save error information to a separate file
            error_file = output_path / (media_path.stem + ".error.json")
            error_info = {
                "file": str(media_path),
                "file_type": "audio" if media_path.suffix.lower() in AUDIO_EXTENSIONS else "video",
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2)
            
            print(f"  Error details saved to: {error_file}")
    
    # Print summary
    stats["end_time"] = datetime.now()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total files:      {stats['total']}")
    print(f"Processed:        {stats['processed']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Duration:         {stats['duration']}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Save batch summary
    summary_file = output_path / f"batch_summary_{stats['start_time'].strftime('%Y%m%d_%H%M%S')}.json"
    summary_data = {
        "input_directory": str(Path(input_dir).absolute()),
        "output_directory": str(output_path.absolute()),
        "start_time": stats["start_time"].isoformat(),
        "end_time": stats["end_time"].isoformat(),
        "duration_seconds": stats["duration"].total_seconds(),
        "statistics": {
            "total": stats["total"],
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "failed": stats["failed"]
        },
        "processed_files": [
            str(mf.name) for mf in media_files
        ]
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos and audio files, saving results to JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos and audio files in a directory
  python backend/batch_process_videos.py -i ./media -o ./results
  
  # Process files, skip if results already exist
  python backend/batch_process_videos.py -i ./media -o ./results --skip-existing
  
  # Use default output directory (./results)
  python backend/batch_process_videos.py -i ./backend/tests
  
  # Enable verbose error output
  python backend/batch_process_videos.py -i ./media -v

Supported formats:
  Videos: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v, .mpeg, .mpg
  Audio:  .mp3, .wav, .ogg, .m4a, .flac, .aac, .wma, .opus
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        required=True,
        help='Directory containing video/audio files to process'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='./results',
        help='Directory to save JSON results (default: ./results)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip videos that already have result files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed error messages and tracebacks'
    )
    
    args = parser.parse_args()
    
    try:
        process_videos_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing
        )
    except Exception as e:
        print(f"\nFatal error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
