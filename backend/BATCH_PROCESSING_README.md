# Batch Video & Audio Processing

This CLI tool allows you to process multiple video and audio files in a folder and save the analysis results to individual JSON files.

## Features

- 🎥 Processes video files (MP4, AVI, MOV, MKV, WebM, etc.)
- 🎵 Processes audio files (MP3, WAV, OGG, M4A, FLAC, etc.)
- 💾 Saves each result to a separate JSON file
- 🔄 Optional skip for already processed files
- 📊 Provides batch processing statistics and summary
- ❌ Saves error details for failed files
- 🎯 Supports multiple formats

## Supported Formats

**Video:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpeg`, `.mpg`

**Audio:** `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac`, `.aac`, `.wma`, `.opus`

## Usage

### Basic Usage

```bash
# Process all videos and audio files in a directory
python backend/batch_process_videos.py -i ./media -o ./results

# Use default output directory (./results)
python backend/batch_process_videos.py -i ./media
```

### Advanced Options

```bash
# Skip files that already have results
python backend/batch_process_videos.py -i ./media -o ./results --skip-existing

# Enable verbose error output
python backend/batch_process_videos.py -i ./media -o ./results -v

# Process test videos
python backend/batch_process_videos.py -i ./backend/tests -o ./test_results

# Process audio files
python backend/batch_process_videos.py -i ./audio_files -o ./audio_results
```

## Command-Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input-dir` | `-i` | Yes | Directory containing video/audio files to process |
| `--output-dir` | `-o` | No | Directory to save JSON results (default: `./results`) |
| `--skip-existing` | - | No | Skip files that already have result files |
| `--verbose` | `-v` | No | Show detailed error messages and tracebacks |

## Output Format

For each video or audio file, the script creates:

1. **Result JSON file** (`{filename}.json`):
   - Full transcription from Whisper
   - Segment-by-segment analysis with:
     - Text content
     - Timing information (start/end times)
     - Intonation features (pitch, energy, emotion)
     - Toxicity classification
     - Extremist content detection
     - Hate target identification
     - Sarcasm detection (if enabled)
   - Overall statistics and classification

2. **Error file** (if processing fails) (`{filename}.error.json`):
   - Error type and message
   - File type (video or audio)
   - Full traceback
   - Timestamp

3. **Batch summary** (`batch_summary_{timestamp}.json`):
   - Processing statistics
   - Duration
   - List of processed files

## Example Output

```bash
$ python backend/batch_process_videos.py -i ./media -o ./results

Scanning for video and audio files in: ./media
Found 5 file(s) to process
--------------------------------------------------------------------------------

[1/5] Processing: example1.mp4
  📹 Video: ./media/example1.mp4
  💾 Output: ./results/example1.json
  ✅ Success! Results saved to: ./results/example1.json

[2/5] Processing: podcast.mp3
  🎵 Audio: ./media/podcast.mp3
  💾 Output: ./results/podcast.json
  ✅ Success! Results saved to: ./results/podcast.json

[3/5] Processing: example2.mov
  📹 Video: ./media/example2.mov
  💾 Output: ./results/example2.json
  ✅ Success! Results saved to: ./results/example2.json

[4/5] Processing: speech.wav
  🎵 Audio: ./media/speech.wav
  💾 Output: ./results/speech.json
  ✅ Success! Results saved to: ./results/speech.json

[5/5] Processing: example3.mp4
  📹 Video: ./media/example3.mp4
  💾 Output: ./results/example3.json
  ✅ Success! Results saved to: ./results/example3.json

================================================================================
BATCH PROCESSING COMPLETE
================================================================================
Total files:      5
Processed:        5 ✅
Skipped:          0 ⏭️
Failed:           0 ❌
Duration:         0:08:45.123456
Output directory: /home/user/project/results
Summary saved to: ./results/batch_summary_20251005_143022.json
```

## Example: Processing Test Files

```bash
# Process videos in the tests directory
python backend/batch_process_videos.py -i ./backend/tests -o ./test_results

# Process temp_uploads directory
python backend/batch_process_videos.py -i ./backend/temp_uploads -o ./upload_results

# Process audio files only
python backend/batch_process_videos.py -i ./audio_recordings -o ./audio_analysis
```

## Result JSON Structure

Each result file contains:

```json
{
  "whisper": {
    "text": "Full transcription...",
    "segments": [...],
    "language": "en"
  },
  "segments": [
    {
      "text": "Segment text",
      "startTime": {"minute": 0, "second": 5},
      "endTime": {"minute": 0, "second": 10},
      "intonation": {
        "emotion": "neutral",
        "emotion_score": 0.85,
        "f0_mean": 120.5,
        "f0_std": 15.3,
        "energy_mean": 0.45
      },
      "classification": {
        "is_toxic": false,
        "overall_toxicity": 0.12,
        "hate_speech_prob": 0.05,
        "offensive_prob": 0.08
      },
      "extreme": 0.12,
      "isExtremist": false,
      "extremistProbability": 0.15,
      "hateTarget": null
    }
  ],
  "full_text_classification": {...},
  "statistics": {
    "total_segments": 25,
    "toxic_segments": 2,
    "avg_toxicity": 0.18,
    "max_toxicity": 0.65,
    "extremist_segments": 1,
    "avg_extremist_probability": 0.22,
    "is_extremist_content": false,
    "language": "en"
  }
}
```

## Error Handling

If a video fails to process:
- The error is logged to console
- An error JSON file is created with full details
- Processing continues with the next video
- Use `-v` flag for detailed traceback

## Performance Tips

1. **Skip Existing Results**: Use `--skip-existing` when re-running to avoid reprocessing
2. **Batch Size**: Process large batches during off-hours as each video can take several minutes
3. **GPU Usage**: The script will automatically use GPU if available for faster processing
4. **Disk Space**: Ensure adequate disk space for JSON results (typically 10-100KB per video)

## Troubleshooting

### No files found
- Check that your input directory path is correct
- Verify files have supported extensions
- Use absolute paths if relative paths don't work

### Processing errors
- Use `-v` flag to see detailed error messages
- Check the `.error.json` files for specific error details
- Ensure required models and dependencies are installed

### Out of memory
- Process files in smaller batches
- Close other applications to free up RAM/VRAM
- Use a machine with more resources for large batches

### Audio file issues
- Ensure audio files are not corrupted
- Check that audio format is supported by the transcription engine
- Some audio formats may need conversion to more common formats (MP3, WAV)

## Integration with Existing System

This script uses the same `evaluate()` function as the API endpoint, ensuring consistent results between:
- Manual batch processing (this CLI tool)
- API uploads (`/evaluate/` endpoint)
- Individual file processing

All results follow the same JSON schema and include the same analysis features.

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Place your media files** in a directory:
   ```bash
   mkdir my_media
   # Copy your videos and audio files to my_media/
   ```

3. **Run the batch processor**:
   ```bash
   python backend/batch_process_videos.py -i my_media -o my_results
   ```

4. **View results**:
   - Individual JSON files in `my_results/`
   - Batch summary in `my_results/batch_summary_*.json`
   - Error logs in `my_results/*.error.json` (if any failures)
