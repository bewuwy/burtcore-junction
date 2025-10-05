# Extremist Speech Recognition in Recordings

> A multimodal system for detecting extremist content in audio/video recordings using AI

This repository contains our submission for the **Junction X Delft Extreme Challenge** by TU Delft. The solution uses advanced machine learning to analyze both speech content and acoustic features to identify potentially extremist content in video/audio files.

## Features

- **Multimodal Analysis**: Combines text (toxicity, hate speech, sentiment) with audio features (emotion, pitch, energy)
- **Sarcasm Detection**: Automatically detects and reduces scores for sarcastic content
- **Hate Target Detection**: Identifies specific targets of hate speech (religion, race, gender, etc.)
- **Intelligent Fallback**: Uses heuristic-based classification when trained model is unavailable
- **Segment-Level Detection**: Identifies specific time segments containing extremist content
- **Web UI & API**: User-friendly interface and RESTful API for integration
- **Batch Processing**: Process multiple files at once via CLI

## Prerequisites

- Python 3.8+
- Node.js 18+ (with pnpm)
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, for faster processing)

## Quick Start

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn backend.endpoints:app --reload
```

The API will be available at `http://localhost:8000`

### 2. Web UI Setup

```bash
# Navigate to web-ui directory
cd web-ui

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

The web interface will be available at `http://localhost:5173`

## Usage Modes

### Mode 1: Web Interface

1. Open `http://localhost:5173` in your browser
2. Upload a video/audio file or provide a URL
3. View results with timestamped segments and hate targets
4. Hover over flagged segments to see details

### Mode 2: API Endpoints

**Analyze Video/Audio:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -F "file=@video.mp4"
```

**Response includes:**
- `isExtremist`: Boolean indicating extremist content detected
- `segments`: Array of analyzed segments with timestamps
- `statistics`: Aggregate metrics (toxicity, extremist ratio, etc.)
- `heuristicUsed`: Whether fallback heuristic was used
- Each segment contains:
  - `extremistProbability`: Confidence score (0-1)
  - `hateTarget`: Identified target (if applicable)
  - `sarcasm`: Sarcasm detection results
  - `classification`: Multi-model outputs
  - `intonation`: Audio feature analysis

### Mode 3: Batch Processing

Process multiple files from a directory:

```bash
# Basic batch processing
python backend/batch_process_videos.py -i ./media -o ./results

# Skip already processed files
python backend/batch_process_videos.py -i ./media -o ./results --skip-existing

# Verbose error output
python backend/batch_process_videos.py -i ./media -o ./results -v
```

**Supported formats:**
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpeg`, `.mpg`
- Audio: `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac`, `.aac`, `.wma`, `.opus`

### Mode 4: Train Custom Classifier

Train your own extremist classifier on labeled data:

```bash
python backend/classify_extremist.py \
  --mode train \
  --extremist_dir ./transcripts/hate \
  --non_extremist_dir ./transcripts/non_hate \
  --model_type random_forest \
  --model_path models/extremist_classifier.pkl
```

**Requirements:**
- Each directory should contain paired files: `filename_multimodel.json` and `filename_intonation.json`
- Use `batch_process_videos.py` to generate these files

## How It Works

### Classification Pipeline

```
Video/Audio Input
    ↓
[1] Whisper Transcription
    → Text segments with timestamps
    ↓
[2] Multi-Model Classification
    → Toxicity, hate speech, offensive language, sentiment, targets
    ↓
[3] Acoustic Analysis (Wav2Vec2)
    → Pitch, energy, emotion, duration
    ↓
[4] Sarcasm Detection
    → Detect sarcastic delivery patterns
    ↓
[5] Final Classification
    ├─ Trained Model Available?
    │  ├─ YES → ExtremistClassifier
    │  └─ NO → Intonation Heuristic
    ↓
Output: Extremist/Non-Extremist with confidence
```

### Key Components

**Multi-Model Classifier:**
- 5 BERT-based models for text analysis
- Toxicity, hate speech, offensive language detection
- Sentiment and target group identification

**Acoustic Analyzer:**
- Emotion recognition (anger, fear, disgust, happy, neutral, sad)
- Pitch (F0) analysis: mean, std, range, slope
- Energy (RMS) analysis: mean, max
- Duration patterns

**Sarcasm Detector:**
- 6 detection patterns (deadpan, monotone, emotion-mismatch, etc.)
- Automatic score reduction (30-80%) when detected
- Pattern identification for transparency

**Intelligent Fallback:**
- When trained model unavailable, uses intonation-based heuristic
- Adjusts toxicity scores based on audio characteristics
- Provides confidence scoring for transparency

### Feature Set (30+ features per segment)

- Toxicity scores (7 dimensions)
- Hate/offensive detection scores
- Sentiment analysis (3 dimensions)
- Target group scores (religion, race, gender, disability, age, ethnicity)
- Emotion type and confidence
- Acoustic features (pitch, energy, duration)
- Sarcasm probability and pattern

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── endpoints.py           # API routes
│   ├── extreme.py             # Main evaluation logic
│   ├── classify_extremist.py # Binary classifier training/inference
│   ├── multi_model_classifier.py # Multi-model text analysis
│   ├── batch_process_videos.py # Batch processing CLI
│   ├── training.py            # Model training utilities
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   └── testing/               # Test scripts and notebooks
├── web-ui/                    # SvelteKit frontend
│   ├── src/                   # Source code
│   │   └── routes/+page.svelte # Main interface
│   ├── package.json           # Node dependencies
│   └── vite.config.ts         # Vite configuration
├── learning/                  # Model training notebooks
├── labeling/                  # Dataset labeling tools
├── transcripts/               # Sample transcripts
│   ├── hate/                  # Labeled extremist content
│   └── non_hate/              # Labeled non-extremist content
└── models/                    # Trained model files (if available)
```

## Configuration

Key settings in `backend/config.py`:

- **Whisper Model**: `base` (can upgrade to `small`, `medium`, `turbo`)
- **Thresholds**: Toxicity (0.5), sarcasm (0.4), compression ratio (2.4)
- **Audio Processing**: 16kHz sampling, mono channel

## API Response Example

```json
{
  "success": true,
  "result": "EXTREMIST CONTENT DETECTED: 5/10 segments (50.0%)",
  "isExtremist": true,
  "heuristicUsed": false,
  "segments": [
    {
      "text": "hateful content here",
      "startTime": {"minute": 0, "second": 15},
      "endTime": {"minute": 0, "second": 18},
      "extreme": 0.875,
      "isExtremist": true,
      "extremistProbability": 0.78,
      "hateTarget": "Religion",
      "hateTargetConfidence": 0.75,
      "sarcasm": {
        "detected": false,
        "probability": 0.18,
        "pattern": null
      },
      "classification": {
        "overall_toxicity": 0.875,
        "is_toxic": true,
        "detected_issues": ["toxic_insult:0.65", "offensive_offensive:0.71"]
      },
      "intonation": {
        "emotion": "angry",
        "emotion_score": 0.82,
        "f0_mean": 187.3,
        "rms_mean": 0.078
      }
    }
  ],
  "statistics": {
    "total_segments": 10,
    "extremist_segments": 5,
    "avg_extremist_probability": 0.72,
    "extremist_ratio": 0.5
  }
}
```

## Acknowledgments

- **TU Delft** for organizing the Junction X Delft Extreme Challenge
- **OpenAI Whisper** for speech recognition
- **Hugging Face** for pre-trained BERT models
- **Meta AI** for Wav2Vec2 emotion recognition
