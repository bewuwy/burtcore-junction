# Extremist Speech Recognition in Recordings

> A multimodal system for detecting extremist content in audio/video recordings

This repository contains our submission for the **Junction X Delft Extreme Challenge** by TU Delft. The solution uses advanced machine learning techniques to analyze both speech content and acoustic features to identify potentially extremist content in video files.

## Features

- **Video Upload & Analysis**: Submit video files via web interface or API
- **Multimodal Classification**: Combines multiple AI models for comprehensive analysis
  - Speech transcription using OpenAI Whisper
  - Text-based toxicity, hate speech, and offensive language detection
  - Acoustic analysis (intonation, emotion, pitch, energy)
  - Sentiment and target analysis
- **Segment-Level Detection**: Identifies specific time segments containing extremist content
- **Real-time Processing**: Fast API backend with responsive web UI
- **Multiple Input Methods**: Upload files directly or provide URLs

## Architecture

The system consists of three main components:

1. **Backend API** (FastAPI + Python)
   - Video processing and transcription
   - Multi-model classification pipeline
   - RESTful API endpoints

2. **Web UI** (SvelteKit + TypeScript)
   - File upload interface
   - Results visualization
   - Timeline view of flagged segments

3. **ML Pipeline**
   - Whisper for speech-to-text transcription
   - BERT-based models for toxicity detection
   - Wav2Vec2 for acoustic feature extraction
   - Ensemble classifier for final prediction

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+ (with pnpm)
- FFmpeg (for audio processing)

### Installation & Running

#### 1. Start the Backend Server

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn backend.endpoints:app --reload
```

The API will be available at `http://localhost:8000`

#### 2. Start the Web UI

```bash
# Navigate to web-ui directory
cd web-ui

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

The web interface will be available at `http://localhost:5173`

### Usage

1. Open the web UI in your browser
2. Upload a video file or provide a URL
3. Review the analysis results with timestamped segments

## How It Works

### Classification Pipeline

1. **Audio Extraction**: Extract audio track from video file
2. **Transcription**: Use Whisper to generate timestamped transcript
3. **Multi-Model Analysis**: Process each segment through multiple classifiers:
   - **Hate Speech Detector**: Binary hate/not-hate classification
   - **Offensive Language Detector**: Offensive content detection
   - **Sentiment Analyzer**: Emotional tone analysis
   - **Target Analyzer**: Detection of targeted groups (race, religion, etc.)
4. **Acoustic Analysis**: Extract audio features:
   - Emotion recognition (anger, fear, disgust, etc.)
   - Pitch (F0) characteristics
   - Energy (RMS) levels
   - Speech duration patterns
5. **Binary Classification**: Ensemble model combines all features to predict extremist content

### Feature Set (30+ features per segment)

- Toxicity scores (7 dimensions)
- Hate/offensive detection scores
- Sentiment analysis (3 dimensions)
- Target group scores (multiple categories)
- Emotion type and confidence
- Acoustic features (pitch, energy, duration)

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── endpoints.py           # API routes
│   ├── extreme.py             # Main evaluation logic
│   ├── classify_extremist.py # Binary classifier
│   ├── multi_model_classifier.py # Multi-model pipeline
│   ├── requirements.txt       # Python dependencies
│   └── testing/               # Test scripts and notebooks
├── web-ui/                    # SvelteKit frontend
│   ├── src/                   # Source code
│   ├── package.json           # Node dependencies
│   └── vite.config.ts         # Vite configuration
├── learning/                  # Model training notebooks
├── labeling/                  # Dataset labeling tools
└── transcripts/               # Sample transcripts
    ├── hate/                  # Labeled extremist content
    └── non_hate/              # Labeled non-extremist content
```

## Acknowledgments

- TU Delft for organizing the Junction X Delft Extreme Challenge
- OpenAI Whisper for speech recognition
- Hugging Face for pre-trained models
