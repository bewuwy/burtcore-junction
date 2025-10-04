# Extremist Classifier Integration

## Overview

The extremist classifier has been integrated into the classification pipeline as the final step. It combines:
1. **Multi-model classifier outputs** (toxicity, hate speech, offensive language, sentiment, targets)
2. **Intonation and emotion features** (pitch, energy, emotion from wav2vec2)

This provides a more robust classification by considering both textual content and audio characteristics.

## How It Works

### Pipeline Flow

1. **Transcription** (`transcript_wav2vec_pipeline.py`)
   - Video/audio → text segments using Whisper

2. **Feature Extraction** (`intonation_wav2vec2_pipeline.py`)
   - Audio segments → intonation features (pitch, energy, emotion)

3. **Multi-Model Classification** (`multi_model_classifier.py`)
   - Text segments → toxicity scores from 5 models
   - Outputs: toxicity, hate speech, offensive language, sentiment, target analysis

4. **Extremist Classification** (`classify_extremist.py`) **← FINAL STEP**
   - Combines multi-model outputs + intonation features
   - Binary classification: extremist vs non-extremist
   - Outputs: `isExtremist` (boolean) and `extremistProbability` (0-1)

### Integration Points

#### In `extreme.py`:
- Added `get_extremist_classifier()` function to load/initialize the model
- Modified segment processing to include extremist classification
- Added extremist statistics to the results

#### In `endpoints.py`:
- Updated API response to include extremist classification
- Added `isExtremist` field to indicate overall content classification
- Enhanced summary message to prioritize extremist detection

## API Response Format

```json
{
  "success": true,
  "result": "⚠️ EXTREMIST CONTENT DETECTED (heuristic-based): 5/10 segments (50.0%). Avg probability: 72.3%",
  "isExtremist": true,
  "heuristicUsed": true,
  "segments": [
    {
      "text": "segment text...",
      "startTime": {"minute": 0, "second": 0},
      "endTime": {"minute": 0, "second": 5},
      "extreme": 0.85,
      "isExtremist": true,
      "extremistProbability": 0.78,
      "heuristicUsed": true,
      "heuristicConfidence": 0.72,
      "classification": { /* multi-model outputs */ },
      "intonation": { /* pitch, emotion data */ }
    }
  ],
  "statistics": {
    "total_segments": 10,
    "toxic_segments": 7,
    "avg_toxicity": 0.65,
    "max_toxicity": 0.92,
    "extremist_segments": 5,
    "avg_extremist_probability": 0.72,
    "max_extremist_probability": 0.95,
    "extremist_ratio": 0.5,
    "is_extremist_content": true
  }
}
```

### New Fields (Heuristic Mode)

- `heuristicUsed`: Boolean indicating if the intonation heuristic was used (true when trained model unavailable)
- `heuristicConfidence`: Confidence in the heuristic adjustment (0-1, per segment)
- Note: `extremistProbability` contains the modified toxicity score when using heuristic

## Training the Classifier

The extremist classifier needs to be trained before it can be used. If no trained model exists, the system will fall back to toxicity-based classification.

### Training Command

```bash
python -m backend.classify_extremist \
  --mode train \
  --extremist_dir /path/to/extremist/outputs \
  --non_extremist_dir /path/to/non_extremist/outputs \
  --model_type random_forest \
  --model_path models/extremist_classifier.pkl
```

### Directory Structure for Training

Your training data directories should contain:
- `*_multimodel.json` - Output from multi-model classifier
- `*_intonation.json` - Output from intonation pipeline

The base names should match (e.g., `video1_multimodel.json` and `video1_intonation.json`).

### Training Data Preparation

1. **Collect labeled videos** (extremist and non-extremist)
2. **Process each video** through the pipeline:
   ```bash
   # This creates multimodel + intonation outputs
   python -m backend.extreme /path/to/video.mp4
   ```
3. **Organize outputs** into separate directories
4. **Train the model** using the command above

## Configuration

Key configuration options in `backend/config.py`:

```python
# Extremist classifier settings
EXTREMIST_CLASSIFIER_TYPE = 'random_forest'  # or 'gradient_boosting', 'logistic'
EXTREMIST_RATIO_THRESHOLD = 0.3  # Overall content threshold
EXTREMIST_MODEL_PATH = 'models/extremist_classifier.pkl'

# Random Forest parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
...
```

## Graceful Degradation

If the trained extremist classifier is not available:
- **Intonation-based heuristic is automatically applied** to modify toxicity scores
- The system continues working with enhanced multi-model toxicity classification
- `isExtremist` and `extremistProbability` fields are still populated
- `heuristicUsed` field is set to `true` to indicate fallback mode
- `heuristicConfidence` field shows confidence in the heuristic adjustment (0-1)
- No errors or crashes occur

### Intonation-Based Heuristic

When the trained model is unavailable, the system uses an intelligent heuristic that considers:

1. **Emotion Analysis** (±15%)
   - Angry, fear, disgust emotions → increase score
   - Happy, neutral emotions → slightly decrease score
   - Weighted by emotion confidence

2. **Pitch Variation** (±10%)
   - High pitch standard deviation or range → increase score
   - Indicates emotional intensity and agitation

3. **Energy/Loudness** (±8%)
   - High RMS energy → increase score
   - Can indicate aggressive or intense speech

4. **Pitch Slope** (±5%)
   - Rapid pitch changes → increase score
   - May indicate emotional volatility

The heuristic combines these factors to adjust the base toxicity score, with confidence calculated based on:
- Number of factors that contributed
- Strength of adjustments
- Original toxicity score (more reliable in mid-range)

**Example:**
```
Original toxicity: 0.60
+ Angry emotion (confidence 0.8): +0.12
+ High pitch variation (normalized 0.7): +0.07
+ High energy (normalized 0.8): +0.06
= Modified score: 0.85 (confidence: 0.72)
```

## Model Features

The extremist classifier uses **40+ features** including:

**From Multi-Model Classifier:**
- Overall toxicity score
- 6 toxic-BERT categories
- Hate/not-hate scores
- Offensive/non-offensive scores
- Sentiment (negative, neutral, positive)
- Target analysis scores

**From Intonation Pipeline:**
- Emotion classification (angry, fear, happy, etc.)
- Pitch features (mean, std, min, max, range, slope)
- Energy features (RMS mean, max)
- Duration

## Performance Considerations

- **Memory**: Classifier models are cached as singletons
- **Speed**: ~50-100ms additional processing per segment
- **Accuracy**: Depends on training data quality and quantity
- **Minimum training data**: 100+ labeled videos recommended

## Monitoring & Debugging

Check console output for:
```
Initializing extremist classifier...
✓ Loaded trained extremist classifier from models/extremist_classifier.pkl
```

Or warnings:
```
Warning: No trained extremist classifier model found at models/extremist_classifier.pkl
Extremist classification will not be available.
```

## Next Steps

1. **Train the model** with your labeled dataset
2. **Validate performance** on held-out test set
3. **Tune thresholds** (`EXTREMIST_RATIO_THRESHOLD`) based on your use case
4. **Monitor predictions** and collect feedback for retraining
