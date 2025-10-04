# Binary Extremist Classifier

A binary classifier that combines outputs from the multi-model classifier and intonation-wav2vec pipeline to classify video segments as extremist or non-extremist content.

## Key Features

- **Multimodal Classification**: Combines toxicity/hate scores with intonation and emotion features
- **Text-Free**: Does NOT use text directly as a feature - only uses derived scores and acoustic features
- **Multiple Model Types**: Supports Random Forest, Gradient Boosting, and Logistic Regression
- **Segment-Level Analysis**: Classifies each segment individually and provides overall video assessment

## Extracted Features

### From Multi-Model Classifier:
- Overall toxicity score
- Toxic-BERT scores (6 categories: toxic, severe_toxic, obscene, threat, insult, identity_hate)
- Hate detection scores (HATE, NOT-HATE)
- Offensive detection scores (offensive, non-offensive)
- Sentiment scores (negative, neutral, positive)
- Target analysis scores (race, religion, etc.)
- Number of detected issues

### From Intonation-Wav2Vec2 Pipeline:
- Emotion type (one-hot encoded: angry, fear, happy, neutral, sad, disgust)
- Emotion confidence score
- Pitch (F0) features: mean, std, min, max, range, slope, final
- Energy (RMS) features: mean, max
- Segment duration

Total: ~30+ features per segment (without using text)

## Usage

### Training Mode

Train the classifier on labeled data (extremist vs non-extremist):

```bash
python classify_extremist.py \
  --mode train \
  --extremist_dir /path/to/extremist/outputs \
  --non_extremist_dir /path/to/non_extremist/outputs \
  --model_type random_forest \
  --model_path extremist_classifier.pkl \
  --multimodel_suffix _multimodel.json \
  --intonation_suffix _intonation.json
```

**Requirements:**
- Each directory should contain paired files:
  - `video_1_multimodel.json` (output from multi_model_classifier.py)
  - `video_1_intonation.json` (output from intonation_wav2vec2_pipeline.py)
- Label all files in `extremist_dir` as positive (extremist)
- Label all files in `non_extremist_dir` as negative (non-extremist)

### Prediction Mode

Classify new content using trained model:

```bash
python classify_extremist.py \
  --mode predict \
  --model_path extremist_classifier.pkl \
  --multimodel_file /path/to/video_multimodel.json \
  --intonation_file /path/to/video_intonation.json \
  --output_file predictions.json
```

## Output Format

The prediction output includes:

```json
{
  "file": "video_name.json",
  "total_segments": 10,
  "extremist_segments": 3,
  "non_extremist_segments": 7,
  "extremist_ratio": 0.3,
  "avg_extremist_probability": 0.42,
  "is_extremist_content": false,
  "predictions": [
    {
      "start": 0.0,
      "end": 5.2,
      "is_extremist": true,
      "extremist_probability": 0.85,
      "text": "segment text (for reference only)"
    }
  ]
}
```

## Pipeline Workflow

1. **Extract Audio**: Process video to get audio file
2. **Transcribe**: Run Whisper to get transcript with timestamps
3. **Multi-Model Classification**: 
   ```bash
   python multi_model_classifier.py --input transcript.json --output video_multimodel.json
   ```
4. **Intonation Analysis**:
   ```bash
   python testing/transcript-proccessing/intonation_wav2vec2_pipeline.py \
     --audio video.mp4 \
     --whisper_json transcript.json \
     --out_json video_intonation.json
   ```
5. **Extremist Classification**:
   ```bash
   python classify_extremist.py --mode predict \
     --multimodel_file video_multimodel.json \
     --intonation_file video_intonation.json \
     --output_file extremist_prediction.json
   ```

## Model Types

- **random_forest** (default): Best for non-linear patterns, provides feature importance
- **gradient_boosting**: Often highest accuracy, good for imbalanced data
- **logistic**: Fast, interpretable, good baseline

## Classification Threshold

- Segment-level: Binary classification (0 or 1)
- Video-level: Overall classified as extremist if >30% of segments are extremist
- Adjustable via modifying `extremist_ratio > 0.3` in the code

## Training Tips

1. **Balanced Dataset**: Try to have similar numbers of extremist and non-extremist examples
2. **Quality Labels**: Ensure ground truth labels are accurate
3. **Cross-Validation**: The script performs 5-fold CV to assess generalization
4. **Feature Importance**: Review top features to understand what drives predictions

## Dependencies

```bash
pip install numpy pandas scikit-learn transformers torch librosa soundfile
```

## Example Training Data Structure

```
extremist_outputs/
  ├── hate_video_1_multimodel.json
  ├── hate_video_1_intonation.json
  ├── hate_video_2_multimodel.json
  └── hate_video_2_intonation.json

non_extremist_outputs/
  ├── normal_video_1_multimodel.json
  ├── normal_video_1_intonation.json
  ├── normal_video_2_multimodel.json
  └── normal_video_2_intonation.json
```

## Performance Metrics

The training process reports:
- Precision, Recall, F1-Score for both classes
- Confusion Matrix
- ROC-AUC Score
- Cross-Validation Accuracy
- Top 15 Most Important Features

## Notes

- Text is included in output for reference only, not used as a feature
- Segments are aligned by timestamp with 0.5s tolerance
- Missing intonation features (NaN) are replaced with 0.0
- Model uses `class_weight='balanced'` to handle imbalanced data

