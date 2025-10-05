#!/usr/bin/env python3
"""
Train extremist classifier from labeling/dataset.csv using:
- Artificial intonation features derived from NLTK VADER (text-based proxies for pitch/energy/emotion)
- MultiModelClassifier outputs (toxicity/hate/offensive/target/sentiment)

The first N rows are assumed to be labeled; we use their label to train a supervised model.
"""
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# NLTK VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Local imports
from config import Config
from multi_model_classifier import MultiModelClassifier
from classify_extremist import ExtremistClassifier


def ensure_vader() -> None:
    """Ensure the VADER lexicon is available."""
    try:
        _ = SentimentIntensityAnalyzer()
    except Exception:
        nltk.download('vader_lexicon')


def text_proxies(text: str) -> Dict[str, float]:
    """Compute simple textual prosody proxies used to synthesize intonation stats."""
    t = text or ""
    words = t.split()
    num_words = len(words)
    num_chars = len(t)
    exclam = t.count('!')
    quest = t.count('?')
    ellipses = t.count('...')
    caps_tokens = sum(1 for w in words if len(w) >= 3 and w.isupper())
    caps_ratio = caps_tokens / max(1, num_words)
    punct_intensity = (exclam * 1.0 + quest * 0.6 + ellipses * 0.8) / max(1, num_words)
    avg_word_len = sum(len(w) for w in words) / max(1, num_words)
    return {
        'num_words': float(num_words),
        'num_chars': float(num_chars),
        'caps_ratio': float(caps_ratio),
        'punct_intensity': float(punct_intensity),
        'avg_word_len': float(avg_word_len),
    }


def synthesize_intonation(text: str, vader: Dict[str, float]) -> Dict[str, Any]:
    """
    Create artificial intonation features shaped like the wav2vec2 pipeline output using VADER sentiment
    and text punctuation proxies. Values are clamped to plausible ranges used by the extractor.
    """
    proxies = text_proxies(text)
    compound = float(vader.get('compound', 0.0))
    pos_score = float(vader.get('pos', 0.0))
    neg_score = float(vader.get('neg', 0.0))
    neu_score = float(vader.get('neu', 0.0))

    # Map sentiment to discrete emotion
    if compound >= 0.4 and pos_score > neg_score:
        emotion = 'happy'
        emotion_strength = pos_score
    elif compound <= -0.4 and neg_score >= max(pos_score, neu_score):
        # Strong negative -> angry/disgust
        emotion = 'angry' if neg_score > 0.6 else 'disgust'
        emotion_strength = neg_score
    elif -0.4 < compound < 0 and neg_score > 0.3:
        emotion = 'sad'
        emotion_strength = max(neg_score, -compound)
    else:
        emotion = 'neutral'
        emotion_strength = neu_score if neu_score > 0 else 0.5 - abs(compound) * 0.5

    # Duration proxy: ~ 180 wpm -> 3 words/sec baseline, adjust with punctuation and emotion
    duration = proxies['num_words'] / 2.5  # seconds
    duration = float(max(0.3, min(duration, 30.0)))

    # Pitch synthesis (Hz)
    # Base mean influenced by sentiment polarity
    f0_mean = 170 + 40 * (pos_score - neg_score)  # center around ~170 Hz
    f0_mean = float(max(Config.F0_MIN, min(f0_mean, Config.F0_MAX)))

    # Variability increases with punctuation/caps and high arousal (|compound|)
    f0_std = 15 + 60 * proxies['punct_intensity'] + 25 * proxies['caps_ratio'] + 20 * abs(compound)
    f0_std = float(max(5.0, min(f0_std, 80.0)))

    f0_range = float(max(30.0, min(f0_std * 2.8, 300.0)))
    f0_min = float(max(Config.F0_MIN, f0_mean - 1.2 * f0_std))
    f0_max = float(min(Config.F0_MAX, f0_mean + 1.2 * f0_std))

    # Slope: rising with excitement, falling with negativity
    slope_base = 50 * (pos_score - neg_score)
    slope_punct = 120 * proxies['punct_intensity']
    f0_slope = float(max(-100.0, min(slope_base + slope_punct, 100.0)))  # Hz/s

    f0_final = float(max(Config.F0_MIN, min(f0_mean + 0.15 * f0_slope, Config.F0_MAX)))

    # Energy synthesis (RMS ~ 0-0.2)
    rms_mean = 0.03 + 0.08 * proxies['caps_ratio'] + 0.06 * proxies['punct_intensity'] + 0.02 * abs(compound)
    rms_mean = float(max(0.005, min(rms_mean, 0.12)))
    rms_max = float(max(rms_mean, min(rms_mean * 1.8, 0.2)))

    return {
        'emotion': emotion,
        'emotion_score': float(max(0.0, min(emotion_strength, 1.0))),
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_min': f0_min,
        'f0_max': f0_max,
        'f0_range': f0_range,
        'f0_slope': f0_slope,
        'f0_final': f0_final,
        'rms_mean': rms_mean,
        'rms_max': rms_max,
        'duration': duration,
    }


def detect_columns(df: pd.DataFrame,
                   text_col: Optional[str],
                   label_col: Optional[str]) -> Tuple[str, str]:
    """Detect text and label columns with sensible fallbacks."""
    if text_col and text_col in df.columns:
        tcol = text_col
    else:
        candidates_t = [
            'text', 'Text', 'sentence', 'content', 'transcript', 'message', 'tweet'
        ]
        tcol = next((c for c in candidates_t if c in df.columns), None)
        if not tcol:
            raise ValueError(f"Could not find a text column in CSV. Available columns: {list(df.columns)}")

    if label_col and label_col in df.columns:
        lcol = label_col
    else:
        candidates_l = [
            'extremism', 'extremism_filled', 'Extremist', 'label', 'Label', 'is_extremist', 'target'
        ]
        lcol = next((c for c in candidates_l if c in df.columns), None)
        if not lcol:
            raise ValueError(f"Could not find a label column in CSV. Available columns: {list(df.columns)}")

    return tcol, lcol


def load_labeled_rows(csv_path: str, n_rows: int,
                      text_column: Optional[str],
                      label_column: Optional[str]) -> pd.DataFrame:
    """Load first N labeled rows from the CSV robustly."""
    # Read only necessary columns and rows if possible
    df_head = pd.read_csv(csv_path, nrows=max(n_rows, 200))
    tcol, lcol = detect_columns(df_head, text_column, label_column)

    # Reload just those cols for performance if file is huge
    usecols = [tcol, lcol]
    df = pd.read_csv(csv_path, usecols=usecols)

    # Keep first N rows assuming they're labeled; drop rows with missing text or label
    df = df.head(n_rows)
    df = df.dropna(subset=[tcol, lcol])

    # Normalize label to {0,1}
    def to_int_label(x):
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in {'1', 'true', 'yes', 'extremist'}:
                return 1
            if xl in {'0', 'false', 'no', 'not extremist', 'non-extremist', 'non extremist'}:
                return 0
        try:
            val = int(float(x))
            return 1 if val == 1 else 0
        except Exception:
            return None

    df['label'] = df[lcol].apply(to_int_label)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Clean text
    df['text_clean'] = df[tcol].astype(str).str.strip()
    df = df[df['text_clean'].str.len() > 0]

    return df[[ 'text_clean', 'label' ]].rename(columns={'text_clean': 'text'})


def main():
    parser = argparse.ArgumentParser(description='Train extremist classifier from CSV using VADER + MultiModel features')
    parser.add_argument('--csv', default=str(Path('labeling') / 'dataset.csv'), help='Path to dataset CSV')
    parser.add_argument('--n_rows', type=int, default=10000, help='Number of labeled rows to use (from top)')
    parser.add_argument('--text_column', default=None, help='Text column name (auto-detect if omitted)')
    parser.add_argument('--label_column', default=None, help='Label column name (auto-detect if omitted)')
    parser.add_argument('--model_type', choices=['random_forest', 'gradient_boosting', 'logistic'],
                        default=Config.EXTREMIST_CLASSIFIER_TYPE, help='Classifier type')
    parser.add_argument('--save_model_path', default=str(Config.EXTREMIST_MODEL_PATH), help='Where to save model')
    parser.add_argument('--save_features', default=None, help='Optional CSV to dump engineered features')
    parser.add_argument('--no_hf', action='store_true', help='Skip HuggingFace multi-model classification (VADER-only)')
    args = parser.parse_args()

    print('Loading labeled dataset...')
    df = load_labeled_rows(args.csv, args.n_rows, args.text_column, args.label_column)
    if df.empty:
        raise RuntimeError('No labeled rows found to train on.')

    print(f"Loaded {len(df)} labeled rows")

    # Prepare analyzers
    ensure_vader()
    vader_analyzer = SentimentIntensityAnalyzer()

    # Try to init multi-model classifier unless disabled
    mm_classifier: Optional[MultiModelClassifier] = None
    if not args.no_hf:
        try:
            print('Initializing multi-model classifier (this may download models on first run)...')
            mm_classifier = MultiModelClassifier()
        except Exception as e:
            print(f"Warning: Failed to initialize MultiModelClassifier: {e}\nProceeding with VADER-only features.")
            mm_classifier = None

    # Build dataset
    feature_dicts: list[Dict[str, float]] = []
    y_list: list[int] = []
    all_keys: set[str] = set()

    # Temporary classifier instance to use feature extraction/combine logic
    ex_clf = ExtremistClassifier(model_type=args.model_type)

    print('Engineering features...')
    rows_iter = tqdm(df.itertuples(index=False), total=len(df))
    feature_dump_rows = [] if args.save_features else None
    for row in rows_iter:
        text: str = getattr(row, 'text')
        label: int = int(getattr(row, 'label'))

        # VADER
        vs = vader_analyzer.polarity_scores(text)
        inton = synthesize_intonation(text, vs)

        # Multi-model
        if mm_classifier is not None:
            mm = mm_classifier.classify_text(text)
            mm_segment = {
                'start': 0.0,
                'end': max(0.1, float(inton.get('duration', 1.0))),
                'text': text,
                'overall_toxicity': mm.get('overall_toxicity', 0.0),
                'is_toxic': mm.get('is_toxic', False),
                'hate_target': mm.get('hate_target', None),
                'hate_target_confidence': mm.get('hate_target_confidence', 0.0),
                'categories': mm.get('categories', {}),
                'model_outputs': mm.get('model_outputs', {}),
                'detected_issues': mm.get('detected_issues', []),
            }
        else:
            # Fallback minimal multi-model features from VADER
            mm_segment = {
                'start': 0.0,
                'end': max(0.1, float(inton.get('duration', 1.0))),
                'text': text,
                'overall_toxicity': float(max(0.0, min((vs['neg']*0.8 + (1-vs['neu'])*0.2), 1.0))),
                'is_toxic': bool(vs['neg'] > 0.6),
                'hate_target': None,
                'hate_target_confidence': 0.0,
                'categories': {},
                'model_outputs': {
                    'toxic_bert': {
                        'toxic': float(vs['neg']),
                        'severe_toxic': float(vs['neg']*0.5),
                        'obscene': 0.0,
                        'threat': 0.0,
                        'insult': float(max(0.0, vs['neg']-0.2)),
                        'identity_hate': 0.0,
                    },
                    'hate_detection': {'HATE': float(max(0.0, vs['neg']-0.3)), 'NOT-HATE': float(vs['neu'])},
                    'offensive_detection': {'offensive': float(max(0.0, vs['neg']-0.25)), 'non-offensive': float(vs['neu'])},
                    'sentiment': {'negative': float(vs['neg']), 'neutral': float(vs['neu']), 'positive': float(vs['pos'])},
                    'target_analysis': {},
                },
                'detected_issues': [],
            }

        feat_dict = ex_clf.build_feature_dict(mm_segment, inton)
        feature_dicts.append(feat_dict)
        y_list.append(label)
        all_keys.update(feat_dict.keys())

        if feature_dump_rows is not None:
            # Flatten for CSV dump (subset)
            row_out = {'label': label}
            for k, v in inton.items():
                row_out[f'inton_{k}'] = v
            row_out['tox_overall'] = mm_segment['overall_toxicity']
            if 'model_outputs' in mm_segment and 'toxic_bert' in mm_segment['model_outputs']:
                for k, v in mm_segment['model_outputs']['toxic_bert'].items():
                    row_out[f'toxic_{k}'] = v
            feature_dump_rows.append(row_out)

    # Build a consistent feature matrix including ALL keys
    feature_names = sorted(all_keys)
    ex_clf.feature_names = feature_names  # preserve for importance display

    X = np.array([[fd.get(name, 0.0) for name in feature_names] for fd in feature_dicts], dtype=float)
    y = np.array(y_list, dtype=int)

    print(f"Feature matrix: {X.shape}, Positives: {int(y.sum())}, Negatives: {len(y)-int(y.sum())}")

    # Train and save
    ex_clf.train(X, y)

    save_path = Path(args.save_model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ex_clf.save_model(str(save_path))

    if feature_dump_rows is not None and feature_dump_rows:
        out_csv = Path(args.save_features)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feature_dump_rows).to_csv(out_csv, index=False)
        print(f"Saved engineered features to {out_csv}")

    print(f"\n✓ Training complete. Model saved at: {save_path}")


if __name__ == '__main__':
    main()
