#!/usr/bin/env python3
"""
Binary extremist classifier using multi-model classifier and intonation features.
Classifies segments as extremist or not based on:
1. Toxicity/hate/offensive scores from multi-model classifier
2. Intonation and emotion features from wav2vec2
3. Does NOT use text directly as a feature
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, fbeta_score
import pickle
import warnings
from config import Config
warnings.filterwarnings("ignore")


class ExtremistClassifier:
    """Binary classifier for extremist content using multimodal features."""

    def __init__(self, model_type=Config.EXTREMIST_CLASSIFIER_TYPE):
        """
        Initialize the classifier.

        Args:
            model_type: Type of classifier ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()

        if self.model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=Config.RF_N_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
                max_features=Config.RF_MAX_FEATURES,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=Config.GB_N_ESTIMATORS,
                max_depth=Config.GB_MAX_DEPTH,
                learning_rate=Config.GB_LEARNING_RATE,
                subsample=Config.GB_SUBSAMPLE,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=Config.LR_MAX_ITER,
                C=Config.LR_C,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.feature_names = []
        self.is_trained = False
        self.decision_threshold = float(Config.BASE_DECISION_THRESHOLD)

    def extract_multimodel_features(self, segment: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from multi-model classifier output.
        Includes all available scores and summary attributes.
        """
        features: Dict[str, float] = {}

        # Overall summary fields
        features['overall_toxicity'] = float(segment.get('overall_toxicity', 0.0) or 0.0)
        features['is_toxic'] = 1.0 if segment.get('is_toxic', False) else 0.0
        features['hate_target_confidence'] = float(segment.get('hate_target_confidence', 0.0) or 0.0)

        # One-hot for hate_target label if present
        hate_target = segment.get('hate_target')
        if isinstance(hate_target, str) and hate_target.strip():
            key = f"hate_target_is_{hate_target.strip().lower().replace(' ', '_')}"
            features[key] = 1.0

        # Extract all model output scores
        model_outputs = segment.get('model_outputs', {}) or {}

        # Toxic-BERT scores (6 categories)
        for key, value in (model_outputs.get('toxic_bert', {}) or {}).items():
            features[f'toxic_{key}'] = float(value)

        # Hate detection scores
        hate_det = model_outputs.get('hate_detection', {}) or {}
        for key, value in hate_det.items():
            features[f'hate_{key}'] = float(value)

        # Offensive detection scores
        offensive_det = (model_outputs.get('offensive_detection', {}) or {})
        for key, value in offensive_det.items():
            features[f'offensive_{key}'] = float(value)

        # Sentiment scores (emotional tone)
        sentiment = (model_outputs.get('sentiment', {}) or {})
        for key, value in sentiment.items():
            features[f'sentiment_{key}'] = float(value)

        # Target analysis - include all raw target scores and a max summary
        target = (model_outputs.get('target_analysis', {}) or {})
        if target:
            try:
                features['target_max_score'] = float(max(target.values()))
            except Exception:
                features['target_max_score'] = 0.0
            for key, value in target.items():
                features[f'target_{key}'] = float(value)
        else:
            features['target_max_score'] = 0.0

        # Number of detected issues
        features['num_detected_issues'] = float(len(segment.get('detected_issues', []) or []))

        # Additionally, if a flat categories dict exists, include any missing keys (without duplication)
        categories = segment.get('categories', {}) or {}
        for cat_key, score in categories.items():
            # Only add if not already covered above
            if cat_key not in features:
                try:
                    features[cat_key] = float(score)
                except Exception:
                    continue

        # Ensure finite values
        for k, v in list(features.items()):
            if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
                features[k] = 0.0

        return features

    def extract_intonation_features(self, segment: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from intonation/emotion output.

        Args:
            segment: Segment from intonation pipeline with pitch/energy/emotion

        Returns:
            Dictionary of numerical features
        """
        features = {}

        # Emotion score (confidence)
        features['emotion_score'] = segment.get('emotion_score', 0.0)

        # Pitch (F0) features
        features['f0_mean'] = segment.get('f0_mean', 0.0)
        features['f0_std'] = segment.get('f0_std', 0.0)
        features['f0_min'] = segment.get('f0_min', 0.0)
        features['f0_max'] = segment.get('f0_max', 0.0)
        features['f0_range'] = segment.get('f0_range', 0.0)
        features['f0_slope'] = segment.get('f0_slope', 0.0)
        features['f0_final'] = segment.get('f0_final', 0.0)

        # Energy features
        features['rms_mean'] = segment.get('rms_mean', 0.0)
        features['rms_max'] = segment.get('rms_max', 0.0)

        # Duration
        features['duration'] = segment.get('duration', 0.0)

        # Encode emotion as one-hot (without using text)
        # Common emotions: angry, fear, happy, neutral, sad, disgust
        emotion = segment.get('emotion', 'neutral').lower()
        emotions_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'disgust']
        for emo in emotions_list:
            features[f'emotion_is_{emo}'] = 1.0 if emo in emotion else 0.0

        # Handle NaN values from pitch extraction
        for key in features:
            if pd.isna(features[key]) or np.isnan(features[key]) or not np.isfinite(features[key]):
                features[key] = 0.0

        return features

    def build_feature_dict(self, multimodel_segment: Dict[str, Any], intonation_segment: Dict[str, Any]) -> Dict[str, float]:
        """Return a flat dict of all features from multi-model and intonation segments."""
        mm_features = self.extract_multimodel_features(multimodel_segment)
        intonation_features = self.extract_intonation_features(intonation_segment)
        return {**mm_features, **intonation_features}

    def combine_features(
        self,
        multimodel_segment: Dict[str, Any],
        intonation_segment: Dict[str, Any]
    ) -> np.ndarray:
        """
        Combine features into a vector in consistent order using self.feature_names.
        If feature_names is empty, initialize from current dict; otherwise ignore new keys.
        Note: For full inclusion across a dataset, prefer build_feature_dict and construct
        a global union externally (as done in the dataset trainer).
        """
        all_features = self.build_feature_dict(multimodel_segment, intonation_segment)
        if not self.feature_names:
            self.feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features.get(name, 0.0) for name in self.feature_names])
        return feature_vector

    def align_segments(
        self,
        multimodel_segments: List[Dict[str, Any]],
        intonation_segments: List[Dict[str, Any]],
        time_tolerance: float = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Align segments from multi-model classifier and intonation pipeline by timestamp.

        Args:
            multimodel_segments: List of segments from multi-model classifier
            intonation_segments: List of segments from intonation pipeline
            time_tolerance: Maximum time difference (seconds) to consider segments aligned
                          If None, uses Config.TIME_TOLERANCE

        Returns:
            List of (multimodel_segment, intonation_segment) pairs
        """
        if time_tolerance is None:
            time_tolerance = Config.TIME_TOLERANCE
        
        aligned = []

        for mm_seg in multimodel_segments:
            mm_start = mm_seg.get('start', 0.0)
            mm_end = mm_seg.get('end', 0.0)
            mm_mid = (mm_start + mm_end) / 2

            # Find closest intonation segment
            best_match = None
            best_distance = float('inf')

            for inton_seg in intonation_segments:
                inton_start = inton_seg.get('start', 0.0)
                inton_end = inton_seg.get('end', 0.0)
                inton_mid = (inton_start + inton_end) / 2

                distance = abs(mm_mid - inton_mid)

                if distance < best_distance and distance < time_tolerance:
                    best_distance = distance
                    best_match = inton_seg

            if best_match is not None:
                aligned.append((mm_seg, best_match))

        return aligned

    def prepare_training_data(
        self,
        multimodel_file: str,
        intonation_file: str,
        label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from multi-model and intonation outputs.

        Args:
            multimodel_file: Path to multi-model classifier JSON output
            intonation_file: Path to intonation pipeline JSON output
            label: Ground truth label (1 for extremist, 0 for non-extremist)

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        # Load data
        with open(multimodel_file, 'r') as f:
            multimodel_data = json.load(f)

        with open(intonation_file, 'r') as f:
            intonation_data = json.load(f)

        # Get segments
        mm_segments = multimodel_data.get('segment_classifications', [])
        intonation_segments = intonation_data if isinstance(intonation_data, list) else []

        # Align segments
        aligned = self.align_segments(mm_segments, intonation_segments)

        if not aligned:
            print(f"Warning: No aligned segments found for {multimodel_file}")
            return np.array([]), np.array([])

        # Extract features
        X = []
        y = []

        for mm_seg, inton_seg in aligned:
            feature_vec = self.combine_features(mm_seg, inton_seg)
            X.append(feature_vec)
            y.append(label)

        return np.array(X), np.array(y)

    def _optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Pick a probability threshold maximizing F-beta (recall-focused) over candidate grid."""
        beta = float(Config.F_BETA_FOR_THRESHOLD)
        # Use unique probabilities plus a small grid for stability
        uniq = np.unique(y_proba)
        grid = np.linspace(0.05, 0.95, 19)
        cands = np.unique(np.concatenate([uniq, grid]))
        best_thr, best_score = 0.5, -1.0
        for thr in cands:
            y_pred = (y_proba >= thr).astype(int)
            try:
                score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score, best_thr = score, float(thr)
        return float(best_thr)

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = None):
        """
        Train the extremist classifier with safeguards for tiny/imbalanced datasets.
        """
        if validation_split is None:
            validation_split = Config.VALIDATION_SPLIT
        if len(X) == 0:
            raise ValueError("Cannot train with empty dataset")

        classes, counts = np.unique(y, return_counts=True)
        # Tiny/imbalanced dataset path
        if len(classes) < 2 or counts.min() < 2 or len(X) < 6:
            print("\nNote: Insufficient class representation for stratified validation (class counts:",
                  {int(c): int(n) for c, n in zip(classes, counts)}, ")")
            print("Training on full dataset and skipping validation/CV.")
            X_scaled = self.scaler.fit_transform(X)
            self.classifier.fit(X_scaled, y)
            # Determine threshold on training set if enabled
            if hasattr(self.classifier, 'predict_proba') and Config.OPTIMIZE_DECISION_THRESHOLD:
                y_proba = self.classifier.predict_proba(X_scaled)[:, 1]
                self.decision_threshold = self._optimize_threshold(y, y_proba)
            else:
                self.decision_threshold = float(Config.BASE_DECISION_THRESHOLD)
            print(f"Selected decision threshold (F{Config.F_BETA_FOR_THRESHOLD:.1f}-max): {self.decision_threshold:.3f}")
            # Metrics (using decision_threshold)
            try:
                y_proba = self.classifier.predict_proba(X_scaled)[:, 1]
                y_pred = (y_proba >= self.decision_threshold).astype(int)
                print("\n" + "="*60)
                print("TRAINING SET RESULTS (no hold-out, thresholded)")
                print("="*60)
                print(classification_report(y, y_pred, target_names=['Non-Extremist', 'Extremist']))
                print("\nConfusion Matrix:")
                print(confusion_matrix(y, y_pred))
                try:
                    print(f"\nROC-AUC (resubstitution): {roc_auc_score(y, y_proba):.4f}")
                except Exception:
                    pass
            except Exception:
                pass
            # Feature importance (tree-based)
            if hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                print("\nTop 15 Most Important Features:")
                for i in range(min(15, len(self.feature_names))):
                    idx = indices[i]
                    print(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
            self.is_trained = True
            return

        # Standard split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=Config.RANDOM_STATE, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"Training {self.model_type} classifier...")
        self.classifier.fit(X_train_scaled, y_train)

        # Threshold optimization on validation set
        if hasattr(self.classifier, 'predict_proba') and Config.OPTIMIZE_DECISION_THRESHOLD:
            y_val_proba = self.classifier.predict_proba(X_val_scaled)[:, 1]
            self.decision_threshold = self._optimize_threshold(y_val, y_val_proba)
        else:
            self.decision_threshold = float(Config.BASE_DECISION_THRESHOLD)
        print(f"Selected decision threshold (F{Config.F_BETA_FOR_THRESHOLD:.1f}-max): {self.decision_threshold:.3f}")

        # Evaluate with chosen threshold
        y_val_proba = self.classifier.predict_proba(X_val_scaled)[:, 1]
        y_pred = (y_val_proba >= self.decision_threshold).astype(int)

        print("\n" + "="*60)
        print("VALIDATION RESULTS (thresholded)")
        print("="*60)
        print(classification_report(y_val, y_pred, target_names=['Non-Extremist', 'Extremist']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print(f"\nROC-AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}")

        # Cross-validation bounded by min class count
        X_all_scaled = self.scaler.transform(X)
        cv_max = int(min(Config.CROSS_VALIDATION_FOLDS, counts.min()))
        if cv_max >= 2:
            cv_scores = cross_val_score(self.classifier, X_all_scaled, y, cv=cv_max)
            print(f"\n{cv_max}-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        else:
            print("\nSkipping cross-validation due to insufficient samples per class.")

        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("\nTop 15 Most Important Features:")
            for i in range(min(15, len(self.feature_names))):
                idx = indices[i]
                print(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

        self.is_trained = True

    def predict(
        self,
        multimodel_segment: Dict[str, Any],
        intonation_segment: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Predict whether a segment is extremist or not.

        Args:
            multimodel_segment: Segment from multi-model classifier
            intonation_segment: Segment from intonation pipeline

        Returns:
            Tuple of (prediction, probability) where prediction is 0 or 1
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        feature_vec = self.combine_features(multimodel_segment, intonation_segment).reshape(1, -1)
        feature_vec_scaled = self.scaler.transform(feature_vec)
        proba = self.classifier.predict_proba(feature_vec_scaled)[0, 1]
        pred = 1 if proba >= self.decision_threshold else 0
        return int(pred), float(proba)

    def predict_file(
        self,
        multimodel_file: str,
        intonation_file: str,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Predict extremist classification for all segments in a file.

        Args:
            multimodel_file: Path to multi-model classifier JSON output
            intonation_file: Path to intonation pipeline JSON output
            output_file: Optional path to save predictions

        Returns:
            Dictionary with predictions for all segments
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")

        # Load data
        with open(multimodel_file, 'r') as f:
            multimodel_data = json.load(f)

        with open(intonation_file, 'r') as f:
            intonation_data = json.load(f)

        # Get segments
        mm_segments = multimodel_data.get('segment_classifications', [])
        intonation_segments = intonation_data if isinstance(intonation_data, list) else []

        # Align segments
        aligned = self.align_segments(mm_segments, intonation_segments)

        # Make predictions
        predictions = []
        extremist_count = 0

        for mm_seg, inton_seg in aligned:
            pred, prob = self.predict(mm_seg, inton_seg)

            predictions.append({
                'start': mm_seg.get('start'),
                'end': mm_seg.get('end'),
                'is_extremist': bool(pred),
                'extremist_probability': prob,
                'text': mm_seg.get('text', '')  # Include for reference only
            })

            if pred == 1:
                extremist_count += 1

        # Calculate statistics
        total_segments = len(predictions)
        extremist_ratio = extremist_count / total_segments if total_segments > 0 else 0.0
        avg_probability = np.mean([p['extremist_probability'] for p in predictions]) if predictions else 0.0
        
        # Calculate weighted extremist score (sum of extremist segment probabilities / total segments)
        # This gives a score that considers both the percentage of extremist segments AND their confidence
        # Only extremist segments contribute to the score, weighted by their probability
        weighted_extremist_score = sum(p['extremist_probability'] for p in predictions if p['is_extremist']) / total_segments if total_segments > 0 else 0.0

        result = {
            'file': Path(multimodel_file).name,
            'total_segments': total_segments,
            'extremist_segments': extremist_count,
            'non_extremist_segments': total_segments - extremist_count,
            'extremist_ratio': extremist_ratio,
            'avg_extremist_probability': avg_probability,
            'weighted_extremist_score': weighted_extremist_score,
            'is_extremist_content': weighted_extremist_score > Config.EXTREMIST_RATIO_THRESHOLD,  # Overall classification threshold
            'predictions': predictions
        }

        # Save to file if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Predictions saved to {output_file}")

        return result

    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'decision_threshold': self.decision_threshold,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.decision_threshold = float(model_data.get('decision_threshold', Config.BASE_DECISION_THRESHOLD))
        self.is_trained = True

        print(f"Model loaded from {filepath} (threshold={self.decision_threshold:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Binary extremist classifier using multimodal features")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                       help='Mode: train or predict')
    parser.add_argument('--model_type', choices=['random_forest', 'gradient_boosting', 'logistic'],
                       default='random_forest', help='Type of classifier')
    parser.add_argument('--model_path', default='extremist_classifier.pkl',
                       help='Path to save/load trained model')

    # Training arguments
    parser.add_argument('--extremist_dir', help='Directory with extremist content outputs')
    parser.add_argument('--non_extremist_dir', help='Directory with non-extremist content outputs')
    parser.add_argument('--multimodel_suffix', default='_multimodel.json',
                       help='Suffix for multi-model classifier files')
    parser.add_argument('--intonation_suffix', default='_intonation.json',
                       help='Suffix for intonation pipeline files')

    # Prediction arguments
    parser.add_argument('--multimodel_file', help='Multi-model classifier output file')
    parser.add_argument('--intonation_file', help='Intonation pipeline output file')
    parser.add_argument('--output_file', help='Output file for predictions')

    args = parser.parse_args()

    classifier = ExtremistClassifier(model_type=args.model_type)

    if args.mode == 'train':
        if not args.extremist_dir or not args.non_extremist_dir:
            parser.error("Training mode requires --extremist_dir and --non_extremist_dir")

        print("Preparing training data...")
        X_list = []
        y_list = []

        # Load extremist samples
        extremist_path = Path(args.extremist_dir)
        extremist_files = list(extremist_path.glob(f"*{args.multimodel_suffix}"))

        for mm_file in extremist_files:
            base_name = str(mm_file).replace(args.multimodel_suffix, '')
            inton_file = base_name + args.intonation_suffix

            if Path(inton_file).exists():
                X, y = classifier.prepare_training_data(str(mm_file), inton_file, label=1)
                if len(X) > 0:
                    X_list.append(X)
                    y_list.append(y)

        print(f"Loaded {len(extremist_files)} extremist files")

        # Load non-extremist samples
        non_extremist_path = Path(args.non_extremist_dir)
        non_extremist_files = list(non_extremist_path.glob(f"*{args.multimodel_suffix}"))

        for mm_file in non_extremist_files:
            base_name = str(mm_file).replace(args.multimodel_suffix, '')
            inton_file = base_name + args.intonation_suffix

            if Path(inton_file).exists():
                X, y = classifier.prepare_training_data(str(mm_file), inton_file, label=0)
                if len(X) > 0:
                    X_list.append(X)
                    y_list.append(y)

        print(f"Loaded {len(non_extremist_files)} non-extremist files")

        # Combine all data
        if not X_list:
            print("Error: No training data found")
            return

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)

        print(f"\nTotal samples: {len(X_all)}")
        print(f"Extremist samples: {np.sum(y_all)}")
        print(f"Non-extremist samples: {len(y_all) - np.sum(y_all)}")
        print(f"Number of features: {X_all.shape[1]}")

        # Train model
        classifier.train(X_all, y_all)

        # Save model
        classifier.save_model(args.model_path)

    elif args.mode == 'predict':
        if not args.multimodel_file or not args.intonation_file:
            parser.error("Prediction mode requires --multimodel_file and --intonation_file")

        # Load model
        classifier.load_model(args.model_path)

        # Make predictions
        result = classifier.predict_file(
            args.multimodel_file,
            args.intonation_file,
            args.output_file
        )

        # Print summary
        print("\n" + "="*60)
        print("EXTREMIST CLASSIFICATION RESULTS")
        print("="*60)
        print(f"File: {result['file']}")
        print(f"Total segments: {result['total_segments']}")
        print(f"Extremist segments: {result['extremist_segments']}")
        print(f"Non-extremist segments: {result['non_extremist_segments']}")
        print(f"Extremist ratio (count-based): {result['extremist_ratio']:.2%}")
        print(f"Average extremist probability: {result['avg_extremist_probability']:.4f}")
        print(f"Weighted extremist score (probability-weighted): {result['weighted_extremist_score']:.4f}")
        print(f"Overall classification: {'EXTREMIST' if result['is_extremist_content'] else 'NON-EXTREMIST'}")


if __name__ == '__main__':
    main()
