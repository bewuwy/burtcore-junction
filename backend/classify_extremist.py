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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings("ignore")


class ExtremistClassifier:
    """Binary classifier for extremist content using multimodal features."""

    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier.

        Args:
            model_type: Type of classifier ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()

        if model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.feature_names = []
        self.is_trained = False

    def extract_multimodel_features(self, segment: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from multi-model classifier output.
        Does NOT include the text itself.

        Args:
            segment: Segment classification from multi-model classifier

        Returns:
            Dictionary of numerical features
        """
        features = {}

        # Overall toxicity score
        features['overall_toxicity'] = segment.get('overall_toxicity', 0.0)

        # Extract all model output scores
        model_outputs = segment.get('model_outputs', {})

        # Toxic-BERT scores (6 categories)
        toxic_bert = model_outputs.get('toxic_bert', {})
        for key, value in toxic_bert.items():
            features[f'toxic_{key}'] = value

        # Hate detection scores
        hate_det = model_outputs.get('hate_detection', {})
        features['hate_score'] = hate_det.get('HATE', 0.0)
        features['not_hate_score'] = hate_det.get('NOT-HATE', 0.0)

        # Offensive detection scores
        offensive_det = model_outputs.get('offensive_detection', {})
        features['offensive_score'] = offensive_det.get('offensive', 0.0)
        features['non_offensive_score'] = offensive_det.get('non-offensive', 0.0)

        # Sentiment scores (emotional tone)
        sentiment = model_outputs.get('sentiment', {})
        features['sentiment_negative'] = sentiment.get('negative', 0.0)
        features['sentiment_neutral'] = sentiment.get('neutral', 0.0)
        features['sentiment_positive'] = sentiment.get('positive', 0.0)

        # Target analysis - get max target score
        target = model_outputs.get('target_analysis', {})
        if target:
            features['max_target_score'] = max(target.values())
            # Individual target categories
            for key, value in target.items():
                features[f'target_{key}'] = value
        else:
            features['max_target_score'] = 0.0

        # Number of detected issues
        features['num_detected_issues'] = len(segment.get('detected_issues', []))

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

    def combine_features(
        self,
        multimodel_segment: Dict[str, Any],
        intonation_segment: Dict[str, Any]
    ) -> np.ndarray:
        """
        Combine features from both models into a single feature vector.

        Args:
            multimodel_segment: Segment from multi-model classifier
            intonation_segment: Segment from intonation pipeline

        Returns:
            Feature vector as numpy array
        """
        mm_features = self.extract_multimodel_features(multimodel_segment)
        intonation_features = self.extract_intonation_features(intonation_segment)

        # Combine all features
        all_features = {**mm_features, **intonation_features}

        # Store feature names if not already stored
        if not self.feature_names:
            self.feature_names = sorted(all_features.keys())

        # Create feature vector in consistent order
        feature_vector = np.array([all_features.get(name, 0.0) for name in self.feature_names])

        return feature_vector

    def align_segments(
        self,
        multimodel_segments: List[Dict[str, Any]],
        intonation_segments: List[Dict[str, Any]],
        time_tolerance: float = 0.5
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Align segments from multi-model classifier and intonation pipeline by timestamp.

        Args:
            multimodel_segments: List of segments from multi-model classifier
            intonation_segments: List of segments from intonation pipeline
            time_tolerance: Maximum time difference (seconds) to consider segments aligned

        Returns:
            List of (multimodel_segment, intonation_segment) pairs
        """
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

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the extremist classifier.

        Args:
            X: Feature matrix
            y: Labels (1 for extremist, 0 for non-extremist)
            validation_split: Fraction of data to use for validation
        """
        if len(X) == 0:
            raise ValueError("Cannot train with empty dataset")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train classifier
        print(f"Training {self.model_type} classifier...")
        self.classifier.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        y_pred = self.classifier.predict(X_val_scaled)
        y_pred_proba = self.classifier.predict_proba(X_val_scaled)[:, 1]

        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(classification_report(y_val, y_pred, target_names=['Non-Extremist', 'Extremist']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print(f"\nROC-AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")

        # Cross-validation
        X_all_scaled = self.scaler.transform(X)
        cv_scores = cross_val_score(self.classifier, X_all_scaled, y, cv=5)
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Feature importance (for tree-based models)
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

        feature_vec = self.combine_features(multimodel_segment, intonation_segment)
        feature_vec = feature_vec.reshape(1, -1)
        feature_vec_scaled = self.scaler.transform(feature_vec)

        prediction = self.classifier.predict(feature_vec_scaled)[0]
        probability = self.classifier.predict_proba(feature_vec_scaled)[0, 1]

        return int(prediction), float(probability)

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

        result = {
            'file': Path(multimodel_file).name,
            'total_segments': total_segments,
            'extremist_segments': extremist_count,
            'non_extremist_segments': total_segments - extremist_count,
            'extremist_ratio': extremist_ratio,
            'avg_extremist_probability': avg_probability,
            'is_extremist_content': extremist_ratio > 0.3,  # Overall classification threshold
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
            'model_type': self.model_type
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
        self.is_trained = True

        print(f"Model loaded from {filepath}")


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
        print(f"Extremist ratio: {result['extremist_ratio']:.2%}")
        print(f"Average extremist probability: {result['avg_extremist_probability']:.4f}")
        print(f"Overall classification: {'EXTREMIST' if result['is_extremist_content'] else 'NON-EXTREMIST'}")


if __name__ == '__main__':
    main()

