#!/home/olafim/PycharmProjects/junction-extreme/backend/testing/transcript-proccessing/.my-env/bin/python3
"""
Multi-model hate speech and toxicity classifier.
Combines outputs from multiple models to provide comprehensive categorization.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
import warnings
from backend.config import Config
warnings.filterwarnings("ignore")


class MultiModelClassifier:
    """Classifier that combines multiple models for comprehensive hate speech analysis."""
    
    def __init__(self, device=None):
        """Initialize all classification models."""
        if device is None:
            self.device = Config.get_device()
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        device_id = Config.get_device_id() if device is None else (0 if self.device == "cuda" else -1)
        
        # Model 1: Toxic-BERT (6 toxicity categories)
        print(f"Loading {Config.TOXIC_BERT_MODEL}...")
        self.toxic_tokenizer = AutoTokenizer.from_pretrained(Config.TOXIC_BERT_MODEL)
        self.toxic_model = AutoModelForSequenceClassification.from_pretrained(Config.TOXIC_BERT_MODEL)
        self.toxic_model.to(self.device)
        self.toxic_model.eval()
        self.toxic_labels = self.toxic_model.config.id2label
        
        # Model 2: Hate speech binary classifier
        print(f"Loading {Config.HATE_DETECTION_MODEL}...")
        self.hate_pipe = pipeline(
            "text-classification",
            model=Config.HATE_DETECTION_MODEL,
            device=device_id,
            top_k=None,
            truncation=True,
            max_length=Config.MAX_TOKEN_LENGTH
        )
        
        # Model 3: Offensive language classifier
        print(f"Loading {Config.OFFENSIVE_DETECTION_MODEL}...")
        self.offensive_pipe = pipeline(
            "text-classification",
            model=Config.OFFENSIVE_DETECTION_MODEL,
            device=device_id,
            top_k=None,
            truncation=True,
            max_length=Config.MAX_TOKEN_LENGTH
        )
        
        # Model 4: Sentiment analysis (for emotional tone)
        print(f"Loading {Config.SENTIMENT_MODEL}...")
        self.sentiment_pipe = pipeline(
            "text-classification",
            model=Config.SENTIMENT_MODEL,
            device=device_id,
            top_k=None,
            truncation=True,
            max_length=Config.MAX_TOKEN_LENGTH
        )

        # Model 5: Target analysis
        print(f"Loading {Config.TARGET_ANALYSIS_MODEL}...")
        self.target_pipeline = pipeline(
            "text-classification",
            model=Config.TARGET_ANALYSIS_MODEL,
            truncation=True,
            max_length=Config.MAX_TOKEN_LENGTH
        )
        
        print("All models loaded successfully!\n")
    
    def classify_toxic(self, text: str) -> Dict[str, float]:
        """Classify using toxic-bert (6 categories)."""
        inputs = self.toxic_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_TOKEN_LENGTH,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.toxic_model(**inputs)
            probs = torch.sigmoid(outputs.logits)
        
        return {
            self.toxic_labels[i]: float(probs[0][i].item())
            for i in range(len(self.toxic_labels))
        }
    
    def classify_hate(self, text: str) -> Dict[str, float]:
        """Classify using hate speech detector."""
        # Truncate text if too long (safety measure)
        text = text[:Config.MAX_TEXT_LENGTH] if len(text) > Config.MAX_TEXT_LENGTH else text
        results = self.hate_pipe(text)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        return {item['label']: float(item['score']) for item in results}
    
    def classify_offensive(self, text: str) -> Dict[str, float]:
        """Classify using offensive language detector."""
        # Truncate text if too long (safety measure)
        text = text[:Config.MAX_TEXT_LENGTH] if len(text) > Config.MAX_TEXT_LENGTH else text
        results = self.offensive_pipe(text)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        return {item['label']: float(item['score']) for item in results}
    
    def classify_sentiment(self, text: str) -> Dict[str, float]:
        """Classify sentiment/emotional tone."""
        # Truncate text if too long (safety measure)
        text = text[:Config.MAX_TEXT_LENGTH] if len(text) > Config.MAX_TEXT_LENGTH else text
        results = self.sentiment_pipe(text)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        return {item['label']: float(item['score']) for item in results}

    def classify_target(self, text: str) -> Dict[str, float]:
        """Classify hate speech target."""
        # Truncate text if too long (safety measure)
        text = text[:Config.MAX_TEXT_LENGTH] if len(text) > Config.MAX_TEXT_LENGTH else text
        results = self.target_pipeline(text)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        return {item['label']: float(item['score']) for item in results}
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text using all models and combine results.
        
        Returns comprehensive categorization with scores from all models.
        """
        if not text or not text.strip():
            return {
                "overall_toxicity": 0.0,
                "is_toxic": False,
                "categories": {},
                "detected_issues": []
            }
        
        # Get classifications from all models
        toxic_scores = self.classify_toxic(text)
        hate_scores = self.classify_hate(text)
        offensive_scores = self.classify_offensive(text)
        sentiment_scores = self.classify_sentiment(text)
        target_scores = self.classify_target(text)
        
        # Combine all scores
        all_categories = {
            **{f"toxic_{k}": v for k, v in toxic_scores.items()},
            **{f"hate_{k}": v for k, v in hate_scores.items()},
            **{f"offensive_{k}": v for k, v in offensive_scores.items()},
            **{f"sentiment_{k}": v for k, v in sentiment_scores.items()},
            **{f"target_{k}": v for k, v in target_scores.items()}
        }
        
        # Detect issues (categories above threshold)
        threshold = Config.TOXICITY_THRESHOLD
        detected_issues = []
        
        for category, score in all_categories.items():
            # Skip non-issue categories
            if any(skip in category.lower() for skip in ['nothate', 'not-offensive', 'neutral', 'positive']):
                continue
            
            if score > threshold:
                detected_issues.append(f"{category}:{score:.3f}")
        
        # Calculate overall toxicity - only consider POSITIVE toxic indicators
        # Exclude negative labels like "NOT-HATE", "not-offensive", etc.
        toxic_scores_list = []
        
        # From toxic-bert: toxic, severe_toxic, obscene, threat, insult, identity_hate
        for key in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            if f"toxic_{key}" in all_categories:
                toxic_scores_list.append(all_categories[f"toxic_{key}"])
        
        # From hate detection: only HATE label (not NOT-HATE)
        if "hate_HATE" in all_categories:
            toxic_scores_list.append(all_categories["hate_HATE"])
        
        # From offensive detection: only offensive label (not not-offensive)
        if "offensive_offensive" in all_categories:
            toxic_scores_list.append(all_categories["offensive_offensive"])
        
        # Calculate overall toxicity as weighted average of top scores
        # Using max gives too much weight to a single model
        if toxic_scores_list:
            # Sort and take average of top 3 scores for more robust estimate
            sorted_scores = sorted(toxic_scores_list, reverse=True)
            top_scores = sorted_scores[:min(3, len(sorted_scores))]
            overall_toxicity = sum(top_scores) / len(top_scores)
        else:
            overall_toxicity = 0.0
        
        # Extract hate target if content is hateful
        hate_target = None
        hate_target_confidence = 0.0
        
        if overall_toxicity > threshold and target_scores:
            # Find the target with highest confidence
            # Filter out "not_cyberbullying" and similar negative labels
            positive_targets = {k: v for k, v in target_scores.items() 
                              if 'not' not in k.lower() and v > 0.3}
            
            if positive_targets:
                hate_target = max(positive_targets, key=positive_targets.get)
                hate_target_confidence = positive_targets[hate_target]
                
                # Clean up the target label for display
                # e.g., "religion_based" -> "Religion"
                hate_target = hate_target.replace('_', ' ').replace('-', ' ').title()
        
        return {
            "overall_toxicity": float(overall_toxicity),
            "is_toxic": overall_toxicity > threshold,
            "hate_target": hate_target,
            "hate_target_confidence": float(hate_target_confidence),
            "categories": all_categories,
            "detected_issues": detected_issues,
            "model_outputs": {
                "toxic_bert": toxic_scores,
                "hate_detection": hate_scores,
                "offensive_detection": offensive_scores,
                "sentiment": sentiment_scores,
                "target_analysis": target_scores
            }
        }
    
    def classify_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Classify a full transcript JSON file using all models.
        
        Args:
            transcript_path: Path to Whisper JSON transcript
        
        Returns:
            Dictionary with comprehensive classification results
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        full_text = data.get("text", "").strip()
        segments = data.get("segments", [])
        
        # Classify full text
        print(f"Classifying full transcript...")
        full_classification = self.classify_text(full_text)
        
        # Classify individual segments
        segment_classifications = []
        print(f"Classifying {len(segments)} segments...")
        for seg in tqdm(segments, desc="Segments"):
            seg_text = seg.get("text", "").strip()
            if seg_text:
                seg_class = self.classify_text(seg_text)
                segment_classifications.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg_text,
                    **seg_class
                })
        
        # Calculate aggregate statistics
        if segment_classifications:
            avg_toxicity = sum(s["overall_toxicity"] for s in segment_classifications) / len(segment_classifications)
            max_toxicity = max(s["overall_toxicity"] for s in segment_classifications)
            toxic_segments_count = sum(1 for s in segment_classifications if s["is_toxic"])
            
            # Aggregate detected issues
            all_issues = {}
            for seg in segment_classifications:
                for issue in seg["detected_issues"]:
                    category = issue.split(':')[0]
                    all_issues[category] = all_issues.get(category, 0) + 1
        else:
            avg_toxicity = 0.0
            max_toxicity = 0.0
            toxic_segments_count = 0
            all_issues = {}
        
        return {
            "file": os.path.basename(transcript_path),
            "full_text_classification": full_classification,
            "segment_classifications": segment_classifications,
            "statistics": {
                "total_segments": len(segment_classifications),
                "toxic_segments": toxic_segments_count,
                "avg_toxicity": float(avg_toxicity),
                "max_toxicity": float(max_toxicity),
                "language": data.get("language", "unknown"),
                "issue_frequency": all_issues
            }
        }


def classify_single_file(input_file: str, output_file: str, classifier: MultiModelClassifier):
    """Classify a single transcript file."""
    print(f"\nClassifying: {input_file}")
    
    result = classifier.classify_transcript(input_file)
    
    # Save result
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Classification saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Overall toxicity score: {result['full_text_classification']['overall_toxicity']:.4f}")
    print(f"Is toxic: {result['full_text_classification']['is_toxic']}")
    print(f"Total segments: {result['statistics']['total_segments']}")
    print(f"Toxic segments: {result['statistics']['toxic_segments']}")
    print(f"Avg segment toxicity: {result['statistics']['avg_toxicity']:.4f}")
    
    if result['full_text_classification']['detected_issues']:
        print(f"\nDetected issues in full text:")
        for issue in result['full_text_classification']['detected_issues'][:10]:
            print(f"  - {issue}")
    
    if result['statistics']['issue_frequency']:
        print(f"\nMost frequent issues across segments:")
        sorted_issues = sorted(result['statistics']['issue_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)
        for issue, count in sorted_issues[:10]:
            print(f"  - {issue}: {count} segments")
    
    return result


def classify_directory(input_dir: str, output_dir: str, classifier: MultiModelClassifier):
    """Classify all JSON transcripts in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} transcript files")
    
    all_results = []
    summary_rows = []
    
    for json_file in json_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {json_file.name}")
            print(f"{'='*60}")
            
            result = classifier.classify_transcript(str(json_file))
            all_results.append(result)
            
            # Add to summary
            summary_rows.append({
                "file": result["file"],
                "overall_toxicity": result["full_text_classification"]["overall_toxicity"],
                "is_toxic": result["full_text_classification"]["is_toxic"],
                "avg_segment_toxicity": result["statistics"]["avg_toxicity"],
                "max_segment_toxicity": result["statistics"]["max_toxicity"],
                "total_segments": result["statistics"]["total_segments"],
                "toxic_segments": result["statistics"]["toxic_segments"],
                "language": result["statistics"]["language"]
            })
            
            # Save individual result
            out_file = output_path / f"{json_file.stem}_multi_classification.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Save summary CSV
    summary_csv = output_path / "multi_classification_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n✓ Summary saved to {summary_csv}")
    
    # Save full results JSON
    full_json = output_path / "all_multi_classifications.json"
    with open(full_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Full results saved to {full_json}")
    
    return all_results, summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model hate speech and toxicity classification"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_file", help="Single JSON transcript file")
    group.add_argument("--input_dir", help="Directory with JSON transcripts")
    
    parser.add_argument("--output_file", help="Output JSON file (for --input_file)")
    parser.add_argument("--output_dir", help="Output directory (for --input_dir)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch", action="store_true", 
                       help="Process subdirectories in batch mode")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_file and not args.output_file:
        parser.error("--output_file is required when using --input_file")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    
    # Initialize classifier
    print("Initializing multi-model classifier...")
    classifier = MultiModelClassifier(device=args.device)
    
    # Single file mode
    if args.input_file:
        classify_single_file(args.input_file, args.output_file, classifier)
        return
    
    # Directory mode
    if args.batch:
        print(f"\nBatch mode: processing subdirectories in {args.input_dir}")
        subdirs = [d for d in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, d))]
        
        for subdir in subdirs:
            input_path = os.path.join(args.input_dir, subdir)
            output_path = os.path.join(args.output_dir, subdir)
            print(f"\n{'='*60}")
            print(f"Processing subdirectory: {subdir}")
            print(f"{'='*60}")
            classify_directory(input_path, output_path, classifier)
    else:
        classify_directory(args.input_dir, args.output_dir, classifier)
    
    print("\n" + "="*60)
    print("✓ ALL CLASSIFICATIONS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

