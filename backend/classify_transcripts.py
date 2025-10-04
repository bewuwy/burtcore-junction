#!/usr/bin/env python3
"""
Classify transcripts using hate-measure-roberta-large model.
Processes Whisper JSON transcripts and outputs hate speech classification scores.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm


class HateSpeechClassifier:
    """Classifier for hate speech using RoBERTa model."""
    
    def __init__(self, model_name="ucberkeley-dlab/hate-measure-roberta-large", device=None):
        """
        Initialize the hate speech classifier.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu), auto-detect if None
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def classify_text(self, text: str) -> Dict[str, float]:
        """
        Classify a single text for hate speech.

        Args:
            text: Text to classify
        
        Returns:
            Dictionary with hate_score (probability of hate speech)
        """
        if not text or not text.strip():
            return {"hate_score": 0.0, "hate_label": "non-hate", "confidence": 0.0}

        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Extract hate speech probability (assuming binary classification)
        # Model output: [non-hate, hate] or similar
        hate_prob = probs[0][1].item()  # Probability of hate speech class

        return {
            "hate_score": float(hate_prob),
            "hate_label": "hate" if hate_prob > 0.5 else "non-hate",
            "confidence": float(max(probs[0]))
        }

    def classify_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Classify a full transcript JSON file.
        
        Args:
            transcript_path: Path to Whisper JSON transcript
        
        Returns:
            Dictionary with classification results
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        full_text = data.get("text", "").strip()
        segments = data.get("segments", [])
        
        # Classify full text
        full_classification = self.classify_text(full_text)
        
        # Classify individual segments
        segment_classifications = []
        for seg in segments:
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
            avg_hate_score = sum(s["hate_score"] for s in segment_classifications) / len(segment_classifications)
            max_hate_score = max(s["hate_score"] for s in segment_classifications)
            hate_segments_count = sum(1 for s in segment_classifications if s["hate_label"] == "hate")
        else:
            avg_hate_score = 0.0
            max_hate_score = 0.0
            hate_segments_count = 0
        
        return {
            "file": os.path.basename(transcript_path),
            "full_text_classification": full_classification,
            "segment_classifications": segment_classifications,
            "statistics": {
                "total_segments": len(segment_classifications),
                "hate_segments": hate_segments_count,
                "avg_hate_score": float(avg_hate_score),
                "max_hate_score": float(max_hate_score),
                "language": data.get("language", "unknown")
            }
        }


def classify_directory(input_dir: str, output_dir: str, classifier: HateSpeechClassifier):
    """
    Classify all JSON transcripts in a directory.
    
    Args:
        input_dir: Directory containing JSON transcripts
        output_dir: Directory to save classification results
        classifier: HateSpeechClassifier instance
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} transcript files")
    
    all_results = []
    summary_rows = []
    
    # Process each file
    for json_file in tqdm(json_files, desc="Classifying transcripts"):
        try:
            result = classifier.classify_transcript(str(json_file))
            all_results.append(result)
            
            # Add to summary
            summary_rows.append({
                "file": result["file"],
                "full_text_hate_score": result["full_text_classification"]["hate_score"],
                "full_text_label": result["full_text_classification"]["hate_label"],
                "avg_segment_hate_score": result["statistics"]["avg_hate_score"],
                "max_segment_hate_score": result["statistics"]["max_hate_score"],
                "total_segments": result["statistics"]["total_segments"],
                "hate_segments": result["statistics"]["hate_segments"],
                "language": result["statistics"]["language"]
            })
            
            # Save individual result
            out_file = output_path / f"{json_file.stem}_classification.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Save summary CSV
    summary_csv = output_path / "classification_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n✓ Summary saved to {summary_csv}")
    
    # Save full results JSON
    full_json = output_path / "all_classifications.json"
    with open(full_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Full results saved to {full_json}")
    
    return all_results, summary_rows


def classify_single_file(input_file: str, output_file: str, classifier: HateSpeechClassifier):
    """
    Classify a single transcript file.
    
    Args:
        input_file: Path to JSON transcript
        output_file: Path to save classification result
        classifier: HateSpeechClassifier instance
    """
    print(f"Classifying: {input_file}")
    
    result = classifier.classify_transcript(input_file)
    
    # Save result
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Classification saved to {output_file}")
    
    # Print summary
    print("\nClassification Summary:")
    print(f"  Full text hate score: {result['full_text_classification']['hate_score']:.4f}")
    print(f"  Full text label: {result['full_text_classification']['hate_label']}")
    print(f"  Total segments: {result['statistics']['total_segments']}")
    print(f"  Hate segments: {result['statistics']['hate_segments']}")
    print(f"  Avg segment hate score: {result['statistics']['avg_hate_score']:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify transcripts for hate speech using RoBERTa model"
    )
    
    # Single file or directory mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file",
        help="Single JSON transcript file to classify"
    )
    group.add_argument(
        "--input_dir",
        help="Directory containing JSON transcript files"
    )
    
    parser.add_argument(
        "--output_file",
        help="Output JSON file (required for --input_file)"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save classification results (required for --input_dir)"
    )
    parser.add_argument(
        "--model",
        default="ucberkeley-dlab/hate-measure-roberta-large",
        help="HuggingFace model name (default: ucberkeley-dlab/hate-measure-roberta-large)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple subdirectories (expects input_dir to contain subdirectories)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_file and not args.output_file:
        parser.error("--output_file is required when using --input_file")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    
    # Initialize classifier
    classifier = HateSpeechClassifier(model_name=args.model, device=args.device)
    
    # Single file mode
    if args.input_file:
        classify_single_file(args.input_file, args.output_file, classifier)
        return
    
    # Directory mode
    if args.batch:
        # Batch mode: process multiple subdirectories
        print(f"\nBatch mode: processing subdirectories in {args.input_dir}")
        subdirs = [d for d in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, d))]
        
        for subdir in subdirs:
            input_path = os.path.join(args.input_dir, subdir)
            output_path = os.path.join(args.output_dir, subdir)
            print(f"\n{'='*60}")
            print(f"Processing: {subdir}")
            print(f"{'='*60}")
            classify_directory(input_path, output_path, classifier)
    else:
        # Single directory mode
        print(f"\nProcessing directory: {args.input_dir}")
        classify_directory(args.input_dir, args.output_dir, classifier)
    
    print("\n✓ All classifications complete!")


if __name__ == "__main__":
    main()

