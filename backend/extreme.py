from backend.testing.transcript_proccessing.transcript_wav2vec_pipeline import transcribe_single_file
from backend.testing.transcript_proccessing.intonation_wav2vec2_pipeline import process_segments
from backend.multi_model_classifier import MultiModelClassifier
from backend.classify_extremist import ExtremistClassifier
from backend.config import Config
import json
from pathlib import Path
import os

# Initialize classifiers once at module level for reuse
_classifier = None
_extremist_classifier = None

def get_classifier():
    """Get or initialize the multi-model classifier (singleton pattern)."""
    global _classifier
    if _classifier is None:
        print("Initializing multi-model classifier...")
        _classifier = MultiModelClassifier()
    return _classifier

def get_extremist_classifier():
    """Get or initialize the extremist classifier (singleton pattern)."""
    global _extremist_classifier
    if _extremist_classifier is None:
        print("Initializing extremist classifier...")
        _extremist_classifier = ExtremistClassifier()
        # Try to load trained model if it exists
        model_path = Config.EXTREMIST_MODEL_PATH
        if os.path.exists(model_path):
            try:
                _extremist_classifier.load_model(str(model_path))
                print(f"✓ Loaded trained extremist classifier from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load extremist classifier model: {e}")
                print("Extremist classification will not be available.")
                _extremist_classifier = None
        else:
            print(f"Warning: No trained extremist classifier model found at {model_path}")
            print("Extremist classification will not be available.")
            _extremist_classifier = None
    return _extremist_classifier

def detect_sarcasm(intonation_data: dict, toxicity_score: float, text: str = "") -> tuple[float, str]:
    """
    Detect sarcasm using intonation patterns and text-emotion mismatch.
    
    Sarcasm indicators:
    1. Wide pitch range + flat trajectory (deadpan sarcasm)
    2. Very flat/monotone delivery with long duration
    3. Mismatch between positive emotion and toxic text
    4. Exaggerated pitch variation (mock enthusiasm)
    
    Args:
        intonation_data: Intonation features from wav2vec2 pipeline
        toxicity_score: Toxicity score from multi-model classifier
        text: Optional text content for additional analysis
        
    Returns:
        Tuple of (sarcasm_probability, primary_pattern) where:
        - sarcasm_probability: 0-1 confidence that segment is sarcastic
        - primary_pattern: Description of detected pattern
    """
    if not intonation_data:
        return 0.0, "none"
    
    # Get features
    f0_range = intonation_data.get('f0_range', 0.0)
    f0_std = intonation_data.get('f0_std', 0.0)
    f0_slope = abs(intonation_data.get('f0_slope', 0.0))
    duration = intonation_data.get('duration', 0.0)
    emotion = intonation_data.get('emotion', '').lower()
    emotion_score = intonation_data.get('emotion_score', 0.0)
    
    sarcasm_indicators = []
    
    # Pattern 1: Deadpan sarcasm (wide range but flat delivery)
    if f0_range > 150 and f0_std < 20 and duration > 2.0:
        sarcasm_indicators.append(('deadpan', 0.35))
    
    # Pattern 2: Monotone delivery (very flat pitch)
    if f0_std < 15 and duration > 3.0:
        sarcasm_indicators.append(('monotone', 0.30))
    
    # Pattern 3: Emotion-text mismatch (happy/neutral emotion but toxic text)
    if emotion in ['happy', 'neutral'] and emotion_score > 0.5 and toxicity_score > 0.6:
        sarcasm_indicators.append(('emotion_mismatch', 0.40))
    
    # Pattern 4: Exaggerated pitch (mock enthusiasm)
    if f0_range > 200 and f0_std > 40:
        sarcasm_indicators.append(('exaggerated', 0.30))
    
    # Pattern 5: Unusually slow delivery (emphasis/mockery)
    if duration > 5.0 and f0_std < 25:
        sarcasm_indicators.append(('slow_delivery', 0.25))
    
    # Pattern 6: Happy emotion with moderate toxicity (classic sarcasm)
    if emotion == 'happy' and emotion_score > 0.6 and 0.4 < toxicity_score < 0.7:
        sarcasm_indicators.append(('happy_toxic', 0.45))
    
    # Calculate overall sarcasm probability
    if not sarcasm_indicators:
        return 0.0, "none"
    
    # Take the strongest indicator and add partial credit for others
    sarcasm_indicators.sort(key=lambda x: x[1], reverse=True)
    primary_pattern = sarcasm_indicators[0][0]
    primary_score = sarcasm_indicators[0][1]
    
    # Add bonus for multiple indicators
    if len(sarcasm_indicators) > 1:
        secondary_bonus = sum(score * 0.3 for _, score in sarcasm_indicators[1:])
        total_score = min(primary_score + secondary_bonus, 0.95)
    else:
        total_score = primary_score
    
    return total_score, primary_pattern


def apply_intonation_heuristic(toxicity_score: float, intonation_data: dict, text: str = "") -> tuple[float, float, dict]:
    """
    Apply intonation-based heuristic to modify toxicity score when extremist classifier is unavailable.
    
    Considers:
    - Sarcasm detection (inverts/reduces extremity when detected) - OPTIONAL via Config
    - Emotion (angry, fear, disgust boost score)
    - Pitch variation (high variability suggests emotional intensity)
    - Energy (high energy can indicate aggressive speech)
    
    Args:
        toxicity_score: Original toxicity score from multi-model classifier (0-1)
        intonation_data: Intonation features from wav2vec2 pipeline
        text: Optional text content for sarcasm detection
        
    Returns:
        Tuple of (modified_score, confidence, sarcasm_info) where:
        - modified_score: Adjusted toxicity score (0-1)
        - confidence: Confidence in the heuristic (0-1)
        - sarcasm_info: Dict with sarcasm detection details
    """
    if not intonation_data:
        return toxicity_score, 0.0, {"detected": False, "probability": 0.0, "pattern": "none"}
    
    # STEP 1: Check for sarcasm first (if enabled in config)
    sarcasm_prob = 0.0
    sarcasm_pattern = "none"
    sarcasm_detected = False
    
    if Config.SARCASM_DETECTION_ENABLED:
        sarcasm_prob, sarcasm_pattern = detect_sarcasm(intonation_data, toxicity_score, text)
        sarcasm_detected = sarcasm_prob > Config.SARCASM_THRESHOLD
    
    sarcasm_info = {
        "detected": sarcasm_detected,
        "probability": float(sarcasm_prob),
        "pattern": sarcasm_pattern
    }
    
    # If sarcasm is detected (and enabled), invert/reduce the extremity
    if sarcasm_detected:
        # Sarcastic toxic content is less threatening - reduce score significantly
        # Higher sarcasm probability = more reduction
        reduction_factor = Config.SARCASM_REDUCTION_MIN + (Config.SARCASM_REDUCTION_MAX * sarcasm_prob)
        modified_score = toxicity_score * (1.0 - reduction_factor)
        
        # Confidence is based on sarcasm detection strength
        confidence = sarcasm_prob * 0.8  # Cap at 0.8 for sarcasm-based modification
        
        return modified_score, confidence, sarcasm_info
    
    # Start with original toxicity score
    modified_score = toxicity_score
    adjustment_factors = []
    
    # Factor 1: Emotion analysis
    emotion = intonation_data.get('emotion', '').lower()
    emotion_score = intonation_data.get('emotion_score', 0.0)
    
    if emotion in ['angry', 'fear', 'disgust'] and emotion_score > 0.5:
        # Negative emotions boost the score
        emotion_boost = 0.15 * emotion_score
        adjustment_factors.append(('emotion', emotion_boost))
    elif emotion in ['happy', 'neutral'] and emotion_score > 0.5:
        # Positive/neutral emotions slightly reduce score
        emotion_reduction = -0.05 * emotion_score
        adjustment_factors.append(('emotion', emotion_reduction))
    
    # Factor 2: Pitch variation (high variability can indicate emotional intensity)
    f0_std = intonation_data.get('f0_std', 0.0)
    f0_range = intonation_data.get('f0_range', 0.0)
    
    # Normalize pitch variation (typical std is 20-50 Hz, range 50-200 Hz)
    normalized_std = min(f0_std / 50.0, 1.0) if f0_std > 0 else 0.0
    normalized_range = min(f0_range / 200.0, 1.0) if f0_range > 0 else 0.0
    
    if normalized_std > 0.6 or normalized_range > 0.6:
        # High pitch variation suggests emotional intensity
        pitch_boost = 0.10 * max(normalized_std, normalized_range)
        adjustment_factors.append(('pitch_variation', pitch_boost))
    
    # Factor 3: Energy/loudness (high energy can indicate aggressive speech)
    rms_mean = intonation_data.get('rms_mean', 0.0)
    rms_max = intonation_data.get('rms_max', 0.0)
    
    # Normalize energy (typical RMS values are 0.01-0.1)
    normalized_energy = min(rms_mean / 0.1, 1.0) if rms_mean > 0 else 0.0
    
    if normalized_energy > 0.6:
        # High energy suggests intensity
        energy_boost = 0.08 * normalized_energy
        adjustment_factors.append(('energy', energy_boost))
    
    # Factor 4: Pitch slope (rapid changes can indicate aggression)
    f0_slope = abs(intonation_data.get('f0_slope', 0.0))
    
    # Normalize slope (typical values -50 to +50 Hz/s)
    normalized_slope = min(f0_slope / 50.0, 1.0) if f0_slope > 0 else 0.0
    
    if normalized_slope > 0.5:
        # Rapid pitch changes
        slope_boost = 0.05 * normalized_slope
        adjustment_factors.append(('pitch_slope', slope_boost))
    
    # Apply adjustments
    total_adjustment = sum(factor[1] for factor in adjustment_factors)
    modified_score = min(max(toxicity_score + total_adjustment, 0.0), 1.0)
    
    # Calculate confidence based on how many factors contributed
    # and the strength of the original toxicity score
    confidence = 0.0
    if adjustment_factors:
        # Base confidence on number of factors and their strength
        num_factors = len(adjustment_factors)
        avg_adjustment = abs(total_adjustment) / max(num_factors, 1)
        
        # Higher confidence if multiple factors agree and adjustments are significant
        confidence = min(0.3 + (num_factors * 0.15) + (avg_adjustment * 2.0), 0.85)
        
        # Reduce confidence if original toxicity is very low or very high
        # (heuristic is less reliable at extremes)
        if toxicity_score < 0.2 or toxicity_score > 0.8:
            confidence *= 0.7
    
    return modified_score, confidence, sarcasm_info

def evaluate(file_path: str, output_file: str = "test.json"):
    """
    Evaluate a video file by transcribing it and extracting intonation/emotion features,
    and classifying content for hate speech, toxicity, and offensive language.
    
    Args:
        file_path: Path to the video file
        output_file: Path to save the combined results JSON file
        
    Returns:
        dict: Contains segments with timing info, intonation results, and toxicity classification
    """
    print("evaluating file", file_path)

    # Get classifier instances
    classifier = get_classifier()
    extremist_classifier = get_extremist_classifier()

    # Get raw transcription result and audio (without saving)
    whisper_result, audio = transcribe_single_file(file_path, output_file=None)
    
    # Audio is already loaded at 16kHz mono
    y_all = audio
    sr = 16000

    # Get original segments from whisper result for processing
    original_segments = whisper_result["segments"]

    # Run intonation/emotion extraction on original segments
    print("Extracting intonation and emotion features...")
    intonation_results = process_segments(y_all, sr, original_segments)
    full_text = whisper_result.get("text", "").strip()
    full_classification = classifier.classify_text(full_text) if full_text else None

    # Classify individual segments (batch processing)
    segment_texts = [seg.get("text", "").strip() for seg in original_segments]
    segment_classifications = []
    for text in segment_texts:
        if text:
            seg_class = classifier.classify_text(text)
            segment_classifications.append(seg_class)
        else:
            segment_classifications.append(None)

    segments_response = []
    extremist_probabilities = []
    extremist_segments_count = 0
    extremist_probabilities = []  # All probabilities for averaging
    extremist_weighted_sum = 0.0  # Sum of probabilities for extremist segments only
    

    for idx, segment in enumerate(original_segments):
        seg_data = {
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text", ""),
            "startTime": {
                "minute": int(segment["start"] / 60),
                "second": int(segment["start"] % 60)
            },
            "endTime": {
                "minute": int(segment["end"] / 60),
                "second": int(segment["end"] % 60)
            },
        }
        # Add intonation results if available
        if idx < len(intonation_results):
            seg_data["intonation"] = intonation_results[idx]
        # Add classification results if available
        if idx < len(segment_classifications) and segment_classifications[idx]:
            classification = segment_classifications[idx]
            seg_data["classification"] = classification
            seg_data["extreme"] = classification["overall_toxicity"]
            if classification.get("hate_target"):
                seg_data["hateTarget"] = classification["hate_target"]
                seg_data["hateTargetConfidence"] = classification["hate_target_confidence"]
            # FINAL STEP: Use extremist classifier to combine features and intonation
            if extremist_classifier and getattr(extremist_classifier, 'is_trained', False) and idx < len(intonation_results):
                try:
                    mm_segment = {
                        'start': segment.get('start'),
                        'end': segment.get('end'),
                        'text': segment.get('text'),
                        'overall_toxicity': classification['overall_toxicity'],
                        'model_outputs': classification.get('model_outputs', {}),
                        'detected_issues': classification.get('detected_issues', [])
                    }
                    inton_segment = intonation_results[idx]
                    is_extremist, extremist_prob = extremist_classifier.predict(mm_segment, inton_segment)
                    seg_data["isExtremist"] = bool(is_extremist)
                    seg_data["extremistProbability"] = float(extremist_prob)
                    seg_data["heuristicUsed"] = False
                    if is_extremist:
                        extremist_segments_count += 1
                        extremist_weighted_sum += extremist_prob
                    extremist_probabilities.append(extremist_prob)
                except Exception as e:
                    print(f"Warning: Could not apply extremist classifier to segment {idx}: {e}")
                    seg_data["isExtremist"] = None
                    seg_data["extremistProbability"] = None
                    seg_data["heuristicUsed"] = False
            elif idx < len(intonation_results):
                # FALLBACK: Use intonation-based heuristic when extremist classifier is unavailable
                inton_data = intonation_results[idx]
                seg_text = segment.get('text', '')
                modified_score, heuristic_confidence, sarcasm_info = apply_intonation_heuristic(
                    classification['overall_toxicity'], 
                    inton_data,
                    seg_text
                )
                seg_data["extreme"] = modified_score
                seg_data["sarcasm"] = sarcasm_info
                threshold = Config.TOXICITY_THRESHOLD - (0.1 * heuristic_confidence)
                is_extremist_heuristic = modified_score > threshold
                seg_data["isExtremist"] = is_extremist_heuristic
                seg_data["extremistProbability"] = float(modified_score)
                seg_data["heuristicUsed"] = True
                seg_data["heuristicConfidence"] = float(heuristic_confidence)
                if is_extremist_heuristic:
                    extremist_segments_count += 1
                    extremist_weighted_sum += modified_score
                extremist_probabilities.append(modified_score)
            else:
                seg_data["isExtremist"] = None
                seg_data["extremistProbability"] = None
                seg_data["heuristicUsed"] = False
        else:
            seg_data["extreme"] = 0.0
            seg_data["isExtremist"] = None
            seg_data["extremistProbability"] = None
            seg_data["heuristicUsed"] = False
        segments_response.append(seg_data)

    # Calculate aggregate statistics
    toxic_segments_count = sum(1 for c in segment_classifications if c and c.get("is_toxic"))
    avg_toxicity = sum(c["overall_toxicity"] for c in segment_classifications if c) / len(segment_classifications) if segment_classifications else 0.0
    max_toxicity = max((c["overall_toxicity"] for c in segment_classifications if c), default=0.0)
    avg_extremist_prob = sum(extremist_probabilities) / len(extremist_probabilities) if extremist_probabilities else 0.0
    max_extremist_prob = max(extremist_probabilities, default=0.0)
    extremist_ratio = extremist_segments_count / len(segments_response) if len(segments_response) > 0 else 0.0
    
    # Calculate weighted extremist score (sum of extremist segment probabilities / total segments)
    # This gives a score that considers both the percentage of extremist segments AND their confidence
    # Only extremist segments contribute to the score, weighted by their probability
    weighted_extremist_score = extremist_weighted_sum / len(segments_response) if len(segments_response) > 0 else 0.0

    # Determine if content is extremist overall (using weighted score with threshold from config)
    is_extremist_content = weighted_extremist_score > Config.EXTREMIST_RATIO_THRESHOLD if extremist_probabilities else None

    # Prepare combined results
    combined_results = {
        "segments": segments_response,
        "classification": full_classification,
        "statistics": {
            "total_segments": len(segments_response),
            "toxic_segments": toxic_segments_count,
            "avg_toxicity": float(avg_toxicity),
            "max_toxicity": float(max_toxicity),
            "extremist_segments": extremist_segments_count,
            "avg_extremist_probability": float(avg_extremist_prob),
            "max_extremist_probability": float(max_extremist_prob),
            "extremist_ratio": float(extremist_ratio),
            "weighted_extremist_score": float(weighted_extremist_score),
            "is_extremist_content": is_extremist_content,
            "language": whisper_result.get("language", "unknown"),
        }
    }

    # Save combined results to JSON
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved combined results to {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total segments: {len(segments_response)}")
    print(f"  Toxic segments: {toxic_segments_count}")
    print(f"  Average toxicity: {avg_toxicity:.3f}")
    print(f"  Max toxicity: {max_toxicity:.3f}")
    if extremist_probabilities:
        heuristic_used = any(seg.get("heuristicUsed", False) for seg in segments_response)
        if heuristic_used:
            print(f"  [Using intonation-based heuristic - no trained extremist classifier]")
        print(f"  Extremist segments: {extremist_segments_count}")
        print(f"  Average extremist probability: {avg_extremist_prob:.3f}")
        print(f"  Max extremist probability: {max_extremist_prob:.3f}")
        print(f"  Extremist ratio (count-based): {extremist_ratio:.3f}")
        print(f"  Weighted extremist score (probability-weighted): {weighted_extremist_score:.3f}")
        print(f"  Overall classification: {'EXTREMIST' if is_extremist_content else 'NON-EXTREMIST'}")

    return {
        "segments": segments_response,
        "classification": full_classification,
        "statistics": combined_results["statistics"],
    }
