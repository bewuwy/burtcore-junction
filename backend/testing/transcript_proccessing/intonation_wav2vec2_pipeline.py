#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa, soundfile as sf
import torch
from transformers import pipeline
import warnings
import whisper
warnings.filterwarnings("ignore", category=UserWarning)

def extract_pitch_features(y, sr, fmin=50.0, fmax=400.0):
    hop_length = int(0.01 * sr)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=2048, hop_length=hop_length
    )
    f0 = np.array(f0, dtype=float)
    vmask = ~np.isnan(f0)
    if np.sum(vmask) == 0:
        # No pitch detected - return 0.0 instead of NaN for JSON compatibility
        return dict(f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0,
                    f0_range=0.0, f0_slope=0.0, f0_final=0.0)
    v = f0[vmask]
    f0_mean, f0_std = np.mean(v), np.std(v)
    f0_min, f0_max = np.min(v), np.max(v)
    f0_range = f0_max - f0_min
    x = np.arange(v.size)
    slope = np.polyfit(x, v, 1)[0]
    n_final = min(25, v.size)
    f0_final = np.mean(v[-n_final:])
    return dict(f0_mean=float(f0_mean), f0_std=float(f0_std),
                f0_min=float(f0_min), f0_max=float(f0_max),
                f0_range=float(f0_range), f0_slope=float(slope),
                f0_final=float(f0_final))

def extract_energy_features(y, sr):
    frame_length, hop_length = int(0.025 * sr), int(0.01 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return dict(rms_mean=float(np.mean(rms)), rms_max=float(np.max(rms)))

def slice_audio(y, sr, start, end):
    i0 = max(0, int(start * sr))
    i1 = min(len(y), int(end * sr))
    return y[i0:i1]

def ensure_wav_mono_16k(input_audio):
    y, sr = librosa.load(input_audio, sr=16000, mono=True)
    norm_path = Path(".cache_intonation") / "audio_16k_mono.wav"
    norm_path.parent.mkdir(exist_ok=True)
    sf.write(norm_path, y, 16000)
    return str(norm_path), y, 16000

def process_segments(y_all, sr, segments, model="superb/wav2vec2-base-superb-er", device=None, fmin=50.0, fmax=400.0, batch_size=8):
    """
    Process audio segments and extract intonation and emotion features using batch processing.
    
    Args:
        y_all: Audio array (numpy array)
        sr: Sample rate (should be 16000)
        segments: List of segment dicts with 'start', 'end', 'text' keys
        model: Wav2Vec2 model for emotion classification
        device: Device to use (None for auto-detect, "cuda" or "cpu")
        fmin: Minimum frequency for pitch extraction
        fmax: Maximum frequency for pitch extraction
        batch_size: Batch size for emotion classification (default: 8)
    
    Returns:
        List of dicts with emotion and intonation features for each segment
    """
    if not segments:
        return []
    
    # Determine device
    device_id = 0 if (device == "cuda" or (device is None and torch.cuda.is_available())) else -1
    emo_pipe = pipeline("audio-classification", model=model, device=device_id, batch_size=batch_size)
    
    # Prepare cache directory
    cache_dir = Path(".cache_intonation")
    cache_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract all audio segments and save them
    audio_files = []
    segment_data = []
    
    for idx, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        text = seg.get("text", "").strip()
        y_seg = slice_audio(y_all, sr, start, end)
        
        # Save segment to temporary file
        seg_path = cache_dir / f"seg_{idx}.wav"
        sf.write(seg_path, y_seg, sr)
        audio_files.append(str(seg_path))
        
        segment_data.append({
            "index": idx,
            "start": start,
            "end": end,
            "text": text,
            "y_seg": y_seg
        })
    
    # Step 2: Batch process all segments for emotion classification
    # The pipeline will automatically batch these efficiently on GPU
    emotion_predictions = emo_pipe(audio_files)
    
    # Step 3: Combine results with pitch/energy features
    results = []
    for seg_data, emotion_preds in zip(segment_data, emotion_predictions):
        # Get top emotion prediction
        top = emotion_preds[0] if emotion_preds else {"label": "unknown", "score": 0.0}
        
        # Extract pitch and energy features (CPU-bound, can't batch easily)
        f0_feats = extract_pitch_features(seg_data["y_seg"], sr, fmin=fmin, fmax=fmax)
        en_feats = extract_energy_features(seg_data["y_seg"], sr)
        
        row = {
            "start": seg_data["start"],
            "end": seg_data["end"],
            "duration": seg_data["end"] - seg_data["start"],
            "text": seg_data["text"],
            "emotion": top["label"],
            "emotion_score": float(top["score"]),
            **f0_feats,
            **en_feats
        }
        results.append(row)
    
    # Clean up temporary files
    for audio_file in audio_files:
        try:
            Path(audio_file).unlink()
        except:
            pass
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Intonation + Emotion (Wav2Vec2) per Whisper segment")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--whisper_json", required=True)
    parser.add_argument("--out_csv", default="emotion_intonation_timeline.csv")
    parser.add_argument("--out_json", default="emotion_intonation_timeline.json")
    parser.add_argument("--model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=400.0)
    args = parser.parse_args()

    norm_wav_path, y_all, sr = ensure_wav_mono_16k(args.audio)
    segments = json.loads(Path(args.whisper_json).read_text()).get("segments", [])
    if not segments:
        raise RuntimeError("No 'segments' found in Whisper JSON")

    device = 0 if (args.device == "cuda" or (args.device is None and torch.cuda.is_available())) else -1
    emo_pipe = pipeline("audio-classification", model=args.model, device=device)

    results = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        text = seg.get("text", "").strip()
        y_seg = slice_audio(y_all, sr, start, end)
        tmp_wav = Path(".cache_intonation/tmp.wav")
        sf.write(tmp_wav, y_seg, sr)

        preds = emo_pipe(str(tmp_wav))
        top = preds[0] if preds else {"label": "unknown", "score": 0.0}
        f0_feats = extract_pitch_features(y_seg, sr, fmin=args.fmin, fmax=args.fmax)
        en_feats = extract_energy_features(y_seg, sr)

        row = {
            "start": start, "end": end, "duration": end - start,
            "text": text, "emotion": top["label"], "emotion_score": float(top["score"]),
            **f0_feats, **en_feats
        }
        results.append(row)

    # Save as CSV
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"[OK] CSV saved to {args.out_csv}")

    # Save as JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON saved to {args.out_json}")

if __name__ == "__main__":
    main()