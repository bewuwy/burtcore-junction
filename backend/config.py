#!/usr/bin/env python3
"""
Configuration file for the extremist content detection system.
Centralizes all configurable parameters for models, devices, and thresholds.
"""
import os
import torch
from pathlib import Path
from typing import Literal


class Config:
    """Central configuration for all models and processing pipelines."""
    
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    
    # Device selection: 'auto', 'cuda', 'cpu'
    # 'auto' will use CUDA if available, otherwise CPU
    # Note: Set to 'auto' if you get CUDA initialization errors
    DEVICE: Literal['auto', 'cuda', 'cpu'] = 'auto'  # Changed from 'cuda' to 'auto' for stability

    @staticmethod
    def get_device() -> str:
        """Get the actual device to use based on configuration and availability."""
        if Config.DEVICE == 'auto':
            try:
                # Force CUDA initialization if available
                if torch.cuda.is_available():
                    # Initialize CUDA context
                    torch.cuda.init()
                    return 'cuda'
                else:
                    return 'cpu'
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {e}")
                return 'cpu'
        elif Config.DEVICE == 'cuda':
            try:
                if not torch.cuda.is_available():
                    print("Warning: CUDA requested but not available. Falling back to CPU.")
                    return 'cpu'
                # Try to initialize CUDA context
                torch.cuda.init()
                # Verify we can actually use it
                _ = torch.cuda.device_count()
                return 'cuda'
            except Exception as e:
                print(f"Warning: CUDA requested but initialization failed: {e}")
                print("Falling back to CPU.")
                return 'cpu'
        return Config.DEVICE
    
    @staticmethod
    def get_device_id() -> int:
        """Get device ID for pipeline (0 for CUDA, -1 for CPU)."""
        return 0 if Config.get_device() == 'cuda' else -1
    
    # ============================================================================
    # MULTI-MODEL CLASSIFIER CONFIGURATION
    # ============================================================================
    
    # Model selection for toxicity detection
    TOXIC_BERT_MODEL = "unitary/toxic-bert"
    HATE_DETECTION_MODEL = "cardiffnlp/twitter-roberta-base-hate-latest"
    OFFENSIVE_DETECTION_MODEL = "cardiffnlp/twitter-roberta-base-offensive"
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    TARGET_ANALYSIS_MODEL = "wesleyacheng/hate-speech-multilabel-classification-with-bert"
    
    # Toxicity thresholds
    TOXICITY_THRESHOLD = 0.5  # Minimum score to consider content toxic
    EXTREMIST_RATIO_THRESHOLD = 0.3  # Ratio of toxic segments to classify video as extremist
    
    # Text processing
    MAX_TEXT_LENGTH = 2000  # Maximum text length before truncation
    MAX_TOKEN_LENGTH = 512  # Maximum tokens for model input
    
    # ============================================================================
    # EXTREMIST CLASSIFIER CONFIGURATION
    # ============================================================================
    
    # Classifier type: 'random_forest', 'gradient_boosting', 'logistic'
    EXTREMIST_CLASSIFIER_TYPE: Literal['random_forest', 'gradient_boosting', 'logistic'] = 'random_forest'
    
    # Random Forest parameters
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2
    RF_MAX_FEATURES = 'sqrt'
    
    # Gradient Boosting parameters
    GB_N_ESTIMATORS = 150
    GB_MAX_DEPTH = 10
    GB_LEARNING_RATE = 0.1
    GB_SUBSAMPLE = 0.8
    
    # Logistic Regression parameters
    LR_MAX_ITER = 1000
    LR_C = 1.0
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    CROSS_VALIDATION_FOLDS = 5
    RANDOM_STATE = 42
    
    # Segment alignment
    TIME_TOLERANCE = 0.5  # Maximum time difference (seconds) for segment alignment
    
    # ============================================================================
    # SARCASM DETECTION CONFIGURATION
    # ============================================================================
    
    # Enable/disable sarcasm detection
    SARCASM_DETECTION_ENABLED = False
    
    # Sarcasm detection threshold (0-1)
    # Higher = more conservative (fewer detections, fewer false positives)
    # Lower = more sensitive (more detections, more false positives)
    SARCASM_THRESHOLD = 0.4
    
    # Sarcasm score reduction
    # When sarcasm is detected, reduce toxicity score by this range
    SARCASM_REDUCTION_MIN = 0.3  # Minimum reduction (30%)
    SARCASM_REDUCTION_MAX = 0.5  # Additional reduction based on confidence (up to 50% more)
    # Total reduction range: 30-80% (0.3 + 0.5 * sarcasm_probability)
    
    # ============================================================================
    # WHISPER CONFIGURATION
    # ============================================================================
    
    # Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'
    WHISPER_MODEL_SIZE = 'base'
    
    # Whisper parameters
    WHISPER_LANGUAGE = "en"
    WHISPER_BEAM_SIZE = 5
    WHISPER_BEST_OF = 5
    
    # Anti-hallucination parameters
    WHISPER_CONDITION_ON_PREVIOUS_TEXT = False  # Prevents repetitions
    WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
    WHISPER_LOGPROB_THRESHOLD = -1.0
    WHISPER_NO_SPEECH_THRESHOLD = 0.6
    WHISPER_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Fallback temperatures
    
    # ============================================================================
    # INTONATION & EMOTION CONFIGURATION
    # ============================================================================
    
    # Emotion detection model
    EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    # Audio processing
    SAMPLE_RATE = 16000
    
    # Pitch extraction
    PITCH_METHOD = 'yin'  # 'yin' or 'pyin'
    F0_MIN = 75  # Minimum F0 in Hz
    F0_MAX = 600  # Maximum F0 in Hz
    
    # ============================================================================
    # FILE & PATH CONFIGURATION
    # ============================================================================
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "backend" / "temp_uploads"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = BASE_DIR / "models"
    
    # Model save paths
    EXTREMIST_MODEL_PATH = MODEL_DIR / "extremist_classifier.pkl"
    
    # Create directories if they don't exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    
    # FastAPI settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = True
    
    # File upload limits
    MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    LOG_LEVEL = 'INFO'
    
    # Verbose output
    VERBOSE = True
    
    # ============================================================================
    # PERFORMANCE CONFIGURATION
    # ============================================================================
    
    # Batch processing
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Cache models in memory (faster but more memory usage)
    CACHE_MODELS = True
    
    @staticmethod
    def print_config():
        """Print current configuration."""
        print("=" * 80)
        print("CURRENT CONFIGURATION")
        print("=" * 80)
        print(f"Device: {Config.get_device()}")
        print(f"Whisper Model: {Config.WHISPER_MODEL_SIZE}")
        print(f"Extremist Classifier: {Config.EXTREMIST_CLASSIFIER_TYPE}")
        print(f"Toxicity Threshold: {Config.TOXICITY_THRESHOLD}")
        print(f"Extremist Ratio Threshold: {Config.EXTREMIST_RATIO_THRESHOLD}")
        print(f"Sarcasm Detection: {'Enabled' if Config.SARCASM_DETECTION_ENABLED else 'Disabled'}")
        if Config.SARCASM_DETECTION_ENABLED:
            print(f"  Sarcasm Threshold: {Config.SARCASM_THRESHOLD}")
            print(f"  Reduction Range: {Config.SARCASM_REDUCTION_MIN*100:.0f}%-{(Config.SARCASM_REDUCTION_MIN + Config.SARCASM_REDUCTION_MAX)*100:.0f}%")
        print(f"API Host: {Config.API_HOST}:{Config.API_PORT}")
        print(f"Verbose: {Config.VERBOSE}")
        print("=" * 80)
