import os
from enum import Enum

class ModelType(Enum):
    RESNET50 = "resnet50"
    EFFICIENTNET = "efficientnet"
    CUSTOM_CNN = "custom_cnn"

# Dataset configuration
DATASET_PATH = "/path/to/deepfake_dataset"
TRAIN_RATIO = 0.8
FRAME_SAMPLE_RATE = 5  # Process every 5th frame to reduce computation

# Face detection configuration
FACE_DETECTION_SIZE = (224, 224)  # MTCNN output size
MIN_FACE_CONFIDENCE = 0.8         # Minimum confidence for face detection

# EVA configuration
EVA_LEVELS = 4                     # Pyramid levels for EVA
EVA_AMPLIFICATION_FACTOR = 10      # Amplification factor
EVA_FREQUENCY_MIN = 0.05          # Minimum frequency to amplify (Hz)
EVA_FREQUENCY_MAX = 0.4           # Maximum frequency to amplify (Hz)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
FEATURE_DIM = 512                  # Output dimension of feature extractor

# Paths
MODEL_SAVE_PATH = "saved_models"
VISUALIZATION_PATH = "visualizations"