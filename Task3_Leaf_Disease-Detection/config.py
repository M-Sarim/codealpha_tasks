"""
Configuration file for Leaf Disease Detection Project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Data configuration
IMAGE_SIZE = (224, 224)  # Standard size for transfer learning models
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model configuration
NUM_CLASSES = 10  # 9 diseases + 1 healthy
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Disease classes
DISEASE_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Binary classification mapping (Healthy vs Diseased)
BINARY_MAPPING = {
    'Tomato___healthy': 'Healthy',
    'Tomato___Bacterial_spot': 'Diseased',
    'Tomato___Early_blight': 'Diseased',
    'Tomato___Late_blight': 'Diseased',
    'Tomato___Leaf_Mold': 'Diseased',
    'Tomato___Septoria_leaf_spot': 'Diseased',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Diseased',
    'Tomato___Target_Spot': 'Diseased',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Diseased',
    'Tomato___Tomato_mosaic_virus': 'Diseased'
}

# Data augmentation parameters
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Model save paths
MULTICLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'tomato_disease_multiclass_model.h5')
BINARY_MODEL_PATH = os.path.join(MODELS_DIR, 'tomato_disease_binary_model.h5')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'model_weights.h5')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
