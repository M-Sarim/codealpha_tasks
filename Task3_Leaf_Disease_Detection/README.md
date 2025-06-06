# Leaf Disease Detection Project

An advanced deep learning project for detecting diseases in tomato plant leaves using computer vision and convolutional neural networks.

## 🌱 Project Overview

This project provides an image-based automatic inspection interface for plant disease detection. It uses state-of-the-art deep learning techniques to categorize tomato plant leaves as healthy or infected with various diseases.

### Key Features

- **Multi-class Classification**: Detect 9 different tomato diseases + healthy plants
- **Binary Classification**: Simple healthy vs diseased classification
- **Transfer Learning**: Uses pre-trained models (EfficientNet, ResNet50, MobileNetV2, VGG16)
- **Custom CNN**: Option to use a custom-built CNN architecture
- **Data Augmentation**: Comprehensive data augmentation for better generalization
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Easy Prediction Interface**: Simple API for predicting new images

## 🗂️ Project Structure

```
leaf-disease-detection/
├── data/
│   ├── raw/                    # Original tomato dataset
│   └── processed/              # Preprocessed and split data
├── src/
│   ├── data_preprocessing.py   # Data analysis and preprocessing
│   ├── model.py               # Model architectures
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Model evaluation
│   └── predict.py             # Prediction interface
├── models/                    # Saved trained models
├── notebooks/                 # Jupyter notebooks for exploration
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🦠 Disease Classes

The model can detect the following tomato diseases:

1. **Tomato\_\_\_Bacterial_spot**
2. **Tomato\_\_\_Early_blight**
3. **Tomato\_\_\_Late_blight**
4. **Tomato\_\_\_Leaf_Mold**
5. **Tomato\_\_\_Septoria_leaf_spot**
6. **Tomato\_\_\_Spider_mites Two-spotted_spider_mite**
7. **Tomato\_\_\_Target_Spot**
8. **Tomato\_\_\_Tomato_Yellow_Leaf_Curl_Virus**
9. **Tomato\_\_\_Tomato_mosaic_virus**
10. **Tomato\_\_\_healthy**

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd leaf-disease-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Complete Pipeline (Recommended)

```bash
# Run everything: preprocessing, training, evaluation, and visualization
python main.py --mode full --task multiclass --model_type efficientnet --epochs 20
```

This will:

- ✅ Analyze the dataset and create comprehensive visualizations
- ✅ Preprocess data with proper train/val/test splits
- ✅ Train the model with transfer learning
- ✅ Evaluate performance with detailed metrics
- ✅ Generate all visualization reports

### 3. Step-by-Step Approach

#### Data Preprocessing & Visualization

```python
from src.data_preprocessing import DataPreprocessor
from src.visualizations import create_all_visualizations

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Analyze the dataset (creates dataset_analysis.png)
df = preprocessor.analyze_dataset()

# Create comprehensive visualizations
viz_dir = create_all_visualizations()

# Create processed dataset with proper splits
preprocessor.create_processed_dataset()
```

#### Training Models

```python
from src.train import train_model

# Train multiclass model
trainer_multi, history_multi = train_model(
    task='multiclass',
    model_type='efficientnet',
    epochs=50
)

# Train binary model
trainer_binary, history_binary = train_model(
    task='binary',
    model_type='efficientnet',
    epochs=50
)
```

#### Model Evaluation

```python
from src.evaluate import evaluate_model

# Evaluate multiclass model
evaluator_multi, results_multi = evaluate_model(
    'models/tomato_disease_multiclass_model_efficientnet.h5',
    task='multiclass'
)

# Evaluate binary model
evaluator_binary, results_binary = evaluate_model(
    'models/tomato_disease_binary_model_efficientnet.h5',
    task='binary'
)
```

#### Making Predictions

```python
from src.predict import predict_image, predict_directory

# Predict single image
result = predict_image(
    'models/tomato_disease_multiclass_model_efficientnet.h5',
    'path/to/your/image.jpg',
    task='multiclass'
)

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")

# Predict all images in a directory
results = predict_directory(
    'models/tomato_disease_multiclass_model_efficientnet.h5',
    'path/to/your/images/',
    task='multiclass'
)
```

## 📊 Comprehensive Visualizations

The project includes extensive visualization capabilities:

### 🎨 **Automatic Visualizations Generated:**

1. **Dataset Analysis** (`dataset_analysis.png`)

   - Class distribution with counts
   - Healthy vs diseased ratio
   - Train/validation splits
   - Class balance analysis
   - Summary statistics

2. **Sample Images Grid** (`visualizations/sample_images_grid.png`)

   - Representative images from each disease class
   - 4 samples per class in organized grid
   - Clear class labels

3. **Image Properties Analysis** (`visualizations/image_properties_analysis.png`)

   - Image dimensions distribution
   - File size analysis
   - Aspect ratio distribution
   - Format and color mode statistics

4. **Data Augmentation Examples** (`visualizations/data_augmentation_examples.png`)

   - Shows original vs augmented images
   - Demonstrates preprocessing effects
   - Helps understand data transformations

5. **Training Progress** (`training_history_[task]_[model].png`)

   - Loss and accuracy curves
   - Validation metrics
   - Learning rate schedules
   - Training convergence analysis

6. **Model Evaluation** (Generated after training)
   - Confusion matrices with heatmaps
   - Classification reports
   - ROC curves (for binary classification)
   - Sample predictions with actual images

### 🔍 **Creating Visualizations:**

```python
# Create all visualizations at once
from src.visualizations import create_all_visualizations
viz_dir = create_all_visualizations()

# Or create specific visualizations
from src.visualizations import VisualizationManager
viz = VisualizationManager()

# Sample images grid
viz.plot_sample_images_grid()

# Image properties analysis
viz.plot_image_properties_analysis()

# Data augmentation examples
viz.plot_data_augmentation_examples()
```

## 🏗️ Model Architectures

The project supports multiple model architectures:

### Transfer Learning Models

- **EfficientNetB0** (Recommended): Efficient and accurate
- **ResNet50**: Deep residual networks
- **MobileNetV2**: Lightweight for mobile deployment
- **VGG16**: Classic CNN architecture

### Custom CNN

- Multi-layer CNN with batch normalization and dropout
- Designed specifically for plant disease detection

## 📊 Configuration

Key configuration parameters in `config.py`:

```python
# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Data splits
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
```

## 📁 **Why is the Models Folder Empty?**

The `models/` folder is initially empty because **no models have been trained yet**. Models are created and saved during the training process:

### 🎯 **After Training, You'll Find:**

```
models/
├── tomato_disease_multiclass_model_efficientnet.h5     # Multiclass model
├── tomato_disease_multiclass_model_efficientnet_info.json  # Training info
├── tomato_disease_binary_model_efficientnet.h5        # Binary model
├── tomato_disease_binary_model_efficientnet_info.json     # Training info
└── [other model architectures if trained]
```

### 🚀 **To Populate the Models Folder:**

```bash
# Quick training (20 epochs)
python main.py --mode train --task multiclass --model_type efficientnet --epochs 20

# Or full pipeline
python main.py --mode full --task multiclass --model_type efficientnet --epochs 50
```

### 📊 **Model Files Include:**

- **`.h5` files**: Complete trained models ready for prediction
- **`_info.json` files**: Training metadata, metrics, and configuration
- **Training plots**: Automatically saved visualization of training progress

## 📈 Performance Metrics

The models are evaluated using comprehensive metrics:

### 🎯 **Classification Metrics:**

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each class (macro/weighted averages)
- **Recall**: Recall for each class (macro/weighted averages)
- **F1-Score**: Harmonic mean of precision and recall
- **Top-3 Accuracy**: For multiclass classification (checks if true class is in top 3 predictions)

### 📊 **Visual Evaluation:**

- **Confusion Matrix**: Detailed heatmap showing classification results
- **Classification Report**: Per-class precision, recall, F1-score
- **ROC Curve**: For binary classification with AUC score
- **Sample Predictions**: Grid showing actual images with predictions
- **Training History**: Loss and accuracy curves over epochs

## 🔧 Advanced Usage

### Custom Training

```python
from src.model import DiseaseDetectionModel
from src.data_preprocessing import DataPreprocessor

# Create custom model
model_builder = DiseaseDetectionModel(task='multiclass', model_type='efficientnet')
model = model_builder.build_model()
model = model_builder.compile_model()

# Prepare data
preprocessor = DataPreprocessor()
train_gen, val_gen, test_gen = preprocessor.create_data_generators(task='multiclass')

# Train with custom parameters
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=model_builder.get_callbacks('custom_model.h5')
)
```

### Fine-tuning

```python
# Unfreeze base model layers for fine-tuning
model_builder.unfreeze_base_model(layers_to_unfreeze=20)

# Continue training with lower learning rate
history_fine = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen
)
```

## 📝 Results and Outputs

The project generates comprehensive outputs organized in different directories:

### 📁 **Generated Files Structure:**

```
leaf-disease-detection/
├── models/                          # 🤖 Trained models (after training)
│   ├── *.h5                        # Model files
│   └── *_info.json                 # Training metadata
├── visualizations/                  # 🎨 All visualizations
│   ├── sample_images_grid.png      # Sample images from each class
│   ├── image_properties_analysis.png # Image statistics
│   ├── data_augmentation_examples.png # Augmentation examples
│   └── dataset_report.txt          # Summary report
├── dataset_analysis.png            # 📊 Main dataset analysis
├── training_history_*.png          # 📈 Training progress plots
├── confusion_matrix_*.png          # 🎯 Model evaluation
├── classification_report_*.png     # 📋 Detailed metrics
├── roc_curve_*.png                 # 📉 ROC analysis (binary)
└── prediction_samples_*.png        # 🔮 Sample predictions
```

### 🎯 **Output Categories:**

#### **Training Outputs:**

- **Model Files**: Complete trained models ready for deployment
- **Training Plots**: Loss/accuracy curves, learning rate schedules
- **Training Info**: JSON files with metrics, hyperparameters, timestamps

#### **Evaluation Outputs:**

- **Confusion Matrices**: Heatmaps showing classification performance
- **Classification Reports**: Per-class precision, recall, F1-scores
- **ROC Curves**: Binary classification performance analysis
- **Sample Predictions**: Visual grid showing model predictions on test images

#### **Data Analysis Outputs:**

- **Dataset Statistics**: Class distributions, balance analysis
- **Sample Images**: Representative images from each disease class
- **Image Properties**: Dimension, size, format analysis
- **Augmentation Examples**: Before/after data transformation examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: PlantVillage Dataset
- Pre-trained models: TensorFlow/Keras model zoo
- Inspiration: Agricultural AI research community

## 📞 Support

For questions or issues, please:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Happy Plant Disease Detection! 🌱🔬**
