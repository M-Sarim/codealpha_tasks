"""
Main script for Leaf Disease Detection Project
Demonstrates the complete workflow from data preprocessing to prediction
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

import config
from src.data_preprocessing import DataPreprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image, predict_directory
from src.visualizations import create_all_visualizations

def main():
    parser = argparse.ArgumentParser(description='Leaf Disease Detection Pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate', 'predict', 'full'],
                       default='full', help='Mode to run')
    parser.add_argument('--task', choices=['multiclass', 'binary'], default='multiclass',
                       help='Classification task')
    parser.add_argument('--model_type', choices=['efficientnet', 'resnet', 'mobilenet', 'vgg16', 'custom_cnn'],
                       default='efficientnet', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images for prediction')
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation/prediction')

    args = parser.parse_args()

    print("🌱 Leaf Disease Detection Project")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_type}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    if args.mode in ['preprocess', 'full']:
        run_preprocessing()

    if args.mode in ['train', 'full']:
        run_training(args.task, args.model_type, args.epochs)

    if args.mode in ['evaluate', 'full']:
        run_evaluation(args.task, args.model_type, args.model_path)

    if args.mode == 'predict':
        run_prediction(args.task, args.model_type, args.image_path, args.image_dir, args.model_path)

def run_preprocessing():
    """Run data preprocessing"""
    print("\n🔄 Starting Data Preprocessing...")
    start_time = time.time()

    try:
        preprocessor = DataPreprocessor()

        # Analyze dataset
        print("📊 Analyzing dataset...")
        df = preprocessor.analyze_dataset()
        print(f"✅ Dataset analysis complete. Found {len(df)} classes with {df['Total_Count'].sum()} total images.")

        # Create comprehensive visualizations
        print("🎨 Creating comprehensive visualizations...")
        viz_dir = create_all_visualizations()
        print(f"✅ Visualizations created in: {viz_dir}")

        # Create processed dataset
        print("🏗️  Creating processed dataset...")
        preprocessor.create_processed_dataset()
        print("✅ Processed dataset created successfully!")

        elapsed_time = time.time() - start_time
        print(f"⏱️  Preprocessing completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return False

    return True

def run_training(task, model_type, epochs):
    """Run model training"""
    print(f"\n🚀 Starting Model Training...")
    print(f"Task: {task}, Model: {model_type}, Epochs: {epochs}")
    start_time = time.time()

    try:
        # Train model
        trainer, history = train_model(
            task=task,
            model_type=model_type,
            epochs=epochs,
            fine_tune=True
        )

        elapsed_time = time.time() - start_time
        print(f"✅ Training completed in {elapsed_time:.2f} seconds")
        print(f"📁 Model saved to: {trainer.model_path}")

        # Print final metrics
        if history and hasattr(history, 'history'):
            final_acc = history.history['val_accuracy'][-1]
            final_loss = history.history['val_loss'][-1]
            print(f"📈 Final validation accuracy: {final_acc:.4f}")
            print(f"📉 Final validation loss: {final_loss:.4f}")

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

    return True

def run_evaluation(task, model_type, model_path=None):
    """Run model evaluation"""
    print(f"\n📊 Starting Model Evaluation...")
    start_time = time.time()

    try:
        # Determine model path
        if model_path is None:
            if task == 'multiclass':
                model_path = config.MULTICLASS_MODEL_PATH.replace('.h5', f'_{model_type}.h5')
            else:
                model_path = config.BINARY_MODEL_PATH.replace('.h5', f'_{model_type}.h5')

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first or provide a valid model path.")
            return False

        # Evaluate model
        evaluator, results = evaluate_model(model_path, task)

        elapsed_time = time.time() - start_time
        print(f"✅ Evaluation completed in {elapsed_time:.2f} seconds")

        # Print key metrics
        print(f"📈 Test Accuracy: {results['accuracy']:.4f}")
        print(f"📈 Test Precision: {results['precision']:.4f}")
        print(f"📈 Test Recall: {results['recall']:.4f}")
        print(f"📈 Test F1-Score: {results['f1_score']:.4f}")

    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        return False

    return True

def run_prediction(task, model_type, image_path=None, image_dir=None, model_path=None):
    """Run prediction on images"""
    print(f"\n🔮 Starting Prediction...")

    try:
        # Determine model path
        if model_path is None:
            if task == 'multiclass':
                model_path = config.MULTICLASS_MODEL_PATH.replace('.h5', f'_{model_type}.h5')
            else:
                model_path = config.BINARY_MODEL_PATH.replace('.h5', f'_{model_type}.h5')

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first or provide a valid model path.")
            return False

        if image_path:
            # Predict single image
            print(f"🖼️  Predicting single image: {image_path}")
            result = predict_image(model_path, image_path, task)

            print(f"✅ Prediction: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"🏥 Health Status: {'Healthy' if result['is_healthy'] else 'Diseased'}")

        elif image_dir:
            # Predict directory of images
            print(f"📁 Predicting images in directory: {image_dir}")
            results = predict_directory(model_path, image_dir, task)

            if results:
                healthy_count = sum(1 for r in results if r.get('is_healthy', False))
                total_count = len([r for r in results if 'error' not in r])

                print(f"✅ Processed {total_count} images")
                print(f"🌱 Healthy: {healthy_count}")
                print(f"🦠 Diseased: {total_count - healthy_count}")
            else:
                print("❌ No images found or processed")

        else:
            print("❌ Please provide either --image_path or --image_dir for prediction")
            return False

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return False

    return True

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")

    # Check if raw data exists
    if not os.path.exists(config.RAW_DATA_DIR):
        print(f"❌ Raw data directory not found: {config.RAW_DATA_DIR}")
        return False

    # Check if train and val directories exist
    train_dir = os.path.join(config.RAW_DATA_DIR, 'train')
    val_dir = os.path.join(config.RAW_DATA_DIR, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ Training data directory not found: {train_dir}")
        return False

    if not os.path.exists(val_dir):
        print(f"❌ Validation data directory not found: {val_dir}")
        return False

    # Check if disease classes exist
    missing_classes = []
    for class_name in config.DISEASE_CLASSES:
        class_train_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_train_dir):
            missing_classes.append(class_name)

    if missing_classes:
        print(f"❌ Missing disease classes: {missing_classes}")
        return False

    print("✅ All requirements met!")
    return True

if __name__ == "__main__":
    # Check requirements first
    if not check_requirements():
        print("\n❌ Requirements check failed. Please ensure the dataset is properly set up.")
        sys.exit(1)

    # Run main function
    try:
        main()
        print("\n🎉 Pipeline completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)
