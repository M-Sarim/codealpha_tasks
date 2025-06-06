"""
Prediction module for Leaf Disease Detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import tensorflow as tf

# Configure matplotlib to handle font issues
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config

class DiseasePredictor:
    def __init__(self, model_path, task='multiclass'):
        """
        Initialize the predictor

        Args:
            model_path: Path to the trained model
            task: 'multiclass' or 'binary'
        """
        self.model_path = model_path
        self.task = task
        self.model = None
        self.class_names = None
        self._load_model()
        self._set_class_names()

    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print("Model loaded successfully!")

    def _set_class_names(self):
        """Set class names based on task"""
        if self.task == 'multiclass':
            self.class_names = config.DISEASE_CLASSES
        else:
            self.class_names = ['Diseased', 'Healthy']

    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image array
        """
        # Load and resize image
        img = load_img(image_path, target_size=config.IMAGE_SIZE)

        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array, img

    def predict_single_image(self, image_path, show_image=True):
        """
        Predict disease for a single image

        Args:
            image_path: Path to the image file
            show_image: Whether to display the image

        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)

        if self.task == 'multiclass':
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
        else:
            confidence = predictions[0][0]
            predicted_class_idx = int(confidence > 0.5)
            predicted_class = self.class_names[predicted_class_idx]

            top_3_predictions = [
                {
                    'class': 'Healthy',
                    'confidence': float(1 - confidence)
                },
                {
                    'class': 'Diseased',
                    'confidence': float(confidence)
                }
            ]
            top_3_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Create result dictionary
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions,
            'is_healthy': predicted_class == 'Tomato___healthy' or predicted_class == 'Healthy'
        }

        # Display image and results
        if show_image:
            self._display_prediction(original_img, result)

        return result

    def predict_batch(self, image_paths, show_results=True):
        """
        Predict diseases for multiple images

        Args:
            image_paths: List of image file paths
            show_results: Whether to display results

        Returns:
            List of prediction results
        """
        results = []

        print(f"Processing {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

            try:
                result = self.predict_single_image(image_path, show_image=False)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })

        if show_results:
            self._display_batch_results(results)

        return results

    def _display_prediction(self, image, result):
        """Display single prediction result"""
        plt.figure(figsize=(12, 6))

        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Input Image\n{os.path.basename(result['image_path'])}")
        plt.axis('off')

        # Display prediction results
        plt.subplot(1, 2, 2)

        # Create bar plot for top predictions
        classes = [pred['class'].replace('Tomato___', '') for pred in result['top_3_predictions']]
        confidences = [pred['confidence'] for pred in result['top_3_predictions']]

        colors = ['green' if result['is_healthy'] and i == 0 else 'red' if not result['is_healthy'] and i == 0 else 'lightblue'
                 for i in range(len(classes))]

        bars = plt.barh(classes, confidences, color=colors)
        plt.xlabel('Confidence')
        plt.title(f"Prediction: {result['predicted_class'].replace('Tomato___', '')}\n"
                 f"Confidence: {result['confidence']:.3f}")
        plt.xlim(0, 1)

        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{conf:.3f}', va='center')

        plt.tight_layout()
        plt.show()

    def _display_batch_results(self, results):
        """Display batch prediction results"""
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            print("No valid predictions to display")
            return

        # Create summary statistics
        healthy_count = sum(1 for r in valid_results if r['is_healthy'])
        diseased_count = len(valid_results) - healthy_count

        print(f"\nBatch Prediction Summary:")
        print(f"Total images processed: {len(valid_results)}")
        print(f"Healthy plants: {healthy_count}")
        print(f"Diseased plants: {diseased_count}")
        print(f"Average confidence: {np.mean([r['confidence'] for r in valid_results]):.3f}")

        # Plot summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Pie chart for healthy vs diseased
        axes[0].pie([healthy_count, diseased_count],
                   labels=['Healthy', 'Diseased'],
                   autopct='%1.1f%%',
                   colors=['green', 'red'])
        axes[0].set_title('Health Status Distribution')

        # Confidence distribution
        confidences = [r['confidence'] for r in valid_results]
        axes[1].hist(confidences, bins=20, alpha=0.7, color='blue')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution')
        axes[1].axvline(np.mean(confidences), color='red', linestyle='--',
                       label=f'Mean: {np.mean(confidences):.3f}')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        # Print detailed results
        print("\nDetailed Results:")
        for i, result in enumerate(valid_results[:10]):  # Show first 10
            status = "✓ Healthy" if result['is_healthy'] else "✗ Diseased"
            print(f"{i+1:2d}. {os.path.basename(result['image_path']):30s} | "
                  f"{status:12s} | Confidence: {result['confidence']:.3f}")

        if len(valid_results) > 10:
            print(f"... and {len(valid_results) - 10} more results")

    def predict_from_directory(self, directory_path, show_results=True):
        """
        Predict diseases for all images in a directory

        Args:
            directory_path: Path to directory containing images
            show_results: Whether to display results

        Returns:
            List of prediction results
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(directory_path, file))

        if not image_paths:
            print(f"No image files found in {directory_path}")
            return []

        print(f"Found {len(image_paths)} images in {directory_path}")

        return self.predict_batch(image_paths, show_results)

def predict_image(model_path, image_path, task='multiclass'):
    """
    Convenience function to predict a single image

    Args:
        model_path: Path to the trained model
        image_path: Path to the image file
        task: 'multiclass' or 'binary'

    Returns:
        Prediction result dictionary
    """
    predictor = DiseasePredictor(model_path, task)
    return predictor.predict_single_image(image_path)

def predict_directory(model_path, directory_path, task='multiclass'):
    """
    Convenience function to predict all images in a directory

    Args:
        model_path: Path to the trained model
        directory_path: Path to directory containing images
        task: 'multiclass' or 'binary'

    Returns:
        List of prediction results
    """
    predictor = DiseasePredictor(model_path, task)
    return predictor.predict_from_directory(directory_path)

if __name__ == "__main__":
    # Example usage
    multiclass_model_path = config.MULTICLASS_MODEL_PATH.replace('.h5', '_efficientnet.h5')

    if os.path.exists(multiclass_model_path):
        print("Testing prediction with multiclass model...")

        # Test with a sample image from the test set
        test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test')
        if os.path.exists(test_dir):
            # Find a sample image
            for class_dir in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_dir)
                if os.path.isdir(class_path):
                    images = os.listdir(class_path)
                    if images:
                        sample_image = os.path.join(class_path, images[0])
                        print(f"Testing with sample image: {sample_image}")

                        result = predict_image(multiclass_model_path, sample_image, 'multiclass')
                        print(f"Prediction: {result['predicted_class']}")
                        print(f"Confidence: {result['confidence']:.3f}")
                        break
    else:
        print(f"Model not found: {multiclass_model_path}")
        print("Please train the model first using train.py")