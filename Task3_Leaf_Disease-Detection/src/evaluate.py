"""
Evaluation module for Leaf Disease Detection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Configure matplotlib to handle font issues
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import config
from data_preprocessing import DataPreprocessor

class ModelEvaluator:
    def __init__(self, model_path, task='multiclass'):
        """
        Initialize the evaluator

        Args:
            model_path: Path to the trained model
            task: 'multiclass' or 'binary'
        """
        self.model_path = model_path
        self.task = task
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.class_names = None

    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print("Model loaded successfully!")
        return self.model

    def prepare_test_data(self):
        """Prepare test data generator"""
        _, _, self.test_generator = self.preprocessor.create_data_generators(task=self.task)

        # Get class names
        if self.task == 'multiclass':
            self.class_names = list(self.test_generator.class_indices.keys())
        else:
            self.class_names = ['Diseased', 'Healthy']  # Binary classification

        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes: {self.class_names}")

        return self.test_generator

    def evaluate_model(self):
        """Evaluate the model on test data"""
        if self.model is None:
            self.load_model()

        if not hasattr(self, 'test_generator'):
            self.prepare_test_data()

        print("Evaluating model on test data...")

        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)

        # Get true labels
        true_labels = self.test_generator.classes

        if self.task == 'multiclass':
            predicted_labels = np.argmax(predictions, axis=1)
        else:
            predicted_labels = (predictions > 0.5).astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)

        if self.task == 'binary':
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)

            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
        else:
            precision = precision_score(true_labels, predicted_labels, average='weighted')
            recall = recall_score(true_labels, predicted_labels, average='weighted')
            f1 = f1_score(true_labels, predicted_labels, average='weighted')

            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision (weighted): {precision:.4f}")
            print(f"Test Recall (weighted): {recall:.4f}")
            print(f"Test F1-Score (weighted): {f1:.4f}")

        # Store results
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels
        }

        return self.evaluation_results

    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        if not hasattr(self, 'evaluation_results'):
            self.evaluate_model()

        cm = confusion_matrix(
            self.evaluation_results['true_labels'],
            self.evaluation_results['predicted_labels']
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {self.task.capitalize()} Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if save_path is None:
            save_path = os.path.join(
                config.PROJECT_ROOT,
                f'confusion_matrix_{self.task}.png'
            )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return cm

    def plot_classification_report(self, save_path=None):
        """Plot classification report as heatmap"""
        if not hasattr(self, 'evaluation_results'):
            self.evaluate_model()

        # Generate classification report
        report = classification_report(
            self.evaluation_results['true_labels'],
            self.evaluation_results['predicted_labels'],
            target_names=self.class_names,
            output_dict=True
        )

        # Convert to DataFrame for plotting
        df_report = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            df_report.iloc[:-3, :3],  # Exclude macro/weighted avg rows and support column
            annot=True,
            cmap='Blues',
            fmt='.3f'
        )
        plt.title(f'Classification Report - {self.task.capitalize()} Classification')

        if save_path is None:
            save_path = os.path.join(
                config.PROJECT_ROOT,
                f'classification_report_{self.task}.png'
            )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(
            self.evaluation_results['true_labels'],
            self.evaluation_results['predicted_labels'],
            target_names=self.class_names
        ))

        return report

    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve (for binary classification)"""
        if self.task != 'binary':
            print("ROC curve is only available for binary classification")
            return

        if not hasattr(self, 'evaluation_results'):
            self.evaluate_model()

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(
            self.evaluation_results['true_labels'],
            self.evaluation_results['predictions']
        )
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)

        if save_path is None:
            save_path = os.path.join(config.PROJECT_ROOT, f'roc_curve_{self.task}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"AUC Score: {roc_auc:.4f}")

        return fpr, tpr, roc_auc

    def plot_prediction_samples(self, num_samples=16, save_path=None):
        """Plot sample predictions with actual images"""
        if not hasattr(self, 'evaluation_results'):
            self.evaluate_model()

        # Get sample images and predictions
        sample_indices = np.random.choice(
            len(self.evaluation_results['true_labels']),
            size=min(num_samples, len(self.evaluation_results['true_labels'])),
            replace=False
        )

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()

        # Get test directory path
        if self.task == 'multiclass':
            test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'test')
        else:
            test_dir = os.path.join(config.PROCESSED_DATA_DIR, 'binary_test')

        # Get image paths from test generator
        test_image_paths = []
        test_labels = []

        if os.path.exists(test_dir):
            for class_name in os.listdir(test_dir):
                class_dir = os.path.join(test_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        test_image_paths.append(os.path.join(class_dir, img_name))
                        test_labels.append(class_name)

        for i, idx in enumerate(sample_indices):
            if i >= 16:  # Limit to 16 samples
                break

            true_label = self.evaluation_results['true_labels'][idx]
            pred_label = self.evaluation_results['predicted_labels'][idx]

            if self.task == 'multiclass':
                confidence = np.max(self.evaluation_results['predictions'][idx])
            else:
                confidence = self.evaluation_results['predictions'][idx][0]
                if pred_label == 0:
                    confidence = 1 - confidence

            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]

            # Try to load actual image
            try:
                if idx < len(test_image_paths):
                    from PIL import Image
                    img = Image.open(test_image_paths[idx])
                    axes[i].imshow(img)
                else:
                    # Fallback to placeholder
                    axes[i].text(0.5, 0.5, 'Image\nNot\nAvailable',
                                ha='center', va='center', transform=axes[i].transAxes,
                                bbox=dict(boxstyle='round', facecolor='lightgray'))
            except Exception as e:
                # Fallback to text display
                axes[i].text(0.5, 0.5, f'Sample {idx}\nError loading image',
                            ha='center', va='center', transform=axes[i].transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightcoral'))

            # Set title with prediction info
            correct = "[OK]" if true_label == pred_label else "[X]"
            title_color = 'green' if true_label == pred_label else 'red'

            axes[i].set_title(f'{correct} True: {true_class.replace("Tomato___", "")}\n'
                             f'Pred: {pred_class.replace("Tomato___", "")} ({confidence:.3f})',
                             fontsize=8, color=title_color, fontweight='bold')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(sample_indices), 16):
            axes[i].axis('off')

        plt.suptitle(f'Sample Predictions - {self.task.capitalize()} Classification',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(config.PROJECT_ROOT, f'prediction_samples_{self.task}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")

        # Evaluate model
        results = self.evaluate_model()

        # Generate all plots
        self.plot_confusion_matrix()
        self.plot_classification_report()

        if self.task == 'binary':
            self.plot_roc_curve()

        self.plot_prediction_samples()

        print("Evaluation report generated successfully!")

        return results

def evaluate_model(model_path, task='multiclass'):
    """Main evaluation function"""
    evaluator = ModelEvaluator(model_path, task)
    results = evaluator.generate_evaluation_report()
    return evaluator, results

if __name__ == "__main__":
    # Evaluate multiclass model
    multiclass_model_path = config.MULTICLASS_MODEL_PATH.replace('.h5', '_efficientnet.h5')
    if os.path.exists(multiclass_model_path):
        print("Evaluating multiclass model...")
        evaluator_multi, results_multi = evaluate_model(multiclass_model_path, 'multiclass')

    # Evaluate binary model
    binary_model_path = config.BINARY_MODEL_PATH.replace('.h5', '_efficientnet.h5')
    if os.path.exists(binary_model_path):
        print("\nEvaluating binary model...")
        evaluator_binary, results_binary = evaluate_model(binary_model_path, 'binary')