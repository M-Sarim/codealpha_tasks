"""
Training module for Leaf Disease Detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime
import json
import config

# Configure matplotlib to handle font issues
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from data_preprocessing import DataPreprocessor
from model import create_model

class ModelTrainer:
    def __init__(self, task='multiclass', model_type='efficientnet'):
        """
        Initialize the trainer

        Args:
            task: 'multiclass' or 'binary'
            model_type: 'efficientnet', 'resnet', 'mobilenet', 'vgg16', 'custom_cnn'
        """
        self.task = task
        self.model_type = model_type
        self.model = None
        self.model_builder = None
        self.history = None
        self.preprocessor = DataPreprocessor()

        # Set model save path
        if task == 'multiclass':
            self.model_path = config.MULTICLASS_MODEL_PATH.replace('.h5', f'_{model_type}.h5')
        else:
            self.model_path = config.BINARY_MODEL_PATH.replace('.h5', f'_{model_type}.h5')

    def prepare_data(self):
        """Prepare data generators"""
        print("Preparing data generators...")

        self.train_generator, self.val_generator, self.test_generator = \
            self.preprocessor.create_data_generators(task=self.task)

        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Number of classes: {self.train_generator.num_classes}")

        # Print class indices
        print("Class indices:", self.train_generator.class_indices)

        return self.train_generator, self.val_generator, self.test_generator

    def build_and_compile_model(self):
        """Build and compile the model"""
        print(f"Building {self.model_type} model for {self.task} classification...")

        self.model, self.model_builder = create_model(
            task=self.task,
            model_type=self.model_type
        )

        print("Model built successfully!")
        print(self.model.summary())

        return self.model

    def train_model(self, epochs=None, fine_tune=True):
        """Train the model"""
        if epochs is None:
            epochs = config.EPOCHS

        print(f"Starting training for {epochs} epochs...")

        # Get callbacks
        callbacks = self.model_builder.get_callbacks(self.model_path)

        # Calculate steps
        steps_per_epoch = self.train_generator.samples // config.BATCH_SIZE
        validation_steps = self.val_generator.samples // config.BATCH_SIZE

        # Initial training
        print("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs // 2,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Fine-tuning (if enabled and not custom CNN)
        if fine_tune and self.model_type != 'custom_cnn':
            print("Phase 2: Fine-tuning with unfrozen layers...")

            # Unfreeze some layers
            self.model_builder.unfreeze_base_model()

            # Continue training with lower learning rate
            history2 = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs // 2,
                validation_data=self.val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )

            # Combine histories
            self.history = self._combine_histories(history1, history2)
        else:
            self.history = history1

        print("Training completed!")
        return self.history

    def _combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined_history = {}

        for key in hist1.history.keys():
            combined_history[key] = hist1.history[key] + hist2.history[key]

        return type('History', (), {'history': combined_history})()

    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot additional metrics based on task
        metric_plotted = False

        # Try to plot top-3 accuracy for multiclass
        if self.task == 'multiclass' and 'top_3_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
            axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
            axes[1, 0].set_title('Top-3 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            metric_plotted = True

        # Try to plot precision for binary classification
        elif self.task == 'binary' and 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            metric_plotted = True

        # If no additional metrics available, show a summary
        if not metric_plotted:
            axes[1, 0].text(0.5, 0.5, f'Additional metrics\nnot available\nfor {self.task} task',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 0].set_title('Additional Metrics')
            axes[1, 0].axis('off')

        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(config.PROJECT_ROOT, f'training_history_{self.task}_{self.model_type}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def save_training_info(self):
        """Save training information"""
        if self.history is None:
            print("No training history available.")
            return

        training_info = {
            'task': self.task,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat(),
            'final_metrics': {
                'train_accuracy': float(self.history.history['accuracy'][-1]),
                'val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'train_loss': float(self.history.history['loss'][-1]),
                'val_loss': float(self.history.history['val_loss'][-1])
            },
            'best_metrics': {
                'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
                'best_val_loss': float(min(self.history.history['val_loss']))
            },
            'epochs_trained': len(self.history.history['accuracy']),
            'class_indices': self.train_generator.class_indices
        }

        info_path = self.model_path.replace('.h5', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)

        print(f"Training info saved to: {info_path}")
        return training_info

def train_model(task='multiclass', model_type='efficientnet', epochs=None, fine_tune=True):
    """Main training function"""
    print(f"Starting training pipeline for {task} classification using {model_type}")

    # Initialize trainer
    trainer = ModelTrainer(task=task, model_type=model_type)

    # Prepare data
    trainer.prepare_data()

    # Build model
    trainer.build_and_compile_model()

    # Train model
    history = trainer.train_model(epochs=epochs, fine_tune=fine_tune)

    # Plot results
    trainer.plot_training_history()

    # Save training info
    trainer.save_training_info()

    print(f"Training completed! Model saved to: {trainer.model_path}")

    return trainer, history

if __name__ == "__main__":
    # Train multiclass model
    print("Training multiclass model...")
    trainer_multi, history_multi = train_model(task='multiclass', model_type='efficientnet')

    # Train binary model
    print("\nTraining binary model...")
    trainer_binary, history_binary = train_model(task='binary', model_type='efficientnet')