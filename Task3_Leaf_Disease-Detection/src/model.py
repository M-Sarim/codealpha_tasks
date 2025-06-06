"""
Model architecture for Leaf Disease Detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    Conv2D, MaxPooling2D, Flatten, BatchNormalization
)
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, MobileNetV2, VGG16
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import config

class DiseaseDetectionModel:
    def __init__(self, task='multiclass', model_type='efficientnet'):
        """
        Initialize the model

        Args:
            task: 'multiclass' or 'binary'
            model_type: 'efficientnet', 'resnet', 'mobilenet', 'vgg16', 'custom_cnn'
        """
        self.task = task
        self.model_type = model_type
        self.input_shape = (*config.IMAGE_SIZE, 3)
        self.num_classes = 2 if task == 'binary' else config.NUM_CLASSES
        self.model = None

    def build_model(self):
        """Build the model based on the specified architecture"""
        if self.model_type == 'efficientnet':
            self.model = self._build_efficientnet()
        elif self.model_type == 'resnet':
            self.model = self._build_resnet()
        elif self.model_type == 'mobilenet':
            self.model = self._build_mobilenet()
        elif self.model_type == 'vgg16':
            self.model = self._build_vgg16()
        elif self.model_type == 'custom_cnn':
            self.model = self._build_custom_cnn()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return self.model

    def _build_efficientnet(self):
        """Build EfficientNetB0 based model"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model layers
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax' if self.task == 'multiclass' else 'sigmoid')
        ])

        return model

    def _build_resnet(self):
        """Build ResNet50 based model"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.3),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax' if self.task == 'multiclass' else 'sigmoid')
        ])

        return model

    def _build_mobilenet(self):
        """Build MobileNetV2 based model"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax' if self.task == 'multiclass' else 'sigmoid')
        ])

        return model

    def _build_vgg16(self):
        """Build VGG16 based model"""
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.4),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax' if self.task == 'multiclass' else 'sigmoid')
        ])

        return model

    def _build_custom_cnn(self):
        """Build custom CNN model (similar to original but improved)"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax' if self.task == 'multiclass' else 'sigmoid')
        ])

        return model

    def compile_model(self, learning_rate=None):
        """Compile the model"""
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE

        optimizer = Adam(learning_rate=learning_rate)

        if self.task == 'multiclass':
            loss = 'categorical_crossentropy'
            # Use compatible metrics for current TensorFlow version
            try:
                from tensorflow.keras.metrics import TopKCategoricalAccuracy
                metrics = ['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
            except ImportError:
                # Fallback to basic metrics if TopKCategoricalAccuracy is not available
                metrics = ['accuracy']
        else:
            loss = 'binary_crossentropy'
            # Use compatible metrics for binary classification
            try:
                from tensorflow.keras.metrics import Precision, Recall
                metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]
            except ImportError:
                # Fallback to basic metrics
                metrics = ['accuracy']

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return self.model

    def get_callbacks(self, model_path):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def unfreeze_base_model(self, layers_to_unfreeze=None):
        """Unfreeze base model for fine-tuning"""
        if self.model_type == 'custom_cnn':
            print("Custom CNN doesn't have a base model to unfreeze")
            return

        base_model = self.model.layers[0]

        if layers_to_unfreeze is None:
            # Unfreeze the last 20% of layers
            layers_to_unfreeze = len(base_model.layers) // 5

        # Unfreeze the top layers
        for layer in base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

        print(f"Unfroze last {layers_to_unfreeze} layers of base model")

        # Recompile with lower learning rate for fine-tuning
        self.compile_model(learning_rate=config.LEARNING_RATE / 10)

    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()

        return self.model.summary()

def create_model(task='multiclass', model_type='efficientnet'):
    """Factory function to create and compile a model"""
    model_builder = DiseaseDetectionModel(task=task, model_type=model_type)
    model = model_builder.build_model()
    model = model_builder.compile_model()

    return model, model_builder