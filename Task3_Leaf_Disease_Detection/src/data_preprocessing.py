"""
Data preprocessing module for Leaf Disease Detection
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import config

# Configure matplotlib to handle font issues
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class DataPreprocessor:
    def __init__(self):
        self.raw_data_dir = config.RAW_DATA_DIR
        self.processed_data_dir = config.PROCESSED_DATA_DIR
        self.image_size = config.IMAGE_SIZE
        self.disease_classes = config.DISEASE_CLASSES
        self.binary_mapping = config.BINARY_MAPPING

    def analyze_dataset(self):
        """Analyze the raw dataset and provide statistics"""
        print("Analyzing dataset...")

        train_dir = os.path.join(self.raw_data_dir, 'train')
        val_dir = os.path.join(self.raw_data_dir, 'val')

        train_stats = {}
        val_stats = {}

        # Analyze training data
        for class_name in self.disease_classes:
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)

            if os.path.exists(train_class_dir):
                train_stats[class_name] = len(os.listdir(train_class_dir))
            else:
                train_stats[class_name] = 0

            if os.path.exists(val_class_dir):
                val_stats[class_name] = len(os.listdir(val_class_dir))
            else:
                val_stats[class_name] = 0

        # Create DataFrame for analysis
        df = pd.DataFrame({
            'Class': list(train_stats.keys()),
            'Train_Count': list(train_stats.values()),
            'Val_Count': list(val_stats.values())
        })
        df['Total_Count'] = df['Train_Count'] + df['Val_Count']
        df['Binary_Class'] = df['Class'].map(self.binary_mapping)

        print("\nDataset Statistics:")
        print(df)

        # Plot distribution
        plt.figure(figsize=(20, 15))

        # Total images per class
        plt.subplot(2, 3, 1)
        colors = ['lightgreen' if cls == 'Tomato___healthy' else 'lightcoral' for cls in df['Class']]
        bars = plt.bar(range(len(df)), df['Total_Count'], color=colors)
        plt.xticks(range(len(df)), [cls.replace('Tomato___', '') for cls in df['Class']], rotation=45, ha='right')
        plt.title('Total Images per Class', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images')
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, df['Total_Count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=8)

        # Binary distribution pie chart
        plt.subplot(2, 3, 2)
        binary_counts = df.groupby('Binary_Class')['Total_Count'].sum()
        colors_pie = ['lightgreen', 'lightcoral']
        wedges, texts, autotexts = plt.pie(binary_counts.values, labels=binary_counts.index,
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        plt.title('Healthy vs Diseased Distribution', fontsize=14, fontweight='bold')

        # Disease types distribution
        plt.subplot(2, 3, 3)
        diseased_df = df[df['Binary_Class'] == 'Diseased']
        bars2 = plt.bar(range(len(diseased_df)), diseased_df['Total_Count'], color='salmon')
        plt.xticks(range(len(diseased_df)), [cls.replace('Tomato___', '') for cls in diseased_df['Class']],
                  rotation=45, ha='right')
        plt.title('Disease Types Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images')
        plt.grid(axis='y', alpha=0.3)

        # Train vs Validation split
        plt.subplot(2, 3, 4)
        x = np.arange(len(df))
        width = 0.35
        plt.bar(x - width/2, df['Train_Count'], width, label='Train', color='lightblue', alpha=0.8)
        plt.bar(x + width/2, df['Val_Count'], width, label='Validation', color='orange', alpha=0.8)
        plt.xticks(x, [cls.replace('Tomato___', '') for cls in df['Class']], rotation=45, ha='right')
        plt.title('Train vs Validation Split', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Class balance analysis
        plt.subplot(2, 3, 5)
        class_percentages = (df['Total_Count'] / df['Total_Count'].sum()) * 100
        bars3 = plt.bar(range(len(df)), class_percentages, color=colors)
        plt.xticks(range(len(df)), [cls.replace('Tomato___', '') for cls in df['Class']], rotation=45, ha='right')
        plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Percentage of Total Images')
        plt.axhline(y=100/len(df), color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Summary statistics
        plt.subplot(2, 3, 6)
        stats_text = f"""Dataset Summary:

Total Images: {df['Total_Count'].sum():,}
Number of Classes: {len(df)}
Healthy Images: {binary_counts['Healthy']:,} ({binary_counts['Healthy']/df['Total_Count'].sum()*100:.1f}%)
Diseased Images: {binary_counts['Diseased']:,} ({binary_counts['Diseased']/df['Total_Count'].sum()*100:.1f}%)

Per Class Statistics:
Mean: {df['Total_Count'].mean():.0f} images
Std Dev: {df['Total_Count'].std():.0f} images
Min: {df['Total_Count'].min()} images
Max: {df['Total_Count'].max()} images

Class Balance:
CV: {(df['Total_Count'].std() / df['Total_Count'].mean()):.3f}
{'[!] Imbalanced' if (df['Total_Count'].std() / df['Total_Count'].mean()) > 0.5 else '[OK] Balanced'}"""

        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(config.PROJECT_ROOT, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return df

    def create_processed_dataset(self):
        """Create processed dataset with proper train/val/test splits"""
        print("Creating processed dataset...")

        # Create processed directories
        for split in ['train', 'val', 'test']:
            for class_name in self.disease_classes:
                os.makedirs(os.path.join(self.processed_data_dir, split, class_name), exist_ok=True)

        # Also create binary classification directories
        for split in ['train', 'val', 'test']:
            for binary_class in ['Healthy', 'Diseased']:
                os.makedirs(os.path.join(self.processed_data_dir, f'binary_{split}', binary_class), exist_ok=True)

        # Process each class
        for class_name in tqdm(self.disease_classes, desc="Processing classes"):
            self._process_class(class_name)

        print("Processed dataset created successfully!")

    def _process_class(self, class_name):
        """Process a single class and split into train/val/test"""
        train_dir = os.path.join(self.raw_data_dir, 'train', class_name)
        val_dir = os.path.join(self.raw_data_dir, 'val', class_name)

        all_images = []

        # Collect all images from train and val directories
        if os.path.exists(train_dir):
            train_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
            all_images.extend(train_images)

        if os.path.exists(val_dir):
            val_images = [os.path.join(val_dir, img) for img in os.listdir(val_dir)]
            all_images.extend(val_images)

        if not all_images:
            return

        # Split into train/val/test (70/20/10)
        train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)  # 0.33 of 0.3 = 0.1

        # Copy images to processed directories
        splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        binary_class = self.binary_mapping[class_name]

        for split_name, img_list in splits.items():
            for i, img_path in enumerate(img_list):
                # Copy to multiclass directory
                dst_path = os.path.join(self.processed_data_dir, split_name, class_name, f"{class_name}_{i}.jpg")
                shutil.copy2(img_path, dst_path)

                # Copy to binary classification directory
                binary_dst_path = os.path.join(self.processed_data_dir, f'binary_{split_name}', binary_class, f"{class_name}_{i}.jpg")
                shutil.copy2(img_path, binary_dst_path)

    def create_data_generators(self, task='multiclass'):
        """Create data generators for training"""
        if task == 'multiclass':
            train_dir = os.path.join(self.processed_data_dir, 'train')
            val_dir = os.path.join(self.processed_data_dir, 'val')
            test_dir = os.path.join(self.processed_data_dir, 'test')
        else:  # binary
            train_dir = os.path.join(self.processed_data_dir, 'binary_train')
            val_dir = os.path.join(self.processed_data_dir, 'binary_val')
            test_dir = os.path.join(self.processed_data_dir, 'binary_test')

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **config.AUGMENTATION_PARAMS
        )

        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical' if task == 'multiclass' else 'binary',
            shuffle=True
        )

        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.image_size,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical' if task == 'multiclass' else 'binary',
            shuffle=False
        )

        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical' if task == 'multiclass' else 'binary',
            shuffle=False
        )

        return train_generator, val_generator, test_generator

if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Analyze dataset
    df = preprocessor.analyze_dataset()

    # Create processed dataset
    preprocessor.create_processed_dataset()

    print("Data preprocessing completed!")