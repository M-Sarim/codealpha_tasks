"""
Comprehensive visualization module for Leaf Disease Detection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from collections import Counter
import config

# Import font configuration
from font_config import configure_matplotlib_fonts, replace_unicode_symbols

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class VisualizationManager:
    def __init__(self):
        self.output_dir = os.path.join(config.PROJECT_ROOT, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_sample_images_grid(self, num_classes=10, images_per_class=4, save_name='sample_images_grid.png'):
        """Display sample images from each class in a grid"""
        fig, axes = plt.subplots(num_classes, images_per_class, figsize=(16, 20))

        train_dir = os.path.join(config.RAW_DATA_DIR, 'train')

        for i, class_name in enumerate(config.DISEASE_CLASSES[:num_classes]):
            class_dir = os.path.join(train_dir, class_name)

            if os.path.exists(class_dir):
                images = os.listdir(class_dir)
                selected_images = random.sample(images, min(images_per_class, len(images)))

                for j, img_name in enumerate(selected_images):
                    img_path = os.path.join(class_dir, img_name)

                    try:
                        img = Image.open(img_path)
                        axes[i, j].imshow(img)
                        axes[i, j].axis('off')

                        if j == 0:  # Add class name to first image
                            class_display = class_name.replace('Tomato___', '').replace('_', ' ')
                            axes[i, j].set_title(class_display, fontsize=10, fontweight='bold', pad=10)

                    except Exception as e:
                        axes[i, j].text(0.5, 0.5, f'Error\nloading\nimage',
                                       ha='center', va='center', transform=axes[i, j].transAxes,
                                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightcoral'))
                        axes[i, j].axis('off')

                # Fill remaining slots if not enough images
                for j in range(len(selected_images), images_per_class):
                    axes[i, j].axis('off')
            else:
                for j in range(images_per_class):
                    axes[i, j].text(0.5, 0.5, f'Class\nnot found',
                                   ha='center', va='center', transform=axes[i, j].transAxes,
                                   fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgray'))
                    axes[i, j].axis('off')

        plt.suptitle('Sample Images from Each Disease Class', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Sample images grid saved to: {save_path}")
        return save_path

    def plot_image_properties_analysis(self, sample_size=200, save_name='image_properties_analysis.png'):
        """Analyze and visualize image properties"""
        train_dir = os.path.join(config.RAW_DATA_DIR, 'train')

        image_info = []

        for class_name in config.DISEASE_CLASSES:
            class_dir = os.path.join(train_dir, class_name)

            if os.path.exists(class_dir):
                images = os.listdir(class_dir)
                sample_images = random.sample(images, min(sample_size//len(config.DISEASE_CLASSES), len(images)))

                for img_name in sample_images:
                    img_path = os.path.join(class_dir, img_name)

                    try:
                        with Image.open(img_path) as img:
                            image_info.append({
                                'class': class_name.replace('Tomato___', ''),
                                'width': img.width,
                                'height': img.height,
                                'aspect_ratio': img.width / img.height,
                                'size_mb': os.path.getsize(img_path) / (1024 * 1024),
                                'format': img.format,
                                'mode': img.mode
                            })
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

        if not image_info:
            print("No image data could be analyzed.")
            return None

        df = pd.DataFrame(image_info)

        # Create comprehensive analysis plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Width distribution
        axes[0, 0].hist(df['width'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['width'].mean(), color='red', linestyle='--', label=f'Mean: {df["width"].mean():.0f}')
        axes[0, 0].legend()

        # Height distribution
        axes[0, 1].hist(df['height'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(df['height'].mean(), color='red', linestyle='--', label=f'Mean: {df["height"].mean():.0f}')
        axes[0, 1].legend()

        # Aspect ratio distribution
        axes[0, 2].hist(df['aspect_ratio'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Aspect Ratio Distribution')
        axes[0, 2].set_xlabel('Aspect Ratio (Width/Height)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(df['aspect_ratio'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["aspect_ratio"].mean():.2f}')
        axes[0, 2].legend()

        # File size distribution
        axes[1, 0].hist(df['size_mb'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('File Size Distribution')
        axes[1, 0].set_xlabel('Size (MB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['size_mb'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["size_mb"].mean():.3f} MB')
        axes[1, 0].legend()

        # Width vs Height scatter
        axes[1, 1].scatter(df['width'], df['height'], alpha=0.6, c=df['aspect_ratio'], cmap='viridis')
        axes[1, 1].set_title('Width vs Height')
        axes[1, 1].set_xlabel('Width (pixels)')
        axes[1, 1].set_ylabel('Height (pixels)')

        # Add diagonal line for square images
        min_dim = min(df['width'].min(), df['height'].min())
        max_dim = max(df['width'].max(), df['height'].max())
        axes[1, 1].plot([min_dim, max_dim], [min_dim, max_dim], 'r--', alpha=0.5, label='Square (1:1)')
        axes[1, 1].legend()

        # Format distribution
        format_counts = df['format'].value_counts()
        axes[1, 2].pie(format_counts.values, labels=format_counts.index, autopct='%1.1f%%')
        axes[1, 2].set_title('Image Format Distribution')

        # Dimensions by class (box plot)
        class_dims = []
        class_labels = []
        for cls in df['class'].unique():
            class_data = df[df['class'] == cls]
            class_dims.extend(class_data['width'].tolist())
            class_labels.extend([f"{cls}_width"] * len(class_data))
            class_dims.extend(class_data['height'].tolist())
            class_labels.extend([f"{cls}_height"] * len(class_data))

        # Simplified box plot for readability
        axes[2, 0].boxplot([df['width'], df['height']], labels=['Width', 'Height'])
        axes[2, 0].set_title('Dimension Distribution')
        axes[2, 0].set_ylabel('Pixels')

        # Size vs Aspect Ratio
        axes[2, 1].scatter(df['aspect_ratio'], df['size_mb'], alpha=0.6, color='purple')
        axes[2, 1].set_title('File Size vs Aspect Ratio')
        axes[2, 1].set_xlabel('Aspect Ratio')
        axes[2, 1].set_ylabel('File Size (MB)')

        # Summary statistics
        stats_text = f"""Image Properties Summary:

Sample Size: {len(df)} images

Dimensions:
Width: {df['width'].mean():.0f} ± {df['width'].std():.0f} px
Height: {df['height'].mean():.0f} ± {df['height'].std():.0f} px
Aspect Ratio: {df['aspect_ratio'].mean():.2f} ± {df['aspect_ratio'].std():.2f}

File Size:
Mean: {df['size_mb'].mean():.3f} MB
Median: {df['size_mb'].median():.3f} MB
Range: {df['size_mb'].min():.3f} - {df['size_mb'].max():.3f} MB

Formats: {', '.join(df['format'].unique())}
Color Modes: {', '.join(df['mode'].unique())}"""

        axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[2, 2].axis('off')

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Image properties analysis saved to: {save_path}")
        return save_path, df

    def plot_data_augmentation_examples(self, class_name='Tomato___healthy', num_examples=8,
                                       save_name='data_augmentation_examples.png'):
        """Show examples of data augmentation"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

        # Create data generator with augmentation
        datagen = ImageDataGenerator(
            **config.AUGMENTATION_PARAMS,
            rescale=1./255
        )

        # Load a sample image
        class_dir = os.path.join(config.RAW_DATA_DIR, 'train', class_name)
        if not os.path.exists(class_dir):
            print(f"Class directory not found: {class_dir}")
            return None

        images = os.listdir(class_dir)
        if not images:
            print(f"No images found in {class_dir}")
            return None

        sample_image_path = os.path.join(class_dir, images[0])

        try:
            # Load and prepare image
            img = load_img(sample_image_path, target_size=config.IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Generate augmented images
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.ravel()

            # Show original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image', fontweight='bold')
            axes[0].axis('off')

            # Generate and show augmented images
            aug_iter = datagen.flow(img_array, batch_size=1)

            for i in range(1, num_examples):
                aug_img = next(aug_iter)[0]
                axes[i].imshow(aug_img)
                axes[i].set_title(f'Augmented {i}')
                axes[i].axis('off')

            plt.suptitle(f'Data Augmentation Examples - {class_name.replace("Tomato___", "")}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"Data augmentation examples saved to: {save_path}")
            return save_path

        except Exception as e:
            print(f"Error creating augmentation examples: {e}")
            return None

    def create_comprehensive_dataset_report(self):
        """Create a comprehensive dataset analysis report"""
        print("Creating comprehensive dataset visualization report...")

        # Generate all visualizations
        sample_grid_path = self.plot_sample_images_grid()
        properties_path, _ = self.plot_image_properties_analysis()
        augmentation_path = self.plot_data_augmentation_examples()

        # Create summary report
        report_path = os.path.join(self.output_dir, 'dataset_report.txt')

        with open(report_path, 'w') as f:
            f.write("LEAF DISEASE DETECTION - DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated visualizations:\n")
            f.write(f"1. Sample Images Grid: {sample_grid_path}\n")
            f.write(f"2. Image Properties Analysis: {properties_path}\n")
            f.write(f"3. Data Augmentation Examples: {augmentation_path}\n\n")
            f.write(f"All visualizations saved in: {self.output_dir}\n")

        print(f"Comprehensive dataset report created!")
        print(f"Report saved to: {report_path}")
        print(f"Visualizations directory: {self.output_dir}")

        return self.output_dir

def create_all_visualizations():
    """Convenience function to create all visualizations"""
    viz_manager = VisualizationManager()
    return viz_manager.create_comprehensive_dataset_report()

if __name__ == "__main__":
    create_all_visualizations()