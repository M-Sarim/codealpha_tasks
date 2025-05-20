#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Diagram Generator

This script creates a visual diagram of the data preprocessing pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patheffects as PathEffects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors
box_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
arrow_color = '#34495e'
text_color = '#ffffff'
bg_color = '#f8f9fa'

# Set background color
ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)

# Define box positions and sizes
box_width = 2.0
box_height = 1.0
y_position = 2.5
x_positions = [1, 4, 7, 10, 13]

# Create boxes and labels
boxes = []
labels = ['Data\nLoading', 'Missing Value\nHandling', 'Outlier\nDetection', 'Feature\nScaling', 'Data\nSplitting']
descriptions = [
    'Load dataset\nExplore structure',
    'Median imputation\nfor numerical features',
    'IQR method\nReplace with bounds',
    'Standardization\nz = (x - μ) / σ',
    'Train (80%)\nTest (20%)'
]

for i, (x, label, desc, color) in enumerate(zip(x_positions, labels, descriptions, box_colors)):
    # Create main box
    box = Rectangle((x, y_position), box_width, box_height,
                   facecolor=color, edgecolor='none', alpha=0.8)
    ax.add_patch(box)
    boxes.append(box)

    # Add main label
    text = ax.text(x + box_width/2, y_position + box_height*0.7, label,
                  ha='center', va='center', color=text_color, fontweight='bold', fontsize=12)
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=(0,0,0,0.3))])

    # Add description
    desc_text = ax.text(x + box_width/2, y_position + box_height*0.3, desc,
                       ha='center', va='center', color=text_color, fontsize=10)
    desc_text.set_path_effects([PathEffects.withStroke(linewidth=1, foreground=(0,0,0,0.3))])

    # Add step number
    step_text = ax.text(x + 0.15, y_position + box_height - 0.15, f"Step {i+1}",
                       ha='left', va='top', color=text_color, fontsize=9,
                       bbox=dict(facecolor=(0,0,0,0.3), edgecolor='none', pad=2))

# Add arrows between boxes
for i in range(len(boxes) - 1):
    arrow = FancyArrowPatch(
        (x_positions[i] + box_width, y_position + box_height/2),
        (x_positions[i+1], y_position + box_height/2),
        arrowstyle='-|>', color=arrow_color, linewidth=2,
        connectionstyle='arc3,rad=0.1', mutation_scale=20
    )
    ax.add_patch(arrow)

# Add title
ax.text(x_positions[2], y_position + box_height + 1, 'Data Preprocessing Pipeline',
       ha='center', va='center', color='#2c3e50', fontsize=18, fontweight='bold')

# Add input and output data representations
# Input data (messy)
np.random.seed(42)
x_input = np.linspace(0, 1, 20)
y_input = np.random.normal(0, 0.3, 20)
ax.scatter(x_input*0.8 + 0.1, y_input*0.8 + 1, color='#e74c3c', alpha=0.7)
ax.text(0.5, 0.5, "Raw Data\n(with outliers, different scales)", ha='center', va='center', fontsize=10)

# Output data (clean)
x_output = np.linspace(0, 1, 20)
y_output = np.sin(x_output * 2 * np.pi)
ax.scatter(x_output*0.8 + 13.1, y_output*0.4 + 1, color='#2ecc71', alpha=0.7)
ax.text(13.5, 0.5, "Preprocessed Data\n(clean, normalized)", ha='center', va='center', fontsize=10)

# Set limits and remove axes
ax.set_xlim(0, 15)
ax.set_ylim(0, 5)
ax.axis('off')

# Save the figure
plt.tight_layout()
plt.savefig('plots/pipeline_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Pipeline diagram created successfully!")
