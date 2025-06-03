#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as PathEffects

# Output directory
output_dir = './assets'
os.makedirs(output_dir, exist_ok=True)

print("Generating LAR-IQA model architecture diagram...")

def draw_model_architecture():
    """Draw LAR-IQA model architecture diagram (English version)"""
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    colors = {
        'input': '#2c7fb8',
        'backbone': '#7fcdbb',
        'kan': '#fc8d59',
        'decoder': '#91bfdb',
        'output': '#4d9221',
        'arrow': '#636363',
        'text': '#252525'
    }
    # Input
    draw_block(0.1, 0.4, 0.15, 0.3, colors['input'], 'Input Image\n(3x384x384)')
    # Backbone
    draw_block(0.3, 0.4, 0.15, 0.3, colors['backbone'], 'MobileNet\nFeature Extractor')
    # KAN
    draw_block(0.5, 0.6, 0.15, 0.3, colors['kan'], 'KAN Module\n(Learnable Activation)')
    # Decoder
    draw_block(0.7, 0.4, 0.15, 0.3, colors['decoder'], 'Decoder\n(Upsampling)')
    # Output
    draw_block(0.9, 0.4, 0.15, 0.3, colors['output'], 'Density Map\n(1x384x384)')
    # Arrows
    draw_arrow(0.25, 0.55, 0.3, 0.55, colors['arrow'])
    draw_arrow(0.45, 0.55, 0.5, 0.75, colors['arrow'])
    draw_arrow(0.65, 0.75, 0.7, 0.55, colors['arrow'])
    draw_arrow(0.85, 0.55, 0.9, 0.55, colors['arrow'])
    # Feature maps
    draw_feature_map(0.3, 0.3, 'Low-level\nFeature (64x96x96)')
    draw_feature_map(0.5, 0.3, 'Mid-level\nFeature (256x24x24)')
    draw_feature_map(0.7, 0.3, 'High-level\nFeature (512x12x12)')
    # Skip connection
    draw_skip_connection(0.3, 0.3, 0.7, 0.3, colors['arrow'])
    # Title
    plt.title('LAR-IQA Model Architecture', fontsize=18, pad=20)
    plt.axis('off')
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='Input/Output'),
        Rectangle((0, 0), 1, 1, facecolor=colors['backbone'], label='Feature Extractor'),
        Rectangle((0, 0), 1, 1, facecolor=colors['kan'], label='KAN Module'),
        Rectangle((0, 0), 1, 1, facecolor=colors['decoder'], label='Decoder'),
        FancyArrowPatch((0, 0), (1, 0), color=colors['arrow'], label='Data Flow')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=5, fontsize=12)
    plt.savefig(os.path.join(output_dir, 'model_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()

def draw_block(x, y, width, height, color, text):
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', 
                     alpha=0.8, linewidth=1, zorder=1)
    plt.gca().add_patch(rect)
    plt.text(x + width/2, y + height/2, text, ha='center', va='center', 
             fontsize=12, color='black', weight='bold', zorder=2,
             path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

def draw_arrow(x1, y1, x2, y2, color):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), color=color, 
                           arrowstyle='-|>', linewidth=2, 
                           connectionstyle='arc3,rad=0.1', zorder=1)
    plt.gca().add_patch(arrow)

def draw_feature_map(x, y, text):
    circle = Circle((x, y), 0.03, facecolor='gray', edgecolor='black', alpha=0.7, zorder=1)
    plt.gca().add_patch(circle)
    plt.text(x, y - 0.06, text, ha='center', va='center', fontsize=10, color='black', 
             path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

def draw_skip_connection(x1, y1, x2, y2, color):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), color=color, 
                           arrowstyle='-|>', linewidth=1.5, linestyle='--',
                           connectionstyle='arc3,rad=-0.2', zorder=1)
    plt.gca().add_patch(arrow)
    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2 - 0.08
    plt.text(midx, midy, 'Skip Connection', ha='center', va='center', fontsize=9, 
             color='gray', rotation=-15)

def draw_kan_detail():
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    colors = {
        'input': '#a6bddb',
        'grid': '#fc8d59',
        'spline': '#fdae61',
        'output': '#99d594',
        'arrow': '#636363',
        'text': '#252525'
    }
    plt.text(0.5, 0.95, 'KAN (Kolmogorov-Arnold Network) Module Details', 
             ha='center', va='center', fontsize=16, weight='bold')
    draw_block(0.1, 0.5, 0.15, 0.25, colors['input'], 'Input Feature\n(C×H×W)')
    draw_block(0.35, 0.65, 0.12, 0.2, colors['grid'], 'Grid Points\n(10×10)')
    draw_block(0.35, 0.4, 0.12, 0.2, colors['grid'], 'Basis Function\nB-spline')
    draw_block(0.6, 0.5, 0.15, 0.25, colors['spline'], 'Spline Activation\n(Learnable)')
    draw_block(0.85, 0.5, 0.15, 0.25, colors['output'], 'Output Feature\n(C×H×W)')
    draw_arrow(0.25, 0.625, 0.35, 0.75, colors['arrow'])
    draw_arrow(0.25, 0.625, 0.35, 0.5, colors['arrow'])
    draw_arrow(0.47, 0.75, 0.6, 0.625, colors['arrow'])
    draw_arrow(0.47, 0.5, 0.6, 0.625, colors['arrow'])
    draw_arrow(0.75, 0.625, 0.85, 0.625, colors['arrow'])
    plt.text(0.5, 0.15, 'Specs: 7-layer KAN, 10×10 grid, B-spline basis', 
             ha='center', va='center', fontsize=12, style='italic')
    formula = r'$f(x) = \sum_{i=1}^{n} c_i \phi_i(x)$'
    plt.text(0.5, 0.25, formula + '    where $\\phi_i(x)$ is basis, $c_i$ is learnable', 
             ha='center', va='center', fontsize=13)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'kan_module_detail.png'), dpi=300, bbox_inches='tight')
    plt.close()

draw_model_architecture()
draw_kan_detail()
print("Model architecture diagrams saved to assets directory.") 