import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_yaml_config(yaml_path):
    """Load the dataset YAML configuration."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def count_annotations(label_dir):
    """Count class instances in the dataset."""
    class_counts = defaultdict(int)
    
    for label_file in Path(label_dir).rglob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
    
    return dict(sorted(class_counts.items()))

def plot_class_distribution(class_counts, class_names, output_dir):
    """Plot and save class distribution."""
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    labels = [f"{class_names[cls]}\n({cls})" for cls in classes]
    
    plt.figure(figsize=(16, 8))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(classes, counts)
    plt.title('Class Distribution (Linear Scale)')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Instances')
    plt.xticks(classes, rotation=90)
    
    # Add counts on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom')
    
    # Log scale
    plt.subplot(1, 2, 2)
    plt.bar(classes, counts, log=True)
    plt.title('Class Distribution (Log Scale)')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Instances (log)')
    plt.xticks(classes, rotation=90)
    
    plt.tight_layout()
    output_path = output_dir / 'class_distribution.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path

def analyze_dataset(data_dir, output_dir):
    """Main function to analyze the dataset."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_yaml_config(data_dir / 'dental_teeth.yaml')
    class_names = config['names']
    
    # Count annotations in each split
    splits = ['train', 'val', 'test']
    results = {}
    
    for split in splits:
        label_dir = data_dir / split / 'labels'
        if label_dir.exists():
            results[split] = count_annotations(label_dir)
    
    # Combine results
    total_counts = defaultdict(int)
    for split_counts in results.values():
        for cls_id, count in split_counts.items():
            total_counts[cls_id] += count
    
    # Plot distribution
    plot_path = plot_class_distribution(total_counts, class_names, output_dir)
    
    # Print summary
    print("\n=== Dataset Analysis ===")
    print(f"\nTotal classes: {len(total_counts)}")
    print(f"Total instances: {sum(total_counts.values())}")
    
    print("\nClass distribution:")
    for cls_id, count in total_counts.items():
        print(f"  {class_names[cls_id]} (ID: {cls_id}): {count}")
    
    print(f"\nClass distribution plot saved to: {plot_path}")
    
    return {
        'class_distribution': total_counts,
        'class_names': class_names,
        'plot_path': plot_path
    }

if __name__ == "__main__":
    data_dir = Path("ToothNumber_TaskDataset")
    output_dir = Path("analysis")
    analyze_dataset(data_dir, output_dir)
