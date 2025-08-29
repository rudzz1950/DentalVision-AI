import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

class DentalDatasetAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        self.image_files = list(self.images_dir.glob('*.jpg'))
        self.label_files = list(self.labels_dir.glob('*.txt'))
        
        # Map class IDs to tooth numbers (Universal Numbering System)
        self.class_to_tooth = {
            0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8',
            8: '9', 9: '10', 10: '11', 11: '12', 12: '13', 13: '14', 14: '15', 15: '16',
            16: '17', 17: '18', 18: '19', 19: '20', 20: '21', 21: '22', 22: '23', 23: '24',
            24: '25', 25: '26', 26: '27', 27: '28', 28: '29', 29: '30', 30: '31', 31: '32'
        }

    def get_dataset_stats(self):
        print(f"Total images: {len(self.image_files)}")
        print(f"Total label files: {len(self.label_files)}")
        
        # Check for missing labels
        image_stems = {f.stem for f in self.image_files}
        label_stems = {f.stem for f in self.label_files}
        missing_labels = image_stems - label_stems
        if missing_labels:
            print(f"\nWarning: {len(missing_labels)} images are missing labels")
        
        # Get image dimensions
        img = cv2.imread(str(self.image_files[0]))
        print(f"\nImage dimensions (HxWxC): {img.shape}")
        
        return len(self.image_files), len(self.label_files)

    def analyze_class_distribution(self):
        class_counts = defaultdict(int)
        
        for label_file in tqdm(self.label_files, desc="Analyzing class distribution"):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
        
        # Sort by class ID
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[0])
        
        print("\nClass distribution:")
        for class_id, count in sorted_classes:
            tooth_num = self.class_to_tooth.get(class_id, 'Unknown')
            print(f"Class {class_id} (Tooth {tooth_num}): {count} instances")
        
        return dict(sorted_classes)

    def plot_class_distribution(self, class_counts):
        classes = [f"{k}\n({self.class_to_tooth.get(k, '?')})" for k in class_counts.keys()]
        counts = list(class_counts.values())
        
        plt.figure(figsize=(15, 6))
        plt.bar(classes, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Class Distribution')
        plt.xlabel('Class ID (Tooth Number)')
        plt.ylabel('Number of Instances')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        print("\nSaved class distribution plot as 'class_distribution.png'")

    def visualize_sample_images(self, num_samples=3):
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.image_files), min(num_samples, len(self.image_files)), replace=False)
        
        for idx in sample_indices:
            img_path = self.image_files[idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                print(f"No label file found for {img_path.name}")
                continue
                
            # Read image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Draw bounding boxes
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
                    
                    # Convert from normalized to pixel coordinates
                    x1 = int((x_center - box_w/2) * w)
                    y1 = int((y_center - box_h/2) * h)
                    x2 = int((x_center + box_w/2) * w)
                    y2 = int((y_center + box_h/2) * h)
                    
                    # Draw rectangle and label
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    tooth_num = self.class_to_tooth.get(int(class_id), '?')
                    cv2.putText(img, f"{tooth_num}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            output_path = f"sample_vis_{img_path.stem}.png"
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Sample: {img_path.name}")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved visualization as {output_path}")

if __name__ == "__main__":
    # Initialize analyzer
    data_dir = r"C:\Users\aniru\Music\Project 1\ToothNumber_TaskDataset"
    analyzer = DentalDatasetAnalyzer(data_dir)
    
    # Get basic stats
    print("=== Dataset Statistics ===")
    analyzer.get_dataset_stats()
    
    # Analyze class distribution
    print("\n=== Class Distribution ===")
    class_counts = analyzer.analyze_class_distribution()
    analyzer.plot_class_distribution(class_counts)
    
    # Visualize sample images with bounding boxes
    print("\n=== Visualizing Sample Images ===")
    analyzer.visualize_sample_images(num_samples=3)
    
    print("\nAnalysis complete! Check the generated plots and visualizations.")
