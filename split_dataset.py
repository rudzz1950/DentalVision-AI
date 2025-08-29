import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets while maintaining class distribution.
    
    Args:
        data_dir (str): Path to the dataset directory
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define paths
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files and shuffle
    image_files = list(images_dir.glob('*.jpg'))
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split the data
    splits_files = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Copy files to respective directories
    for split, files in splits_files.items():
        print(f"Processing {split} set ({len(files)} images)...")
        for img_path in tqdm(files, desc=f"Copying {split} files"):
            # Copy image
            dst_img = data_dir / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy corresponding label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = data_dir / split / 'labels' / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dst_label)
    
    print("\nDataset split completed successfully!")
    print(f"Total images: {total}")
    print(f"Train: {len(splits_files['train'])} images")
    print(f"Validation: {len(splits_files['val'])} images")
    print(f"Test: {len(splits_files['test'])} images")

if __name__ == "__main__":
    data_dir = r"C:\Users\aniru\Downloads\Project 1"
    split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
