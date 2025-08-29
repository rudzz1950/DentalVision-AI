import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
import torch

def create_yaml_config(data_dir):
    """Create YAML configuration file for YOLOv8 training."""
    data = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',  # relative to path
        'val': 'val/images',      # relative to path
        'test': 'test/images',    # relative to path
        'names': {i: str(i+1) for i in range(32)},  # Tooth numbers 1-32
        'nc': 32  # number of classes
    }
    
    yaml_path = data_dir / 'dental_teeth.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return yaml_path

def train_yolov8(data_yaml, epochs=200, imgsz=640, batch=16, model_size='m', patience=30, lr0=0.01, lrf=0.01, optimizer='auto'):
    """
    Train an optimized YOLOv8 model on the dental dataset with advanced features.
    
    Args:
        data_yaml (str): Path to the YAML config file
        epochs (int): Number of training epochs
        imgsz (int): Image size for training
        batch (int): Batch size (auto-scales with GPU memory)
        model_size (str): Model size (n, s, m, l, x)
        patience (int): Early stopping patience
        lr0 (float): Initial learning rate
        lrf (float): Final learning rate (lr0 * lrf)
        optimizer (str): Optimizer to use (SGD, Adam, AdamW, Adamax, AdamP, RMSProp, auto)
    """
    # Load a pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Calculate batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        batch = min(batch, max(4, int(gpu_mem * 0.8 / 3)))  # 80% of GPU mem, ~3GB per batch
    
    # Advanced training configuration
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'project': 'dental_teeth_detection',
        'name': f'yolov8{model_size}_optimized',
        'exist_ok': True,
        'optimizer': optimizer,
        'lr0': lr0,  # Initial learning rate
        'lrf': lrf,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # Distribution Focal Loss gain
        'hsv_h': 0.015,  # Image HSV-Hue augmentation
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation
        'hsv_v': 0.4,  # Image HSV-Value augmentation
        'degrees': 10.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 2.0,  # Image shear (+/- deg)
        'perspective': 0.0001,  # Image perspective
        'flipud': 0.0,  # Flip up-down probability
        'fliplr': 0.5,  # Flip left-right probability
        'mosaic': 1.0,  # Mosaic augmentation probability
        'mixup': 0.1,  # MixUp augmentation probability
        'copy_paste': 0.1,  # Copy-paste augmentation probability
        'erasing': 0.4,  # Random erasing probability
        'crop_fraction': 0.9,  # Random crop fraction
        'patience': patience,  # Early stopping patience
        'save_period': -1,  # Save checkpoint every x epochs (-1 = last only)
        'single_cls': False,  # Train as single-class dataset
        'rect': False,  # Rectangular training
        'cos_lr': True,  # Cosine LR scheduler
        'close_mosaic': 10,  # Disable mosaic for last N epochs
        'amp': True,  # Automatic Mixed Precision (AMP) training
        'overlap_mask': True,  # Overlap mask during training
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Use dropout regularization (probability)
        'val': True,  # Validate during training
        'plots': True,  # Save plots during training
        'optimize': True,  # Optimize ONNX and TorchScript models
        'seed': 42,  # Global training seed
    }
    
    # Train the model with optimized parameters
    results = model.train(**train_args)
    
    return model

def evaluate_model(model, data_yaml, split='val'):
    """Evaluate the trained model on the validation or test set."""
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        project='dental_teeth_detection',
        name=f'yolov8m_val',
        exist_ok=True
    )
    return metrics

def export_model(model, format='onnx'):
    """Export the model to different formats."""
    export_path = model.export(format=format)
    print(f"Model exported to {export_path}")
    return export_path

if __name__ == "__main__":
    # Set up paths
    data_dir = Path(r"C:\Users\aniru\Downloads\Project 1")
    
    # Create YAML config
    print("Creating YAML configuration...")
    yaml_path = create_yaml_config(data_dir)
    print(f"YAML config created at: {yaml_path}")
    
    # Train the model with optimized parameters
    print("\nStarting optimized model training...")
    model = train_yolov8(
        yaml_path,
        epochs=200,  # Increased epochs for better convergence
        imgsz=640,
        batch=32,  # Increased batch size for better GPU utilization
        model_size='m',  # Medium-sized model (good balance between speed and accuracy)
        patience=30,  # Early stopping patience
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        optimizer='AdamW'  # Using AdamW optimizer
    )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, yaml_path, split='val')
    print("\nValidation metrics:", val_metrics)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, yaml_path, split='test')
    print("\nTest metrics:", test_metrics)
    
    # Export model to ONNX format
    print("\nExporting model to ONNX format...")
    export_path = export_model(model, format='onnx')
    
    print("\nTraining and evaluation complete!")
