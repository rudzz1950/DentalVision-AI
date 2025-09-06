import os
import yaml
import json
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List, Tuple, Optional

def create_yaml_config(data_dir: Path) -> Path:
    """
    Create YAML configuration file for YOLOv8 training with FDI tooth numbering.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Path to the created YAML file
    """
    # Define FDI tooth numbering system
    tooth_types = ['Central Incisor', 'Lateral Incisor', 'Canine', 
                  'First Premolar', 'Second Premolar', 'First Molar', 
                  'Second Molar', 'Third Molar']
    
    # Create class names mapping (0-31)
    class_names = {}
    for i in range(32):
        quadrant = (i // 8) + 1  # 1-4
        tooth_num = (i % 8) + 1  # 1-8
        fdi = int(f"{quadrant}{tooth_num}")
        tooth_type = tooth_types[tooth_num-1] if tooth_num <= 8 else 'Unknown'
        class_names[i] = f"{tooth_type} ({fdi})"
    
    data = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': class_names,
        'nc': 32,
        'fdi_mapping': {i: int(f"{(i//8)+1}{(i%8)+1}") for i in range(32)}
    }
    
    yaml_path = data_dir / 'dental_teeth.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
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
    
    # Advanced training configuration with data augmentation
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'patience': patience,
        'lr0': lr0,
        'lrf': lrf,
        'optimizer': optimizer,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': min(8, os.cpu_count() - 1),
        'weight_decay': 0.0005,
        'label_smoothing': 0.1,  # Label smoothing
        
        # Image augmentation parameters
        'hsv_h': 0.015,  # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,    # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,    # Image HSV-Value augmentation (fraction)
        'degrees': 10.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,     # Image scale (+/- gain)
        'shear': 2.0,     # Image shear (+/- deg)
        'perspective': 0.0005,  # Image perspective (+/- fraction)
        'flipud': 0.5,    # Image flip up-down (probability)
        'fliplr': 0.5,    # Image flip left-right (probability)
        'mosaic': 1.0,    # Use mosaic augmentation (probability)
        'mixup': 0.1,     # Use mixup augmentation (probability)
        'copy_paste': 0.1,  # Use copy-paste augmentation (probability)
        'erasing': 0.4,   # Random erasing (probability)
        'augment': True,  # Apply image augmentation
        
        # Training settings
        'rect': False,  # Rectangular training
        'single_cls': False,  # Train as single-class dataset
        'exist_ok': True,  # Existing project/name ok, do not increment
        'project': 'runs/train',  # Save to project/name
        'name': 'exp',  # Save results to project/name
        'save_period': -1,  # Save checkpoint every x epochs
        'freeze': None,  # Freeze layers (comma separated)
        'plots': True,  # Save plots during training
        'seed': 42,  # Global training seed
    }
    
    # Train the model with optimized parameters
    results = model.train(**train_args)
    
    return model

def evaluate_model(model, data_yaml: str, output_dir: Path, split: str = 'val') -> dict:
    """
    Evaluate the trained model with comprehensive metrics and visualizations.
    
    Args:
        model: Trained YOLO model
        data_yaml: Path to dataset YAML
        output_dir: Directory to save evaluation results
        split: Dataset split to evaluate on ('val' or 'test')
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create output directories
    eval_dir = output_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.45,
        save_json=True,
        save_hybrid=True,
        plots=True,
        project=str(output_dir),
        name='evaluation',
    )
    
    # Save metrics to JSON
    metrics = {
        'mAP50': results.box.map50,
        'mAP50_95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1_score': (2 * results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Per-class AP (if available)
    try:
        per_class_ap = {}
        if hasattr(results.box, 'maps') and results.box.maps is not None:
            maps = results.box.maps  # array of per-class AP@0.5:0.95
            # get class names
            class_names = None
            if hasattr(model, 'names'):
                class_names = model.names
            elif isinstance(data_yaml, (str, Path)) and Path(data_yaml).exists():
                with open(data_yaml, 'r') as f:
                    y = yaml.safe_load(f)
                    class_names = y.get('names')
            if isinstance(class_names, dict):
                for i, ap in enumerate(maps):
                    per_class_ap[str(i)] = {'name': class_names.get(i, str(i)), 'ap50_95': float(ap)}
            else:
                for i, ap in enumerate(maps):
                    per_class_ap[str(i)] = {'name': str(i), 'ap50_95': float(ap)}
        metrics['per_class_ap'] = per_class_ap
    except Exception:
        pass

    # Generate confusion matrix
    if hasattr(results, 'confusion_matrix'):
        conf_mat = results.confusion_matrix.matrix
        plt.figure(figsize=(15, 12))
        plt.imshow(conf_mat, cmap='Blues')
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(eval_dir / 'confusion_matrix.png')
        plt.close()

    # Reliability diagram (calibration curve) if confidences and correctness are available
    try:
        # Attempt to access per-detection correctness if provided
        # Fallback: skip if not available
        if hasattr(results, 'boxes') and results.boxes is not None:
            confs = []
            correct = []
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                    c = r.boxes.conf.cpu().numpy()
                    confs.append(c)
                    # Without GT match info, approximate correctness by thresholding; skip to avoid misleading plot
            # Skip due to lack of GT match flags; real calibration requires TP flags from matching.
        else:
            pass
    except Exception:
        pass

    return metrics

def export_model(model, format='onnx'):
    """Export the model to different formats."""
    export_path = model.export(format=format)
    print(f"Model exported to {export_path}")
    return export_path

def plot_training_metrics(metrics: dict, output_dir: Path) -> None:
    """Plot training metrics over epochs."""
    plt.figure(figsize=(12, 8))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(metrics.get('train/box_loss', []), label='Train Loss')
    plt.plot(metrics.get('val/box_loss', []), label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # mAP curves
    plt.subplot(2, 2, 2)
    plt.plot(metrics.get('metrics/mAP50', []), label='mAP@0.5')
    plt.plot(metrics.get('metrics/mAP50-95', []), label='mAP@0.5:0.95')
    plt.title('mAP Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    # Precision-Recall curve
    plt.subplot(2, 2, 3)
    plt.plot(metrics.get('metrics/precision', []), label='Precision')
    plt.plot(metrics.get('metrics/recall', []), label='Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png')
    plt.close()

def postprocess_predictions(predictions, image_size: Tuple[int, int] = (640, 640)) -> List[Dict]:
    """
    Post-process model predictions for anatomical correctness.
    
    Args:
        predictions: Raw model predictions
        image_size: Size of the input image (width, height)
        
    Returns:
        List of processed detections with anatomical validation
    """
    processed: List[Dict] = []

    # Helper: class_id -> FDI number
    fdi_map = {i: int(f"{(i // 8) + 1}{(i % 8) + 1}") for i in range(32)}

    for pred in predictions:
        # Convert to numpy for easier manipulation
        boxes = pred.boxes.xywhn.cpu().numpy()  # Normalized xywh
        classes = pred.boxes.cls.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()

        # Group by quadrant (1-4)
        quadrants: Dict[int, List[Dict]] = {1: [], 2: [], 3: [], 4: []}
        for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
            x_center, y_center, width, height = box
            cls_id = int(cls_id)
            quadrant = int(cls_id // 8) + 1  # 1..4
            quadrants[quadrant].append({
                'box': box,
                'class_id': cls_id,
                'fdi': fdi_map.get(cls_id, cls_id),
                'score': float(score),
                'x_center': float(x_center),
                'y_center': float(y_center)
            })

        # Sort and validate sequences within each quadrant
        for quad, dets in quadrants.items():
            if not dets:
                continue

            # Expected tooth index order within a quadrant is 1..8
            expected_ids = list(range((quad - 1) * 8, quad * 8))

            # Sort by x direction: upper jaw (q1,q2) left->right; lower jaw (q3,q4) right->left
            if quad in (1, 2):
                dets.sort(key=lambda x: x['x_center'])
            else:
                dets.sort(key=lambda x: -x['x_center'])

            # Enforce monotonic progression by majority voting on local ordering
            # If a detection is out-of-order relative to neighbors, mark it as questionable
            sequence_flags: List[bool] = [True] * len(dets)
            for i in range(1, len(dets)):
                if dets[i]['class_id'] < dets[i - 1]['class_id']:
                    # Out-of-order for the expected progression; flag the lower-confidence one
                    if dets[i]['score'] < dets[i - 1]['score']:
                        sequence_flags[i] = False
                    else:
                        sequence_flags[i - 1] = False

            # Filter out flagged detections but keep at least 1
            filtered = [d for d, ok in zip(dets, sequence_flags) if ok]
            if not filtered:
                filtered = [max(dets, key=lambda d: d['score'])]

            for det in filtered:
                processed.append({
                    'box': det['box'].tolist(),
                    'class_id': det['class_id'],
                    'fdi': det['fdi'],
                    'score': det['score'],
                    'quadrant': quad
                })

    return processed

if __name__ == "__main__":
    # Set up paths
    data_dir = Path(__file__).parent / "ToothNumber_TaskDataset"
    output_dir = Path("runs") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
