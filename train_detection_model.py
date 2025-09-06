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
    
    with open(eval_dir / f'metrics_{split}.json', 'w') as f:
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
        plt.title(f'Confusion Matrix ({split})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(eval_dir / f'confusion_matrix_{split}.png')
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

def _generate_html_report(output_dir: Path) -> None:
    """Generate a simple HTML report aggregating metrics, calibration, and heatmaps."""
    eval_dir = output_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    # Load metrics if present
    def _load_json(p: Path):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    m_val = _load_json(eval_dir / 'metrics_val.json')
    m_test = _load_json(eval_dir / 'metrics_test.json')
    calib_val = _load_json(eval_dir / 'calibration_val.json')
    calib_test = _load_json(eval_dir / 'calibration_test.json')

    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html><head><meta charset="utf-8"><title>DentalVision-AI Report</title>')
    html.append('<style>body{font-family:Arial, sans-serif; margin:24px;} h2{margin-top:28px;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:6px 10px;} img{max-width:100%;height:auto;border:1px solid #ddd;padding:2px;margin:6px 0;}</style>')
    html.append('</head><body>')
    html.append('<h1>DentalVision-AI Training & Evaluation Report</h1>')

    def _metrics_section(title: str, m: dict, split: str):
        if not m:
            return ''
        s = [f'<h2>{title}</h2>']
        s.append('<table><tbody>')
        for k in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']:
            if k in m:
                s.append(f'<tr><th>{k}</th><td>{m[k]:.4f}</td></tr>')
        s.append('</tbody></table>')
        # Confusion matrix
        s.append(f'<h3>Confusion Matrix ({split})</h3>')
        s.append(f'<img src="confusion_matrix_{split}.png" alt="Confusion Matrix {split}">')
        # Calibration
        s.append(f'<h3>Calibration & Reliability ({split})</h3>')
        s.append(f'<img src="reliability_{split}.png" alt="Reliability {split}">')
        # Heatmaps
        s.append(f'<h3>Error Heatmaps ({split})</h3>')
        s.append(f'<div><img src="fp_heatmap_{split}.png" alt="FP Heatmap {split}">')
        s.append(f'<img src="fn_heatmap_{split}.png" alt="FN Heatmap {split}"></div>')
        # Per-class AP
        pc = m.get('per_class_ap') if isinstance(m, dict) else None
        if pc:
            s.append('<h3>Per-Class AP (0.5:0.95)</h3><table><tr><th>Class ID</th><th>Name</th><th>AP</th></tr>')
            for cid, info in pc.items():
                name = info.get('name', cid)
                ap = info.get('ap50_95', 0.0)
                s.append(f'<tr><td>{cid}</td><td>{name}</td><td>{ap:.4f}</td></tr>')
            s.append('</table>')
        return '\n'.join(s)

    html.append(_metrics_section('Validation Metrics', m_val, 'val'))
    html.append(_metrics_section('Test Metrics', m_test, 'test'))
    html.append('<p>Report generated automatically.</p>')
    html.append('</body></html>')

    with open(eval_dir / 'report.html', 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

def export_model(model, format='onnx'):
    """Export the model to different formats."""
    export_path = model.export(format=format)
    print(f"Model exported to {export_path}")
    return export_path

def _xywhn_to_xyxyn(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=np.float32)

def _iou_xyxyn(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)

def compute_calibration_and_heatmaps(model, data_yaml: str, split: str, output_dir: Path,
                                     iou_thr: float = 0.5, bins: int = 15, grid: int = 32) -> None:
    """
    Compute reliability diagram, Expected Calibration Error (ECE), and error heatmaps for a dataset split.
    Saves plots under output_dir / 'evaluation'.
    """
    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg['path']) if 'path' in cfg else Path(data_yaml).parent
    img_dir = base / cfg[split]  # e.g., 'val/images'
    if not img_dir.is_absolute():
        img_dir = base / cfg[split]
    # Deduce labels dir
    labels_dir = img_dir.parent / 'labels'

    # Gather files
    image_files = sorted([p for p in Path(img_dir).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    if not image_files:
        print(f"No images found for split {split} at {img_dir}")
        return

    # Accumulators for calibration
    confs_all: List[float] = []
    correct_all: List[int] = []

    # Heatmaps: FP and FN counts on a grid
    fp_grid = np.zeros((grid, grid), dtype=np.float32)
    fn_grid = np.zeros((grid, grid), dtype=np.float32)

    for img_path in image_files:
        # Load GT labels
        lab_path = labels_dir / (img_path.stem + '.txt')
        gt_classes = []
        gt_boxes_xyxyn = []
        if lab_path.exists():
            with open(lab_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:5])
                        gt_classes.append(cid)
                        gt_boxes_xyxyn.append(_xywhn_to_xyxyn(np.array([x, y, w, h], dtype=np.float32)))
        gt_boxes_xyxyn = np.array(gt_boxes_xyxyn, dtype=np.float32) if gt_boxes_xyxyn else np.zeros((0, 4), dtype=np.float32)
        gt_classes = np.array(gt_classes, dtype=np.int32) if gt_classes else np.zeros((0,), dtype=np.int32)

        # Predict
        pred = model(img_path, conf=0.001, iou=0.7, verbose=False)[0]
        if len(pred.boxes) == 0:
            # All GT become FN
            for g in gt_boxes_xyxyn:
                # center cell
                cx = (g[0] + g[2]) / 2
                cy = (g[1] + g[3]) / 2
                gx = min(grid - 1, max(0, int(cx * grid)))
                gy = min(grid - 1, max(0, int(cy * grid)))
                fn_grid[gy, gx] += 1
            continue

        p_cls = pred.boxes.cls.cpu().numpy().astype(np.int32)
        p_conf = pred.boxes.conf.cpu().numpy().astype(np.float32)
        # Convert to xyxyn normalized
        xyxyn = pred.boxes.xyxyn.cpu().numpy().astype(np.float32)

        # Match predictions to GT by IoU and class
        used_gt = np.zeros((len(gt_boxes_xyxyn),), dtype=bool)
        for i, (pc, pcfg, pb) in enumerate(zip(p_cls, p_conf, xyxyn)):
            best_iou = 0.0
            best_j = -1
            for j, (gc, gb) in enumerate(zip(gt_classes, gt_boxes_xyxyn)):
                if used_gt[j] or gc != pc:
                    continue
                iou = _iou_xyxyn(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            is_tp = best_iou >= iou_thr and best_j >= 0
            confs_all.append(float(pcfg))
            correct_all.append(1 if is_tp else 0)
            if is_tp:
                used_gt[best_j] = True
            else:
                # FP contributes to heatmap
                cx = (pb[0] + pb[2]) / 2
                cy = (pb[1] + pb[3]) / 2
                gx = min(grid - 1, max(0, int(cx * grid)))
                gy = min(grid - 1, max(0, int(cy * grid)))
                fp_grid[gy, gx] += 1

        # Unmatched GTs are FN
        for j, used in enumerate(used_gt):
            if not used:
                gb = gt_boxes_xyxyn[j]
                cx = (gb[0] + gb[2]) / 2
                cy = (gb[1] + gb[3]) / 2
                gx = min(grid - 1, max(0, int(cx * grid)))
                gy = min(grid - 1, max(0, int(cy * grid)))
                fn_grid[gy, gx] += 1

    # Reliability diagram & ECE
    if confs_all:
        confs = np.array(confs_all, dtype=np.float32)
        correct = np.array(correct_all, dtype=np.int32)
        # Bin by confidence
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_ids = np.clip(np.digitize(confs, bin_edges) - 1, 0, bins - 1)
        accs = []
        conf_means = []
        counts = []
        ece = 0.0
        for b in range(bins):
            mask = bin_ids == b
            n = int(mask.sum())
            counts.append(n)
            if n > 0:
                acc = float(correct[mask].mean())
                cm = float(confs[mask].mean())
            else:
                acc, cm = 0.0, (bin_edges[b] + bin_edges[b+1]) / 2
            accs.append(acc)
            conf_means.append(cm)
            ece += (n / max(1, len(confs))) * abs(acc - cm)

        # Plot reliability diagram
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(conf_means, accs, marker='o', label='Model')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram ({split})\nECE={ece:.3f}')
        plt.legend()
        (output_dir / 'evaluation').mkdir(exist_ok=True)
        plt.savefig(output_dir / 'evaluation' / f'reliability_{split}.png', dpi=200, bbox_inches='tight')
        plt.close()

        with open(output_dir / 'evaluation' / f'calibration_{split}.json', 'w') as f:
            json.dump({'ece': ece, 'bins': bins, 'conf_means': conf_means, 'acc': accs, 'counts': counts}, f, indent=2)

    # Heatmaps
    def _save_heatmap(arr: np.ndarray, title: str, filename: str):
        plt.figure(figsize=(6, 5))
        plt.imshow(arr, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Count')
        plt.title(title)
        plt.xlabel('X (grid)')
        plt.ylabel('Y (grid)')
        plt.savefig(output_dir / 'evaluation' / filename, dpi=200, bbox_inches='tight')
        plt.close()

    (output_dir / 'evaluation').mkdir(exist_ok=True)
    _save_heatmap(fp_grid, f'False Positives Heatmap ({split})', f'fp_heatmap_{split}.png')
    _save_heatmap(fn_grid, f'False Negatives Heatmap ({split})', f'fn_heatmap_{split}.png')

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

    # Calibration & Heatmaps for val and test
    print("\nComputing calibration and heatmaps (val)...")
    compute_calibration_and_heatmaps(model, str(yaml_path), split='val', output_dir=output_dir)
    print("Computing calibration and heatmaps (test)...")
    compute_calibration_and_heatmaps(model, str(yaml_path), split='test', output_dir=output_dir)
    
    # Export model to ONNX format
    print("\nExporting model to ONNX format...")
    export_path = export_model(model, format='onnx')
    
    # Generate HTML report aggregating metrics and plots
    print("\nGenerating HTML report...")
    _generate_html_report(output_dir)
    
    print("\nTraining and evaluation complete!")
