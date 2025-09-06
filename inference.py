import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml
from typing import Optional
import torch
try:
    from ensemble_boxes import weighted_boxes_fusion
except Exception:
    weighted_boxes_fusion = None

try:
    # For anatomical postprocessing summary
    from train_detection_model import postprocess_predictions as _anatomical_post
except Exception:
    _anatomical_post = None

class ToothDetector:
    def __init__(self, model_path, data_yaml_path: Optional[str] = None):
        """
        Initialize the tooth detector with a trained YOLOv8 model.
        
        Args:
            model_path (str): Path to the trained YOLOv8 model (.pt file)
            data_yaml_path (str | None): Optional path to dataset YAML to read FDI mapping
        """
        self.model = YOLO(model_path)
        # Prefer FDI mapping from dataset YAML if provided, else fallback to computed mapping
        self.class_to_fdi = self._load_fdi_mapping(data_yaml_path)

    @staticmethod
    def _compute_fdi_mapping(num_classes: int = 32) -> dict:
        """Compute class_id -> FDI number mapping (0..31 -> 11..48)."""
        return {i: int(f"{(i // 8) + 1}{(i % 8) + 1}") for i in range(num_classes)}

    def _load_fdi_mapping(self, data_yaml_path: Optional[str]) -> dict:
        # Try loading mapping from YAML 'fdi_mapping' or derive from class count
        if data_yaml_path and Path(data_yaml_path).exists():
            try:
                with open(data_yaml_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    if 'fdi_mapping' in cfg and isinstance(cfg['fdi_mapping'], dict):
                        # Keys may be strings in YAML; normalize to int
                        return {int(k): int(v) for k, v in cfg['fdi_mapping'].items()}
                    # Fallback: compute from nc if present
                    if 'nc' in cfg and isinstance(cfg['nc'], int):
                        return self._compute_fdi_mapping(cfg['nc'])
            except Exception:
                pass
        # Final fallback
        return self._compute_fdi_mapping(32)
        
    def detect_teeth(self, image_path, conf_threshold=0.5, iou_threshold=0.5, tta: bool = False):
        """
        Detect teeth in an image.
        
        Args:
            image_path (str): Path to the input image
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
            tta (bool): Enable Test-Time Augmentation
            
        Returns:
            tuple: (image with detections, list of detection results)
        """
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(
            image_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            augment=tta,
            verbose=False
        )
        
        return results[0].plot(), results[0]

    @staticmethod
    def _soft_nms_single_class(xyxyn: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5,
                               sigma: float = 0.5, method: str = 'gaussian', score_thresh: float = 0.001):
        """
        Basic Soft-NMS on a set of boxes for a single class.
        xyxyn: Nx4 normalized [x1,y1,x2,y2]
        scores: N
        method: 'gaussian' or 'linear'
        Returns indices kept and updated scores.
        """
        boxes = xyxyn.copy()
        scores = scores.copy()
        N = boxes.shape[0]
        indices = np.arange(N)
        keep = []

        def iou(box, others):
            x1 = np.maximum(box[0], others[:, 0])
            y1 = np.maximum(box[1], others[:, 1])
            x2 = np.minimum(box[2], others[:, 2])
            y2 = np.minimum(box[3], others[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
            union = area1 + area2 - inter + 1e-9
            return inter / union

        while indices.size > 0:
            # pick max score
            max_i = np.argmax(scores[indices])
            cur = indices[max_i]
            keep.append(cur)
            indices = np.delete(indices, max_i)
            if indices.size == 0:
                break
            ious = iou(boxes[cur], boxes[indices])
            if method == 'linear':
                decay = np.ones_like(ious)
                mask = ious > iou_thresh
                decay[mask] = 1 - ious[mask]
            else:  # gaussian
                decay = np.exp(-(ious ** 2) / sigma)
            scores[indices] = scores[indices] * decay
            # prune low scores
            keep_mask = scores[indices] >= score_thresh
            indices = indices[keep_mask]

        return np.array(keep, dtype=int), scores

    def apply_soft_nms(self, detections, iou_thresh=0.5, sigma=0.5, method='gaussian', score_thresh=0.001, conf_filter=0.25):
        """
        Apply Soft-NMS per class on Ultralytics detections in-place and filter by conf_filter after decay.
        Returns filtered indices list.
        """
        # Extract arrays
        boxes_xyxyn = detections.boxes.xyxyn.cpu().numpy()  # Nx4 normalized
        scores = detections.boxes.conf.cpu().numpy()        # N
        classes = detections.boxes.cls.cpu().numpy().astype(int)  # N

        kept_global = []
        updated_scores = scores.copy()
        for c in np.unique(classes):
            idxs = np.where(classes == c)[0]
            if idxs.size == 0:
                continue
            keep_c, new_scores_c = self._soft_nms_single_class(
                boxes_xyxyn[idxs], scores[idxs], iou_thresh=iou_thresh, sigma=sigma, method=method, score_thresh=score_thresh
            )
            # map back
            updated_scores[idxs] = new_scores_c
            kept_global.extend(idxs[keep_c].tolist())

        # Apply confidence filter after decay
        kept_global = [i for i in kept_global if updated_scores[i] >= conf_filter]

        # Update detections in-place
        import torch as _torch
        mask = _torch.zeros(len(scores), dtype=_torch.bool)
        if kept_global:
            mask[_torch.tensor(kept_global, dtype=_torch.long)] = True
        detections.boxes.conf = detections.boxes.conf[mask]
        detections.boxes.cls = detections.boxes.cls[mask]
        detections.boxes.xyxy = detections.boxes.xyxy[mask]
        detections.boxes.xyxyn = detections.boxes.xyxyn[mask]
        detections.boxes.xywh = detections.boxes.xywh[mask]
        detections.boxes.xywhn = detections.boxes.xywhn[mask]
        return kept_global

    def apply_wbf(self, detections, iou_thr=0.55, skip_box_thr=0.001, weights=None, conf_filter=0.25):
        """Apply Weighted Boxes Fusion to detections in-place. Requires ensemble-boxes package."""
        if weighted_boxes_fusion is None:
            raise RuntimeError("weighted_boxes_fusion not available. Please install ensemble-boxes.")
        # Prepare inputs (normalized xyxyn)
        boxes = detections.boxes.xyxyn.cpu().numpy().tolist()  # list of [x1,y1,x2,y2]
        scores = detections.boxes.conf.cpu().numpy().tolist()
        labels = detections.boxes.cls.cpu().numpy().astype(int).tolist()
        # WBF requires list per model
        wb, ws, wl = weighted_boxes_fusion([boxes], [scores], [labels],
                                           weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # Filter by conf
        keep_idx = [i for i, s in enumerate(ws) if s >= conf_filter]
        if not keep_idx:
            # Clear all
            detections.boxes.conf = detections.boxes.conf[:0]
            detections.boxes.cls = detections.boxes.cls[:0]
            detections.boxes.xyxy = detections.boxes.xyxy[:0]
            detections.boxes.xyxyn = detections.boxes.xyxyn[:0]
            detections.boxes.xywh = detections.boxes.xywh[:0]
            detections.boxes.xywhn = detections.boxes.xywhn[:0]
            return []
        wb = [wb[i] for i in keep_idx]
        ws = [ws[i] for i in keep_idx]
        wl = [wl[i] for i in keep_idx]
        # Update detections tensors
        xyxyn = torch.tensor(wb, dtype=detections.boxes.xyxyn.dtype, device=detections.boxes.xyxyn.device)
        confs = torch.tensor(ws, dtype=detections.boxes.conf.dtype, device=detections.boxes.conf.device)
        clses = torch.tensor(wl, dtype=detections.boxes.cls.dtype, device=detections.boxes.cls.device)
        # Convert normalized to absolute xyxy based on original shape if needed
        # Ultralytics stores original image size in detections.orig_shape
        h, w = detections.orig_shape[:2]
        xyxy = xyxyn.clone()
        xyxy[:, [0, 2]] *= w
        xyxy[:, [1, 3]] *= h
        # xywh/xywhn can be recomputed if needed
        detections.boxes.xyxyn = xyxyn
        detections.boxes.xyxy = xyxy
        detections.boxes.conf = confs
        detections.boxes.cls = clses
        # Recompute xywh and xywhn
        xywh = xyxy.clone()
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        xywh[:, 0] = xyxy[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = xyxy[:, 1] + xywh[:, 3] / 2
        detections.boxes.xywh = xywh
        xywhn = xywh.clone()
        xywhn[:, [0, 2]] = xywhn[:, [0, 2]] / w
        xywhn[:, [1, 3]] = xywhn[:, [1, 3]] / h
        detections.boxes.xywhn = xywhn
        return list(range(len(ws)))

    def apply_anatomical_post(self, detections):
        """
        Apply anatomical postprocessing (quadrant sort and out-of-order filtering)
        using train_detection_model.postprocess_predictions and rebuild detection tensors.
        """
        if _anatomical_post is None:
            return
        # postprocess_predictions expects a list of predictions-like with .boxes fields
        processed = _anatomical_post([detections])
        if not processed:
            return
        # Build tensors from processed list (expects normalized xywhn)
        import torch as _torch
        h, w = detections.orig_shape[:2]
        xywhn_list = []
        confs_list = []
        cls_list = []
        for p in processed:
            # box is [x, y, w, h] normalized
            bx = p['box']
            xywhn_list.append([bx[0], bx[1], bx[2], bx[3]])
            confs_list.append(float(p['score']))
            cls_list.append(int(p['class_id']))
        if not xywhn_list:
            return
        xywhn = _torch.tensor(xywhn_list, dtype=detections.boxes.xywhn.dtype, device=detections.boxes.xywhn.device)
        confs = _torch.tensor(confs_list, dtype=detections.boxes.conf.dtype, device=detections.boxes.conf.device)
        clses = _torch.tensor(cls_list, dtype=detections.boxes.cls.dtype, device=detections.boxes.cls.device)
        # Derive other formats
        xyxyn = _torch.zeros((xywhn.shape[0], 4), dtype=xywhn.dtype, device=xywhn.device)
        xyxyn[:, 0] = xywhn[:, 0] - xywhn[:, 2] / 2.0
        xyxyn[:, 1] = xywhn[:, 1] - xywhn[:, 3] / 2.0
        xyxyn[:, 2] = xywhn[:, 0] + xywhn[:, 2] / 2.0
        xyxyn[:, 3] = xywhn[:, 1] + xywhn[:, 3] / 2.0
        xyxy = xyxyn.clone()
        xyxy[:, [0, 2]] *= w
        xyxy[:, [1, 3]] *= h
        xywh = xywhn.clone()
        xywh[:, [0, 2]] *= w
        xywh[:, [1, 3]] *= h
        # Assign back
        detections.boxes.xywhn = xywhn
        detections.boxes.xyxyn = xyxyn
        detections.boxes.xyxy = xyxy
        detections.boxes.xywh = xywh
        detections.boxes.conf = confs
        detections.boxes.cls = clses
    
    def plot_detections(self, image, detections, output_path=None):
        """
        Plot detections on the image.
        
        Args:
            image: Input image (numpy array)
            detections: YOLO detection results
            output_path (str, optional): Path to save the output image
            
        Returns:
            numpy.ndarray: Image with detections drawn
        """
        # Get the image with detections drawn
        img_with_dets = detections.plot()
        
        # Save the result if output path is provided
        if output_path:
            cv2.imwrite(str(output_path), cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
            print(f"Saved detection result to {output_path}")
            
        return img_with_dets
    
    def process_directory(self, input_dir, output_dir, conf_threshold=0.5, iou_threshold=0.5, tta: bool = False,
                          soft_nms: bool = False, soft_nms_method: str = 'gaussian', soft_nms_sigma: float = 0.5):
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Path to directory containing input images
            output_dir (str): Path to save output images with detections
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
            tta (bool): Enable Test-Time Augmentation
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # Process each image
        for img_path in image_files:
            try:
                # Get detections
                img_with_dets, detections = self.detect_teeth(
                    str(img_path),
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    tta=tta
                )
                if soft_nms and len(detections.boxes) > 0:
                    self.apply_soft_nms(detections, iou_thresh=iou_threshold, sigma=soft_nms_sigma,
                                        method=soft_nms_method, conf_filter=conf_threshold)
                
                # Save the result
                output_path = output_dir / f"det_{img_path.name}"
                cv2.imwrite(str(output_path), cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
                
                # Print detection summary
                print(f"\nDetections in {img_path.name}:")
                for box in detections.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    fdi = self.class_to_fdi.get(class_id, class_id)
                    print(f"- Tooth FDI {fdi}: {conf:.2f} confidence")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect teeth in dental images using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLOv8 model (.pt file)')
    parser.add_argument('--source', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--data-yaml', type=str, default=None, help='Optional dataset YAML to load FDI mapping')
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation')
    # Soft-NMS flags
    parser.add_argument('--soft-nms', action='store_true', help='Enable Soft-NMS postprocessing')
    parser.add_argument('--soft-nms-method', type=str, choices=['gaussian', 'linear'], default='gaussian', help='Soft-NMS decay method')
    parser.add_argument('--soft-nms-sigma', type=float, default=0.5, help='Soft-NMS sigma for gaussian method')
    # WBF flags (takes precedence if set)
    parser.add_argument('--wbf', action='store_true', help='Enable Weighted Boxes Fusion (overrides Soft-NMS if both set)')
    parser.add_argument('--wbf-iou', type=float, default=0.55, help='WBF IoU threshold')
    parser.add_argument('--wbf-skip-thr', type=float, default=0.001, help='WBF skip box threshold')
    # Anatomical postprocessing summary
    parser.add_argument('--anatomical-post', action='store_true', help='Print anatomical postprocessing summary (quadrant ordering)')
    args = parser.parse_args()
    
    # Initialize the detector
    detector = ToothDetector(args.model, data_yaml_path=args.data_yaml)
    
    # Check if source is a file or directory
    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if source_path.is_file():
        # Process single image
        try:
            img_with_dets, detections = detector.detect_teeth(
                str(source_path),
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                tta=args.tta
            )
            if args.wbf and len(detections.boxes) > 0:
                detector.apply_wbf(detections, iou_thr=args.wbf_iou, skip_box_thr=args.wbf_skip_thr, conf_filter=args.conf)
            elif args.soft_nms and len(detections.boxes) > 0:
                detector.apply_soft_nms(detections, iou_thresh=args.iou, sigma=args.soft_nms_sigma,
                                        method=args.soft_nms_method, conf_filter=args.conf)
            
            # Optional anatomical postprocessing apply + summary
            if args.anatomical_post and _anatomical_post is not None and len(detections.boxes) > 0:
                try:
                    detector.apply_anatomical_post(detections)
                    # Summary print
                    processed = _anatomical_post([detections])
                    if processed:
                        print("\nAnatomical ordering summary (FDI, quadrant, conf):")
                        for p in processed:
                            print(f"  FDI {p['fdi']} | Q{p['quadrant']} | conf {p['score']:.2f}")
                except Exception:
                    pass

            # Save the result
            output_path = output_dir / f"det_{source_path.name}"
            cv2.imwrite(str(output_path), cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
            print(f"Saved detection result to {output_path}")
            
            # Print detection summary
            print("\nDetections:")
            for box in detections.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                fdi = detector.class_to_fdi.get(class_id, class_id)
                print(f"- Tooth FDI {fdi}: {conf:.2f} confidence")
            
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
    else:
        # Process directory
        # For directory mode, we will still apply Soft-NMS if requested inside the loop
        detector.process_directory(
            args.source,
            args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            tta=args.tta,
            soft_nms=args.soft_nms,
            soft_nms_method=args.soft_nms_method,
            soft_nms_sigma=args.soft_nms_sigma
        )

if __name__ == "__main__":
    main()
