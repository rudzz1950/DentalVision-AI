import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

class ToothDetector:
    def __init__(self, model_path):
        """
        Initialize the tooth detector with a trained YOLOv8 model.
        
        Args:
            model_path (str): Path to the trained YOLOv8 model (.pt file)
        """
        self.model = YOLO(model_path)
        self.class_to_tooth = {i: str(i+1) for i in range(32)}  # Map class IDs to tooth numbers
        
    def detect_teeth(self, image_path, conf_threshold=0.5, iou_threshold=0.5):
        """
        Detect teeth in an image.
        
        Args:
            image_path (str): Path to the input image
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
            
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
            verbose=False
        )
        
        return results[0].plot(), results[0]
    
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
    
    def process_directory(self, input_dir, output_dir, conf_threshold=0.5, iou_threshold=0.5):
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Path to directory containing input images
            output_dir (str): Path to save output images with detections
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
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
                    iou_threshold=iou_threshold
                )
                
                # Save the result
                output_path = output_dir / f"det_{img_path.name}"
                cv2.imwrite(str(output_path), cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
                
                # Print detection summary
                print(f"\nDetections in {img_path.name}:")
                for box in detections.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"- Tooth {self.class_to_tooth[class_id]}: {conf:.2f} confidence")
                
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
    args = parser.parse_args()
    
    # Initialize the detector
    detector = ToothDetector(args.model)
    
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
                iou_threshold=args.iou
            )
            
            # Save the result
            output_path = output_dir / f"det_{source_path.name}"
            cv2.imwrite(str(output_path), cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
            print(f"Saved detection result to {output_path}")
            
            # Print detection summary
            print("\nDetections:")
            for box in detections.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"- Tooth {detector.class_to_tooth[class_id]}: {conf:.2f} confidence")
            
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
    else:
        # Process directory
        detector.process_directory(
            args.source,
            args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )

if __name__ == "__main__":
    main()
