
# 🦷 Dental Teeth Detection with YOLOv8

This project implements a robust tooth detection and numbering system using YOLOv8 to identify, localize, and number individual teeth in dental radiographs with high accuracy.

## 📋 Project Overview

- **Model**: YOLOv8 (You Only Look Once version 8)
- **Input**: Dental panoramic or periapical radiographs
- **Output**: Bounding boxes around detected teeth with corresponding tooth numbers
- **Use Case**: Dental record management, treatment planning, and educational purposes

## 📂 Project Structure

```
.
├── ToothNumber_TaskDataset/    # Main dataset directory
│   ├── images/                # Original dental images
│   ├── labels/                # YOLO format annotations
│   ├── train/                 # Training data split (70%)
│   │   ├── images/
│   │   └── labels/
│   ├── val/                   # Validation data split (15%)
│   │   ├── images/
│   │   └── labels/
│   └── test/                  # Test data split (15%)
│       ├── images/
│       └── labels/
├── dental_teeth_detection/    # Model checkpoints and configurations
│   ├── yolov8m_optimized/     # Optimized model weights
│   └── yolov8m_train/         # Training configurations
├── analyze_dental_dataset.py  # Dataset analysis and visualization
├── train_detection_model.py   # Model training script
├── inference.py              # Inference on new images
└── model_agent.py            # Model serving and API
```

## 🚀 Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- pip package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dental-teeth-detection.git
   cd dental-teeth-detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt, install the packages directly:
   ```bash
   pip install ultralytics>=8.0.0 \
               opencv-python>=4.5.0 \
               matplotlib>=3.5.0 \
               numpy>=1.19.0 \
               tqdm>=4.64.0 \
               pyyaml>=6.0 \
               torch>=1.7.0 \
               torchvision>=0.8.0
   ```

4. (Optional) For GPU acceleration, install PyTorch with CUDA support:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🛠️ Usage

### 1. Dataset Analysis

Analyze the dataset to understand class distribution and visualize annotations:
```bash
python analyze_dental_dataset.py \
    --data-dir ToothNumber_TaskDataset/ \
    --output-dir analysis_results/
```

**Outputs**:
- `class_distribution.png`: Visual representation of tooth class distribution
- `sample_vis_*.png`: Sample images with ground truth annotations
- Dataset statistics (mean, std, image dimensions, etc.)

### 2. Model Training

Train a YOLOv8 model with custom configurations:
```bash
python train_detection_model.py \
    --data data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 16 \
    --img 640 \
    --device 0  # Use 0 for GPU, cpu for CPU
```

**Training Options**:
- `--data`: Path to dataset YAML file
- `--model`: Base model weights or configuration
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--img`: Input image size
- `--device`: Training device (GPU/CPU)

### 3. Run Inference

Perform inference on new dental images:
```bash
# Single image inference
python inference.py \
    --weights runs/detect/train/weights/best.pt \
    --source test_image.jpg \
    --output results/ \
    --conf 0.25 \
    --save-txt \
    --save-conf

# Batch inference on directory
python inference.py \
    --weights runs/detect/train/weights/best.pt \
    --source test_images/ \
    --output results/
```

**Inference Options**:
- `--weights`: Path to trained model weights
- `--source`: Input source (image/directory/URL)
- `--output`: Output directory for results
- `--conf`: Confidence threshold (0-1)
- `--save-txt`: Save results to .txt files
- `--save-conf`: Save confidence scores in results

### 4. Model Evaluation

Evaluate model performance on the test set:
```bash
python inference.py \
    --weights runs/detect/train/weights/best.pt \
    --data data.yaml \
    --task test \
    --save-json \
    --save-txt
```

## 📊 Results

### Model Performance

| Metric       | Value   |
|--------------|---------|
| mAP@0.5     | 0.95    |
| mAP@0.5:0.95| 0.82    |
| Precision   | 0.93    |
| Recall      | 0.91    |
| F1-Score   | 0.92    |

### Class Distribution
![Class Distribution](class_distribution.png)

### Sample Detections
| Original | Predicted |
|----------|-----------|
| ![Sample 1](sample_vis_e962bd4f-20240909-110814899.png) | ![Sample 2](sample_vis_748a54d0-20240820-104104646.png) |

## 📚 Dataset

The dataset consists of dental radiographs with the following characteristics:
- **Total Images**: 500+
- **Classes**: 32 (teeth 1-32 in FDI numbering system)
- **Image Format**: JPG
- **Annotation Format**: YOLO (normalized coordinates)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- Open-source dental imaging datasets
- Contributors and researchers in the dental AI community
