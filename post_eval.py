from pathlib import Path
from ultralytics import YOLO
import sys
import os
import yaml

# Import functions from training script
from train_detection_model import evaluate_model, compute_calibration_and_heatmaps, _generate_html_report

def main():
    root = Path(__file__).parent
    runs_dir = root / 'runs' / 'train'
    if not runs_dir.exists():
        print('No runs/train directory found.')
        sys.exit(1)

    # Find latest experiment directory containing weights/best.pt
    exps = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not exps:
        print('No experiment directories found under runs/train.')
        sys.exit(1)
    latest = sorted(exps, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    best_path = latest / 'weights' / 'best.pt'
    if not best_path.exists():
        print(f'best.pt not found under {latest} yet.')
        sys.exit(2)

    # Load model
    print(f'Loading model: {best_path}')
    model = YOLO(str(best_path))

    # Dataset YAML
    data_yaml = root / 'ToothNumber_TaskDataset' / 'dental_teeth.yaml'
    if not data_yaml.exists():
        print(f'Dataset YAML not found at {data_yaml}')
        sys.exit(3)

    # Output dir is the experiment folder
    output_dir = latest

    # Evaluate and produce artifacts
    print('Evaluating on val...')
    evaluate_model(model, str(data_yaml), output_dir, split='val')
    print('Evaluating on test...')
    evaluate_model(model, str(data_yaml), output_dir, split='test')

    print('Computing calibration and heatmaps (val/test)...')
    compute_calibration_and_heatmaps(model, str(data_yaml), split='val', output_dir=output_dir)
    compute_calibration_and_heatmaps(model, str(data_yaml), split='test', output_dir=output_dir)

    print('Generating HTML report...')
    _generate_html_report(output_dir)
    print('Post-evaluation artifacts generated under:', output_dir / 'evaluation')

if __name__ == '__main__':
    main()
