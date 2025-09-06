import os
from pathlib import Path
import shutil
from typing import List

import cv2
import numpy as np

from inference import ToothDetector

README_MARK_START = "<!-- SAMPLE_PREDICTIONS_START -->"
README_MARK_END = "<!-- SAMPLE_PREDICTIONS_END -->"


def generate_samples(model_path: Path, data_yaml: Path, images_dir: Path, out_dir: Path, max_images: int = 4) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector with FDI mapping
    det = ToothDetector(str(model_path), data_yaml_path=str(data_yaml))

    # Pick a few images
    image_files = sorted([p for p in images_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])[:max_images]
    saved: List[Path] = []

    for img_path in image_files:
        try:
            # Run detection
            img_with_dets, d = det.detect_teeth(str(img_path), conf_threshold=0.35, iou_threshold=0.5, tta=False)
            # Prefer WBF if available; otherwise Soft-NMS isn't exposed via this script
            try:
                if len(d.boxes) > 0:
                    det.apply_wbf(d, iou_thr=0.55, skip_box_thr=0.001, conf_filter=0.35)
            except Exception:
                pass
            # Apply anatomical postprocessing (default behavior from CLI, mirror here)
            try:
                if len(d.boxes) > 0:
                    det.apply_anatomical_post(d)
            except Exception:
                pass
            # Re-render with final boxes and add quadrant overlay
            img_with_dets = d.plot()
            img_with_dets = det.draw_quadrant_overlay(img_with_dets)

            # Save
            out_path = out_dir / f"sample_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
            saved.append(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return saved


def update_readme_with_samples(readme_path: Path, sample_paths: List[Path], root: Path) -> None:
    if not sample_paths:
        print("No sample paths provided; skipping README update.")
        return

    rel_paths = [p.relative_to(root).as_posix() for p in sample_paths]

    section = [README_MARK_START, "\n", "## Sample Predictions\n", "\n",
               "Below are a few sample predictions with FDI labels (11–48), anatomical ordering, and quadrant overlays:\n", "\n"]
    # Make a simple 2-column layout
    for rp in rel_paths:
        section.append(f"![Sample]({rp})\n\n")
    section.append(README_MARK_END)
    section_text = "\n".join(section)

    # Insert or replace between markers
    text = readme_path.read_text(encoding="utf-8")
    if README_MARK_START in text and README_MARK_END in text:
        start = text.index(README_MARK_START)
        end = text.index(README_MARK_END) + len(README_MARK_END)
        new_text = text[:start] + section_text + text[end:]
    else:
        # Append at the end
        new_text = text.rstrip() + "\n\n" + section_text + "\n"

    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Updated README with {len(rel_paths)} sample images.")


def main():
    root = Path(__file__).parent
    runs_dir = root / 'runs' / 'train'
    exps = [p for p in runs_dir.iterdir() if p.is_dir()] if runs_dir.exists() else []
    if not exps:
        print('No experiments found under runs/train. Exiting.')
        return
    latest = sorted(exps, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    best = latest / 'weights' / 'best.pt'
    if not best.exists():
        print(f'best.pt not found under {latest}. Exiting.')
        return

    data_yaml = root / 'ToothNumber_TaskDataset' / 'dental_teeth.yaml'
    img_dir = root / 'ToothNumber_TaskDataset' / 'test' / 'images'
    out_dir = root / 'results' / 'readme_samples'

    print('Generating README sample images...')
    samples = generate_samples(best, data_yaml, img_dir, out_dir, max_images=4)

    readme = root / 'README.md'
    print('Updating README...')
    update_readme_with_samples(readme, samples, root)


if __name__ == '__main__':
    main()
