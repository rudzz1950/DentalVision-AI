import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import torch
from ultralytics import YOLO

@dataclass
class ModelInfo:
    """Stores information about the YOLOv8 model."""
    model_size: str = 'm'
    num_classes: int = 32
    input_size: int = 640
    batch_size: int = 16
    optimizer: str = 'AdamW'
    learning_rate: float = 0.01
    epochs: int = 200
    patience: int = 30
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def model_name(self) -> str:
        return f'yolov8{self.model_size}.pt'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'patience': self.patience,
            'device': self.device
        }

class ModelAgent:
    """Agent for answering questions about the dental teeth detection model."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the model agent.
        
        Args:
            model_path: Path to the trained YOLOv8 model weights
            config_path: Path to the dataset YAML config file
        """
        self.model = None
        self.model_info = ModelInfo()
        self.config = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_config(self, config_path: str) -> None:
        """Load dataset configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        if not self.model:
            return "No model loaded. Please load a model first."
            
        summary = []
        model_info = self.model.model
        
        if hasattr(model_info, 'model'):
            num_params = sum(p.numel() for p in model_info.parameters())
            summary.append(f"Model: YOLOv8{self.model_info.model_size}")
            summary.append(f"Parameters: {num_params:,}")
            summary.append(f"Input size: {self.model_info.input_size}x{self.model_info.input_size}")
            summary.append(f"Number of classes: {self.model_info.num_classes}")
            summary.append(f"Device: {self.model_info.device}")
        
        return "\n".join(summary)
    
    def get_training_info(self) -> str:
        """Get information about the training configuration."""
        info = [
            "Training Configuration:",
            f"- Epochs: {self.model_info.epochs}",
            f"- Batch size: {self.model_info.batch_size}",
            f"- Optimizer: {self.model_info.optimizer}",
            f"- Learning rate: {self.model_info.learning_rate}",
            f"- Early stopping patience: {self.model_info.patience} epochs",
            f"- Input size: {self.model_info.input_size}x{self.model_info.input_size}"
        ]
        return "\n".join(info)
    
    def get_dataset_info(self) -> str:
        """Get information about the dataset."""
        if not self.config:
            return "No dataset configuration loaded."
            
        info = ["Dataset Configuration:"]
        for key, value in self.config.items():
            if key == 'names' and isinstance(value, dict):
                class_info = [f"  - {i+1}: {name}" for i, name in value.items()]
                info.append(f"- Classes ({len(value)}):")
                info.extend(class_info)
            else:
                info.append(f"- {key}: {value}")
        
        return "\n".join(info)
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the model and dataset."""
        question = question.lower()
        
        if any(q in question for q in ['model', 'architecture', 'parameters']):
            return self.get_model_summary()
        
        elif any(q in question for q in ['train', 'training', 'epoch', 'batch', 'optimizer']):
            return self.get_training_info()
        
        elif any(q in question for q in ['dataset', 'data', 'class', 'label']):
            return self.get_dataset_info()
        
        elif any(q in question for q in ['help', 'what can you do']):
            return """I can answer questions about:
- Model architecture and parameters
- Training configuration and hyperparameters
- Dataset information and class labels
- Model usage and inference

Try asking about the model, training, or dataset!"""
        
        return "I'm not sure how to answer that. Try asking about the model, training, or dataset."

def main():
    """Run the model agent in interactive mode."""
    # Initialize with default paths (update these to your actual paths)
    model_path = "yolov8m.pt"  # Update with your model path
    config_path = "dental_teeth.yaml"  # Update with your config path
    
    agent = ModelAgent(model_path=model_path, config_path=config_path)
    
    print("Dental Teeth Detection Model Assistant")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("Ask me about the model: ").strip()
            if question.lower() in ['exit', 'quit']:
                break
                
            if question:
                response = agent.answer_question(question)
                print("\n" + response + "\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
