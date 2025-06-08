import torch
import argparse
import os
from typing import Dict, Any, List, Optional
import logging
from torch.utils.data import DataLoader

from .base_model import BaseModel
from .inference import ModelInference
from .evaluation import ModelEvaluator
from .trainer import ModelTrainer
from ..multidatasets import DetectionDataset

class TestPipeline:
    """Complete testing pipeline for multi-modal detection models."""
    
    def __init__(self, model_type: str = "rtdetr", model_name: str = "rtdetr_r50vd_6x_coco",
                 device: str = "auto", num_classes: int = 80):
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.base_model = BaseModel(model_type, model_name, device, num_classes)
        self.inference = ModelInference(self.base_model)
        self.evaluator = ModelEvaluator(self.base_model)
        self.trainer = ModelTrainer(self.base_model, self.evaluator)
    
    def test_inference(self, image_path: str, confidence_threshold: float = 0.5,
                      visualize: bool = True, save_path: str = None) -> Dict[str, Any]:
        """Test inference on a single image."""
        
        self.logger.info(f"Testing inference on: {image_path}")
        
        results = self.inference.predict(
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            visualize=visualize,
            save_path=save_path
        )
        
        self.logger.info(f"Detected {len(results['detections'])} objects")
        return results
    
    def test_batch_inference(self, image_dir: str, confidence_threshold: float = 0.5,
                           batch_size: int = 8, save_dir: str = None) -> List[Dict[str, Any]]:
        """Test inference on a batch of images."""
        
        self.logger.info(f"Testing batch inference on directory: {image_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        results = []
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            
            for image_path in batch_files:
                save_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.basename(image_path)
                    save_path = os.path.join(save_dir, f"result_{filename}")
                
                result = self.inference.predict(
                    image_path=image_path,
                    confidence_threshold=confidence_threshold,
                    visualize=save_dir is not None,
                    save_path=save_path
                )
                results.append(result)
        
        self.logger.info(f"Completed batch inference on {len(results)} images")
        return results
    
    def test_evaluation(self, dataset_path: str, dataset_type: str = "coco",
                       confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Test evaluation on a dataset."""
        
        self.logger.info(f"Testing evaluation on {dataset_type} dataset: {dataset_path}")
        
        if dataset_type.lower() == "coco":
            metrics = self.evaluator.evaluate_coco(
                dataset_path=dataset_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        elif dataset_type.lower() == "kitti":
            metrics = self.evaluator.evaluate_kitti(
                dataset_path=dataset_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        else:
            # Custom evaluation
            metrics = self.evaluator.evaluate_custom(
                dataset_path=dataset_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        
        self.logger.info(f"Evaluation completed. mAP: {metrics.get('mAP', 'N/A')}")
        return metrics
    
    def test_training(self, train_dataset_path: str, val_dataset_path: str = None,
                     num_epochs: int = 5, batch_size: int = 8, lr: float = 1e-4,
                     save_dir: str = "./test_checkpoints") -> Dict[str, Any]:
        """Test training pipeline."""
        
        self.logger.info("Testing training pipeline")
        
        # Create datasets
        train_dataset = DetectionDataset(
            dataset_path=train_dataset_path,
            split="train",
            transform=None  # Add appropriate transforms
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_dataloader = None
        if val_dataset_path:
            val_dataset = DetectionDataset(
                dataset_path=val_dataset_path,
                split="val",
                transform=None
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # Run training
        training_results = self.trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
            lr=lr,
            save_dir=save_dir,
            save_every=2,
            validate_every=1
        )
        
        self.logger.info("Training test completed")
        return training_results
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        return self.evaluator._collate_fn(batch)
    
    def run_full_pipeline_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete pipeline test with the given configuration."""
        
        self.logger.info("Running full pipeline test")
        
        results = {}
        
        # Test inference if image provided
        if config.get('test_image'):
            results['inference'] = self.test_inference(
                image_path=config['test_image'],
                confidence_threshold=config.get('confidence_threshold', 0.5),
                visualize=config.get('visualize', True)
            )
        
        # Test evaluation if dataset provided
        if config.get('eval_dataset'):
            results['evaluation'] = self.test_evaluation(
                dataset_path=config['eval_dataset'],
                dataset_type=config.get('dataset_type', 'coco'),
                confidence_threshold=config.get('confidence_threshold', 0.5)
            )
        
        # Test training if training dataset provided
        if config.get('train_dataset'):
            results['training'] = self.test_training(
                train_dataset_path=config['train_dataset'],
                val_dataset_path=config.get('val_dataset'),
                num_epochs=config.get('num_epochs', 2),
                batch_size=config.get('batch_size', 4),
                lr=config.get('lr', 1e-4)
            )
        
        self.logger.info("Full pipeline test completed")
        return results


def test_rtdetr():
    """Test RT-DETR model functionality."""
    
    print("Testing RT-DETR model...")
    
    # Initialize test pipeline
    pipeline = TestPipeline(
        model_type="rtdetr",
        model_name="rtdetr_r50vd_6x_coco",
        device="auto",
        num_classes=80
    )
    
    # Test configuration
    config = {
        'test_image': "./sampledata/bus.jpg",
        'confidence_threshold': 0.5,
        'visualize': True
    }
    
    # Run tests
    results = pipeline.run_full_pipeline_test(config)
    
    print(f"Test completed. Results: {results}")
    return results


def test_multimodels():
    """Test multiple model types."""
    
    print("Testing multiple model types...")
    
    model_configs = [
        {"model_type": "rtdetr", "model_name": "rtdetr_r50vd_6x_coco"},
        {"model_type": "detr", "model_name": "detr_resnet50"},
        {"model_type": "yolo", "model_name": "yolov8n"}
    ]
    
    results = {}
    
    for config in model_configs:
        try:
            print(f"Testing {config['model_type']} - {config['model_name']}")
            
            pipeline = TestPipeline(
                model_type=config["model_type"],
                model_name=config["model_name"],
                device="auto",
                num_classes=80
            )
            
            test_config = {
                'test_image': "./sampledata/bus.jpg",
                'confidence_threshold': 0.5,
                'visualize': False
            }
            
            model_results = pipeline.run_full_pipeline_test(test_config)
            results[f"{config['model_type']}_{config['model_name']}"] = model_results
            
        except Exception as e:
            print(f"Error testing {config['model_type']}: {e}")
            results[f"{config['model_type']}_{config['model_name']}"] = {"error": str(e)}
    
    print("Multi-model testing completed")
    return results


def main():
    """Main function for command-line testing."""
    
    parser = argparse.ArgumentParser(description="Test Detection Models Pipeline")
    parser.add_argument("--model_type", type=str, default="rtdetr",
                       choices=["rtdetr", "detr", "yolo", "yolov8", "yolov9", "yolov10"],
                       help="Type of model to test")
    parser.add_argument("--model_name", type=str, default="rtdetr_r50vd_6x_coco",
                       help="Name of the model")
    parser.add_argument("--test_image", type=str, default="./sampledata/bus.jpg",
                       help="Path to test image")
    parser.add_argument("--eval_dataset", type=str, default=None,
                       help="Path to evaluation dataset")
    parser.add_argument("--train_dataset", type=str, default=None,
                       help="Path to training dataset")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--num_classes", type=int, default=80,
                       help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training/evaluation")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of epochs for training test")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = TestPipeline(
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.device,
        num_classes=args.num_classes
    )
    
    # Create test configuration
    config = {
        'test_image': args.test_image,
        'eval_dataset': args.eval_dataset,
        'train_dataset': args.train_dataset,
        'confidence_threshold': args.confidence_threshold,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'visualize': True
    }
    
    # Run tests
    results = pipeline.run_full_pipeline_test(config)
    
    print("\n=== Test Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Run basic tests
    test_rtdetr()
    test_multimodels()
    
    # Run command-line interface
    main()