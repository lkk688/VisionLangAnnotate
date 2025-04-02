import os
import contextlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
import re
from typing import Dict, List, Optional, Tuple, Union
import time
import torchvision
import json
import glob
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MultiWorld:
    """
    MultiWorld class that implements object detection model training, evaluation, and inference
    based on the Hugging Face framework. Supports multiple model architectures including:
    - YOLOv8
    - DETR
    - RT-DETR
    - RT-DETRv2
    - ViTDet
    and other Hugging Face compatible object detection models.
    """
    def __init__(self, model=None, config=None, model_type="yolov8", model_name=None, scale='s', device=None):
        """
        Initialize MultiWorld with a model or create a new one.
        
        Args:
            model: Existing detection model or None to create a new one
            config: Model config object or None to create a default one
            model_type: Type of model to use ('yolov8', 'detr', 'rt-detr', 'rt-detrv2', 'vitdet')
            model_name: Specific model name/checkpoint from Hugging Face Hub (e.g., 'facebook/detr-resnet-50')
            scale: Model scale ('n', 's', 'm', 'l', 'x') if creating a new YOLO model
            device: Device to use (None for auto-detection)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        
        # Create or use provided model
        if model is None:
            # If model_name is provided, load from Hugging Face Hub
            if model_name:
                self.model = self._load_from_hub(model_name)
                self.model_type = self._detect_model_type(self.model, model_name)
            else:
                # Create a new model based on model_type
                self.model = self._create_model(model_type, config, scale, model_name)
        else:
            self.model = model
            # Try to detect model type if not explicitly provided
            if model_type == "auto":
                self.model_type = self._detect_model_type(self.model)
            
        self.model = self.model.to(self.device)
        
        # Store or create config
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        elif config is not None:
            self.config = config
        else:
            # Create default config based on model type
            self.config = self._create_default_config(model_type, scale)
        
        # Initialize appropriate image processor based on model type
        self.processor = self._create_processor()
        
        # COCO class names
        self.class_names = coco_names
        
        # Update model config with COCO class names if needed
        self._update_model_config_with_class_names()
    
    def _update_model_config_with_class_names(self):
        """
        Update the model's configuration with COCO class names if needed.
        """
        if hasattr(self.model, 'config'):
            # Check if id2label is missing or empty
            if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
                self.model.config.id2label = {str(k): v for k, v in self.class_names.items()}
                self.model.config.label2id = {v: str(k) for k, v in self.class_names.items()}
            # Check if using numeric keys instead of string keys
            elif all(isinstance(k, int) for k in self.model.config.id2label.keys()):
                self.model.config.id2label = {str(k): v for k, v in self.model.config.id2label.items()}
                self.model.config.label2id = {v: str(k) for k, v in self.model.config.id2label.items()}
    
    def _create_model(self, model_type, config=None, scale='s', model_name=None):
        """
        Create a new model based on the specified type.
        
        Args:
            model_type: Type of model to create
            config: Optional config for the model
            scale: Scale for YOLO models
            model_name: Optional model name for loading from HF Hub
            
        Returns:
            New model instance
        """
        if model_type == 'yolov8':
            # Create YOLOv8 model
            if config is None:
                config = YoloConfig(
                    scale=scale,
                    nc=80,
                    ch=3,
                    min_size=640,
                    max_size=640,
                    use_fp16=True if self.device.type == 'cuda' else False
                )
                
            # Check if a specific model name is provided to load from HF Hub
            if model_name and 'yolo' in model_name.lower():
                # Register YOLO architecture with HF
                from modeling_yolohfdetr import register_yolo_architecture
                register_yolo_architecture()
                
                # Load model from Hugging Face Hub
                from transformers import AutoModelForObjectDetection
                try:
                    print(f"Loading YOLO model from Hugging Face Hub: {model_name}")
                    model = AutoModelForObjectDetection.from_pretrained(model_name)
                    # Update model's config with COCO class names if needed
                    if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                        if not model.config.id2label or len(model.config.id2label) == 0:
                            model.config.id2label = {str(k): v for k, v in coco_names.items()}
                            model.config.label2id = {v: str(k) for k, v in coco_names.items()}
                    return model
                except Exception as e:
                    print(f"Error loading YOLO model from HF Hub: {e}")
                    print("Falling back to local YOLOv8 model creation")
            
            # Create local model if no HF model was loaded
            return YoloDetectionModel(cfg=config, device=self.device)
            
    def _load_from_hub(self, model_name):
        """
        Load a model from Hugging Face Hub.
        
        Args:
            model_name: Model name/path on Hugging Face Hub
            
        Returns:
            Loaded model
        """
        try:
            # Check if this is a YOLO model
            if 'yolo' in model_name.lower():
                # Register YOLO architecture with HF
                from modeling_yolohfdetr import register_yolo_architecture
                register_yolo_architecture()
                
            from transformers import AutoModelForObjectDetection
            print(f"Loading model from Hugging Face Hub: {model_name}")
            model = AutoModelForObjectDetection.from_pretrained(model_name)
            
            # Update model's config with COCO class names if needed
            if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                if not model.config.id2label or len(model.config.id2label) == 0:
                    model.config.id2label = {str(k): v for k, v in coco_names.items()}
                    model.config.label2id = {v: str(k) for k, v in coco_names.items()}
            
            return model
        except Exception as e:
            print(f"Error loading model from hub: {e}")
            print("Falling back to YOLOv8")
            return YoloDetectionModel(
                cfg=YoloConfig(
                    scale='s',
                    nc=80,
                    ch=3,
                    min_size=640,
                    max_size=640,
                    use_fp16=True if self.device.type == 'cuda' else False
                ),
                device=self.device
            )
    
    def _detect_model_type(self, model, model_name=None):
        """
        Detect the type of model based on its architecture or name.
        
        Args:
            model: Model to detect type for
            model_name: Optional model name to help with detection
            
        Returns:
            Detected model type as string
        """
        # First check model name if provided
        if model_name:
            model_name_lower = model_name.lower()
            if 'yolo' in model_name_lower:
                return 'yolov8'
            elif 'detr-' in model_name_lower:
                return 'detr'
            elif 'rt-detrv2' in model_name_lower:
                return 'rt-detrv2'
            elif 'rt-detr' in model_name_lower:
                return 'rt-detr'
            elif 'vit-det' in model_name_lower:
                return 'vitdet'
        
        # Check model architecture
        model_class_name = model.__class__.__name__
        if 'Yolo' in model_class_name:
            return 'yolov8'
        elif 'Detr' in model_class_name:
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                if 'rt-detr' in model.config.model_type.lower():
                    return 'rt-detr'
            return 'detr'
        elif 'ViT' in model_class_name or 'Vit' in model_class_name:
            return 'vitdet'
        
        # Check model config
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            model_type = model.config.model_type.lower()
            if 'yolo' in model_type:
                return 'yolov8'
            elif 'rt-detrv2' in model_type:
                return 'rt-detrv2'
            elif 'rt-detr' in model_type:
                return 'rt-detr'
            elif 'detr' in model_type:
                return 'detr'
            elif 'vit' in model_type and 'det' in model_type:
                return 'vitdet'
        
        # Default to YOLO if can't determine
        print(f"Could not determine model type, defaulting to 'yolov8'")
        return 'yolov8'
    
    def _create_default_config(self, model_type, scale='s'):
        """
        Create a default config for the specified model type.
        
        Args:
            model_type: Type of model
            scale: Scale for YOLO models
            
        Returns:
            Config object
        """
        if model_type == 'yolov8':
            return YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if self.device.type == 'cuda' else False
            )
        elif model_type in ['detr', 'rt-detr', 'rt-detrv2']:
            from transformers import DetrConfig
            return DetrConfig(num_labels=91)  # COCO has 80 classes + background
        elif model_type == 'vitdet':
            from transformers import AutoConfig
            try:
                return AutoConfig.from_pretrained("facebook/vit-det-base")
            except:
                from transformers import DetrConfig
                return DetrConfig(num_labels=91)
        else:
            # Default to YOLO config
            return YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if self.device.type == 'cuda' else False
            )
    
    def _create_processor(self):
        """
        Create an appropriate image processor based on model type.
        
        Returns:
            Image processor instance
        """
        # Check if model has a processor attribute (from HF Hub)
        if hasattr(self.model, 'processor') and self.model.processor is not None:
            return self.model.processor
            
        # Check if model has a config with processor_class
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'processor_class'):
            try:
                from transformers import AutoImageProcessor
                return AutoImageProcessor.from_pretrained(self.model.config.processor_class)
            except Exception as e:
                print(f"Failed to load processor from config: {e}")
                
        if self.model_type == 'yolov8':
            from modeling_yolohf import YoloImageProcessor
            return YoloImageProcessor(
                do_resize=True,
                size=640,
                do_normalize=False,
                do_rescale=True,
                rescale_factor=1/255.0,
                do_pad=True,
                pad_size_divisor=32,
                pad_value=114,
                do_convert_rgb=True,
                letterbox=True,
                auto=False,
                stride=32
            )
        elif self.model_type == 'detr':
            from transformers import DetrImageProcessor
            return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        elif self.model_type == 'rt-detr':
            from transformers import AutoImageProcessor
            return AutoImageProcessor.from_pretrained("mindee/rt-detr-resnet-50")
        elif self.model_type == 'rt-detrv2':
            from transformers import AutoImageProcessor
            try:
                return AutoImageProcessor.from_pretrained("mindee/rt-detrv2-resnet-50")
            except:
                return AutoImageProcessor.from_pretrained("mindee/rt-detr-resnet-50")
        elif self.model_type == 'vitdet':
            from transformers import AutoImageProcessor
            try:
                return AutoImageProcessor.from_pretrained("facebook/vit-det-base")
            except:
                from transformers import DetrImageProcessor
                return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        else:
            # Default to DETR processor which works with most models
            from transformers import DetrImageProcessor
            return DetrImageProcessor(
                do_resize=True,
                size={"height": 640, "width": 640},
                do_normalize=True,
                do_rescale=True
            )
    
    def change_model(self, model_type=None, model_name=None, config=None, scale='s'):
        """
        Change the current model to a different type or specific model.
        
        Args:
            model_type: Type of model to use ('yolov8', 'detr', 'rt-detr', 'rt-detrv2', 'vitdet')
            model_name: Specific model name/checkpoint from Hugging Face Hub
            config: Optional config for the new model
            scale: Scale for YOLO models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If model_name is provided, load from Hugging Face Hub
            if model_name:
                self.model = self._load_from_hub(model_name)
                self.model_type = self._detect_model_type(self.model, model_name)
            # Otherwise create a new model based on model_type
            elif model_type:
                self.model_type = model_type.lower()
                self.model = self._create_model(self.model_type, config, scale)
            else:
                return False
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Update config
            if hasattr(self.model, 'config'):
                self.config = self.model.config
            elif config is not None:
                self.config = config
            else:
                self.config = self._create_default_config(self.model_type, scale)
            
            # Update processor
            self.processor = self._create_processor()
            
            return True
        except Exception as e:
            print(f"Error changing model: {e}")
            return False
            
    def load_weights(self, weights_path):
        """
        Load weights from a file.
        
        Args:
            weights_path: Path to weights file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            
            # Handle case where state_dict is inside a 'model' or 'state_dict' key
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # Try to load state dict, handling potential key mismatches
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Strict loading failed: {e}")
                print("Attempting to load with strict=False...")
                self.model.load_state_dict(state_dict, strict=False)
                
            print(f"Loaded weights from {weights_path}")
            
            # Update processor after loading weights
            self.processor = self._create_processor()
            
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
            
    def load_pretrained(self, repo_id):
        """
        Load a pretrained model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if this is a YOLO model
            if 'yolo' in repo_id.lower():
                # Register YOLO architecture with HF
                from modeling_yolohf import register_yolo_architecture
                register_yolo_architecture()
                
            from transformers import AutoModelForObjectDetection
            print(f"Loading model from Hugging Face Hub: {repo_id}")
            self.model = AutoModelForObjectDetection.from_pretrained(repo_id)
            self.model = self.model.to(self.device)
            
            # Detect model type from the loaded model
            self.model_type = self._detect_model_type(self.model, repo_id)
            
            # Update model's config with COCO class names if needed
            self._update_model_config_with_class_names()
            
            # Update config reference
            if hasattr(self.model, 'config'):
                self.config = self.model.config
                
            # Update processor
            self.processor = self._create_processor()
            
            print(f"Loaded pretrained model from {repo_id}")
            return True
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            return False
    
    def evaluate_coco(self, dataset, output_dir=None, batch_size=16, conf_thres=0.25, 
                      iou_thres=0.45, max_det=300, convert_format=None):
        """
        Evaluate the model on a dataset using COCO evaluation metrics.
        
        Args:
            dataset: Evaluation dataset
            output_dir: Directory to save evaluation results
            batch_size: Batch size for inference
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections per image
            convert_format: Function to convert dataset format to COCO format if needed
            
        Returns:
            Dictionary with COCO evaluation metrics
        """
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        # Initialize COCO format results
        coco_results = []
        image_ids = []
        
        # Progress bar
        pbar = tqdm(dataloader, desc="Running inference for COCO evaluation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch_images = batch['img'].to(self.device)
                
                # Get image metadata if available
                if 'image_id' in batch:
                    batch_image_ids = batch['image_id']
                else:
                    # Generate sequential IDs if not provided
                    batch_image_ids = list(range(
                        batch_idx * batch_size, 
                        min((batch_idx + 1) * batch_size, len(dataset))
                    ))
                
                # Store image IDs
                image_ids.extend(batch_image_ids)
                
                # Run inference
                outputs = self.model(
                    pixel_values=batch_images,
                    postprocess=True,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_det=max_det
                )
                
                # Process each image in the batch
                for i, (output, img_id) in enumerate(zip(outputs, batch_image_ids)):
                    # Get image size if available
                    if 'orig_size' in batch:
                        img_h, img_w = batch['orig_size'][i]
                    else:
                        # Use default size if not provided
                        img_h, img_w = batch_images.shape[2:]
                    
                    # Convert predictions to COCO format
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()
                    
                    # Convert each detection to COCO format and add to results
                    for box_idx in range(len(pred_boxes)):
                        x1, y1, x2, y2 = pred_boxes[box_idx]
                        
                        # COCO format uses [x, y, width, height]
                        coco_box = [
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1)
                        ]
                        
                        # Create COCO detection entry
                        detection = {
                            'image_id': int(img_id),
                            'category_id': int(pred_labels[box_idx]),
                            'bbox': coco_box,
                            'score': float(pred_scores[box_idx]),
                            'area': float((x2 - x1) * (y2 - y1)),
                            'iscrowd': 0
                        }
                        
                        coco_results.append(detection)
        
        # Save detections to file
        if output_dir:
            detections_file = os.path.join(output_dir, "coco_detections.json")
            with open(detections_file, 'w') as f:
                json.dump(coco_results, f)
            print(f"Saved detections to {detections_file}")
        
        # Get or create ground truth COCO annotations
        if hasattr(dataset, 'coco_gt'):
            # Dataset already has COCO ground truth
            coco_gt = dataset.coco_gt
        elif convert_format:
            # Use provided conversion function
            coco_gt = convert_format(dataset)
        else:
            # Try to convert dataset to COCO format
            coco_gt = self._convert_dataset_to_coco(dataset)
        
        # Run COCO evaluation
        results = self._run_coco_evaluation(coco_gt, coco_results, output_dir)
        
        return results
    
    def _convert_dataset_to_coco(self, dataset):
        """
        Convert a generic dataset to COCO format.
        
        Args:
            dataset: Dataset to convert
            
        Returns:
            COCO object with ground truth annotations
        """
        # Create COCO structure
        coco_dict = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for category_id, category_name in self.class_names.items():
            coco_dict['categories'].append({
                'id': int(category_id),
                'name': category_name,
                'supercategory': 'none'
            })
        
        # Process each image in the dataset
        annotation_id = 0
        for idx in range(len(dataset)):
            # Get sample
            sample = dataset[idx]
            
            # Get image ID
            if hasattr(sample, 'image_id'):
                image_id = sample.image_id
            else:
                image_id = idx
            
            # Get image size
            if hasattr(sample, 'orig_size'):
                img_h, img_w = sample.orig_size
            elif 'orig_size' in sample:
                img_h, img_w = sample['orig_size']
            elif hasattr(dataset, 'img_size'):
                img_h, img_w = dataset.img_size
            else:
                # Try to get from image tensor
                if 'img' in sample and isinstance(sample['img'], torch.Tensor):
                    _, img_h, img_w = sample['img'].shape
                else:
                    # Default size
                    img_h, img_w = 640, 640
            
            # Add image entry
            coco_dict['images'].append({
                'id': int(image_id),
                'width': int(img_w),
                'height': int(img_h),
                'file_name': f"{image_id}.jpg"  # Placeholder filename
            })
            
            # Get ground truth boxes and labels
            if 'target' in sample:
                target = sample['target']
                if isinstance(target, dict):
                    gt_boxes = target.get('boxes', [])
                    gt_labels = target.get('labels', [])
                    gt_areas = target.get('area', [])
                    gt_iscrowd = target.get('iscrowd', [])
                else:
                    # Try to parse target format
                    gt_boxes = []
                    gt_labels = []
                    gt_areas = []
                    gt_iscrowd = []
            else:
                continue  # Skip if no ground truth
            
            # Convert boxes to COCO format and add annotations
            for box_idx in range(len(gt_boxes)):
                if isinstance(gt_boxes[box_idx], torch.Tensor):
                    x1, y1, x2, y2 = gt_boxes[box_idx].cpu().numpy()
                else:
                    x1, y1, x2, y2 = gt_boxes[box_idx]
                
                # COCO format uses [x, y, width, height]
                coco_box = [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ]
                
                # Get label
                if isinstance(gt_labels[box_idx], torch.Tensor):
                    label = int(gt_labels[box_idx].item())
                else:
                    label = int(gt_labels[box_idx])
                
                # Get area if available, otherwise compute it
                if gt_areas and box_idx < len(gt_areas):
                    if isinstance(gt_areas[box_idx], torch.Tensor):
                        area = float(gt_areas[box_idx].item())
                    else:
                        area = float(gt_areas[box_idx])
                else:
                    area = float((x2 - x1) * (y2 - y1))
                
                # Get iscrowd if available
                if gt_iscrowd and box_idx < len(gt_iscrowd):
                    if isinstance(gt_iscrowd[box_idx], torch.Tensor):
                        iscrowd = int(gt_iscrowd[box_idx].item())
                    else:
                        iscrowd = int(gt_iscrowd[box_idx])
                else:
                    iscrowd = 0
                
                # Add annotation
                coco_dict['annotations'].append({
                    'id': annotation_id,
                    'image_id': int(image_id),
                    'category_id': label,
                    'bbox': coco_box,
                    'area': area,
                    'iscrowd': iscrowd,
                    'segmentation': []  # No segmentation for detection
                })
                
                annotation_id += 1
        
        # Create COCO object from dictionary
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(coco_dict, f)
            f.flush()
            coco_gt = COCO(f.name)
        
        return coco_gt
    
    def _run_coco_evaluation(self, coco_gt, coco_results, output_dir=None):
        """
        Run COCO evaluation on detections.
        
        Args:
            coco_gt: COCO ground truth object
            coco_results: List of COCO format detections
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create COCO detection object
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(coco_results, f)
            f.flush()
            coco_dt = coco_gt.loadRes(f.name)
        
        # Create COCO evaluator
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'AP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
            'AP50': coco_eval.stats[1],  # AP at IoU=0.50
            'AP75': coco_eval.stats[2],  # AP at IoU=0.75
            'APs': coco_eval.stats[3],  # AP for small objects
            'APm': coco_eval.stats[4],  # AP for medium objects
            'APl': coco_eval.stats[5],  # AP for large objects
            'ARmax1': coco_eval.stats[6],  # AR given 1 detection per image
            'ARmax10': coco_eval.stats[7],  # AR given 10 detections per image
            'ARmax100': coco_eval.stats[8],  # AR given 100 detections per image
            'ARs': coco_eval.stats[9],  # AR for small objects
            'ARm': coco_eval.stats[10],  # AR for medium objects
            'ARl': coco_eval.stats[11],  # AR for large objects
        }
        
        # Save evaluation results if output directory is provided
        if output_dir:
            with open(os.path.join(output_dir, "coco_eval_results.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"COCO evaluation results saved to {output_dir}/coco_eval_results.json")
        
        return metrics
    
    def convert_kitti_to_coco(self, kitti_dataset, kitti_label_dir=None):
        """
        Convert KITTI dataset to COCO format for evaluation.
        
        Args:
            kitti_dataset: KITTI dataset object
            kitti_label_dir: Directory containing KITTI labels (if not in dataset)
            
        Returns:
            COCO object with ground truth annotations
        """
        # Create COCO structure
        coco_dict = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # KITTI to COCO category mapping
        # Adjust this mapping based on your specific KITTI dataset
        kitti_to_coco = {
            'Car': 2,
            'Van': 2,
            'Truck': 7,
            'Pedestrian': 0,
            'Person_sitting': 0,
            'Cyclist': 1,
            'Tram': 6,
            'Misc': 90,  # Other
            'DontCare': 91  # Ignore
        }
        
        # Add categories
        added_categories = set()
        for kitti_cat, coco_id in kitti_to_coco.items():
            if coco_id not in added_categories and coco_id < 90:  # Skip 'Other' and 'Ignore'
                coco_dict['categories'].append({
                    'id': coco_id,
                    'name': self.class_names.get(coco_id, f"class_{coco_id}"),
                    'supercategory': 'none'
                })
                added_categories.add(coco_id)
        
        # Process each image in the dataset
        annotation_id = 0
        
        # Try to get label directory from dataset if not provided
        if kitti_label_dir is None and hasattr(kitti_dataset, 'label_dir'):
            kitti_label_dir = kitti_dataset.label_dir
        
        for idx in range(len(kitti_dataset)):
            # Get sample
            if hasattr(kitti_dataset, '__getitem__'):
                sample = kitti_dataset[idx]
            else:
                continue
            
            # Get image ID and path
            if hasattr(sample, 'image_id'):
                image_id = sample.image_id
            elif 'image_id' in sample:
                image_id = sample['image_id']
            else:
                image_id = idx
            
            # Get image path
            if hasattr(sample, 'img_path'):
                img_path = sample.img_path
            elif 'img_path' in sample:
                img_path = sample['img_path']
            elif hasattr(kitti_dataset, 'image_paths'):
                img_path = kitti_dataset.image_paths[idx]
            else:
                img_path = f"{image_id}.png"
            
            # Get image size
            if hasattr(sample, 'orig_size'):
                img_h, img_w = sample.orig_size
            elif 'orig_size' in sample:
                img_h, img_w = sample['orig_size']
            else:
                # Try to get from image
                try:
                    img = cv2.imread(img_path)
                    img_h, img_w = img.shape[:2]
                except:
                    # Default size
                    img_h, img_w = 375, 1242  # Common KITTI image size
            
            # Add image entry
            coco_dict['images'].append({
                'id': int(image_id),
                'width': int(img_w),
                'height': int(img_h),
                'file_name': os.path.basename(img_path)
            })
            
            # Get ground truth annotations
            if 'target' in sample and isinstance(sample['target'], dict):
                # Dataset already provides parsed annotations
                boxes = sample['target'].get('boxes', [])
                labels = sample['target'].get('labels', [])
                
                # Convert to COCO format
                for box_idx in range(len(boxes)):
                    if isinstance(boxes[box_idx], torch.Tensor):
                        x1, y1, x2, y2 = boxes[box_idx].cpu().numpy()
                    else:
                        x1, y1, x2, y2 = boxes[box_idx]
                    
                    # COCO format uses [x, y, width, height]
                    coco_box = [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)
                    ]
                    
                    # Get label
                    if isinstance(labels[box_idx], torch.Tensor):
                        label = int(labels[box_idx].item())
                    else:
                        label = int(labels[box_idx])
                    
                    # Add annotation
                    coco_dict['annotations'].append({
                        'id': annotation_id,
                        'image_id': int(image_id),
                        'category_id': label,
                        'bbox': coco_box,
                        'area': float(coco_box[2] * coco_box[3]),
                        'iscrowd': 0,
                        'segmentation': []
                    })
                    
                    annotation_id += 1
            
            elif kitti_label_dir:
                # Parse KITTI label file
                label_file = os.path.join(
                    kitti_label_dir, 
                    os.path.splitext(os.path.basename(img_path))[0] + '.txt'
                )
                
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 9:
                                continue
                            
                            # Parse KITTI format
                            obj_type = parts[0]
                            truncated = float(parts[1])
                            occluded = int(parts[2])
                            alpha = float(parts[3])
                            x1, y1, x2, y2 = map(float, parts[4:8])
                            
                            # Skip DontCare or objects with very high truncation/occlusion
                            if obj_type == 'DontCare' or truncated > 0.8 or occluded > 2:
                                continue
                            
                            # Map KITTI category to COCO category
                            if obj_type in kitti_to_coco:
                                category_id = kitti_to_coco[obj_type]
                            else:
                                category_id = 90  # Other
                            
                            # Skip 'Other' and 'Ignore' categories
                            if category_id >= 90:
                                continue
                            
                            # COCO format uses [x, y, width, height]
                            coco_box = [
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1)
                            ]
                            
                            # Add annotation
                            coco_dict['annotations'].append({
                                'id': annotation_id,
                                'image_id': int(image_id),
                                'category_id': category_id,
                                'bbox': coco_box,
                                'area': float(coco_box[2] * coco_box[3]),
                                'iscrowd': 0,
                                'segmentation': []
                            })
                            
                            annotation_id += 1
        
        # Create COCO object from dictionary
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(coco_dict, f)
            f.flush()
            coco_gt = COCO(f.name)
        
        return coco_gt
    
    def evaluate_kitti(self, kitti_dataset, kitti_label_dir=None, output_dir=None, 
                       batch_size=16, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        Evaluate the model on KITTI dataset using COCO metrics.
        
        Args:
            kitti_dataset: KITTI dataset object
            kitti_label_dir: Directory containing KITTI labels (if not in dataset)
            output_dir: Directory to save evaluation results
            batch_size: Batch size for inference
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections per image
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert KITTI dataset to COCO format
        coco_gt = self.convert_kitti_to_coco(kitti_dataset, kitti_label_dir)
        
        # Run COCO evaluation
        return self.evaluate_coco(
            kitti_dataset, 
            output_dir=output_dir, 
            batch_size=batch_size, 
            conf_thres=conf_thres, 
            iou_thres=iou_thres, 
            max_det=max_det,
            convert_format=lambda _: coco_gt  # Use already converted ground truth
        )
        
    def predict(self, image, conf_thres=0.25, iou_thres=0.45, max_det=300, visualize=False, output_path=None):
        """
        Run inference on an image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            visualize: Whether to create a visualization
            output_path: Path to save visualization (if visualize=True)
            
        Returns:
            Dictionary with detection results and optionally visualization
        """
        # Prepare image
        if isinstance(image, str):
            # Load image from file
            img_orig = cv2.imread(image)
            if img_orig is None:
                raise FileNotFoundError(f"Image not found at {image}")
            img_path = image
        elif isinstance(image, np.ndarray):
            # Use provided numpy array
            img_orig = image.copy()
            img_path = output_path or "detection_result.jpg"
        else:
            # Try to convert from PIL Image
            try:
                img_orig = np.array(image)
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
                img_path = output_path or "detection_result.jpg"
            except Exception as e:
                raise ValueError(f"Unsupported image type: {type(image)}. Error: {e}")
        
        # Convert BGR to RGB for processor
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Store original image size
        orig_size = pil_image.size[::-1]  # (height, width)
        
        # Preprocess image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Apply FP16 if enabled
        if self.config.use_fp16 and self.device.type == 'cuda':
            inputs = {k: v.half() for k, v in inputs.items()}
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Run inference
        with torch.no_grad():
            raw_outputs = self.model(pixel_values=inputs["pixel_values"], 
                                    postprocess=False,
                                    conf_thres=conf_thres,
                                    iou_thres=iou_thres,
                                    max_det=max_det)
            
            # Convert to DETR format
            if isinstance(raw_outputs, list):
                outputs = self.model.convert_to_detr_format(raw_outputs)
            elif isinstance(raw_outputs, torch.Tensor):
                # Extract boxes, scores, and labels
                boxes = raw_outputs[..., :4]
                scores = raw_outputs[..., 4]
                labels = raw_outputs[..., 5].long()
                
                # Create a list of dictionaries for convert_to_detr_format
                detection_results = [{"boxes": boxes[i], "scores": scores[i], "labels": labels[i]} 
                                    for i in range(raw_outputs.shape[0])]
                
                outputs = self.model.convert_to_detr_format(detection_results)
            else:
                outputs = raw_outputs
            
            # Post-process
            if 'pred_logits' in outputs and 'pred_boxes' in outputs:
                detections = self.processor.post_process_object_detection(
                    outputs, 
                    threshold=conf_thres,
                    target_sizes=[orig_size]
                )[0]
            else:
                # Create compatible format
                compatible_outputs = {
                    "logits": outputs.get("pred_logits", torch.zeros((1, 0, 80), device=self.device)),
                    "pred_boxes": outputs.get("pred_boxes", torch.zeros((1, 0, 4), device=self.device))
                }
                detections = self.processor.post_process_object_detection(
                    compatible_outputs, 
                    threshold=conf_thres,
                    target_sizes=[orig_size]
                )[0]
        
        # Create visualization if requested
        if visualize and len(detections["scores"]) > 0:
            vis_img = self._visualize_detections(img_orig, detections, output_path)
            detections["visualization"] = vis_img
            if output_path:
                detections["visualization_path"] = output_path
                
        return detections
    
    def _visualize_detections(self, image, detections, output_path=None):
        """
        Visualize detections on an image.
        
        Args:
            image: Original image (numpy array, BGR format)
            detections: Detection results from processor
            output_path: Path to save visualization
            
        Returns:
            Numpy array with visualized detections
        """
        # Create a copy of the image
        img_vis = image.copy()
        
        # Extract detection components
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        
        # Draw boxes on the image
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            label = int(labels[i])
            
            # Generate a color based on the class label
            color_factor = (label * 50) % 255
            color = (color_factor, 255 - color_factor, 128)
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Add class name and confidence
            class_name = self.class_names.get(int(label), f"class_{label}")
            label_text = f"{class_name}: {score:.2f}"
            
            # Add a filled rectangle behind text for better visibility
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_vis, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # Add text with white color for better contrast
            cv2.putText(img_vis, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save the visualization if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            cv2.imwrite(output_path, img_vis)
            print(f"Visualization saved to {output_path}")
        
        return img_vis
    
    def train(self, train_dataset, val_dataset=None, epochs=100, batch_size=16, 
              learning_rate=0.01, weight_decay=0.0005, output_dir="./runs/train", 
              resume=None, save_interval=10):
        """
        Train the model on a dataset.
        
        Args:
            train_dataset: Training dataset (must be compatible with YOLO format)
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            output_dir: Directory to save checkpoints and logs
            resume: Path to checkpoint to resume training from
            save_interval: Save checkpoint every N epochs
            
        Returns:
            Dictionary with training results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to training mode
        self.model.train()
        
        # Initialize criterion if not already done
        if not hasattr(self.model, 'criterion'):
            self.model.criterion = self.model.init_criterion()
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(8, os.cpu_count() or 1),
                pin_memory=True,
                collate_fn=self._collate_fn
            )
        else:
            val_loader = None
        
        # Initialize optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.937,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=learning_rate / 100
        )
        
        # Resume training if checkpoint provided
        start_epoch = 0
        if resume:
            if os.path.isfile(resume):
                checkpoint = torch.load(resume, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed training from epoch {start_epoch}")
            else:
                print(f"No checkpoint found at {resume}, starting from scratch")
        
        # Initialize training metrics
        best_val_loss = float('inf')
        training_results = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(start_epoch, epochs):
            # Train for one epoch
            train_loss = self._train_one_epoch(train_loader, optimizer, epoch)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validate if validation dataset is provided
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader, epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(output_dir, "best.pt"))
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
                print(f"Saved checkpoint at epoch {epoch+1}")
            
            # Save last model
            torch.save(self.model.state_dict(), os.path.join(output_dir, "last.pt"))
            
            # Update training results
            training_results['train_losses'].append(train_loss)
            training_results['val_losses'].append(val_loss)
            training_results['learning_rates'].append(current_lr)
            training_results['epochs'].append(epoch + 1)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f if val_loss else 'N/A'}, lr={current_lr:.6f}")
            
            # Save training results
            with open(os.path.join(output_dir, "training_results.json"), 'w') as f:
                json.dump(training_results, f, indent=2)
        
        print(f"Training completed. Final model saved to {output_dir}/last.pt")
        if val_loader:
            print(f"Best model saved to {output_dir}/best.pt with validation loss: {best_val_loss:.4f}")
            
        return training_results
    
    def _train_one_epoch(self, dataloader, optimizer, epoch):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss, loss_items = self.model.loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
        
        return avg_loss
    
    def _validate(self, dataloader, epoch):
        """
        Validate the model on a dataset.
        
        Args:
            dataloader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Validating epoch {epoch+1}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss, loss_items = self.model.loss(batch)
                
                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'val_loss': f"{avg_loss:.4f}"})
        
        return avg_loss
    
    def _collate_fn(self, batch):
        """
        Custom collate function for YOLO datasets.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Collated batch
        """
        # This is a simplified version - adjust based on your dataset format
        images = []
        targets = []
        
        for sample in batch:
            images.append(sample['img'])
            targets.append(sample['target'])
            
        # Stack images
        images = torch.stack(images)
        
        return {
            'img': images,
            'target': targets
        }
    
    def evaluate(self, dataset, batch_size=16, conf_thres=0.25, iou_thres=0.45, 
                max_det=300, output_dir=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Evaluation dataset
            batch_size: Batch size
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        # Initialize metrics
        stats = []
        
        # Progress bar
        pbar = tqdm(dataloader, desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch_images = batch['img'].to(self.device)
                
                # Get ground truth
                targets = batch['target']
                
                # Run inference
                outputs = self.model(
                    pixel_values=batch_images,
                    postprocess=True,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_det=max_det
                )
                
                # Process each image in the batch
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    # Convert predictions to expected format
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()
                    
                    # Convert ground truth to expected format
                    gt_boxes = target['boxes'].cpu().numpy()
                    gt_labels = target['labels'].cpu().numpy()
                    
                    # Compute metrics for this image
                    image_metrics = self._compute_metrics(
                        pred_boxes, pred_scores, pred_labels,
                        gt_boxes, gt_labels,
                        iou_thres=iou_thres
                    )
                    
                    # Add image index and batch index
                    image_metrics['batch_idx'] = batch_idx
                    image_metrics['image_idx'] = batch_idx * batch_size + i
                    
                    # Add to stats
                    stats.append(image_metrics)
        
        # Compute overall metrics
        eval_results = self._summarize_metrics(stats)
        
        # Save evaluation results if output directory is provided
        if output_dir:
            with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Evaluation results saved to {output_dir}/eval_results.json")
        
        return eval_results
    
    def _compute_metrics(self, pred_boxes, pred_scores, pred_labels, 
                        gt_boxes, gt_labels, iou_thres=0.5):
        """
        Compute metrics for a single image.
        
        Args:
            pred_boxes: Predicted boxes (x1, y1, x2, y2)
            pred_scores: Predicted confidence scores
            pred_labels: Predicted class labels
            gt_boxes: Ground truth boxes (x1, y1, x2, y2)
            gt_labels: Ground truth class labels
            iou_thres: IoU threshold for matching predictions to ground truth
            
        Returns:
            Dictionary with metrics for this image
        """
        # Initialize metrics
        metrics = {
            'tp': 0,  # True positives
            'fp': 0,  # False positives
            'fn': 0,  # False negatives
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'ap': 0,  # Average precision
        }
        
        # If no ground truth, all predictions are false positives
        if len(gt_boxes) == 0:
            metrics['fp'] = len(pred_boxes)
            return metrics
        
        # If no predictions, all ground truth are false negatives
        if len(pred_boxes) == 0:
            metrics['fn'] = len(gt_boxes)
            return metrics
        
        # Compute IoU between predictions and ground truth
        ious = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                ious[i, j] = self._box_iou(pred_box, gt_box)
        
        # Match predictions to ground truth
        matched_gt = set()
        
        # Sort predictions by confidence score (high to low)
        sorted_indices = np.argsort(-pred_scores)
        
        for i in sorted_indices:
            pred_label = pred_labels[i]
            
            # Find best matching ground truth for this prediction
            best_iou = iou_thres
            best_gt = -1
            
            for j in range(len(gt_boxes)):
                # Skip if ground truth already matched or labels don't match
                if j in matched_gt or gt_labels[j] != pred_label:
                    continue
                
                # Check if IoU is better than current best
                if ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt = j
            
            # If a match is found, it's a true positive
            if best_gt >= 0:
                metrics['tp'] += 1
                matched_gt.add(best_gt)
            else:
                # No match found, it's a false positive
                metrics['fp'] += 1
        
        # Unmatched ground truth are false negatives
        metrics['fn'] = len(gt_boxes) - len(matched_gt)
        
        # Compute precision, recall, F1
        if metrics['tp'] + metrics['fp'] > 0:
            metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
        
        if metrics['tp'] + metrics['fn'] > 0:
            metrics['recall'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        # Compute AP (simplified version)
        # For a full implementation, you would need to compute the precision-recall curve
        metrics['ap'] = metrics['precision'] * metrics['recall']
        
        return metrics
    
    def _box_iou(self, box1, box2):
        """
        Compute IoU between two boxes.
        
        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)
            
        Returns:
            IoU value
        """
        # Get coordinates of intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute area of intersection
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        
        # Compute area of union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # Compute IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _summarize_metrics(self, stats):
        """
        Summarize metrics across all images.
        
        Args:
            stats: List of metrics dictionaries, one per image
            
        Returns:
            Dictionary with summarized metrics
        """
        # Initialize summary
        summary = {
            'images': len(stats),
            'total_tp': sum(s['tp'] for s in stats),
            'total_fp': sum(s['fp'] for s in stats),
            'total_fn': sum(s['fn'] for s in stats),
            'mean_precision': 0,
            'mean_recall': 0,
            'mean_f1': 0,
            'mean_ap': 0,
        }
        
        # Compute mean metrics
        if summary['images'] > 0:
            summary['mean_precision'] = sum(s['precision'] for s in stats) / summary['images']
            summary['mean_recall'] = sum(s['recall'] for s in stats) / summary['images']
            summary['mean_f1'] = sum(s['f1'] for s in stats) / summary['images']
            summary['mean_ap'] = sum(s['ap'] for s in stats) / summary['images']
        
        # Compute overall precision, recall, F1
        if summary['total_tp'] + summary['total_fp'] > 0:
            summary['overall_precision'] = summary['total_tp'] / (summary['total_tp'] + summary['total_fp'])
        else:
            summary['overall_precision'] = 0
        
        if summary['total_tp'] + summary['total_fn'] > 0:
            summary['overall_recall'] = summary['total_tp'] / (summary['total_tp'] + summary['total_fn'])
        else:
            summary['overall_recall'] = 0
        
        if summary['overall_precision'] + summary['overall_recall'] > 0:
            summary['overall_f1'] = 2 * (summary['overall_precision'] * summary['overall_recall']) / (summary['overall_precision'] + summary['overall_recall'])
        else:
            summary['overall_f1'] = 0
        
        return summary

# COCO class names dictionary
coco_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class DetectionDataset(Dataset):
    """
    Universal detection dataset class that supports COCO, KITTI, and other formats.
    Handles class mapping to standardize on COCO's 80 classes.
    """
    def __init__(self, 
                 dataset_type='coco',
                 data_dir=None, 
                 annotation_file=None,
                 image_dir=None,
                 split='train',
                 transforms=None,
                 target_size=(640, 640),
                 cache_images=False,
                 class_map=None):
        """
        Initialize the detection dataset.
        
        Args:
            dataset_type: Type of dataset ('coco', 'kitti', 'voc', etc.)
            data_dir: Root directory of the dataset
            annotation_file: Path to annotation file (for COCO)
            image_dir: Directory containing images
            split: Dataset split ('train', 'val', 'test')
            transforms: Image transformations
            target_size: Target image size (height, width)
            cache_images: Whether to cache images in memory
            class_map: Custom class mapping dictionary
        """
        self.dataset_type = dataset_type.lower()
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.split = split
        self.transforms = transforms
        self.target_size = target_size
        self.cache_images = cache_images
        
        # Initialize class mapping
        self.class_map = class_map or self._get_default_class_map(dataset_type)
        
        # Initialize COCO class names (standard 80 classes)
        self.coco_names = coco_names
        
        # Load dataset based on type
        self._load_dataset()
        
        # Cache for images
        self.img_cache = {} if cache_images else None
        
    def _get_default_class_map(self, dataset_type):
        """
        Get default class mapping for the specified dataset type.
        Maps dataset-specific classes to COCO classes.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Dictionary mapping dataset classes to COCO classes
        """
        if dataset_type == 'coco':
            # COCO is already using the standard classes
            return {i: i for i in range(80)}
        
        elif dataset_type == 'kitti':
            # KITTI to COCO class mapping
            return {
                'Car': 2,         # car in COCO
                'Van': 2,         # car in COCO
                'Truck': 7,       # truck in COCO
                'Pedestrian': 0,  # person in COCO
                'Person_sitting': 0,  # person in COCO
                'Cyclist': 1,     # bicycle in COCO
                'Tram': 6,        # train in COCO
                'Misc': -1,       # ignore
                'DontCare': -1    # ignore
            }
        
        elif dataset_type == 'voc':
            # Pascal VOC to COCO class mapping
            return {
                'aeroplane': 4,    # airplane in COCO
                'bicycle': 1,      # bicycle in COCO
                'bird': 14,        # bird in COCO
                'boat': 8,         # boat in COCO
                'bottle': 39,      # bottle in COCO
                'bus': 5,          # bus in COCO
                'car': 2,          # car in COCO
                'cat': 15,         # cat in COCO
                'chair': 56,       # chair in COCO
                'cow': 19,         # cow in COCO
                'diningtable': 60, # dining table in COCO
                'dog': 16,         # dog in COCO
                'horse': 17,       # horse in COCO
                'motorbike': 3,    # motorcycle in COCO
                'person': 0,       # person in COCO
                'pottedplant': 58, # potted plant in COCO
                'sheep': 18,       # sheep in COCO
                'sofa': 57,        # couch in COCO
                'train': 6,        # train in COCO
                'tvmonitor': 62    # tv in COCO
            }
        
        else:
            # Default empty mapping
            return {}
    
    def _load_dataset(self):
        """
        Load dataset based on type.
        """
        if self.dataset_type == 'coco':
            self._load_coco_dataset()
        elif self.dataset_type == 'kitti':
            self._load_kitti_dataset()
        elif self.dataset_type == 'voc':
            self._load_voc_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _load_coco_dataset(self):
        """
        Load COCO dataset.
        """
        # Determine annotation file if not provided
        if self.annotation_file is None and self.data_dir is not None:
            if self.split == 'train':
                self.annotation_file = os.path.join(self.data_dir, 'annotations', 'instances_train2017.json')
            elif self.split == 'val':
                self.annotation_file = os.path.join(self.data_dir, 'annotations', 'instances_val2017.json')
            else:
                raise ValueError(f"Unsupported COCO split: {self.split}")
        
        # Determine image directory if not provided
        if self.image_dir is None and self.data_dir is not None:
            if self.split == 'train':
                self.image_dir = os.path.join(self.data_dir, 'train2017')
            elif self.split == 'val':
                self.image_dir = os.path.join(self.data_dir, 'val2017')
            else:
                raise ValueError(f"Unsupported COCO split: {self.split}")
        
        # Load COCO annotations
        self.coco = COCO(self.annotation_file)
        
        # Get image IDs
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        
        # Create category ID to COCO class ID mapping
        self.cat_id_to_coco_id = {}
        for cat_id in self.coco.cats.keys():
            # Map COCO category ID to standard COCO class ID (0-79)
            # This is needed because COCO category IDs are not sequential
            cat_name = self.coco.cats[cat_id]['name']
            for coco_id, name in self.coco_names.items():
                if name.lower() == cat_name.lower():
                    self.cat_id_to_coco_id[cat_id] = coco_id
                    break
            else:
                # If not found, map to -1 (ignore)
                self.cat_id_to_coco_id[cat_id] = -1
        
        print(f"Loaded COCO dataset with {len(self.image_ids)} images")
    
    def _load_kitti_dataset(self):
        """
        Load KITTI dataset.
        """
        # Determine data directories if not provided
        if self.data_dir is not None:
            if self.image_dir is None:
                self.image_dir = os.path.join(self.data_dir, 'training', 'image_2')
            
            self.label_dir = os.path.join(self.data_dir, 'training', 'label_2')
            
            # For validation split, we'll use a subset of the training data
            # since KITTI doesn't have an official train/val split
            if self.split == 'val':
                val_ratio = 0.2
                all_images = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
                num_val = int(len(all_images) * val_ratio)
                self.image_paths = all_images[-num_val:]
            else:
                self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        else:
            raise ValueError("Data directory must be provided for KITTI dataset")
        
        # Create KITTI class name to class ID mapping
        self.kitti_name_to_id = {
            'Car': 0,
            'Van': 1,
            'Truck': 2,
            'Pedestrian': 3,
            'Person_sitting': 4,
            'Cyclist': 5,
            'Tram': 6,
            'Misc': 7,
            'DontCare': 8
        }
        
        print(f"Loaded KITTI dataset with {len(self.image_paths)} images")
    
    def _load_voc_dataset(self):
        """
        Load Pascal VOC dataset.
        """
        # Determine data directories if not provided
        if self.data_dir is not None:
            if self.image_dir is None:
                self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
            
            # Set annotation directory
            self.annotation_dir = os.path.join(self.data_dir, 'Annotations')
            
            # Load image IDs from the appropriate split
            split_file = os.path.join(
                self.data_dir, 'ImageSets', 'Main', 
                f"{self.split}.txt"
            )
            
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f.readlines()]
        else:
            raise ValueError("Data directory must be provided for VOC dataset")
        
        # Create VOC class name to class ID mapping
        self.voc_name_to_id = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }
        
        print(f"Loaded VOC dataset with {len(self.image_ids)} images")
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        if self.dataset_type == 'coco':
            return len(self.image_ids)
        elif self.dataset_type == 'kitti':
            return len(self.image_paths)
        elif self.dataset_type == 'voc':
            return len(self.image_ids)
        else:
            return 0
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and target
        """
        if self.dataset_type == 'coco':
            return self._get_coco_item(idx)
        elif self.dataset_type == 'kitti':
            return self._get_kitti_item(idx)
        elif self.dataset_type == 'voc':
            return self._get_voc_item(idx)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _get_coco_item(self, idx):
        """
        Get a sample from COCO dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and target
        """
        # Get image ID
        img_id = self.image_ids[idx]
        
        # Check if image is cached
        if self.cache_images and img_id in self.img_cache:
            img = self.img_cache[img_id]
        else:
            # Load image
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Cache image if enabled
            if self.cache_images:
                self.img_cache[img_id] = img
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Skip annotations with no area or marked as crowd
            if ann['area'] < 1 or ann['iscrowd']:
                continue
            
            # Get category ID and map to COCO class ID
            cat_id = ann['category_id']
            coco_class_id = self.cat_id_to_coco_id.get(cat_id, -1)
            
            # Skip categories that don't map to a COCO class
            if coco_class_id == -1:
                continue
            
            # Get bounding box
            x, y, w, h = ann['bbox']
            
            # Convert to (x1, y1, x2, y2) format
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(coco_class_id)
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([img.shape[0], img.shape[1]], dtype=torch.int64)
        }
        
        # Apply transformations if provided
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            # Resize image to target size
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(self.target_size[::-1])  # PIL uses (width, height)
            img = np.array(img_pil)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Scale boxes to new image size
            if len(boxes) > 0:
                h_ratio = self.target_size[0] / img_info['height']
                w_ratio = self.target_size[1] / img_info['width']
                
                # Scale boxes
                scaled_boxes = boxes.clone()
                scaled_boxes[:, 0] *= w_ratio  # x1
                scaled_boxes[:, 1] *= h_ratio  # y1
                scaled_boxes[:, 2] *= w_ratio  # x2
                scaled_boxes[:, 3] *= h_ratio  # y2
                
                target['boxes'] = scaled_boxes
        
        return {'img': img, 'target': target, 'image_id': img_id}
    
    def _get_kitti_item(self, idx):
        """
        Get a sample from KITTI dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and target
        """
        # Get image path
        img_path = self.image_paths[idx]
        img_id = os.path.basename(img_path).split('.')[0]
        
        # Check if image is cached
        if self.cache_images and img_id in self.img_cache:
            img = self.img_cache[img_id]
        else:
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Cache image if enabled
            if self.cache_images:
                self.img_cache[img_id] = img
        
        # Get label file path
        label_path = os.path.join(self.label_dir, f"{img_id}.txt")
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    
                    # Parse KITTI format
                    obj_type = parts[0]
                    truncated = float(parts[1])
                    occluded = int(parts[2])
                    alpha = float(parts[3])
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    
                    # Skip DontCare or objects with very high truncation/occlusion
                    if obj_type == 'DontCare' or truncated > 0.8 or occluded > 2:
                        continue
                    
                    # Map KITTI class to COCO class
                    if obj_type in self.class_map:
                        coco_class_id = self.class_map[obj_type]
                    else:
                        # Skip classes that don't map to a COCO class
                        continue
                    
                    # Skip ignored classes
                    if coco_class_id == -1:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(coco_class_id)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([int(img_id) if img_id.isdigit() else hash(img_id) % 10000]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'orig_size': torch.as_tensor([img.shape[0], img.shape[1]], dtype=torch.int64)
        }
        
        # Apply transformations if provided
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            # Resize image to target size
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(self.target_size[::-1])  # PIL uses (width, height)
            img = np.array(img_pil)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Scale boxes to new image size
            if len(boxes) > 0:
                h_ratio = self.target_size[0] / target['orig_size'][0].item()
                w_ratio = self.target_size[1] / target['orig_size'][1].item()
                
                # Scale boxes
                scaled_boxes = boxes.clone()
                scaled_boxes[:, 0] *= w_ratio  # x1
                scaled_boxes[:, 1] *= h_ratio  # y1
                scaled_boxes[:, 2] *= w_ratio  # x2
                scaled_boxes[:, 3] *= h_ratio  # y2
                
                target['boxes'] = scaled_boxes
        
        return {'img': img, 'target': target, 'image_id': img_id}
    
    def _get_voc_item(self, idx):
        """
        Get a sample from VOC dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and target
        """
        # Get image ID
        img_id = self.image_ids[idx]
        
        # Check if image is cached
        if self.cache_images and img_id in self.img_cache:
            img = self.img_cache[img_id]
        else:
            # Load image
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Cache image if enabled
            if self.cache_images:
                self.img_cache[img_id] = img
        
        # Get annotation file path
        ann_path = os.path.join(self.annotation_dir, f"{img_id}.xml")
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        # Parse XML annotation file
        import xml.etree.ElementTree as ET
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Process each object
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text.lower()
            
            # Map VOC class to COCO class
            if class_name in self.class_map:
                coco_class_id = self.class_map[class_name]
            else:
                # Skip classes that don't map to a COCO class
                continue
            
            # Skip ignored classes
            if coco_class_id == -1:
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(coco_class_id)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([int(img_id) if img_id.isdigit() else hash(img_id) % 10000]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'orig_size': torch.as_tensor([height, width], dtype=torch.int64)
        }
        
        # Apply transformations if provided
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            # Resize image to target size
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(self.target_size[::-1])  # PIL uses (width, height)
            img = np.array(img_pil)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Scale boxes to new image size
            if len(boxes) > 0:
                h_ratio = self.target_size[0] / height
                w_ratio = self.target_size[1] / width
                
                # Scale boxes
                scaled_boxes = boxes.clone()
                scaled_boxes[:, 0] *= w_ratio  # x1
                scaled_boxes[:, 1] *= h_ratio  # y1
                scaled_boxes[:, 2] *= w_ratio  # x2
                scaled_boxes[:, 3] *= h_ratio  # y2
                
                target['boxes'] = scaled_boxes
        
        return {'img': img, 'target': target, 'image_id': img_id}
    
    def get_coco_gt(self):
        """
        Get COCO ground truth object for evaluation.
        
        Returns:
            COCO object with ground truth annotations
        """
        if self.dataset_type == 'coco':
            # For COCO dataset, return the loaded COCO object
            return self.coco
        else:
            # For other datasets, convert to COCO format
            return self.convert_to_coco_format()
    
    def convert_to_coco_format(self):
        """
        Convert dataset to COCO format for evaluation.
        
        Returns:
            COCO object with ground truth annotations
        """
        # Create COCO structure
        coco_dict = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories (COCO classes)
        for coco_id, name in self.coco_names.items():
            coco_dict['categories'].append({
                'id': int(coco_id),
                'name': name,
                'supercategory': 'none'
            })
        
        # Process each image in the dataset
        annotation_id = 0
        
        for idx in range(len(self)):
            # Get sample
            sample = self[idx]
            target = sample['target']
            
            # Get image ID
            if isinstance(target['image_id'], torch.Tensor):
                image_id = target['image_id'].item()
            else:
                image_id = target['image_id']
            
            # Get image size
            if 'orig_size' in target:
                if isinstance(target['orig_size'], torch.Tensor):
                    img_h, img_w = target['orig_size'].tolist()
                else:
                    img_h, img_w = target['orig_size']
            else:
                # Default size
                img_h, img_w = self.target_size
            
            # Add image entry
            coco_dict['images'].append({
                'id': int(image_id),
                'width': int(img_w),
                'height': int(img_h),
                'file_name': f"{image_id}.jpg"  # Placeholder filename
            })
            
            # Get boxes and labels
            boxes = target['boxes']
            labels = target['labels']
            
            # Convert boxes to COCO format and add annotations
            for box_idx in range(len(boxes)):
                if isinstance(boxes, torch.Tensor):
                    x1, y1, x2, y2 = boxes[box_idx].tolist()
                else:
                    x1, y1, x2, y2 = boxes[box_idx]
                
                # COCO format uses [x, y, width, height]
                coco_box = [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ]
                
                # Get label
                if isinstance(labels, torch.Tensor):
                    label = int(labels[box_idx].item())
                else:
                    label = int(labels[box_idx])
                
                # Get area if available, otherwise compute it
                if 'area' in target and box_idx < len(target['area']):
                    if isinstance(target['area'], torch.Tensor):
                        area = float(target['area'][box_idx].item())
                    else:
                        area = float(target['area'][box_idx])
                else:
                    area = float((x2 - x1) * (y2 - y1))
                
                # Get iscrowd if available
                if 'iscrowd' in target and box_idx < len(target['iscrowd']):
                    if isinstance(target['iscrowd'], torch.Tensor):
                        iscrowd = int(target['iscrowd'][box_idx].item())
                    else:
                        iscrowd = int(target['iscrowd'][box_idx])
                else:
                    iscrowd = 0
                
                # Add annotation
                coco_dict['annotations'].append({
                    'id': annotation_id,
                    'image_id': int(image_id),
                    'category_id': label,
                    'bbox': coco_box,
                    'area': area,
                    'iscrowd': iscrowd,
                    'segmentation': []  # No segmentation for detection
                })
                
                annotation_id += 1
        
        # Create COCO object from dictionary
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(coco_dict, f)
            f.flush()
            coco = COCO(f.name)
        
        return coco
                
def test_multiworld():
    """
    Test function for MultiWorld class.
    """
    # Initialize MultiWorld
    yolo = MultiWorld(scale='s')
    
    # Load weights
    yolo.load_weights("../modelzoo/yolov8s_statedicts.pt")
    
    # Test prediction on an image
    image_path = "ModelDev/sampledata/bus.jpg"
    results = yolo.predict(
        image_path,
        conf_thres=0.25,
        visualize=True,
        output_path="output/yoloworld_detection.jpg"
    )
    
    # Print results
    print(f"Detected {len(results['scores'])} objects:")
    for i, (score, label) in enumerate(zip(results['scores'], results['labels'])):
        class_name = yolo.class_names.get(label.item(), f"class_{label.item()}")
        print(f"  {i+1}. {class_name}: {score.item():.4f}")
    
    # Display the image
    cv2.imshow("MultiWorld Detection", results["visualization"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_rtdetr():
    import torch
    import requests

    from PIL import Image
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            print(f"{model.config.id2label[label]}: {score:.2f} {box}")
            
        
def myevaluate_kitti():
    # Initialize YOLOWorld
    yolo_model = YOLOWorld(model="PekingU/rtdetr_v2_r18vd")
    # Evaluate on KITTI dataset
    results = yolo_model.evaluate_kitti(
        kitti_dataset,
        kitti_label_dir="/path/to/kitti/labels",
        output_dir="./evaluation_results",
        conf_thres=0.25
    )

    # Print results
    print(f"AP: {results['AP']:.4f}")
    print(f"AP50: {results['AP50']:.4f}")
    print(f"AP75: {results['AP75']:.4f}")
    
if __name__ == "__main__":
    #myevaluate_kitti()
    test_rtdetr()