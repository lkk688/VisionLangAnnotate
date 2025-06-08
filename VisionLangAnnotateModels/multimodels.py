"""Module providing multiple model evaluation and training."""
import os
import numpy as np
import torch
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from PIL import Image

from modeling_yolohf import register_yolo_architecture, YoloConfig          
from transformers import AutoModelForObjectDetection
from multidatasets import coco_names, original_coco_id_mapping, kitti_to_coco, DetectionDataset
from modeling_yolohf import (
    YoloDetectionModel,
    register_yolo_architecture,
    YoloConfig
)

class MultiModels:
    """
    MultiModels class that implements object detection model training, evaluation, and inference
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
        Initialize MultiModels with a model or create a new one.
        
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
            
        if self.model is not None:
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

        self.dataset = None
    
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
                #from modeling_yolohfd import register_yolo_architecture
                register_yolo_architecture()
                
                # Load model from Hugging Face Hub
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
                #from modeling_yolohf import register_yolo_architecture
                register_yolo_architecture()
                
            #from transformers import AutoModelForObjectDetection
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
        Evaluate the model on a COCO dataset.
        
        Args:
            dataset: COCO evaluation dataset
            output_dir: Directory to save evaluation results
            batch_size: Batch size for inference
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections per image
            convert_format: Function to convert dataset to COCO format
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.dataset = dataset
        # Run inference on dataset and get detections
        coco_results = self._run_inference_on_dataset(
            dataset=dataset,
            batch_size=batch_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            output_dir=output_dir
        )#list of dicts, each dict is one object with 'image_id'
        
        # Save detections to file
        if output_dir:
            detections_file = os.path.join(output_dir, "coco_detections.json")
            with open(detections_file, 'w') as f:
                json.dump(coco_results, f)
            print(f"Saved detections to {detections_file}")
        
        # Get or create ground truth COCO annotations
        if hasattr(dataset, '_coco') and dataset._coco is not None:
            # Dataset already has COCO ground truth
            print("Using dataset's existing COCO ground truth")
            coco_gt = dataset._coco #pycocotools.coco.COCO object
        elif hasattr(dataset, 'coco_gt') and dataset.coco_gt is not None:
            # Dataset already has COCO ground truth
            print("Using dataset's existing COCO ground truth")
            coco_gt = dataset.coco_gt
        elif convert_format:
            # Use provided conversion function
            print("Using provided conversion function")
            coco_gt = convert_format(dataset)
        else:
            # Try to convert dataset to COCO format
            print("Converting dataset to COCO format")
            coco_gt = self._convert_dataset_to_coco(dataset)
        
        # Debug: Print information about ground truth and results
        gt_img_ids = set(coco_gt.getImgIds()) #5000 images
        result_img_ids = set(r['image_id'] for r in coco_results) #4945 images
        
        print(f"Ground truth has {len(gt_img_ids)} images")
        print(f"Results contain {len(result_img_ids)} unique image IDs")
        print(f"Overlap: {len(gt_img_ids.intersection(result_img_ids))} images")
        
        # Check if there are any categories in the ground truth
        gt_cat_ids = set(coco_gt.getCatIds()) #1-90, total 80
        result_cat_ids = set(r['category_id'] for r in coco_results) #0-79, total 80
        
        print(f"Ground truth has {len(gt_cat_ids)} categories: {list(gt_cat_ids)[:10]}")
        print(f"Results contain {len(result_cat_ids)} unique category IDs: {list(result_cat_ids)[:10]}")
        
        # Save ground truth to file for debugging
        if output_dir:
            gt_file = os.path.join(output_dir, "coco_ground_truth.json")
            with open(gt_file, 'w') as f:
                json.dump({
                    'images': coco_gt.dataset.get('images', []),
                    'annotations': coco_gt.dataset.get('annotations', []),
                    'categories': coco_gt.dataset.get('categories', [])
                }, f)
            print(f"Saved ground truth to {gt_file}")
        
        # Run COCO evaluation
        metrics = self._run_custom_evaluation(coco_gt, coco_results, output_dir)
        results = self._run_coco_evaluation(coco_gt, coco_results, output_dir)
        
        return results
    
    def _run_inference_on_dataset(self, dataset, batch_size=16, conf_thres=0.25, 
                                 iou_thres=0.45, max_det=300, output_dir=None,
                                 output_format='coco', visualize=True):
        """
        Run inference on a dataset and return detections in specified format.
        
        Args:
            dataset: Dataset to run inference on
            batch_size: Batch size for inference
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections per image
            output_dir: Directory to save visualization results
            output_format: Format of output detections ('coco' or 'kitti')
            
        Returns:
            List of detections in specified format
        """
        # Set model to evaluation mode
        self.model.eval()
        
        #num_workers = min(4, os.cpu_count() or 1)
        dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # No multiprocessing
                pin_memory=False,
                collate_fn=getattr(dataset, 'collate_fn', self._collate_fn)
            )
        
        # Initialize results
        results = []
        
        # Store image ID mapping for consistent IDs between GT and detections
        self.image_id_map = {}
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Running inference for {output_format.upper()} evaluation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Handle different batch formats
                if isinstance(batch, dict):
                    # Dictionary format
                    batch_images = batch.get('img', batch.get('pixel_values', batch.get('images'))) #[4, 3, 640, 640]
                    
                    # Store original images for visualization if not already present
                    if visualize and 'img_orig' not in batch and isinstance(batch_images, torch.Tensor):
                        # Convert tensor back to numpy for visualization
                        if batch_images.dim() == 4:  # batch of images
                            batch['img_orig'] = [img.permute(1, 2, 0).cpu().numpy() for img in batch_images]
                        elif batch_images.dim() == 3:  # single image
                            batch['img_orig'] = [batch_images.permute(1, 2, 0).cpu().numpy()]
                            
                    if isinstance(batch_images, list):
                         # Store original images before processing
                        if visualize and 'img_orig' not in batch:
                            batch['img_orig'] = batch_images.copy()
                        # Process images with processor if they're not tensors
                        inputs = self.processor(images=batch_images, return_tensors="pt")
                        batch_images = inputs["pixel_values"]
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Tuple/list format (images, targets)
                    batch_images = batch[0]
                    if isinstance(batch[1], dict):
                        batch = batch[1]
                        batch['img'] = batch_images
                        
                        # Store original images for visualization
                        if visualize and 'img_orig' not in batch and isinstance(batch_images, torch.Tensor):
                            if batch_images.dim() == 4:  # batch of images
                                batch['img_orig'] = [img.permute(1, 2, 0).cpu().numpy() for img in batch_images]
                            elif batch_images.dim() == 3:  # single image
                                batch['img_orig'] = [batch_images.permute(1, 2, 0).cpu().numpy()]
                else:
                    print(f"Warning: Unsupported batch format: {type(batch)}")
                    continue
                
                # Get image metadata if available
                batch_image_ids = [target['image_id'][0].numpy() for target in batch['target']]
                
                # Move batch to device
                if isinstance(batch_images, torch.Tensor):
                    batch_images = batch_images.to(self.device) #[4, 3, 640, 640]
                
                # Run inference based on model type
                try:
                    if self.model_type == 'yolov8':
                        # YOLO models expect pixel_values and have postprocess parameter
                        outputs = self.model(
                            pixel_values=batch_images, #[4, 3, 640, 640]
                            postprocess=True,
                            conf_thres=conf_thres,
                            iou_thres=iou_thres,
                            max_det=max_det
                        )
                    else:
                        # Try standard HF format first
                        outputs = self.model(
                            pixel_values=batch_images
                        )
                        
                        # Post-process if needed
                        if hasattr(self.processor, 'post_process_object_detection'):
                            # Get original sizes for proper scaling
                            if isinstance(batch, dict) and 'orig_size' in batch:
                                target_sizes = batch['orig_size']
                            else:
                                # Use image size if original size not provided
                                target_sizes = [(batch_images.shape[2], batch_images.shape[3])] * len(batch_images)
                            
                            outputs = self.processor.post_process_object_detection(
                                outputs,
                                threshold=conf_thres,
                                target_sizes=target_sizes
                            )
                except Exception as e:
                    print(f"Error during inference: {e}")
                    continue
                
                # Process each image in the batch
                for i, (output, img_id) in enumerate(zip(outputs, batch_image_ids)):
                    # # Get image size if available
                    # if isinstance(batch, dict) and 'orig_size' in batch:
                    #     img_h, img_w = batch['orig_size'][i] if isinstance(batch['orig_size'], list) else batch['orig_size']
                    # else:
                    #     # Use default size if not provided
                    #     img_h, img_w = batch_images.shape[2:] if len(batch_images.shape) == 4 else (batch_images.shape[1], batch_images.shape[2])
                    img_h, img_w=batch['target'][i]['orig_size']
                    # Handle different output formats
                    if isinstance(output, dict):
                        # Standard format with boxes, scores, labels
                        pred_boxes = output.get('boxes', [])
                        pred_scores = output.get('scores', [])
                        pred_labels = output.get('labels', [])
                    elif isinstance(output, (list, tuple)) and len(output) >= 3:
                        # Tuple format (boxes, scores, labels)
                        pred_boxes, pred_scores, pred_labels = output[:3]
                    else:
                        print(f"Warning: Unsupported output format: {type(output)}")
                        continue
                    
                    # Convert tensors to numpy arrays
                    if isinstance(pred_boxes, torch.Tensor):
                        pred_boxes = pred_boxes.cpu().numpy()
                    if isinstance(pred_scores, torch.Tensor):
                        pred_scores = pred_scores.cpu().numpy()
                    if isinstance(pred_labels, torch.Tensor):
                        pred_labels = pred_labels.cpu().numpy()
                    
                    # Store consistent image ID mapping
                    if isinstance(img_id, str):
                        # Convert string IDs to integers for COCO
                        if img_id not in self.image_id_map:
                            self.image_id_map[img_id] = len(self.image_id_map)
                        numeric_img_id = self.image_id_map[img_id]
                    else:
                        numeric_img_id = int(img_id)
                        
                    # Process detections based on output format
                    if output_format == 'coco':
                        # Convert each detection to COCO format
                        for box_idx in range(len(pred_boxes)):
                            x1, y1, x2, y2 = pred_boxes[box_idx]
                            
                            # COCO format uses [x, y, width, height]
                            coco_box = [
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1)
                            ]
                            
                            # Get category ID, handling both string and integer labels
                            label = pred_labels[box_idx]
                            if isinstance(label, (str, np.str_)):
                                # Try to convert string label to integer
                                if hasattr(self.model, 'config') and hasattr(self.model.config, 'label2id'):
                                    category_id = int(self.model.config.label2id.get(label, 0))
                                else:
                                    # Default to 0 if can't convert
                                    category_id = 0
                            else:
                                category_id = int(label)
                            
                            # Map model's continuous category IDs to COCO's non-continuous IDs
                            # COCO uses specific IDs that aren't sequential
                            
                            # Create detection entry
                            # detection = {
                            #     'image_id': int(img_id) if not isinstance(img_id, str) else hash(img_id) % 10000,
                            #     'category_id': category_id,
                            #     'bbox': coco_box,
                            #     'score': float(pred_scores[box_idx]),
                            #     'area': float((x2 - x1) * (y2 - y1)),
                            #     'iscrowd': 0
                            # }
                            # Use the mapping if available, otherwise add 1 (0->1, 1->2, etc.)
                            if category_id in original_coco_id_mapping:
                                mapped_category_id = original_coco_id_mapping[category_id]
                            else:
                                # Fallback to simple +1 mapping for categories not in the mapping
                                mapped_category_id = category_id + 1
                            detection = {
                                'image_id': numeric_img_id,
                                'category_id': mapped_category_id,
                                'bbox': coco_box,
                                'score': float(pred_scores[box_idx]),
                                'area': float((x2 - x1) * (y2 - y1)),
                                'iscrowd': 0
                            }
                            
                            results.append(detection)
                    
                    elif output_format == 'kitti':
                        # Convert each detection to KITTI format
                        image_results = []
                        
                        for box_idx in range(len(pred_boxes)):
                            x1, y1, x2, y2 = pred_boxes[box_idx]
                            
                            # Get category ID and map to KITTI class
                            label = pred_labels[box_idx]
                            if isinstance(label, (str, np.str_)):
                                # Try to convert string label to integer
                                if hasattr(self.model, 'config') and hasattr(self.model.config, 'label2id'):
                                    category_id = int(self.model.config.label2id.get(label, 0))
                                else:
                                    # Default to 0 if can't convert
                                    category_id = 0
                            else:
                                category_id = int(label)
                            
                            # Map COCO class ID to KITTI class name
                            coco_to_kitti = {
                                0: 'Pedestrian',  # person
                                1: 'Cyclist',     # bicycle
                                2: 'Car',         # car
                                3: 'Cyclist',     # motorcycle
                                5: 'Car',         # bus
                                7: 'Truck',       # truck
                                9: 'Misc'         # traffic light
                            }
                            
                            kitti_class = coco_to_kitti.get(category_id, 'DontCare')
                            
                            # Skip classes that don't map to KITTI
                            if kitti_class == 'DontCare' and pred_scores[box_idx] < 0.5:
                                continue
                            
                            # Format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y score
                            kitti_line = f"{kitti_class} 0.0 0 0.0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 {pred_scores[box_idx]:.6f}"
                            image_results.append(kitti_line)
                        
                        # Add to results
                        # results.append({
                        #     'image_id': img_id,
                        #     'detections': image_results
                        # })
                        
                        # Add to results with consistent image ID
                        results.append({
                            'image_id': numeric_img_id,
                            'original_id': img_id,  # Keep original ID for file naming
                            'detections': image_results
                        })
                        
                        # Save detections to file if output directory is provided
                        if output_dir:
                            # Use original ID for file naming
                            result_file = os.path.join(output_dir, f"{img_id}.txt")
                            with open(result_file, 'w') as f:
                                for line in image_results:
                                    f.write(line + '\n')
                    
                    # Visualize if output directory is provided
                    if visualize and output_dir and batch_idx % 10 == 0 and i == 0:  # Visualize every 10th batch, first image
                        # Try to get original image for visualization
                        if isinstance(batch, dict) and 'img_orig' in batch:
                            #img_orig = batch['img_orig'][i]
                            img_orig = batch['img_orig'][i] if isinstance(batch['img_orig'], list) else batch['img_orig']
                            
                            # Convert tensor to numpy if needed
                            if isinstance(img_orig, torch.Tensor):
                                img_orig = img_orig.permute(1, 2, 0).cpu().numpy()
                                
                            # Convert normalized image to uint8 if needed
                            if img_orig.dtype == np.float32 and img_orig.max() <= 1.0:
                                img_orig = (img_orig * 255).astype(np.uint8)
                                
                            # Convert RGB to BGR for OpenCV if needed
                            if img_orig.shape[2] == 3:  # RGB image
                                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
                                
                        elif isinstance(batch, dict) and 'img_path' in batch:
                            try:
                                img_path = batch['img_path'][i] if isinstance(batch['img_path'], list) else batch['img_path']
                                img_orig = cv2.imread(img_path)
                            except:
                                img_orig = None
                        else:
                            # Try to reconstruct from processed tensor
                            try:
                                if isinstance(batch_images, torch.Tensor) and batch_images.dim() == 4:
                                    img = batch_images[i].permute(1, 2, 0).cpu().numpy()
                                    img_orig = (img * 255).astype(np.uint8)
                                    if img_orig.shape[2] == 3:  # RGB image
                                        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
                                else:
                                    img_orig = None
                            except:
                                img_orig = None
                        
                        if img_orig is not None:
                            # Create visualization
                            vis_result = {
                                'boxes': pred_boxes,
                                'scores': pred_scores,
                                'labels': pred_labels
                            }
                            
                            # Visualize detections
                            img_path = self.dataset.image_id_to_path[int(img_id)]
                            img = cv2.imread(img_path)
                            img_orig = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #HWC
                            self._visualize_detections_metrics(
                                image = img_orig,
                                detections=vis_result,
                                ground_truth=batch['target'][i],
                                output_path=os.path.join(output_dir, "visualize_detections_metrics", f"vis_{img_id}.jpg")
                            )
        
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
        
        # Create image ID mapping if it doesn't exist
        if not hasattr(self, 'image_id_map'):
            self.image_id_map = {}
        
        # Process each image in the dataset
        annotation_id = 0
        for idx in range(len(dataset)):
            # Get sample
            sample = dataset[idx]
            
            # Get image ID
            if hasattr(sample, 'image_id'):
                image_id = sample.image_id
            elif isinstance(sample, dict) and 'image_id' in sample:
                image_id = sample['image_id']
            else:
                image_id = idx
            
            # Ensure consistent image ID mapping
            if isinstance(image_id, str):
                if image_id not in self.image_id_map:
                    self.image_id_map[image_id] = len(self.image_id_map)
                numeric_image_id = self.image_id_map[image_id]
            else:
                numeric_image_id = int(image_id)
            
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
                'file_name': f"{image_id}.jpg"  # Keep original ID for filename
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
                # coco_dict['annotations'].append({
                #     'id': annotation_id,
                #     'image_id': int(image_id),
                #     'category_id': label,
                #     'bbox': coco_box,
                #     'area': area,
                #     'iscrowd': iscrowd,
                #     'segmentation': []  # No segmentation for detection
                # })
                # Add annotation with consistent image ID
                coco_dict['annotations'].append({
                    'id': annotation_id,
                    'image_id': numeric_image_id,
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
    
    def _visualize_evaluation_results(self, img_id, gt_by_img, pred_by_img, img_metrics, output_dir):
        """
        Visualize detection results with ground truth and predictions on the image.
        
        Args:
            img_id: Image ID
            gt_by_img: Dictionary of ground truth boxes by image and category
            pred_by_img: Dictionary of prediction boxes by image and category
            img_metrics: Metrics for this image (precision, recall)
            output_dir: Directory to save visualization
        """
        # Try to find the image file
        img_path = None
        
        # First check if we have a dataset with image paths
        if hasattr(self, 'dataset'):
            # Try to get image path from dataset's image_id_to_path dictionary
            if hasattr(self.dataset, 'image_id_to_path') and img_id in self.dataset.image_id_to_path:
                img_path = self.dataset.image_id_to_path[img_id]
            # If not found in the dictionary, try the get_image_path method
            elif hasattr(self.dataset, 'get_image_path'):
                img_path = self.dataset.get_image_path(img_id)
        
        # If not found, try to find in COCO format
        if img_path is None and hasattr(self, 'coco_gt'):
            try:
                img_info = self.coco_gt.loadImgs(img_id)[0]
                if 'file_name' in img_info:
                    # Try common image directories
                    for img_dir in ['images', 'data/images', 'data']:
                        candidate = os.path.join(img_dir, img_info['file_name'])
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
            except:
                pass
        
        # If still not found, try to construct from image ID
        if img_path is None:
            # Try common image extensions and directories
            for ext in ['.jpg', '.jpeg', '.png']:
                for img_dir in ['images', 'data/images', 'data']:
                    candidate = os.path.join(img_dir, f"{img_id}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break
        
        # If we found an image path but it doesn't exist, set to None
        if img_path and not os.path.exists(img_path):
            print(f"Warning: Image path found but file does not exist: {img_path}")
            img_path = None
            
        # Get image dimensions from ground truth if available
        width, height = 640, 480  # Default size
        if hasattr(self, 'coco_gt'):
            try:
                img_info = self.coco_gt.loadImgs(img_id)[0]
                if 'width' in img_info and 'height' in img_info:
                    width, height = img_info['width'], img_info['height']
            except:
                pass
        
        # If we couldn't find the image, create a blank canvas
        if img_path is None:
            print(f"Warning: Could not find image for ID {img_id}, using blank canvas")
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
        else:
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image from {img_path}, using blank canvas")
                img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Define colors for different categories and status
        # Ground truth: Green
        # True positive: Blue
        # False positive: Red
        gt_color = (0, 255, 0)  # Green for ground truth
        tp_color = (255, 0, 0)  # Blue for true positives
        fp_color = (0, 0, 255)  # Red for false positives
        
        # Draw ground truth boxes
        if img_id in gt_by_img:
            for cat_id, boxes in gt_by_img[img_id].items():
                for gt in boxes:
                    # Get box coordinates
                    x, y, w, h = gt['bbox']
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), gt_color, 2)
                    
                    # Get category name
                    cat_name = self.class_names.get(cat_id, f"class_{cat_id}")
                    
                    # Draw label
                    cv2.putText(img, f"GT: {cat_name}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)
        
        # Draw prediction boxes
        if img_id in pred_by_img:
            for cat_id, boxes in pred_by_img[img_id].items():
                for pred in boxes:
                    # Get box coordinates
                    x, y, w, h = pred['bbox']
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    
                    # Determine color based on match status
                    color = tp_color if pred['matched'] else fp_color
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Get category name
                    cat_name = self.class_names.get(cat_id, f"class_{cat_id}")
                    
                    # Draw label with score
                    label = f"TP: {cat_name}" if pred['matched'] else f"FP: {cat_name}"
                    score_text = f"{pred['score']:.2f}"
                    cv2.putText(img, f"{label} {score_text}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add metrics text to the image
        precision = img_metrics.get('precision', 0)
        recall = img_metrics.get('recall', 0)
        correct = img_metrics.get('correct', 0)
        gt_count = img_metrics.get('gt_count', 0)
        pred_count = img_metrics.get('pred_count', 0)
        
        # Create metrics text
        metrics_text = [
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"Correct: {correct}/{pred_count} predictions",
            f"Ground Truth: {gt_count} objects"
        ]
        
        # Add metrics text to the top of the image
        y_offset = 30
        for text in metrics_text:
            cv2.putText(img, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 30
        
        # Create a legend
        legend_offset = height - 120
        cv2.putText(img, "Legend:", (10, legend_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(img, "Ground Truth", (30, legend_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)
        
        cv2.putText(img, "True Positive", (30, legend_offset + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tp_color, 2)
        
        cv2.putText(img, "False Positive", (30, legend_offset + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fp_color, 2)
        
        # Save the visualization
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Use original image ID for file naming
        vis_path = os.path.join(vis_dir, f"eval_{img_id}.jpg")
        cv2.imwrite(vis_path, img)
        
        return vis_path

    def _run_custom_evaluation(self, coco_gt, coco_results, output_dir=None, iou_thres=0.5):
        """
        Run custom evaluation on detections with per-image metrics.
        
        Args:
            coco_gt: COCO ground truth object
            coco_results: List of COCO format detections
            output_dir: Directory to save evaluation results
            iou_thres: IoU threshold for considering a detection as correct
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if there are any results
        if not coco_results:
            print("Warning: No detection results to evaluate")
            return {
                'mAP': 0, 'precision': 0, 'recall': 0,
                'correct_detections': 0, 'total_gt': 0, 'total_pred': 0
            }
        
        # Verify that all image IDs in results exist in ground truth
        gt_img_ids = set(coco_gt.getImgIds())
        result_img_ids = set(r['image_id'] for r in coco_results)
        
        # Filter out results with image IDs not in ground truth
        valid_results = [r for r in coco_results if r['image_id'] in gt_img_ids]
        
        if len(valid_results) < len(coco_results):
            print(f"Warning: Filtered out {len(coco_results) - len(valid_results)} results with invalid image IDs")
            
        if not valid_results:
            print("Error: No valid detection results to evaluate after filtering")
            return {
                'mAP': 0, 'precision': 0, 'recall': 0,
                'correct_detections': 0, 'total_gt': 0, 'total_pred': 0
            }
        
        # Get all category IDs
        gt_cat_ids = set(coco_gt.getCatIds())
        result_cat_ids = set(r['category_id'] for r in valid_results)
        all_cat_ids = sorted(gt_cat_ids.union(result_cat_ids))
        
        # Group ground truth by image and category
        gt_by_img = {} #[img_id][cat_id]
        for img_id in gt_img_ids:
            gt_by_img[img_id] = {}
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in gt_by_img[img_id]:
                    gt_by_img[img_id][cat_id] = []
                gt_by_img[img_id][cat_id].append({
                    'bbox': ann['bbox'],  # [x, y, width, height]
                    'matched': False  # Track if this GT has been matched
                })
        
        # Group predictions by image and category
        pred_by_img = {}
        for pred in valid_results:
            img_id = pred['image_id']
            cat_id = pred['category_id']
            
            if img_id not in pred_by_img:
                pred_by_img[img_id] = {}
            
            if cat_id not in pred_by_img[img_id]:
                pred_by_img[img_id][cat_id] = []
            
            pred_by_img[img_id][cat_id].append({
                'bbox': pred['bbox'],  # [x, y, width, height]
                'score': pred['score'],
                'matched': False  # Track if this prediction has been matched
            })
        
        # Calculate IoU between two boxes
        def calculate_iou(box1, box2):
            # Convert from [x, y, width, height] to [x1, y1, x2, y2]
            box1_x1, box1_y1 = box1[0], box1[1]
            box1_x2, box1_y2 = box1_x1 + box1[2], box1_y1 + box1[3]
            
            box2_x1, box2_y1 = box2[0], box2[1]
            box2_x2, box2_y2 = box2_x1 + box2[2], box2_y1 + box2[3]
            
            # Calculate intersection area
            x_left = max(box1_x1, box2_x1)
            y_top = max(box1_y1, box2_y1)
            x_right = min(box1_x2, box2_x2)
            y_bottom = min(box1_y2, box2_y2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union area
            box1_area = box1[2] * box1[3]
            box2_area = box2[2] * box2[3]
            union_area = box1_area + box2_area - intersection_area
            
            # Calculate IoU
            iou = intersection_area / union_area if union_area > 0 else 0.0
            
            return iou
        
        # Initialize metrics
        per_image_metrics = {}
        total_correct = 0
        total_gt = 0
        total_pred = 0
        
        # Calculate per-category metrics
        category_metrics = {cat_id: {'TP': 0, 'FP': 0, 'FN': 0, 'AP': 0.0} for cat_id in all_cat_ids}
        
        # Process each image
        for img_id in gt_img_ids:
            # Initialize per-image metrics
            per_image_metrics[img_id] = {
                'correct': 0,
                'gt_count': 0,
                'pred_count': 0,
                'precision': 0.0,
                'recall': 0.0,
                'categories': {}
            }
            
            # Count ground truth for this image
            img_gt_count = 0
            if img_id in gt_by_img:
                for cat_id in gt_by_img[img_id]:
                    img_gt_count += len(gt_by_img[img_id][cat_id])
            
            per_image_metrics[img_id]['gt_count'] = img_gt_count #6
            total_gt += img_gt_count
            
            # Count predictions for this image
            img_pred_count = 0
            if img_id in pred_by_img:
                for cat_id in pred_by_img[img_id]:
                    img_pred_count += len(pred_by_img[img_id][cat_id])
            
            per_image_metrics[img_id]['pred_count'] = img_pred_count #5
            total_pred += img_pred_count
            
            # Match predictions to ground truth
            img_correct = 0
            
            # Process each category in this image
            for cat_id in all_cat_ids:
                # Skip if no ground truth or predictions for this category
                if (img_id not in gt_by_img or cat_id not in gt_by_img[img_id]) and \
                   (img_id not in pred_by_img or cat_id not in pred_by_img[img_id]):
                    continue
                
                # Initialize category metrics for this image
                if cat_id not in per_image_metrics[img_id]['categories']:
                    per_image_metrics[img_id]['categories'][cat_id] = {
                        'TP': 0, 'FP': 0, 'FN': 0, 'precision': 0.0, 'recall': 0.0
                    }
                
                # Get ground truth and predictions for this category
                gt_boxes = []
                if img_id in gt_by_img and cat_id in gt_by_img[img_id]:
                    gt_boxes = gt_by_img[img_id][cat_id] #list of dicts 'bbox'
                
                pred_boxes = []
                if img_id in pred_by_img and cat_id in pred_by_img[img_id]:
                    # Sort predictions by score (descending)
                    pred_boxes = sorted(pred_by_img[img_id][cat_id], key=lambda x: x['score'], reverse=True)
                
                # Match predictions to ground truth
                for pred in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    # Find the best matching ground truth
                    for gt_idx, gt in enumerate(gt_boxes):
                        if gt['matched']:
                            continue  # Skip already matched ground truths
                        
                        iou = calculate_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Check if match is good enough
                    if best_iou >= iou_thres and best_gt_idx >= 0:
                        # Mark as matched
                        gt_boxes[best_gt_idx]['matched'] = True
                        pred['matched'] = True
                        
                        # Count as true positive
                        per_image_metrics[img_id]['categories'][cat_id]['TP'] += 1
                        category_metrics[cat_id]['TP'] += 1
                        img_correct += 1
                    else:
                        # Count as false positive
                        per_image_metrics[img_id]['categories'][cat_id]['FP'] += 1
                        category_metrics[cat_id]['FP'] += 1
                
                # Count false negatives (unmatched ground truths)
                fn_count = sum(1 for gt in gt_boxes if not gt['matched'])
                per_image_metrics[img_id]['categories'][cat_id]['FN'] += fn_count
                category_metrics[cat_id]['FN'] += fn_count
                
                # Calculate precision and recall for this category in this image
                tp = per_image_metrics[img_id]['categories'][cat_id]['TP']
                fp = per_image_metrics[img_id]['categories'][cat_id]['FP']
                fn = per_image_metrics[img_id]['categories'][cat_id]['FN']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                per_image_metrics[img_id]['categories'][cat_id]['precision'] = precision
                per_image_metrics[img_id]['categories'][cat_id]['recall'] = recall
            
            # Update image metrics
            per_image_metrics[img_id]['correct'] = img_correct
            total_correct += img_correct
            
            # Calculate overall precision and recall for this image
            if img_pred_count > 0:
                per_image_metrics[img_id]['precision'] = img_correct / img_pred_count
            
            if img_gt_count > 0:
                per_image_metrics[img_id]['recall'] = img_correct / img_gt_count
            
            # Visualize detection results if output directory is provided
            if output_dir:
                self._visualize_evaluation_results(img_id, gt_by_img, pred_by_img, per_image_metrics[img_id], output_dir)
        
        # Calculate AP for each category using precision-recall curve
        for cat_id in all_cat_ids:
            # Get all predictions for this category
            all_preds = []
            for img_id in pred_by_img:
                if cat_id in pred_by_img[img_id]:
                    for pred in pred_by_img[img_id][cat_id]:
                        all_preds.append({
                            'score': pred['score'],
                            'matched': pred['matched']
                        })
            
            # Sort by score (descending)
            all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
            
            # Calculate precision and recall at each threshold
            precisions = []
            recalls = []
            tp_count = 0
            fp_count = 0
            total_positives = category_metrics[cat_id]['TP'] + category_metrics[cat_id]['FN']
            
            for pred in all_preds:
                if pred['matched']:
                    tp_count += 1
                else:
                    fp_count += 1
                
                precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                recall = tp_count / total_positives if total_positives > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate AP using 11-point interpolation
            if not precisions:
                category_metrics[cat_id]['AP'] = 0.0
                continue
                
            # Ensure precision is monotonically decreasing
            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = max(precisions[i - 1], precisions[i])
            
            # Calculate AP using 11-point interpolation
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if not recalls or recalls[-1] < t:
                    p = 0.0
                else:
                    # Find precision at recall >= t
                    p = max([precisions[i] for i in range(len(recalls)) if recalls[i] >= t], default=0.0)
                
                ap += p / 11.0
            
            category_metrics[cat_id]['AP'] = ap
        
        # Calculate overall metrics
        overall_precision = total_correct / total_pred if total_pred > 0 else 0.0
        overall_recall = total_correct / total_gt if total_gt > 0 else 0.0
        
        # Calculate mAP as average of category APs
        mean_ap = sum(cat_metrics['AP'] for cat_metrics in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
        
        # Prepare summary metrics
        summary_metrics = {
            'mAP': mean_ap,
            'precision': overall_precision,
            'recall': overall_recall,
            'correct_detections': total_correct,
            'total_gt': total_gt,
            'total_pred': total_pred,
            'per_category': {cat_id: {
                'AP': metrics['AP'],
                'precision': metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0.0,
                'recall': metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0.0,
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN']
            } for cat_id, metrics in category_metrics.items()},
            'per_image': per_image_metrics
        }
        
        # Print summary
        print("\n===== Custom Evaluation Results =====")
        print(f"mAP: {mean_ap:.4f}")
        print(f"Precision: {overall_precision:.4f}")
        print(f"Recall: {overall_recall:.4f}")
        print(f"Correct Detections: {total_correct} / {total_pred} predictions, {total_gt} ground truths")
        
        # Print per-category results
        print("\nPer-Category Results:")
        for cat_id in sorted(category_metrics.keys()):
            cat_name = self.class_names.get(cat_id, f"class_{cat_id}")
            metrics = category_metrics[cat_id]
            cat_precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0.0
            cat_recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0.0
            
            print(f"  {cat_name} (ID: {cat_id}): AP={metrics['AP']:.4f}, Precision={cat_precision:.4f}, Recall={cat_recall:.4f}, TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}")
        
        # Save evaluation results if output directory is provided
        if output_dir:
            with open(os.path.join(output_dir, "custom_eval_results.json"), 'w') as f:
                json.dump(summary_metrics, f, indent=2)
            print(f"\nCustom evaluation results saved to {output_dir}/custom_eval_results.json")
        
        return summary_metrics

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
        # Check if there are any results
        if not coco_results:
            print("Warning: No detection results to evaluate")
            return {
                'AP': 0, 'AP50': 0, 'AP75': 0,
                'APs': 0, 'APm': 0, 'APl': 0,
                'ARmax1': 0, 'ARmax10': 0, 'ARmax100': 0,
                'ARs': 0, 'ARm': 0, 'ARl': 0
            }
        
        # Verify that all image IDs in results exist in ground truth
        gt_img_ids = set(coco_gt.getImgIds())
        result_img_ids = set(r['image_id'] for r in coco_results)
        
        # Filter out results with image IDs not in ground truth
        valid_results = [r for r in coco_results if r['image_id'] in gt_img_ids]
        
        if len(valid_results) < len(coco_results):
            print(f"Warning: Filtered out {len(coco_results) - len(valid_results)} results with invalid image IDs")
            
        if not valid_results:
            print("Error: No valid detection results to evaluate after filtering")
            return {
                'AP': 0, 'AP50': 0, 'AP75': 0,
                'APs': 0, 'APm': 0, 'APl': 0,
                'ARmax1': 0, 'ARmax10': 0, 'ARmax100': 0,
                'ARs': 0, 'ARm': 0, 'ARl': 0
            }
        
        # Create COCO detection object
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(valid_results, f)
            f.flush()
            try:
                coco_dt = coco_gt.loadRes(f.name)
            except Exception as e:
                print(f"Error loading results: {e}")
                # Try to debug the issue
                print(f"Ground truth has {len(gt_img_ids)} images")
                print(f"Results contain {len(result_img_ids)} unique image IDs")
                print(f"First few ground truth image IDs: {list(gt_img_ids)[:5]}")
                print(f"First few result image IDs: {list(result_img_ids)[:5]}")
                
                # Return empty metrics
                return {
                    'AP': 0, 'AP50': 0, 'AP75': 0,
                    'APs': 0, 'APm': 0, 'APl': 0,
                    'ARmax1': 0, 'ARmax10': 0, 'ARmax100': 0,
                    'ARs': 0, 'ARm': 0, 'ARl': 0
                }
        
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
    
    def evaluate_kitti(self, dataset, output_dir=None, batch_size=16, conf_thres=0.25, 
                       iou_thres=0.45, max_det=300, kitti_label_dir=None):
        """
        Evaluate the model on KITTI dataset using COCO metrics.
        
        Args:
            dataset: KITTI dataset object
            output_dir: Directory to save evaluation results
            batch_size: Batch size for inference
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections per image
            kitti_label_dir: Directory containing KITTI labels (if not in dataset)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.dataset = dataset
        # First, convert KITTI dataset to COCO format for ground truth
        # This will establish our image ID mapping
        coco_gt = self.convert_kitti_to_coco(dataset, kitti_label_dir)
        
        # Get the valid image IDs from ground truth
        gt_img_ids = set(coco_gt.getImgIds())
        
        # Create a mapping from original image IDs to COCO image IDs
        # This is crucial for matching detection results with ground truth
        img_id_mapping = {}
        for img_id in gt_img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            original_id = os.path.splitext(img_info['file_name'])[0]
            img_id_mapping[original_id] = img_id
        
        # Store original image paths or IDs for later matching
        original_img_ids = []
        for idx in range(len(dataset)):
            # Get sample
            if hasattr(dataset, '__getitem__'):
                sample = dataset[idx]
            else:
                continue
                
            # Get image path or ID
            if hasattr(sample, 'img_path'):
                img_path = sample.img_path
            elif 'img_path' in sample:
                img_path = sample['img_path']
            elif hasattr(dataset, 'image_paths') and idx < len(dataset.image_paths):
                img_path = dataset.image_paths[idx]
            else:
                # Use index as fallback
                img_path = f"{idx}.png"
                
            # Extract original ID from path
            original_id = os.path.splitext(os.path.basename(img_path))[0]
            original_img_ids.append(original_id)
            
        # Run inference on dataset and get detections in KITTI format
        kitti_results = self._run_inference_on_dataset(
            dataset=dataset,
            batch_size=batch_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            output_dir=output_dir,
            output_format='kitti'
        )
        
        # Convert KITTI dataset to COCO format for ground truth
        #coco_gt = self.convert_kitti_to_coco(dataset, kitti_label_dir)
        
        # Convert KITTI detections to COCO format for evaluation
        coco_detections = []
        
        print(f"Got {len(kitti_results)} detection results from inference")
        print(f"Original image IDs: {original_img_ids[:5]}... (total: {len(original_img_ids)})")
        
        # Match results with original image IDs
        for i, result in enumerate(kitti_results):
            # Get the original image ID using the index
            if i < len(original_img_ids):
                original_id = original_img_ids[i]
            else:
                # Skip if index is out of range
                print(f"Warning: Result index {i} is out of range for original_img_ids")
                continue
            
            # Map to the correct COCO image ID using our mapping
            if original_id in img_id_mapping:
                coco_img_id = img_id_mapping[original_id]
            else:
                # Skip if we can't find a mapping
                print(f"Warning: Could not find mapping for image ID: {original_id}")
                continue
            
            # Parse KITTI format detections
            for detection in result['detections']:
                parts = detection.strip().split()
                if len(parts) < 15:  # KITTI format should have at least 15 parts
                    continue
                
                obj_type = parts[0]
                x1, y1, x2, y2 = map(float, parts[4:8])
                score = float(parts[14])
                
                category_id = kitti_to_coco.get(obj_type, 90)
                
                # Skip DontCare
                if category_id == 91:
                    continue
                
                # Create COCO detection entry with consistent image ID
                coco_detections.append({
                    'image_id': coco_img_id,
                    'category_id': category_id,
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': score,
                    'area': float((x2-x1) * (y2-y1)),
                    'iscrowd': 0
                })
        
        # Print debug info
        print(f"Ground truth has {len(gt_img_ids)} images")
        print(f"Converted {len(coco_detections)} detections for evaluation")
        
        # Additional debug info
        if len(coco_detections) == 0:
            print("No valid detections found. Checking detection results:")
            for i, result in enumerate(kitti_results[:5]):  # Print first 5 for debugging
                print(f"Result {i}:")
                print(f"  Image ID: {result.get('original_id', str(result['image_id']))}")
                print(f"  Number of detections: {len(result['detections'])}")
                if len(result['detections']) > 0:
                    print(f"  First detection: {result['detections'][0]}")
                    
        # Run COCO evaluation
        results = self._run_coco_evaluation(coco_gt, coco_detections, output_dir)
        
        return results
        
    def evaluate_kitti_old(self, kitti_dataset, kitti_label_dir=None, output_dir=None, 
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
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Run inference on dataset and get detections in KITTI format
        kitti_results = self._run_inference_on_dataset(
            dataset=kitti_dataset,
            batch_size=batch_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            output_dir=output_dir,
            output_format='kitti'
        )
        
        # Convert KITTI dataset to COCO format for ground truth
        coco_gt = self.convert_kitti_to_coco(kitti_dataset, kitti_label_dir)
        
        # Convert KITTI detections to COCO format for evaluation
        coco_detections = []
        annotation_id = 0
        
        for result in kitti_results:
            img_id = result['image_id']
            
            # Parse KITTI format detections
            for detection in result['detections']:
                parts = detection.strip().split()
                if len(parts) < 15:  # KITTI format should have at least 15 parts
                    continue
                
                obj_type = parts[0]
                x1, y1, x2, y2 = map(float, parts[4:8])
                score = float(parts[14])
                
                # Map KITTI class to COCO class
                kitti_to_coco = {
                    'Car': 2,
                    'Van': 2,
                    'Truck': 7,
                    'Pedestrian': 0,
                    'Person_sitting': 0,
                    'Cyclist': 1,
                    'Tram': 6,
                    'Misc': 90,
                    'DontCare': 91
                }
                
                category_id = kitti_to_coco.get(obj_type, 90)
                
                # Skip DontCare
                if category_id == 91:
                    continue
                
                # Create COCO detection entry
                coco_detections.append({
                    'image_id': int(img_id) if not isinstance(img_id, str) else hash(img_id) % 10000,
                    'category_id': category_id,
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': score,
                    'area': float((x2-x1) * (y2-y1)),
                    'iscrowd': 0
                })
        
        # Run COCO evaluation
        results = self._run_coco_evaluation(coco_gt, coco_detections, output_dir)
        
        return results
        
    def predict(self, image, conf_thres=0.25, iou_thres=0.45, max_det=300, visualize=False, output_path=None):
        """
        Perform object detection on an image.
        
        Args:
            image: Path to image file or PIL Image or numpy array
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            visualize: Whether to visualize the results
            output_path: Path to save visualization
            
        Returns:
            Dictionary with detection results
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Load and preprocess image
        if isinstance(image, str):
            # Load image from file
            if os.path.exists(image):
                if self.model_type in ['yolov8', 'yolov7']:
                    # For YOLO models, use cv2 to load image
                    img_orig = cv2.imread(image)
                    if img_orig is None:
                        raise FileNotFoundError(f"Image not found or invalid: {image}")
                    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                else:
                    # For other models, use PIL to load image
                    pil_image = Image.open(image).convert("RGB")
                    img_rgb = np.array(pil_image)
                    img_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        elif isinstance(image, np.ndarray):
            # Handle numpy array input
            img_orig = image.copy()
            if img_orig.shape[2] == 3 and img_orig.dtype == np.uint8:
                if img_orig.shape[2] == 3:  # Check if it's a color image
                    # Assume BGR format (cv2 default)
                    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                else:
                    raise ValueError("Input numpy array must have 3 channels (color image)")
            else:
                raise ValueError("Input numpy array must be uint8 with 3 channels")
        elif isinstance(image, Image.Image):
            # Handle PIL Image input
            pil_image = image
            img_rgb = np.array(pil_image)
            img_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Image must be a path, PIL Image, or numpy array")
        
        # Store original image size
        orig_size = (img_orig.shape[0], img_orig.shape[1])  # (height, width)
        
        # Process image using the appropriate processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Extract scale factors and padding info if available (for accurate box coordinates)
        scale_factors = inputs.get("scale_factors", None)
        padding_info = inputs.get("padding_info", None)
        
        # Move inputs to the device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            # Get outputs
            outputs = self.model(**inputs)
        
        # Post-process with scale factors and padding info for accurate box coordinates
        if hasattr(self.processor, 'post_process_object_detection'):
            # Use processor's post-processing if available
            if scale_factors is not None and padding_info is not None:
                # Use scale factors and padding info for accurate box coordinates
                results = self.processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=conf_thres,
                    target_sizes=inputs.get("original_sizes", [orig_size]),
                    scale_factors=scale_factors,
                    padding_info=padding_info
                )
            else:
                # Fallback to standard post-processing
                results = self.processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=conf_thres,
                    target_sizes=[orig_size]
                )
            
            # Extract the first result (batch size is 1)
            result = results[0]
        else:
            # Manual post-processing if processor doesn't support it
            if isinstance(outputs, dict) and "pred_boxes" in outputs:
                # DETR-style outputs
                boxes = outputs["pred_boxes"][0].cpu()
                scores = outputs["pred_logits"][0].sigmoid().max(dim=1)[0].cpu()
                labels = outputs["pred_logits"][0].sigmoid().max(dim=1)[1].cpu()
                
                # Filter by confidence
                keep = scores > conf_thres
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Convert normalized boxes to pixel coordinates
                boxes[:, 0::2] *= orig_size[1]  # scale x by width
                boxes[:, 1::2] *= orig_size[0]  # scale y by height
                
                result = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                }
            elif isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
                # YOLO-style outputs (list of dicts)
                result = outputs[0]
            else:
                raise ValueError(f"Unsupported output format from model: {type(outputs)}")
        
        # Visualize results if requested
        if visualize:
            result_vis = self._visualize_detections(image=img_orig, detections=result, output_path=output_path)
            result["visualization"] = result_vis
        
        return result
    
    def _visualize_detections_metrics(self, image, detections, ground_truth=None, output_path=None, score_threshold=0.25):
        """
        Visualize detection results on an image with metrics.
        
        Args:
            image: Image to visualize detections on (numpy array)
            detections: Dictionary with boxes, scores, labels
            ground_truth: Optional ground truth annotations
            output_path: Path to save visualization
            score_threshold: Minimum score to display a detection
        """
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
        
        # Define colors for predictions and ground truth
        pred_color = (0, 0, 255)  # Red for predictions
        gt_color = (0, 255, 0)    # Green for ground truth
        
        # Calculate metrics if ground truth is provided
        metrics = {}
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        if ground_truth is not None:
            gt_boxes = ground_truth.get('boxes', [])
            gt_labels = ground_truth.get('labels', [])
            
            # Convert tensors to numpy if needed
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()
            
            # Get prediction boxes and labels
            pred_boxes = detections.get('boxes', [])
            pred_scores = detections.get('scores', [])
            pred_labels = detections.get('labels', [])
            
            # Convert tensors to numpy if needed
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.cpu().numpy()
            
            # Filter predictions by score threshold
            valid_pred_indices = []
            for i, score in enumerate(pred_scores):
                if score >= score_threshold:
                    valid_pred_indices.append(i)
            
            filtered_pred_boxes = [pred_boxes[i] for i in valid_pred_indices]
            filtered_pred_labels = [pred_labels[i] for i in valid_pred_indices]
            
            # Calculate IoU for each ground truth and prediction pair
            for gt_idx, gt_box in enumerate(gt_boxes):
                gt_label = gt_labels[gt_idx]
                best_iou = 0.5  # IoU threshold
                best_pred_idx = -1
                
                for pred_idx, pred_box in enumerate(filtered_pred_boxes):
                    # Skip if prediction already matched or labels don't match
                    if pred_idx in matched_pred_indices or filtered_pred_labels[pred_idx] != gt_label:
                        continue
                    
                    # Calculate IoU
                    # Convert to [x1, y1, x2, y2] format if needed
                    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
                    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
                    
                    # Calculate intersection area
                    x_left = max(gt_x1, pred_x1)
                    y_top = max(gt_y1, pred_y1)
                    x_right = min(gt_x2, pred_x2)
                    y_bottom = min(gt_y2, pred_y2)
                    
                    if x_right < x_left or y_bottom < y_top:
                        iou = 0.0
                    else:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                        union_area = gt_area + pred_area - intersection_area
                        iou = intersection_area / union_area if union_area > 0 else 0.0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                if best_pred_idx >= 0:
                    matched_gt_indices.add(gt_idx)
                    matched_pred_indices.add(best_pred_idx)
            
            # Calculate precision and recall
            true_positives = len(matched_pred_indices)
            false_positives = len(valid_pred_indices) - true_positives
            false_negatives = len(gt_boxes) - len(matched_gt_indices)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'total_predictions': len(valid_pred_indices),
                'total_ground_truth': len(gt_boxes)
            }
        
        # Draw ground truth boxes if provided
        if ground_truth is not None:
            gt_boxes = ground_truth.get('boxes', [])
            gt_labels = ground_truth.get('labels', [])
            
            # Convert tensors to numpy if needed
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()
            
            # Draw each ground truth box
            for i, box in enumerate(gt_boxes):
                # Get box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw box with different style if matched
                box_color = gt_color
                thickness = 2
                if i in matched_gt_indices:
                    thickness = 3
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
                
                # Get label
                if i < len(gt_labels):
                    label_id = int(gt_labels[i])
                    label_name = self.class_names.get(label_id, f"class_{label_id}")
                    
                    # Draw label
                    cv2.putText(img, f"GT: {label_name}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Draw prediction boxes
        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        labels = detections.get('labels', [])
        
        # Convert tensors to numpy if needed
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Draw each prediction box
        for i, box in enumerate(boxes):
            # Skip low-confidence detections
            if i < len(scores) and scores[i] < score_threshold:
                continue
                
            # Get box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw box with different style if matched
            box_color = pred_color
            thickness = 2
            if ground_truth is not None and i in matched_pred_indices:
                box_color = (255, 165, 0)  # Orange for matched predictions
                thickness = 3
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
            
            # Get label and score
            if i < len(labels) and i < len(scores):
                label_id = int(labels[i])
                label_name = self.class_names.get(label_id, f"class_{label_id}")
                score = scores[i]
                
                # Draw label with score
                status = "TP" if i in matched_pred_indices else "FP"
                cv2.putText(img, f"{status}: {label_name} ({score:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Create a padded image to show metrics on the right side and legend at the bottom
        h, w = img.shape[:2]
        metrics_width = 300  # Width of the metrics panel
        legend_height = 120  # Height of the legend panel
        
        # Create padded image with extra space for metrics on right and legend at bottom
        padded_img = np.ones((h + legend_height, w + metrics_width, 3), dtype=np.uint8) * 255
        
        # Copy the original image to the top-left corner
        padded_img[:h, :w] = img
        
        # Draw metrics on the right side
        if metrics:
            # Adjust font size and spacing based on image height
            base_font_size = 0.7
            line_spacing = 30
            
            # Scale down for small images
            if h < 400:
                scale_factor = max(0.5, h / 400)  # Don't go below 50%
                base_font_size *= scale_factor
                line_spacing = int(line_spacing * scale_factor)
                
            title_font_size = base_font_size + 0.1
            metric_font_size = base_font_size
            count_font_size = base_font_size - 0.1
            
            # Draw metrics title
            cv2.putText(padded_img, "Detection Metrics", (w + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, title_font_size, (0, 0, 0), 2)
            
            # Draw metrics values (adaptive based on available space)
            y_offset = 30 + line_spacing
            
            # For very small images, combine metrics on fewer lines
            if h < 250:
                cv2.putText(padded_img, f"P:{metrics['precision']:.2f} R:{metrics['recall']:.2f} F1:{metrics['f1_score']:.2f}", 
                            (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, metric_font_size, (0, 0, 0), 1)
                y_offset += line_spacing
                
                cv2.putText(padded_img, f"TP:{metrics['true_positives']} FP:{metrics['false_positives']} FN:{metrics['false_negatives']}", 
                            (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, count_font_size, (0, 0, 0), 1)
            else:
                # Standard metrics display
                cv2.putText(padded_img, f"Precision: {metrics['precision']:.4f}", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, metric_font_size, (0, 0, 0), 1)
                y_offset += line_spacing
                
                cv2.putText(padded_img, f"Recall: {metrics['recall']:.4f}", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, metric_font_size, (0, 0, 0), 1)
                y_offset += line_spacing
                
                cv2.putText(padded_img, f"F1 Score: {metrics['f1_score']:.4f}", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, metric_font_size, (0, 0, 0), 1)
                y_offset += line_spacing
                
                # Counts section
                cv2.putText(padded_img, "Counts:", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, metric_font_size, (0, 0, 0), 1)
                y_offset += line_spacing
                
                cv2.putText(padded_img, f"TP: {metrics['true_positives']}", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, count_font_size, (0, 100, 0), 1)
                y_offset += line_spacing
                
                cv2.putText(padded_img, f"FP: {metrics['false_positives']}", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, count_font_size, (0, 0, 255), 1)
                y_offset += line_spacing
                
                cv2.putText(padded_img, f"FN: {metrics['false_negatives']}", (w + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, count_font_size, (255, 0, 0), 1)
        
        # Add legend at the bottom of the image
        legend_font_size = 0.7
        legend_spacing = 30
        
        # Draw legend title
        cv2.putText(padded_img, "Legend:", (10, h + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, legend_font_size, (0, 0, 0), 2)
        
        # Draw color boxes and labels in a horizontal layout
        box_size = 20
        box_margin = 10
        text_offset = 25
        
        # Starting positions
        x_pos = 10
        y_pos = h + 60
        
        # Ground Truth
        cv2.rectangle(padded_img, (x_pos, y_pos), (x_pos + box_size, y_pos + box_size), gt_color, -1)
        cv2.putText(padded_img, "Ground Truth", (x_pos + box_size + box_margin, y_pos + box_size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # True Positive
        x_pos += 150
        cv2.rectangle(padded_img, (x_pos, y_pos), (x_pos + box_size, y_pos + box_size), (255, 165, 0), -1)
        cv2.putText(padded_img, "True Positive", (x_pos + box_size + box_margin, y_pos + box_size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # False Positive
        x_pos += 150
        cv2.rectangle(padded_img, (x_pos, y_pos), (x_pos + box_size, y_pos + box_size), pred_color, -1)
        cv2.putText(padded_img, "False Positive", (x_pos + box_size + box_margin, y_pos + box_size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add IoU threshold info if metrics were calculated
        if metrics:
            y_pos = h + 90
            cv2.putText(padded_img, f"IoU Threshold: 0.5", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    def _visualize_detections(self, image, detections, ground_truth=None, output_path=None, score_threshold=0.25):
        """
        Visualize detection results on an image.
        
        Args:
            image: Image to visualize detections on (numpy array)
            detections: Dictionary with boxes, scores, labels
            ground_truth: Optional ground truth annotations
            output_path: Path to save visualization
            score_threshold: Minimum score to display a detection
        """
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
        
        # Define colors for predictions and ground truth
        pred_color = (0, 0, 255)  # Red for predictions
        gt_color = (0, 255, 0)    # Green for ground truth
        
        # Draw ground truth boxes if provided
        if ground_truth is not None:
            # Now process the ground truth dictionary
            gt_boxes = ground_truth.get('boxes', [])
            gt_labels = ground_truth.get('labels', [])
            
            # Convert tensors to numpy if needed
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()
            
            # Draw each ground truth box
            for i, box in enumerate(gt_boxes):
                # Get box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), gt_color, 2)
                
                # Get label
                if i < len(gt_labels):
                    label_id = int(gt_labels[i])
                    label_name = self.class_names.get(label_id, f"class_{label_id}")
                    
                    # Draw label
                    cv2.putText(img, f"GT: {label_name}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)
        
        # Draw prediction boxes
        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        labels = detections.get('labels', [])
        
        # Draw each prediction box
        for i, box in enumerate(boxes):
            # Skip low-confidence detections
            if i < len(scores) and scores[i] < score_threshold:
                continue
                
            # Get box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, 2)
            
            # Get label and score
            if i < len(labels) and i < len(scores):
                label_id = int(labels[i])
                label_name = self.class_names.get(label_id, f"class_{label_id}")
                score = scores[i]
                
                # Draw label with score
                cv2.putText(img, f"Pred: {label_name} ({score:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2)
        
        # Add legend
        legend_y = 30
        if ground_truth is not None:
            cv2.putText(img, "Green: Ground Truth", (10, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
        cv2.putText(img, "Red: Predictions", (10, legend_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
        
        # Save or display the image
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
        
        return img
    
    def train(self, train_dataset, val_dataset=None, epochs=100, batch_size=16, 
              learning_rate=0.01, weight_decay=0.0005, output_dir="./runs/train", 
              resume=None, save_interval=10, optimizer_type='sgd', scheduler_type='cosine',
              warmup_epochs=3, gradient_accumulation_steps=1, mixed_precision=False):
        """
        Train the model on a dataset.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            output_dir: Directory to save checkpoints and logs
            resume: Path to checkpoint to resume training from
            save_interval: Save checkpoint every N epochs
            optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
            scheduler_type: Type of scheduler ('cosine', 'step', 'linear', 'constant')
            warmup_epochs: Number of warmup epochs for learning rate
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Whether to use mixed precision training
            
        Returns:
            Dictionary with training results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to training mode
        self.model.train()
        
        # Initialize criterion if not already done and model supports it
        if hasattr(self.model, 'init_criterion') and not hasattr(self.model, 'criterion'):
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
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.937,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Initialize learning rate scheduler
        if scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=epochs - warmup_epochs,
                eta_min=learning_rate / 100
            )
        elif scheduler_type.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif scheduler_type.lower() == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=epochs - warmup_epochs
            )
        elif scheduler_type.lower() == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Initialize warmup scheduler if needed
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
        else:
            warmup_scheduler = None
        
        # Setup mixed precision training if requested
        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Resume training if checkpoint provided
        start_epoch = 0
        if resume:
            if os.path.isfile(resume):
                checkpoint = torch.load(resume, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
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
            train_loss = self._train_one_epoch(
                train_loader, 
                optimizer, 
                epoch, 
                scaler=scaler, 
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            # Update learning rate
            if epoch < warmup_epochs and warmup_scheduler:
                warmup_scheduler.step()
            else:
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
    
    def _train_one_epoch(self, dataloader, optimizer, epoch, scaler=None, gradient_accumulation_steps=1):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            scaler: GradScaler for mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        # Reset gradients at the beginning of the epoch
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision if scaler is provided
            if scaler:
                with torch.cuda.amp.autocast():
                    # Handle different model types
                    if hasattr(self.model, 'loss'):
                        # YOLO-style models
                        loss, loss_items = self.model.loss(batch)
                    elif 'labels' in batch:
                        # HuggingFace-style models
                        outputs = self.model(
                            pixel_values=batch.get('pixel_values', batch.get('img')),
                            labels=batch.get('labels', batch.get('target'))
                        )
                        loss = outputs.loss
                        loss_items = {'loss': loss.item()}
                    else:
                        # Generic case - try to infer inputs and outputs
                        outputs = self.model(**batch)
                        if isinstance(outputs, dict) and 'loss' in outputs:
                            loss = outputs['loss']
                        else:
                            loss = outputs
                        loss_items = {'loss': loss.item()}
                
                # Backward pass with gradient scaling
                scaler.scale(loss / gradient_accumulation_steps).backward()
                
                # Update weights with gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                # Handle different model types
                if hasattr(self.model, 'loss'):
                    # YOLO-style models
                    loss, loss_items = self.model.loss(batch)
                elif 'labels' in batch:
                    # HuggingFace-style models
                    outputs = self.model(
                        pixel_values=batch.get('pixel_values', batch.get('img')),
                        labels=batch.get('labels', batch.get('target'))
                    )
                    loss = outputs.loss
                    loss_items = {'loss': loss.item()}
                else:
                    # Generic case - try to infer inputs and outputs
                    outputs = self.model(**batch)
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        loss = outputs
                    loss_items = {'loss': loss.item()}
                
                # Backward pass with gradient accumulation
                (loss / gradient_accumulation_steps).backward()
                
                # Update weights with gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
            
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
                
                # Handle different model types
                if hasattr(self.model, 'loss'):
                    # YOLO-style models
                    loss, loss_items = self.model.loss(batch)
                elif 'labels' in batch:
                    # HuggingFace-style models
                    outputs = self.model(
                        pixel_values=batch.get('pixel_values', batch.get('img')),
                        labels=batch.get('labels', batch.get('target'))
                    )
                    loss = outputs.loss
                else:
                    # Generic case - try to infer inputs and outputs
                    outputs = self.model(**batch)
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        loss = outputs
                
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
    
    #not used
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
            

def test_multimodels():
    """
    Test function for MultiModels class demonstrating single image inference,
    COCO dataset evaluation, and KITTI dataset evaluation.
    """
    import argparse
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test MultiModels object detection')
    parser.add_argument('--model', type=str, default='yolov8', help='Model type (yolov8, detr, rt-detr, rt-detrv2, vitdet)')
    parser.add_argument('--weights', type=str, default="", help='Path to model weights: ../modelzoo/yolov8s_statedicts.pt')
    parser.add_argument('--hub_model', type=str, default="lkk688/yolov8x-model", help='HF Hub model name (e.g., "facebook/detr-resnet-50")')
    parser.add_argument('--image', type=str, default="ModelDev/sampledata/sjsupeople.jpg", help='Path to test image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    parser.add_argument('--eval_coco', action='store_true', default=True, help='Evaluate on COCO dataset')
    parser.add_argument('--coco_dir', type=str, default="/DATA10T/Datasets/COCOoriginal", help='COCO dataset directory')
    parser.add_argument('--eval_kitti', action='store_true', default=True, help='Evaluate on KITTI dataset')
    parser.add_argument('--kitti_dir', type=str, default="/DATA10T/Datasets/Kitti/", help='KITTI dataset directory')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize MultiModels
    print(f"Initializing MultiModels with model type: {args.model}")
    if args.hub_model:
        # Load model from Hugging Face Hub
        print(f"Loading model from Hugging Face Hub: {args.hub_model}")
        yolo = MultiModels(model_name=args.hub_model)
    else:
        # Create model and load weights
        yolo = MultiModels(model_type=args.model, scale='s')
        if os.path.exists(args.weights):
            print(f"Loading weights from: {args.weights}")
            yolo.load_weights(args.weights)
    
    # Part 1: Single image inference
    if os.path.exists(args.image):
        print(f"\n--- Running inference on {args.image} ---")
        output_path = os.path.join(args.output_dir, "detection_result.jpg")
        
        # Run prediction
        results = yolo.predict(
            args.image,
            conf_thres=args.conf,
            iou_thres=args.iou,
            visualize=True,
            output_path=output_path
        )
        
        # Print results
        if 'scores' in results and len(results['scores']) > 0:
            print(f"Detected {len(results['scores'])} objects:")
            for i, (score, label, box) in enumerate(zip(
                results['scores'], results['labels'], results['boxes']
            )):
                class_name = yolo.class_names.get(label.item(), f"class_{label.item()}")
                print(f"  {i+1}. {class_name}: {score.item():.4f} at {box.tolist()}")
            
            print(f"Visualization saved to: {output_path}")
            
        else:
            print("No objects detected.")
    
    # Part 2: COCO evaluation
    if args.eval_coco:
        if not args.coco_dir:
            print("\nSkipping COCO evaluation: No dataset directory provided.")
            print("Use --coco_dir to specify the COCO dataset directory.")
        else:
            print(f"\n--- Running COCO evaluation ---")
            
            # Create COCO validation dataset
            val_dataset = DetectionDataset(
                dataset_type='coco',
                data_dir=args.coco_dir,
                split='val',
                target_size=(640, 640)
            )
            
            # Run evaluation
            eval_results = yolo.evaluate_coco(
                dataset=val_dataset,
                output_dir=args.output_dir,
                batch_size=4,  # Adjust based on available memory
                conf_thres=args.conf,
                iou_thres=args.iou
            )
            
            # Print evaluation results
            print("\nCOCO Evaluation Results:")
            if isinstance(eval_results, dict):
                for metric, value in eval_results.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print("  No valid evaluation results returned.")
    
    # Part 3: KITTI evaluation
    if args.eval_kitti:
        if not args.kitti_dir:
            print("\nSkipping KITTI evaluation: No dataset directory provided.")
            print("Use --kitti_dir to specify the KITTI dataset directory.")
        else:
            print(f"\n--- Running KITTI evaluation ---")
            
            # Create KITTI validation dataset
            kitti_dataset = DetectionDataset(
                dataset_type='kitti',
                data_dir=args.kitti_dir,
                split='val',
                target_size=(640, 640)
            )
            
            # Run evaluation
            kitti_results = yolo.evaluate_kitti(
                dataset=kitti_dataset,
                output_dir=os.path.join(args.output_dir, "kitti_eval"),
                batch_size=4,  # Adjust based on available memory
                conf_thres=args.conf,
                iou_thres=args.iou
            )
            
            # Print evaluation results
            print("\nKITTI Evaluation Results:")
            if isinstance(kitti_results, dict):
                for metric, value in kitti_results.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
                        
                # Print per-class results if available
                if 'per_class' in kitti_results:
                    print("\nPer-class Results:")
                    for class_name, metrics in kitti_results['per_class'].items():
                        print(f"  {class_name}:")
                        for metric, value in metrics.items():
                            print(f"    {metric}: {value:.4f}")
            else:
                print("  No valid evaluation results returned.")

if __name__ == "__main__":
    test_multimodels()