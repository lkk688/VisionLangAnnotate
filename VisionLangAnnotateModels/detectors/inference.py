import torch
import numpy as np
import os
import gc
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from VisionLangAnnotateModels.detectors.base_model import BaseMultiModel
from VisionLangAnnotateModels.detectors.detector_class_mapper import class_name_mapper, step1_classes, coco_to_step1_map

class ModelInference(BaseMultiModel):
    """
    Handles model inference operations including prediction and visualization.
    """
    
    # Class-level caches to store loaded models
    _model_cache = {}
    _sam_model_cache = {}
    _sam_processor_cache = {}
    # Note: GroundingDINO model caches have been moved to BaseMultiModel
    
    @classmethod
    def clear_cache(cls, model_key=None):
        """
        Clear model caches to free up memory.
        
        Args:
            model_key: Optional specific model key to clear from cache.
                      If None, all caches will be cleared.
        """
        if model_key is not None:
            # Clear specific model from cache
            if model_key in cls._model_cache:
                print(f"Removing {model_key} from model cache")
                del cls._model_cache[model_key]
            if model_key in cls._sam_model_cache:
                print(f"Removing {model_key} from SAM model cache")
                del cls._sam_model_cache[model_key]
            if model_key in cls._sam_processor_cache:
                print(f"Removing {model_key} from SAM processor cache")
                del cls._sam_processor_cache[model_key]
        else:
            # Clear all caches
            print("Clearing all model caches")
            cls._model_cache.clear()
            cls._sam_model_cache.clear()
            cls._sam_processor_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
    
    @classmethod
    def get_cache_info(cls):
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "model_cache_size": len(cls._model_cache),
            "sam_model_cache_size": len(cls._sam_model_cache),
            "sam_processor_cache_size": len(cls._sam_processor_cache),
            "model_keys": list(cls._model_cache.keys()),
            "sam_model_keys": list(cls._sam_model_cache.keys()),
            "sam_processor_keys": list(cls._sam_processor_cache.keys())
        }
    
    def __init__(self, model=None, config=None, model_type="myyolohf", model_name=None, scale='s', class_list=None, device=None):
        """
        Initialize ModelInference with a model or create a new one.
        
        Args:
            model: Existing detection model or None to create a new one
            config: Model config object or None to create a default one
            model_type: Type of model to use ('myyolohf', 'detr', 'rtdetr', 'groundingdino', 'yolo')
            model_name: Specific model name/checkpoint from Hugging Face Hub
            scale: Model scale ('n', 's', 'm', 'l', 'x') if creating a new YOLO model
            device: Device to use (None for auto-detection)
        """
        # Check if we already have this model in cache
        cache_key = None
        if model is None and model_name is not None:
            cache_key = f"{model_type}_{model_name}"
            if cache_key in ModelInference._model_cache:
                print(f"Using cached model: {model_name}")
                model = ModelInference._model_cache[cache_key]
        
        super().__init__(model, config, model_type, model_name, scale, device)
        
        # Cache the model if it was loaded from scratch
        if cache_key is not None and model is None and self.model is not None:
            ModelInference._model_cache[cache_key] = self.model
            print(f"Cached model: {model_name}")
        
        # Initialize SAM model and processor as None (will be loaded on demand)
        self.sam_processor = None
        self.sam_model = None
        
        print("Class names:", self.class_names)
        self.text_prompt = None
        self.classifier = None
        if class_list is not None:
            self.class_names = class_list #update the class_list
            if model_type=='groundingdino':
                #build prompt list
                self.text_prompt = ", ".join(class_list)
        

    def predict(self, image_input, text_prompt=None, conf_thres=0.25, iou_thres=0.45, max_det=300, 
                visualize=True, save_path=None, ground_truth=None, show_metrics=False,
                detection_model="yolo", box_threshold=0.25, max_width_ratio=0.9, 
                max_height_ratio=0.9, max_area_ratio=0.8, mask_region=None, free_memory=False):
        """
        Run inference on a single image.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            text_prompt: Text prompt for GroundingDINO (comma-separated categories)
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            visualize: Whether to visualize detections
            save_path: Path to save visualization
            ground_truth: Ground truth annotations for metrics calculation
            show_metrics: Whether to calculate and show metrics
            detection_model: Model to use for detection ('yolo' or 'groundingdino')
            box_threshold: Box threshold for GroundingDINO
            max_width_ratio: Maximum allowed width ratio for filtering large detections
            max_height_ratio: Maximum allowed height ratio for filtering large detections
            max_area_ratio: Maximum allowed area ratio for filtering large detections
            mask_region: Optional dictionary specifying a region to mask out:
                - 'bottom_percent': Percentage of the bottom to mask (e.g., 20 for bottom 20%)
                - 'top_percent': Percentage of the top to mask
                - 'left_percent': Percentage of the left to mask
                - 'right_percent': Percentage of the right to mask
                - 'custom': List of [x1, y1, x2, y2] coordinates to mask
            free_memory: Whether to free memory after inference (useful for batch processing)
            
        Returns:
            Dictionary with detections and optional metrics
        """
        # Convert input to PIL Image and apply mask if specified
        image, original_size = self._prepare_image(image_input, mask_region)
        image_width, image_height = original_size
        
        # Parse text prompt to get categories if provided
        categories = []
        if text_prompt:
            categories = [category.strip() for category in text_prompt.split(",")]
        
        # Different inference process based on detection model
        if detection_model.lower() == "groundingdino" and text_prompt:
            # Use GroundingDINO for detection
            results = self.detect_with_groundingdino(image, text_prompt, box_threshold, conf_thres)#4 tuples output
            labels, boxes, scores = self.process_detection_labels(results, categories)
            
            # Convert to standard detection format
            detections = []
            for i in range(len(boxes)):
                if isinstance(boxes, torch.Tensor):
                    box = boxes[i].cpu().numpy().tolist()
                else:
                    box = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else boxes[i]
                
                detections.append({
                    'bbox': box,
                    'score': float(scores[i]),
                    'class_name': labels[i],
                    'class_id': i  # Using index as class_id since GroundingDINO doesn't provide class IDs
                })
                
        elif self.model_type == 'yolo':
            # For YOLO models, use self.yolo_model
            if self.yolo_model is None:
                raise ValueError("YOLO model not loaded. Please call load_yolo_model() first.")
            
            # YOLO models from ultralytics have a different inference pattern
            yolo_results = self.yolo_model(image, conf=conf_thres, iou=iou_thres, max_det=max_det)
            
            # Convert YOLO results to standard detection format
            detections = []
            for result in yolo_results:
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        class_id = int(class_ids[i])
                        class_name = result.names[class_id] if class_id in result.names else f"class_{class_id}"
                        
                        detections.append({
                            'bbox': boxes[i].tolist(),
                            'score': float(scores[i]),
                            'class_name': class_name,
                            'class_id': class_id
                        })
        else:
            # For other model types, use the standard self.model
            if self.model is None:
                raise ValueError("Model not initialized")
            self.model.eval()
            
            # Standard inference process for other model types
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process outputs based on model type
            detections = self._post_process_outputs(outputs, original_size, conf_thres, iou_thres, max_det)
        #array of dicts ['bbox', 'score', 'class_id', 'class_name']
        
        # Apply filtering to remove large detections that are likely false positives
        filtered_detections = self.filter_standard_detections(
            detections, image_width, image_height,
            max_width_ratio, max_height_ratio, max_area_ratio
        )
        
        # Update class names to match user's interested labels if categories are provided
        # This is a second pass of class name updating, which works on the final detection format
        # The first pass happens in process_detection_labels for the raw model outputs
        if categories:
            # Extract class names from detections
            detected_class_names = [det['class_name'] for det in filtered_detections]
            
            # Update class names using zero-shot text classification to map detected labels to user categories
            # For example, this can map 'person' to 'human', 'car'/'truck' to 'vehicle', etc.
            #updated_class_names = self.class_names_update(detected_class_names, categories)
            updated_class_names = class_name_mapper(detected_class_names)
            # Update the detections with the new class names
            for i, det in enumerate(filtered_detections):
                if i < len(updated_class_names):
                    det['class_name'] = updated_class_names[i]
        
        result = {
            'detections': filtered_detections,
            'image_size': original_size
        }
        
        # Generate segmentation masks for GroundingDINO detections if needed
        segmentation_masks = []
        if detection_model.lower() == "groundingdino" and text_prompt:
            # Extract boxes from filtered detections
            detection_boxes = []
            for det in filtered_detections:
                detection_boxes.append(det['bbox'])
            
            # Convert to tensor if needed
            if detection_boxes and not isinstance(detection_boxes, torch.Tensor):
                detection_boxes = torch.tensor(detection_boxes, device=self.device)
            
            # Generate segmentation masks using SAM
            if len(detection_boxes) > 0:
                segmentation_masks = self.generate_segmentation_masks(image, detection_boxes)
                result['segmentation_masks'] = segmentation_masks
        
        # Calculate metrics if ground truth provided
        if ground_truth is not None and show_metrics:
            metrics = self._calculate_metrics(filtered_detections, ground_truth)
            result['metrics'] = metrics
        
        # Visualize if requested
        if visualize:
            self._visualize_detections(image, filtered_detections, save_path, ground_truth, 
                                     result.get('metrics'), segmentation_masks if segmentation_masks else None)
        
        # Free memory if requested (useful for batch processing)
        if free_memory:
            # Clear any intermediate tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
        
        return result

    def _prepare_image(self, image_input, mask_region=None):
        """
        Convert various input types to PIL Image for processing and optionally apply a mask region.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            mask_region: Optional dictionary specifying a region to mask out:
                - 'bottom_percent': Percentage of the bottom to mask (e.g., 20 for bottom 20%)
                - 'top_percent': Percentage of the top to mask
                - 'left_percent': Percentage of the left to mask
                - 'right_percent': Percentage of the right to mask
                - 'custom': List of [x1, y1, x2, y2] coordinates to mask
                
        Returns:
            tuple: (PIL Image, original_size)
        """
        # Case 1: String path to image file
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
            original_size = image.size
        # Case 2: PIL Image
        elif isinstance(image_input, Image.Image):
            image = image_input.copy()
            original_size = image.size
            
        # Case 3: NumPy array
        elif isinstance(image_input, np.ndarray):
            # Check if the image is in BGR format (OpenCV default)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Convert BGR to RGB if needed
                if image_input.dtype == np.uint8:
                    # Assume BGR format from OpenCV
                    rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_image)
                else:
                    # Assume already in RGB format
                    image = Image.fromarray(image_input.astype(np.uint8))
            else:
                raise ValueError("NumPy array must have shape (H, W, 3)")
                
            original_size = (image.width, image.height)
            
        # Case 4: PyTorch tensor
        elif isinstance(image_input, torch.Tensor):
            # Assume tensor in format (C, H, W) with values in [0, 1] or [0, 255]
            if len(image_input.shape) == 3 and image_input.shape[0] == 3:
                # Convert to numpy, transpose to (H, W, C)
                np_image = image_input.cpu().numpy().transpose(1, 2, 0)
                
                # Scale to [0, 255] if needed
                if np_image.max() <= 1.0:
                    np_image = (np_image * 255).astype(np.uint8)
                else:
                    np_image = np_image.astype(np.uint8)
                    
                image = Image.fromarray(np_image)
                original_size = (image.width, image.height)
            else:
                raise ValueError("PyTorch tensor must have shape (3, H, W)")
        
        # Unsupported type
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
            
        # Apply mask region if specified
        if mask_region is not None:
            # Convert image to numpy array for masking
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Create a mask based on the specified region
            if 'bottom_percent' in mask_region:
                # Mask the bottom portion of the image
                bottom_percent = mask_region['bottom_percent']
                if 0 <= bottom_percent <= 100:
                    mask_height = int(height * bottom_percent / 100)
                    img_array[height - mask_height:, :, :] = 0  # Black out the bottom region
            
            if 'top_percent' in mask_region:
                # Mask the top portion of the image
                top_percent = mask_region['top_percent']
                if 0 <= top_percent <= 100:
                    mask_height = int(height * top_percent / 100)
                    img_array[:mask_height, :, :] = 0  # Black out the top region
            
            if 'left_percent' in mask_region:
                # Mask the left portion of the image
                left_percent = mask_region['left_percent']
                if 0 <= left_percent <= 100:
                    mask_width = int(width * left_percent / 100)
                    img_array[:, :mask_width, :] = 0  # Black out the left region
            
            if 'right_percent' in mask_region:
                # Mask the right portion of the image
                right_percent = mask_region['right_percent']
                if 0 <= right_percent <= 100:
                    mask_width = int(width * right_percent / 100)
                    img_array[:, width - mask_width:, :] = 0  # Black out the right region
            
            if 'custom' in mask_region:
                # Mask a custom region defined by [x1, y1, x2, y2]
                x1, y1, x2, y2 = mask_region['custom']
                if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 <= width and 0 <= y2 <= height:
                    img_array[y1:y2, x1:x2, :] = 0  # Black out the custom region
            
            # Convert back to PIL Image
            image = Image.fromarray(img_array)
        
        return image, original_size
    
    def _post_process_outputs(self, outputs, original_size, conf_thres, iou_thres, max_det):
        """
        Post-process model outputs to get final detections.
        """
        if self.model_type in ['detr', 'rt-detr', 'rt-detrv2']:
            return self._post_process_detr_outputs(outputs, original_size, conf_thres)
        elif self.model_type == 'yolov8':
            return self._post_process_yolo_outputs(outputs, original_size, conf_thres, iou_thres, max_det)
        else:
            return self._post_process_generic_outputs(outputs, original_size, conf_thres)
    
    def _post_process_detr_outputs(self, outputs, original_size, conf_thres):
        """Post-process DETR-style outputs."""
        logits = outputs.logits
        boxes = outputs.pred_boxes
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get confidence scores and class predictions
        scores, labels = probs[..., :-1].max(-1)
        
        # Filter by confidence threshold
        keep = scores > conf_thres
        
        detections = []
        for i in range(len(keep)):
            valid_indices = keep[i]
            if valid_indices.sum() == 0:
                continue
                
            valid_boxes = boxes[i][valid_indices]
            valid_scores = scores[i][valid_indices]
            valid_labels = labels[i][valid_indices]
            
            # Convert from center format to corner format and scale to original size
            w, h = original_size
            valid_boxes = valid_boxes * torch.tensor([w, h, w, h], device=valid_boxes.device)
            
            # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
            x_c, y_c, width, height = valid_boxes.unbind(-1)
            x1 = x_c - 0.5 * width
            y1 = y_c - 0.5 * height
            x2 = x_c + 0.5 * width
            y2 = y_c + 0.5 * height
            valid_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            for j in range(len(valid_boxes)):
                detections.append({
                    'bbox': valid_boxes[j].cpu().numpy(),
                    'score': valid_scores[j].cpu().item(),
                    'class_id': valid_labels[j].cpu().item(),
                    'class_name': self.class_names.get(valid_labels[j].cpu().item(), 'unknown')
                })
        
        return detections
    
    def _post_process_yolo_outputs(self, outputs, original_size, conf_thres, iou_thres, max_det):
        """Post-process YOLO-style outputs."""
        # YOLO outputs are typically in a different format
        # This is a simplified version - actual implementation depends on YOLO model structure
        if hasattr(outputs, 'prediction'):
            predictions = outputs.prediction
        elif hasattr(outputs, 'logits'):
            predictions = outputs.logits
        else:
            predictions = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        
        detections = []
        # Apply NMS and confidence filtering
        # This would need to be implemented based on the specific YOLO output format
        
        return detections
    
    def _post_process_generic_outputs(self, outputs, original_size, conf_thres):
        """Generic post-processing for other model types."""
        # Fallback to DETR-style processing
        return self._post_process_detr_outputs(outputs, original_size, conf_thres)
    
    def _calculate_metrics(self, detections, ground_truth):
        """Calculate precision, recall, and other metrics."""
        # Implementation for metrics calculation
        tp = fp = fn = 0
        
        # This is a simplified version - full implementation would need IoU calculation
        # and proper matching between predictions and ground truth
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def _visualize_detections(self, image, detections, save_path=None, ground_truth=None, metrics=None, masks=None):
        """Visualize detections on the image."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Apply masks with semi-transparency if provided
        if masks is not None:
            import random
            import numpy as np
            # Generate random colors for masks
            mask_colors = [(random.random(), random.random(), random.random(), 0.3) for _ in range(len(masks))]
            
            # Plot each mask with its color
            for i, mask in enumerate(masks):
                mask_img = np.ma.masked_where(mask == 0, mask)
                ax.imshow(mask_img, alpha=0.5, cmap=plt.cm.get_cmap('jet'), interpolation='none')
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"{det['class_name']}: {det['score']:.2f}"
            ax.text(x1, y1-5, label, fontsize=10, color='red', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Draw ground truth if provided
        if ground_truth:
            for gt in ground_truth:
                bbox = gt['bbox']
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
                ax.add_patch(rect)
        
        ax.set_title('Object Detection Results')
        ax.axis('off')
        
        # Add metrics text if available
        if metrics:
            metrics_text = f"Precision: {metrics['precision']:.3f}\nRecall: {metrics['recall']:.3f}\nF1: {metrics['f1_score']:.3f}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        
        #plt.show()
    
    def batch_predict(self, image_inputs, batch_size=8, **kwargs):
        """Run inference on multiple images.
        
        Args:
            image_inputs: List of image inputs (paths, PIL Images, numpy arrays, or tensors)
            batch_size: Number of images to process in each batch
            **kwargs: Additional arguments to pass to predict method
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_inputs), batch_size):
            batch_inputs = image_inputs[i:i+batch_size]
            
            for image_input in batch_inputs:
                result = self.predict(image_input, visualize=False, **kwargs)
                results.append(result)
        
        return results
        
    def perform_box_segmentation(self, image_input, text_prompt=None, detection_model="groundingdino", 
                               confidence_threshold=0.35, box_threshold=0.25, 
                               visualize=True, save_path=None, export_json=None, 
                               export_labelstudio=None, max_width_ratio=0.9, 
                               max_height_ratio=0.9, max_area_ratio=0.8):
        """
        Perform object detection and segmentation on an image using YOLO/GroundingDINO and SAM.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            text_prompt: Text prompt for GroundingDINO (comma-separated categories)
            detection_model: Model to use for detection ('yolo' or 'groundingdino')
            confidence_threshold: Confidence threshold for detections
            box_threshold: Box threshold for GroundingDINO
            visualize: Whether to visualize the results
            save_path: Path to save the visualization
            export_json: Path to export results as JSON
            export_labelstudio: Path to export results in Label Studio format
            max_width_ratio: Maximum allowed width ratio for filtering large detections
            max_height_ratio: Maximum allowed height ratio for filtering large detections
            max_area_ratio: Maximum allowed area ratio for filtering large detections
            
        Returns:
            Dictionary containing detection and segmentation results
        """
        import torch
        import numpy as np
        from PIL import Image
        import json
        import os
        
        # Convert input to PIL Image
        image, (image_width, image_height) = self._prepare_image(image_input)
        
        # Get image path for result reporting
        if isinstance(image_input, str):
            image_path = image_input
        else:
            # For non-path inputs, use a placeholder
            image_path = "memory_image"
        
        # Parse text prompt to get categories
        categories = []
        if text_prompt:
            categories = [category.strip() for category in text_prompt.split(",")]
        
        # Perform object detection based on the selected model
        if detection_model.lower() == "yolo":
            # Use YOLO for detection
            results = self.detect_with_yolo(image, confidence_threshold)
            labels, boxes, scores = self.process_detection_labels(results, categories)
        else:
            # Use GroundingDINO for detection
            results = self.detect_with_groundingdino(image, text_prompt, box_threshold, confidence_threshold)
            labels, boxes, scores = self.process_detection_labels(results, categories)
        
        # Filter out large detections that are likely false positives
        labels, boxes, scores = self.filter_large_detections(
            labels, boxes, scores, image_width, image_height,
            max_width_ratio, max_height_ratio, max_area_ratio
        )
        
        # Generate segmentation masks using SAM
        segmentation_masks = self.generate_segmentation_masks(image, boxes)
        
        # Prepare results dictionary
        results_dict = {
            "image_path": image_path,
            "image_width": image_width,
            "image_height": image_height,
            "labels": labels,
            "boxes": boxes.tolist() if isinstance(boxes, (torch.Tensor, np.ndarray)) else boxes,
            "scores": scores,
            "masks": [mask.tolist() if isinstance(mask, np.ndarray) else mask for mask in segmentation_masks]
        }
        
        # Visualize results if requested
        if visualize:
            # Format detections as a list of dictionaries for visualization
            formatted_detections = []
            for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
                box_list = box.tolist() if isinstance(box, torch.Tensor) else box
                formatted_detections.append({
                    'bbox': box_list,
                    'class_name': label,
                    'score': score
                })
            
            self._visualize_detections(
                image, 
                formatted_detections, 
                save_path=save_path,
                masks=segmentation_masks
            )
        
        # Export results to JSON if requested
        if export_json:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                "image_path": image_path,
                "image_width": image_width,
                "image_height": image_height,
                "detections": []
            }
            
            for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
                box_data = box.tolist() if hasattr(box, 'tolist') else box
                mask_data = segmentation_masks[i].tolist() if i < len(segmentation_masks) and hasattr(segmentation_masks[i], 'tolist') else None
                
                json_results["detections"].append({
                    "label": label,
                    "box": box_data,
                    "score": float(score),
                    "mask": mask_data
                })
            
            with open(export_json, "w") as f:
                json.dump(json_results, f, indent=2)
        
        # Export results in Label Studio format if requested
        if export_labelstudio:
            labelstudio_data = {
                "data": {
                    "image": os.path.basename(image_path) if isinstance(image_input, str) else "memory_image"
                },
                "annotations": [{
                    "result": []
                }]
            }
            
            for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
                # Normalize coordinates for Label Studio
                box_data = box.tolist() if hasattr(box, 'tolist') else box
                x1, y1, x2, y2 = box_data
                
                # Add bounding box annotation
                bbox_annotation = {
                    "id": f"bbox_{i}",
                    "type": "rectanglelabels",
                    "value": {
                        "x": 100 * x1 / image_width,
                        "y": 100 * y1 / image_height,
                        "width": 100 * (x2 - x1) / image_width,
                        "height": 100 * (y2 - y1) / image_height,
                        "rotation": 0,
                        "rectanglelabels": [label]
                    },
                    "score": float(score),
                    "to_name": "image",
                    "from_name": "bbox"
                }
                labelstudio_data["annotations"][0]["result"].append(bbox_annotation)
                
                # Add polygon annotation if mask is available
                if i < len(segmentation_masks):
                    mask = segmentation_masks[i]
                    # Convert mask to polygon points
                    import cv2
                    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Use the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        # Simplify the contour to reduce points
                        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        # Format points for Label Studio
                        points = []
                        for point in approx.reshape(-1, 2):
                            x, y = point
                            points.append([100 * x / image_width, 100 * y / image_height])
                        
                        polygon_annotation = {
                            "id": f"polygon_{i}",
                            "type": "polygonlabels",
                            "value": {
                                "points": points,
                                "polygonlabels": [label]
                            },
                            "score": float(score),
                            "to_name": "image",
                            "from_name": "polygon"
                        }
                        labelstudio_data["annotations"][0]["result"].append(polygon_annotation)
            
            with open(export_labelstudio, "w") as f:
                json.dump(labelstudio_data, f, indent=2)
        
        return results_dict
    
    # Note: load_yolo_model method has been moved to BaseMultiModel class
    
    # Method moved to BaseMultiModel
    def load_groundingdino_model(self, model_name=None):
        """
        Load GroundingDINO model and processor.
        This method is now a wrapper around the BaseMultiModel implementation.
        
        Args:
            model_name: Name of the GroundingDINO model to load, defaults to 'IDEA-Research/grounding-dino-base'
            
        Returns:
            Tuple of (processor, model)
        """
        if model_name is None:
            model_name = "IDEA-Research/grounding-dino-base"
        return super().load_groundingdino_model(model_name)
    
    def load_sam_model(self, model_name="facebook/sam-vit-base"):
        """
        Load Segment Anything Model (SAM) and processor.
        
        Args:
            model_name: Model name or path (default: "facebook/sam-vit-base")
            
        Returns:
            Tuple of (processor, model)
        """
        # Check if model and processor are already in cache
        if model_name in ModelInference._sam_model_cache and model_name in ModelInference._sam_processor_cache:
            print(f"Using cached SAM model: {model_name}")
            self.sam_processor = ModelInference._sam_processor_cache[model_name]
            self.sam_model = ModelInference._sam_model_cache[model_name]
            return self.sam_processor, self.sam_model
            
        try:
            from transformers import SamProcessor, SamModel
            
            self.sam_processor = SamProcessor.from_pretrained(model_name)
            self.sam_model = SamModel.from_pretrained(model_name).to(self.device)
            
            # Cache the model and processor
            ModelInference._sam_processor_cache[model_name] = self.sam_processor
            ModelInference._sam_model_cache[model_name] = self.sam_model
            print(f"Loaded and cached SAM model: {model_name}")
            
            return self.sam_processor, self.sam_model
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            return None, None

    def detect_with_yolo(self, image_input, confidence=0.25):
        """
        Perform object detection using YOLO model.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            confidence: Confidence threshold for detections
            
        Returns:
            Tuple of (boxes, labels, scores)
        """
        # Check if YOLO model is loaded
        if self.yolo_model is None:
            print("YOLO model not loaded. Please initialize with model_type='yolo' and a valid model_name.")
            return torch.zeros((0, 4)), [], []
        
        # Prepare image for inference
        image, original_size = self._prepare_image(image_input)
        
        # Run inference with YOLO
        yolo_results = self.yolo_model(image, conf=confidence)
        
        boxes = []
        labels = []
        scores = []
        
        for result in yolo_results:
            # Check if there are any detections
            if len(result.boxes) > 0:
                # Convert boxes from xywh to xyxy format
                boxes_tensor = result.boxes.xyxy
                boxes.append(boxes_tensor)
                
                # Get class labels
                cls_indices = result.boxes.cls.cpu().numpy()
                cls_names = [result.names[int(idx)] for idx in cls_indices]
                labels.extend(cls_names)
                
                # Get confidence scores
                scores.extend(result.boxes.conf.cpu().numpy())
        
        # Convert list of tensors to single tensor
        if boxes:
            boxes = torch.cat(boxes, dim=0)
        else:
            boxes = torch.zeros((0, 4))  # Empty tensor if no detections
        
        return boxes, labels, scores
    
    def detect_with_groundingdino(self, image_input, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Perform object detection using GroundingDINO model.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            text_prompt: Text prompt describing objects to detect (comma-separated)
            box_threshold: Confidence threshold for bounding box predictions
            text_threshold: Confidence threshold for text-to-box associations
            
        Returns:
            Tuple of (boxes, labels, scores, text_labels)
        """
        import torch
        
        # Load GroundingDINO model if not already loaded
        if self.grounding_processor is None or self.grounding_model is None:
            self.load_groundingdino_model()
            if self.grounding_processor is None or self.grounding_model is None:
                print("Failed to load GroundingDINO model. Make sure the model is properly initialized.")
                return torch.zeros((0, 4)), [], [], []
        
        # Format text prompt for better detection
        # Split the text prompt into individual categories
        categories = [cat.strip() for cat in text_prompt.split(',')]
        # Format for GroundingDINO: use a period after each category
        formatted_prompt = ". ".join(categories) + "."
        print(f"Using formatted prompt: {formatted_prompt}")
        
        # Prepare image for inference
        image, original_size = self._prepare_image(image_input)
        
        # Prepare inputs for GroundingDINO
        grounding_inputs = self.grounding_processor(text=formatted_prompt, images=image, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.grounding_model(**grounding_inputs)
        
        # Process results
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=grounding_inputs.input_ids,
            target_sizes=target_sizes,
            text_threshold=text_threshold,
            threshold=box_threshold
        )[0]
        
        return results["boxes"], results["labels"], results["scores"], results["text_labels"]
        #The key `labels` is will return integer ids, Use `text_labels` instead

    def generate_segmentation_masks(self, image, boxes):
        """
        Generate segmentation masks using SAM model.
        
        Args:
            image: PIL Image
            boxes: Tensor of bounding boxes in xyxy format
            
        Returns:
            List of segmentation masks
        """
        import torch
        
        # Load SAM model if not already loaded
        if self.sam_processor is None or self.sam_model is None:
            self.load_sam_model()
            if self.sam_processor is None or self.sam_model is None:
                print("Failed to load SAM model.")
                return []
        
        segmentation_masks = []
        
        if len(boxes) > 0:
            for box in boxes:
                # Convert box to list if it's a tensor
                box_list = box.tolist() if isinstance(box, torch.Tensor) else box
                
                # Process the image and bounding box with SAM
                sam_inputs = self.sam_processor(image, input_boxes=[[box_list]], return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    sam_outputs = self.sam_model(**sam_inputs)
                
                # Get predicted segmentation mask
                masks = self.sam_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                segmentation_masks.append(masks[0][0][0].numpy())
        
        return segmentation_masks
    
    def process_detection_labels(self, results, categories, empty_label_option="fallback"):
        """
        Process detection labels based on model results and handle empty labels according to the specified option.
        
        Args:
            results: Dictionary containing detection results from the model
            categories: List of category names from the text prompt
            empty_label_option: How to handle empty labels ('ignore', 'unknown', or 'fallback')
            
        Returns:
            Tuple of (labels, boxes, scores)
        """
        import torch
        import numpy as np
        
        # Extract boxes and scores from results
        if isinstance(results, dict):
            # For GroundingDINO results
            boxes = results["boxes"]
            scores = results["scores"]
            text_labels = results.get("text_labels", [])
        elif isinstance(results, tuple) and len(results) == 4:
            # For GroundingDINO results returned as a tuple (boxes, labels, scores, text_labels)
            boxes, labels, scores, text_labels = results
            # If text_labels is empty but labels contains integer IDs (from FutureWarning), use generic labels
            if not text_labels and labels is not None and len(labels) > 0:
                text_labels = [f"object_{i}" for i in range(len(labels))]
        else:
            # For YOLO results or other formats
            # This case should not happen with current implementation
            # but kept for backward compatibility
            boxes, text_labels, scores = results
        
        # Check if text_labels are empty and handle accordingly
        if text_labels and any(label != "" for label in text_labels):
            # Use text_labels if they're not empty
            labels = text_labels
            # Create a mask for valid detections (all are valid since we have labels)
            valid_detections = [True] * len(boxes)
        else:
            # If text_labels are empty, handle according to empty_label_option
            print("Warning: Empty text_labels detected")
            
            if empty_label_option == "ignore":
                # Create a mask to filter out all detections with empty labels
                print("Ignoring detections with empty labels")
                valid_detections = [False] * len(boxes)
                labels = []
            elif empty_label_option == "unknown":
                # Label all detections as "unknown"
                print("Labeling detections with empty labels as 'unknown'")
                labels = ["unknown"] * len(boxes)
                valid_detections = [True] * len(boxes)
            else:  # "fallback" option - use intelligent fallback labeling
                print("Using intelligent fallback labeling for empty labels")
                valid_detections = [True] * len(boxes)
                if not categories:
                    # If no categories provided, use generic "object" label
                    labels = ["object"] * len(boxes)
                elif len(categories) == 1:
                    # If only one category, use it for all detections
                    labels = [categories[0]] * len(boxes)
                else:
                    # Distribute categories across detections
                    labels = []
                    for i in range(len(boxes)):
                        # Cycle through categories
                        category_index = i % len(categories)
                        labels.append(categories[category_index])
        
        # Apply the filter to keep only valid detections
        if not all(valid_detections):
            # Filter boxes, scores, and labels to keep only valid detections
            if isinstance(boxes, torch.Tensor):
                valid_tensor = torch.tensor(valid_detections, device=boxes.device)
                boxes = boxes[valid_tensor]
            else:
                boxes = np.array(boxes)[valid_detections]
            
            scores = [score for i, score in enumerate(scores) if valid_detections[i]]
            labels = [label for i, label in enumerate(labels) if valid_detections[i]]
        
        return labels, boxes, scores
    
    def filter_large_detections(self, labels, boxes, scores, image_width, image_height, 
                               max_width_ratio=0.9, max_height_ratio=0.9, max_area_ratio=0.8):
        """
        Filter out detections that are too large relative to the image dimensions.
        
        Args:
            labels: List of detection labels
            boxes: Tensor of bounding boxes in xyxy format
            scores: List of confidence scores
            image_width: Width of the image
            image_height: Height of the image
            max_width_ratio: Maximum allowed width ratio (default: 0.9)
            max_height_ratio: Maximum allowed height ratio (default: 0.9)
            max_area_ratio: Maximum allowed area ratio (default: 0.8)
            
        Returns:
            Tuple of (filtered_labels, filtered_boxes, filtered_scores)
        """
        import torch
        import numpy as np
        
        # Convert boxes to numpy for easier handling if it's a tensor
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = np.array(boxes)
        
        # Create a mask for valid detections
        valid_detections = []
        
        for box in boxes_np:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Calculate ratios
            width_ratio = width / image_width
            height_ratio = height / image_height
            area_ratio = area / (image_width * image_height)
            
            # Check if the detection is too large
            if (width_ratio > max_width_ratio or 
                height_ratio > max_height_ratio or 
                area_ratio > max_area_ratio):
                valid_detections.append(False)
                print(f"Filtering out large detection: width_ratio={width_ratio:.2f}, height_ratio={height_ratio:.2f}, area_ratio={area_ratio:.2f}")
            else:
                valid_detections.append(True)
        
        # Apply the filter
        if not all(valid_detections):
            # Convert valid_detections to a tensor for indexing if boxes is a tensor
            if isinstance(boxes, torch.Tensor):
                valid_tensor = torch.tensor(valid_detections, device=boxes.device)
                filtered_boxes = boxes[valid_tensor]
            else:
                filtered_boxes = boxes_np[valid_detections]
            
            filtered_scores = [score for i, score in enumerate(scores) if valid_detections[i]]
            filtered_labels = [label for i, label in enumerate(labels) if valid_detections[i]]
            
            print(f"Filtered out {len(boxes) - len(filtered_boxes)} large detections")
            return filtered_labels, filtered_boxes, filtered_scores
        
        return labels, boxes, scores
        
    def filter_standard_detections(self, detections, image_width, image_height, 
                                  max_width_ratio=0.9, max_height_ratio=0.9, max_area_ratio=0.8):
        """
        Filter out detections that are too large relative to the image dimensions.
        This method works with the standard detection format (list of dictionaries).
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'score', 'class_name', 'class_id'
            image_width: Width of the image
            image_height: Height of the image
            max_width_ratio: Maximum allowed width ratio (default: 0.9)
            max_height_ratio: Maximum allowed height ratio (default: 0.9)
            max_area_ratio: Maximum allowed area ratio (default: 0.8)
            
        Returns:
            List of filtered detection dictionaries
        """
        filtered_detections = []
        
        for det in detections:
            box = det['bbox']
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Calculate ratios
            width_ratio = width / image_width
            height_ratio = height / image_height
            area_ratio = area / (image_width * image_height)
            
            # Check if the detection is too large
            if (width_ratio > max_width_ratio or 
                height_ratio > max_height_ratio or 
                area_ratio > max_area_ratio):
                print(f"Filtering out large detection: width_ratio={width_ratio:.2f}, height_ratio={height_ratio:.2f}, area_ratio={area_ratio:.2f}")
            else:
                filtered_detections.append(det)
        
        if len(filtered_detections) < len(detections):
            print(f"Filtered out {len(detections) - len(filtered_detections)} large detections")
        
        return filtered_detections
        
def test_rtdetrv2():
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

    

if __name__ == "__main__":
    #huggingface-cli login
    #test_rtdetrv2()
    #base_model = ModelInference(model_type='rt-detr', model_name="PekingU/rtdetr_v2_r18vd", device='cuda')
    #base_model = ModelInference(model_type='detr', model_name="facebook/detr-resnet-50", device='cuda')
    base_model = ModelInference(model_type='yolo', model_name="/DATA10T/models/yolo11l.pt", device='cuda') #"yolo11n.pt" yolov8l
    results = base_model.predict(
            image_input="VisionLangAnnotateModels/sampledata/sjsupeople.jpg",
            visualize=True,
            save_path="./output/testsjsudetection.jpg"
        )
    
    base_model = ModelInference(model_type='yolo', model_name="/DATA10T/models/yolov11face/model.pt", device='cuda')
    results = base_model.predict(
            image_input="VisionLangAnnotateModels/sampledata/bus.jpg",
            visualize=True,
            save_path="./output/testface.jpg"
        )
    
    base_model = ModelInference(model_type='yolo', model_name="/DATA10T/models/yolov11licenseplate/license-plate-finetune-v1l.pt", device='cuda')
    results = base_model.predict(
            image_input="VisionLangAnnotateModels/sampledata/sanjoselicenseplate.jpg",
            visualize=True,
            save_path="./output/sanjoselicenseplate.jpg"
        )
    
    base_model = ModelInference(model_type='groundingdino', model_name="IDEA-Research/grounding-dino-base", device='cuda') #"yolo11n.pt" yolov8l
    results = base_model.predict(
            image_input="VisionLangAnnotateModels/sampledata/sjsupeople.jpg",
            text_prompt="person, car, dog",  # Comma-separated categories
            detection_model="groundingdino",
            visualize=True,
            save_path="./output/testsjsudetection_grounding.jpg"
        )
    
    results2 = base_model.perform_box_segmentation(
            image_input="VisionLangAnnotateModels/sampledata/sjsupeople.jpg",
            text_prompt="person, car, dog",
            detection_model="groundingdino",
            visualize=True,
            save_path="./output/testsjsusegmentation_grounding2.jpg"
        )

