import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from VisionLangAnnotateModels.detectors.base_model import BaseMultiModel

class ModelInference(BaseMultiModel):
    """
    Handles model inference operations including prediction and visualization.
    """
    
    def predict(self, image_path, conf_thres=0.25, iou_thres=0.45, max_det=300, 
                visualize=True, save_path=None, ground_truth=None, show_metrics=False):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image or PIL Image object
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            visualize: Whether to show visualization
            save_path: Path to save visualization
            ground_truth: Ground truth annotations for metrics calculation
            show_metrics: Whether to calculate and show metrics
            
        Returns:
            Dictionary with detections and optional metrics
        """
        self.model.eval()
        
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        original_size = image.size
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs based on model type
        detections = self._post_process_outputs(outputs, original_size, conf_thres, iou_thres, max_det)
        
        result = {
            'detections': detections,
            'image_size': original_size
        }
        
        # Calculate metrics if ground truth provided
        if ground_truth is not None and show_metrics:
            metrics = self._calculate_metrics(detections, ground_truth)
            result['metrics'] = metrics
        
        # Visualize if requested
        if visualize:
            self._visualize_detections(image, detections, save_path, ground_truth, 
                                     result.get('metrics'))
        
        return result
    
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
    
    def _visualize_detections(self, image, detections, save_path=None, ground_truth=None, metrics=None):
        """Visualize detections on the image."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
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
        
        plt.show()
    
    def batch_predict(self, image_paths, batch_size=8, **kwargs):
        """Run inference on multiple images."""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for image_path in batch_paths:
                result = self.predict(image_path, visualize=False, **kwargs)
                results.append(result)
        
        return results

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
    test_rtdetrv2()
    base_model = BaseMultiModel(model_type='rt-detr', model_name='rt-detrv2', device='cuda')
    inference = ModelInference(base_model)
    inference.predict(
            image_path="VisionLangAnnotateModels/sampledata/sjsupeople.jpg",
            visualize=True,
            save_path="./output"
        )