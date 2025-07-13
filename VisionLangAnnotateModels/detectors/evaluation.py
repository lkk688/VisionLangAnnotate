import json
import os
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torch.utils.data import DataLoader
from base_model import BaseMultiModel

class ModelEvaluator(BaseMultiModel):
    """
    Handles model evaluation on datasets with comprehensive metrics.
    """
    
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
        )
        
        self.print_category_id_stats()
        
        # Save detections to file
        if output_dir:
            detections_file = os.path.join(output_dir, "coco_detections.json")
            with open(detections_file, 'w') as f:
                json.dump(coco_results, f)
            print(f"Saved detections to {detections_file}")
        
        # Get or create ground truth COCO annotations
        if hasattr(dataset, '_coco') and dataset._coco is not None:
            print("Using dataset's existing COCO ground truth")
            coco_gt = dataset._coco
        elif hasattr(dataset, 'coco_gt') and dataset.coco_gt is not None:
            coco_gt = dataset.coco_gt
        else:
            print("Creating COCO ground truth from dataset")
            coco_gt = self._create_coco_gt_from_dataset(dataset, output_dir)
        
        # Run COCO evaluation if we have results
        if coco_results and len(coco_results) > 0:
            try:
                coco_dt = coco_gt.loadRes(coco_results)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                # Extract metrics
                metrics = {
                    'mAP': coco_eval.stats[0],
                    'mAP_50': coco_eval.stats[1],
                    'mAP_75': coco_eval.stats[2],
                    'mAP_small': coco_eval.stats[3],
                    'mAP_medium': coco_eval.stats[4],
                    'mAP_large': coco_eval.stats[5],
                    'mAR_1': coco_eval.stats[6],
                    'mAR_10': coco_eval.stats[7],
                    'mAR_100': coco_eval.stats[8],
                    'mAR_small': coco_eval.stats[9],
                    'mAR_medium': coco_eval.stats[10],
                    'mAR_large': coco_eval.stats[11]
                }
                
                if output_dir:
                    metrics_file = os.path.join(output_dir, "coco_metrics.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    print(f"Saved metrics to {metrics_file}")
                
                return metrics
                
            except Exception as e:
                print(f"Error during COCO evaluation: {e}")
                return self._run_custom_evaluation(dataset, coco_results, output_dir)
        else:
            print("No detections found, running custom evaluation")
            return self._run_custom_evaluation(dataset, coco_results, output_dir)
    
    def _run_inference_on_dataset(self, dataset, batch_size=16, conf_thres=0.25, 
                                  iou_thres=0.45, max_det=300, output_dir=None):
        """Run inference on entire dataset and return COCO-format results."""
        # For YOLO models, use self.yolo_model instead of self.model
        if self.model_type == 'yolo':
            if self.yolo_model is None:
                raise ValueError("YOLO model not loaded. Please call load_yolo_model() first.")
            # YOLO models from ultralytics don't need explicit eval() call
            # They handle it internally during inference
        else:
            # For other model types, use the standard self.model
            if self.model is None:
                raise ValueError("Model not initialized")
            self.model.eval()
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=4
        )
        
        coco_results = []
        
        print(f"Running inference on {len(dataset)} images...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(dataloader)}")
                
                # Move batch to device
                images = batch['pixel_values'].to(self.device)
                image_ids = batch['image_id']
                original_sizes = batch.get('original_size', None)
                
                # Run inference based on model type
                if self.model_type == 'yolo':
                    # For YOLO models, we need to convert the batch tensor to a list of images
                    # and run inference on each image
                    outputs = []
                    for i in range(len(images)):
                        # Convert tensor to PIL Image for YOLO
                        img = self._tensor_to_pil(images[i])
                        # Run YOLO inference
                        result = self.yolo_model(img, conf=conf_thres, iou=iou_thres, max_det=max_det)
                        outputs.append(result)
                else:
                    # Standard inference for other model types
                    outputs = self.model(pixel_values=images)
                
                # Process each image in the batch
                for i in range(len(images)):
                    image_id = image_ids[i].item() if torch.is_tensor(image_ids[i]) else image_ids[i]
                    
                    if original_sizes is not None:
                        orig_size = original_sizes[i]
                        if torch.is_tensor(orig_size):
                            orig_size = orig_size.cpu().numpy()
                    else:
                        orig_size = (640, 640)  # Default size
                    
                    # Extract single image outputs
                    single_output = self._extract_single_output(outputs, i)
                    
                    # Post-process to get detections
                    detections = self._post_process_single_output(
                        single_output, orig_size, conf_thres, iou_thres, max_det
                    )
                    
                    # Convert to COCO format
                    for det in detections:
                        coco_result = {
                            'image_id': int(image_id),
                            'category_id': int(det['class_id']),
                            'bbox': [float(x) for x in det['bbox']],  # [x, y, width, height]
                            'score': float(det['score'])
                        }
                        coco_results.append(coco_result)
        
        print(f"Generated {len(coco_results)} detections")
        return coco_results
    
    def _run_custom_evaluation(self, dataset, coco_results, output_dir=None):
        """Run custom evaluation when COCO evaluation fails."""
        print("Running custom evaluation...")
        
        # Initialize metrics
        metrics = {
            'total_images': len(dataset),
            'total_detections': len(coco_results),
            'per_category_metrics': {},
            'overall_metrics': {}
        }
        
        # Group detections by image
        detections_by_image = defaultdict(list)
        for det in coco_results:
            detections_by_image[det['image_id']].append(det)
        
        # Calculate per-image metrics
        total_tp = total_fp = total_fn = 0
        category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0})
        
        for idx in range(len(dataset)):
            try:
                # Get ground truth for this image
                sample = dataset[idx]
                image_id = sample.get('image_id', idx)
                
                if torch.is_tensor(image_id):
                    image_id = image_id.item()
                
                gt_boxes = sample.get('boxes', torch.tensor([]))
                gt_labels = sample.get('labels', torch.tensor([]))
                
                if torch.is_tensor(gt_boxes):
                    gt_boxes = gt_boxes.cpu().numpy()
                if torch.is_tensor(gt_labels):
                    gt_labels = gt_labels.cpu().numpy()
                
                # Get predictions for this image
                pred_detections = detections_by_image.get(image_id, [])
                
                # Calculate metrics for this image
                image_metrics = self._calculate_image_metrics(
                    gt_boxes, gt_labels, pred_detections
                )
                
                # Accumulate overall metrics
                total_tp += image_metrics['tp']
                total_fp += image_metrics['fp']
                total_fn += image_metrics['fn']
                
                # Accumulate per-category metrics
                for cat_id, cat_metrics in image_metrics['per_category'].items():
                    category_stats[cat_id]['tp'] += cat_metrics['tp']
                    category_stats[cat_id]['fp'] += cat_metrics['fp']
                    category_stats[cat_id]['fn'] += cat_metrics['fn']
                    category_stats[cat_id]['total_gt'] += cat_metrics['total_gt']
                    category_stats[cat_id]['total_pred'] += cat_metrics['total_pred']
                    
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall_metrics'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
        
        # Calculate per-category metrics
        for cat_id, stats in category_stats.items():
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            category_name = self.class_names.get(cat_id, f'class_{cat_id}')
            metrics['per_category_metrics'][category_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': stats['tp'],
                'fp': stats['fp'],
                'fn': stats['fn'],
                'total_gt': stats['total_gt'],
                'total_pred': stats['total_pred']
            }
        
        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1-Score: {overall_f1:.4f}")
        print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        
        # Save metrics
        if output_dir:
            metrics_file = os.path.join(output_dir, "custom_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved custom metrics to {metrics_file}")
        
        return metrics
    
    def _calculate_image_metrics(self, gt_boxes, gt_labels, pred_detections, iou_threshold=0.5):
        """Calculate metrics for a single image."""
        metrics = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'per_category': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0})
        }
        
        if len(gt_boxes) == 0 and len(pred_detections) == 0:
            return metrics
        
        # Count ground truth per category
        for label in gt_labels:
            metrics['per_category'][int(label)]['total_gt'] += 1
        
        # Count predictions per category
        for pred in pred_detections:
            metrics['per_category'][pred['category_id']]['total_pred'] += 1
        
        if len(gt_boxes) == 0:
            # No ground truth, all predictions are false positives
            metrics['fp'] = len(pred_detections)
            for pred in pred_detections:
                metrics['per_category'][pred['category_id']]['fp'] += 1
            return metrics
        
        if len(pred_detections) == 0:
            # No predictions, all ground truth are false negatives
            metrics['fn'] = len(gt_boxes)
            for label in gt_labels:
                metrics['per_category'][int(label)]['fn'] += 1
            return metrics
        
        # Convert prediction bboxes to proper format
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        
        for pred in pred_detections:
            bbox = pred['bbox']
            # Convert from [x, y, w, h] to [x1, y1, x2, y2] if needed
            if len(bbox) == 4:
                x, y, w, h = bbox
                pred_boxes.append([x, y, x + w, y + h])
            else:
                pred_boxes.append(bbox)
            pred_labels.append(pred['category_id'])
            pred_scores.append(pred['score'])
        
        pred_boxes = np.array(pred_boxes)
        pred_labels = np.array(pred_labels)
        pred_scores = np.array(pred_scores)
        
        # Convert gt_boxes to [x1, y1, x2, y2] format if needed
        if len(gt_boxes) > 0 and len(gt_boxes[0]) == 4:
            # Assume gt_boxes might be in different format, convert to [x1, y1, x2, y2]
            gt_boxes_converted = []
            for box in gt_boxes:
                if len(box) == 4:
                    # Could be [x, y, w, h] or [x1, y1, x2, y2]
                    # Check if it looks like [x, y, w, h] (w and h should be positive and reasonable)
                    x, y, w_or_x2, h_or_y2 = box
                    if w_or_x2 > x and h_or_y2 > y and w_or_x2 < 2000 and h_or_y2 < 2000:
                        # Likely [x1, y1, x2, y2]
                        gt_boxes_converted.append([x, y, w_or_x2, h_or_y2])
                    else:
                        # Likely [x, y, w, h]
                        gt_boxes_converted.append([x, y, x + w_or_x2, y + h_or_y2])
                else:
                    gt_boxes_converted.append(box)
            gt_boxes = np.array(gt_boxes_converted)
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(gt_boxes, pred_boxes)
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for pred_idx in sorted_indices:
            pred_label = pred_labels[pred_idx]
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                    
                gt_label = gt_labels[gt_idx]
                if pred_label != gt_label:
                    continue
                    
                iou = iou_matrix[gt_idx, pred_idx]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                # True positive
                metrics['tp'] += 1
                metrics['per_category'][pred_label]['tp'] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
            else:
                # False positive
                metrics['fp'] += 1
                metrics['per_category'][pred_label]['fp'] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                metrics['fn'] += 1
                gt_label = int(gt_labels[gt_idx])
                metrics['per_category'][gt_label]['fn'] += 1
        
        return metrics
    
    def _calculate_iou_matrix(self, boxes1, boxes2):
        """Calculate IoU matrix between two sets of boxes."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        # Ensure boxes are in [x1, y1, x2, y2] format
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        # Calculate intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1[:, None] + area2[None, :] - intersection
        
        # Calculate IoU
        iou = intersection / np.maximum(union, 1e-8)
        
        return iou
    
    def print_category_id_stats(self):
        """Print statistics about category IDs in the dataset."""
        if self.dataset is None:
            print("No dataset loaded")
            return
        
        print("\nCategory ID Statistics:")
        print(f"Model has {len(self.class_names)} classes")
        print(f"Class names: {list(self.class_names.values())[:10]}...")  # Show first 10
        
        # Count category occurrences in dataset
        category_counts = defaultdict(int)
        
        try:
            for idx in range(min(100, len(self.dataset))):  # Sample first 100 images
                sample = self.dataset[idx]
                labels = sample.get('labels', [])
                
                if torch.is_tensor(labels):
                    labels = labels.cpu().numpy()
                
                for label in labels:
                    category_counts[int(label)] += 1
            
            print(f"\nCategory distribution (first 100 images):")
            for cat_id, count in sorted(category_counts.items()):
                cat_name = self.class_names.get(cat_id, f'unknown_{cat_id}')
                print(f"  {cat_id}: {cat_name} ({count} instances)")
                
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Handle different batch formats
        if isinstance(batch[0], dict):
            # Dictionary format
            keys = batch[0].keys()
            collated = {}
            
            for key in keys:
                if key == 'pixel_values':
                    # Stack images
                    collated[key] = torch.stack([item[key] for item in batch])
                elif key in ['image_id', 'labels', 'boxes']:
                    # Keep as list for variable length data
                    collated[key] = [item[key] for item in batch]
                else:
                    # Try to stack, fallback to list
                    try:
                        collated[key] = torch.stack([item[key] for item in batch])
                    except:
                        collated[key] = [item[key] for item in batch]
            
            return collated
        else:
            # Tuple format (image, target)
            images, targets = zip(*batch)
            
            # Process images
            if torch.is_tensor(images[0]):
                pixel_values = torch.stack(images)
            else:
                # Convert PIL images to tensors
                pixel_values = torch.stack([self.processor(img, return_tensors="pt")['pixel_values'][0] for img in images])
            
            # Extract information from targets
            image_ids = []
            labels_list = []
            boxes_list = []
            
            for i, target in enumerate(targets):
                if isinstance(target, dict):
                    image_ids.append(target.get('image_id', i))
                    labels_list.append(target.get('labels', torch.tensor([])))
                    boxes_list.append(target.get('boxes', torch.tensor([])))
                else:
                    image_ids.append(i)
                    labels_list.append(torch.tensor([]))
                    boxes_list.append(torch.tensor([]))
            
            return {
                'pixel_values': pixel_values,
                'image_id': image_ids,
                'labels': labels_list,
                'boxes': boxes_list
            }
    
    def _tensor_to_pil(self, tensor):
        """Convert a PyTorch tensor to a PIL Image for YOLO inference.
        
        Args:
            tensor: PyTorch tensor in (C, H, W) format
            
        Returns:
            PIL Image
        """
        # Convert tensor to numpy array
        if tensor.dim() == 4:  # (B, C, H, W)
            tensor = tensor[0]  # Take the first image in the batch
            
        # Convert to numpy and transpose from (C, H, W) to (H, W, C)
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize if needed (assuming values are in [0, 1])
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to PIL Image
        from PIL import Image
        return Image.fromarray(img_np)
    
    def _extract_single_output(self, outputs, index):
        """Extract outputs for a single image from batch outputs."""
        # For YOLO models, outputs is a list of Results objects
        if self.model_type == 'yolo':
            return outputs[index]
        
        # For other models, extract the tensors for the specific index
        single_output = {}
        
        for key, value in outputs.items():
            if torch.is_tensor(value) and len(value.shape) > 0:
                single_output[key] = value[index:index+1]  # Keep batch dimension
            else:
                single_output[key] = value
        
        return single_output
    
    def _post_process_single_output(self, outputs, original_size, conf_thres, iou_thres, max_det):
        """Post-process outputs for a single image."""
        # For YOLO models, outputs is a Results object from ultralytics
        if self.model_type == 'yolo':
            detections = []
            # Check if there are any detections
            if len(outputs.boxes) > 0:
                boxes = outputs.boxes.xyxy.cpu().numpy()
                scores = outputs.boxes.conf.cpu().numpy()
                class_ids = outputs.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    class_id = int(class_ids[i])
                    class_name = outputs.names[class_id] if class_id in outputs.names else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': boxes[i].tolist(),
                        'score': float(scores[i]),
                        'class_id': class_id,
                        'class_name': class_name
                    })
            return detections
        
        # For other models, use the standard post-processing
        # This would use the same logic as in the inference module
        # For now, return empty list as placeholder for non-YOLO models
        return []
    
    def _create_coco_gt_from_dataset(self, dataset, output_dir=None):
        """Create COCO ground truth object from dataset."""
        # This would create a COCO object from the dataset annotations
        # Implementation depends on the dataset format
        pass