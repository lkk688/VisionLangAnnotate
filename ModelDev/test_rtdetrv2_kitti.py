import os
import torch
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from modeling_yoloworld import YOLOWorldProcessor, DetectionDataset

def test_rtdetrv2_on_kitti(
    model_path,
    kitti_data_dir,
    output_dir="./results",
    batch_size=8,
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
    device=None
):
    """
    Evaluate a pretrained RT-DETRv2 model on KITTI dataset.
    
    Args:
        model_path: Path to the pretrained model
        kitti_data_dir: Path to KITTI dataset
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        img_size: Image size for evaluation
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections per image
        device: Device to run evaluation on (None for auto-detection)
        
    Returns:
        Dictionary with evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing RT-DETRv2 model...")
    processor = YOLOWorldProcessor.from_pretrained(
        model_path,
        device=device,
        model_type="rtdetrv2"
    )
    
    # Load KITTI validation dataset
    print("Loading KITTI validation dataset...")
    val_dataset = DetectionDataset(
        dataset_type='kitti',
        data_dir=kitti_data_dir,
        split='val',
        target_size=(img_size, img_size)
    )
    
    print(f"Loaded {len(val_dataset)} validation images")
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = processor.evaluate(
        val_dataset,
        batch_size=batch_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        output_dir=output_dir
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Number of images: {eval_results['images']}")
    print(f"True positives: {eval_results['total_tp']}")
    print(f"False positives: {eval_results['total_fp']}")
    print(f"False negatives: {eval_results['total_fn']}")
    print(f"Precision: {eval_results['overall_precision']:.4f}")
    print(f"Recall: {eval_results['overall_recall']:.4f}")
    print(f"F1 Score: {eval_results['overall_f1']:.4f}")
    print(f"mAP: {eval_results['mean_ap']:.4f}")
    
    # Run COCO evaluation
    print("\nRunning COCO evaluation...")
    coco_results = run_coco_evaluation(processor, val_dataset, output_dir)
    
    return {**eval_results, **coco_results}

def run_coco_evaluation(processor, dataset, output_dir):
    """
    Run COCO evaluation on the dataset.
    
    Args:
        processor: YOLOWorldProcessor instance
        dataset: Dataset to evaluate on
        output_dir: Directory to save results
        
    Returns:
        Dictionary with COCO evaluation results
    """
    # Set model to evaluation mode
    processor.model.eval()
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,  # Process one image at a time for simplicity
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=processor._collate_fn
    )
    
    # Initialize COCO format results
    coco_results = []
    
    # Process each image
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch_images = batch['img'].to(processor.device)
        image_id = batch['image_id']
        
        # Run inference
        with torch.no_grad():
            outputs = processor.model(
                pixel_values=batch_images,
                postprocess=True,
                conf_thres=0.001,  # Use low confidence threshold for COCO evaluation
                iou_thres=0.65,
                max_det=300
            )
        
        # Convert predictions to COCO format
        output = outputs[0]  # Single image batch
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        # Add predictions to COCO results
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            
            # COCO format uses [x, y, width, height]
            coco_results.append({
                'image_id': int(image_id) if isinstance(image_id, int) else int(image_id[0]),
                'category_id': int(labels[i]),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(scores[i])
            })
    
    # Save COCO results
    import json
    with open(os.path.join(output_dir, 'coco_results.json'), 'w') as f:
        json.dump(coco_results, f)
    
    # Create COCO ground truth from dataset
    coco_gt = create_coco_gt(dataset)
    
    # Initialize COCO evaluation
    coco_dt = coco_gt.loadRes(os.path.join(output_dir, 'coco_results.json'))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract results
    coco_results = {
        'coco_precision/mAP': coco_eval.stats[0],
        'coco_precision/mAP_50': coco_eval.stats[1],
        'coco_precision/mAP_75': coco_eval.stats[2],
        'coco_precision/mAP_small': coco_eval.stats[3],
        'coco_precision/mAP_medium': coco_eval.stats[4],
        'coco_precision/mAP_large': coco_eval.stats[5],
        'coco_recall/AR_1': coco_eval.stats[6],
        'coco_recall/AR_10': coco_eval.stats[7],
        'coco_recall/AR_100': coco_eval.stats[8],
        'coco_recall/AR_small': coco_eval.stats[9],
        'coco_recall/AR_medium': coco_eval.stats[10],
        'coco_recall/AR_large': coco_eval.stats[11]
    }
    
    # Print COCO results
    print("\nCOCO Evaluation Results:")
    print(f"mAP (IoU=0.50:0.95): {coco_results['coco_precision/mAP']:.4f}")
    print(f"mAP (IoU=0.50): {coco_results['coco_precision/mAP_50']:.4f}")
    print(f"mAP (IoU=0.75): {coco_results['coco_precision/mAP_75']:.4f}")
    print(f"mAP (small): {coco_results['coco_precision/mAP_small']:.4f}")
    print(f"mAP (medium): {coco_results['coco_precision/mAP_medium']:.4f}")
    print(f"mAP (large): {coco_results['coco_precision/mAP_large']:.4f}")
    
    return coco_results

def create_coco_gt(dataset):
    """
    Create COCO ground truth from dataset.
    
    Args:
        dataset: Dataset to create ground truth from
        
    Returns:
        COCO object with ground truth annotations
    """
    # If dataset has a get_coco_gt method, use it
    if hasattr(dataset, 'get_coco_gt'):
        return dataset.get_coco_gt()
    
    # Otherwise, create COCO ground truth from dataset
    import tempfile
    import json
    
    # Create COCO structure
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Add categories (COCO classes)
    for coco_id, name in dataset.coco_names.items():
        coco_dict['categories'].append({
            'id': int(coco_id),
            'name': name,
            'supercategory': 'none'
        })
    
    # Process each image in the dataset
    annotation_id = 0
    
    for idx in range(len(dataset)):
        # Get sample
        sample = dataset[idx]
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
            img_h, img_w = dataset.target_size
        
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
                label = labels[box_idx].item()
            else:
                label = labels[box_idx]
            
            # Add annotation
            coco_dict['annotations'].append({
                'id': annotation_id,
                'image_id': int(image_id),
                'category_id': int(label),
                'bbox': coco_box,
                'area': coco_box[2] * coco_box[3],
                'iscrowd': 0
            })
            
            annotation_id += 1
    
    # Save COCO dict to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_dict, f)
        temp_file = f.name
    
    # Load COCO object from file
    coco_gt = COCO(temp_file)
    
    # Delete temporary file
    os.unlink(temp_file)
    
    return coco_gt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETRv2 on KITTI dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--kitti_data_dir", type=str, required=True, help="Path to KITTI dataset")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum number of detections per image")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # Run evaluation
    test_rtdetrv2_on_kitti(
        model_path=args.model_path,
        kitti_data_dir=args.kitti_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        device=args.device
    )