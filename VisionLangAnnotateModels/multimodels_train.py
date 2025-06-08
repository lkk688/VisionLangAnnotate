import argparse
import os
import json
import glob
import os
import random
import cv2
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from multidatasets import DetectionDataset, coco_names
from multimodels import MultiModels
from torch.utils.data import random_split


class MixedDetectionDataset(Dataset):
    """
    Dataset class for mixing multiple detection datasets with different weights.
    Handles class mapping to standardize on COCO's 80 classes and supports
    masking unused classes during training.
    """
    def __init__(self, datasets, weights=None, num_classes=80, augment=True):
        """
        Initialize the mixed detection dataset.
        
        Args:
            datasets: List of DetectionDataset objects to mix
            weights: List of weights for each dataset (if None, equal weights are used)
            num_classes: Number of classes to use (fixed to 80 for COCO compatibility)
            augment: Whether to apply data augmentation
        """
        self.datasets = datasets
        self.num_datasets = len(datasets)
        
        # Validate datasets
        if self.num_datasets == 0:
            raise ValueError("No datasets provided for mixing")
        
        # Set dataset weights
        if weights is None:
            self.weights = [1.0 / self.num_datasets] * self.num_datasets
        else:
            if len(weights) != self.num_datasets:
                raise ValueError("Number of weights must match number of datasets")
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # Calculate cumulative weights for sampling
        self.cumulative_weights = [sum(self.weights[:i+1]) for i in range(self.num_datasets)]
        
        # Calculate dataset sizes
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.total_size = sum(self.dataset_sizes)
        
        # Store other parameters
        self.num_classes = num_classes
        self.augment = augment
        
        # Create class mapping for each dataset
        self.class_maps = []
        for dataset in datasets:
            if hasattr(dataset, 'class_map'):
                self.class_maps.append(dataset.class_map)
            else:
                # Default identity mapping
                self.class_maps.append({i: i for i in range(num_classes)})
        
        # Create active classes mask (classes that are present in at least one dataset)
        self.active_classes = set()
        for class_map in self.class_maps:
            self.active_classes.update(class_map.values())
        # Remove ignored classes (-1)
        if -1 in self.active_classes:
            self.active_classes.remove(-1)
        
        # Convert to list and sort
        self.active_classes = sorted(list(self.active_classes))
        
        print(f"Created MixedDetectionDataset with {self.num_datasets} datasets:")
        for i, (dataset, weight) in enumerate(zip(datasets, self.weights)):
            print(f"  Dataset {i+1}: {dataset.dataset_type}, size={len(dataset)}, weight={weight:.4f}")
        print(f"Total samples: {self.total_size}")
        print(f"Active classes: {len(self.active_classes)}/{num_classes}")
    
    def __len__(self):
        """
        Get the length of the mixed dataset.
        
        Returns:
            Total number of samples across all datasets
        """
        return self.total_size
    
    def __getitem__(self, idx):
        """
        Get a sample from the mixed dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and target
        """
        # Map global index to dataset and local index
        dataset_idx, local_idx = self._get_dataset_and_index(idx)
        
        # Get sample from the selected dataset
        sample = self.datasets[dataset_idx][local_idx]
        
        # Apply class masking to indicate which classes should be trained
        sample = self._apply_class_masking(sample)
        
        # Apply additional augmentation if needed
        if self.augment:
            sample = self._apply_augmentation(sample)
        
        return sample
    
    def _get_dataset_and_index(self, idx):
        """
        Map global index to dataset index and local index.
        
        Args:
            idx: Global sample index
            
        Returns:
            Tuple of (dataset_index, local_index)
        """
        # Option 1: Sequential sampling (go through each dataset in order)
        # This preserves the original order of samples within each dataset
        offset = 0
        for dataset_idx, size in enumerate(self.dataset_sizes):
            if idx < offset + size:
                return dataset_idx, idx - offset
            offset += size
        
        # Should never reach here
        raise IndexError(f"Index {idx} out of range for mixed dataset of size {self.total_size}")
    
    def get_weighted_sample(self):
        """
        Get a sample based on dataset weights.
        
        Returns:
            Dictionary with image and target
        """
        # Generate random number between 0 and 1
        r = random.random()
        
        # Find the dataset to sample from based on cumulative weights
        for dataset_idx, cum_weight in enumerate(self.cumulative_weights):
            if r <= cum_weight:
                # Sample from this dataset
                local_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
                sample = self.datasets[dataset_idx][local_idx]
                
                # Apply class masking
                sample = self._apply_class_masking(sample)
                
                # Apply augmentation if needed
                if self.augment:
                    sample = self._apply_augmentation(sample)
                
                return sample
        
        # Should never reach here
        return self.datasets[-1][0]
    
    def _apply_class_masking(self, sample):
        """
        Apply class masking to indicate which classes should be trained.
        
        Args:
            sample: Sample dictionary with image and target
            
        Returns:
            Updated sample with class masking
        """
        # Get target
        target = sample['target']
        
        # Add active classes mask to target
        target['active_classes'] = torch.zeros(self.num_classes, dtype=torch.bool)
        for class_idx in self.active_classes:
            if 0 <= class_idx < self.num_classes:
                target['active_classes'][class_idx] = True
        
        # Return updated sample
        return sample
    
    def _apply_augmentation(self, sample):
        """
        Apply additional augmentation to the sample.
        
        Args:
            sample: Sample dictionary with image and target
            
        Returns:
            Augmented sample
        """
        # Simple augmentation example - can be expanded with more complex transforms
        img = sample['img']
        target = sample['target']
        
        # Apply random horizontal flip with 50% probability
        if random.random() > 0.5:
            # Flip image
            if isinstance(img, torch.Tensor):
                img = torch.flip(img, dims=[-1])  # Flip along width dimension
            else:
                img = np.fliplr(img)
            
            # Flip boxes
            if len(target['boxes']) > 0:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    width = img.shape[-1] if isinstance(img, torch.Tensor) else img.shape[1]
                    # Flip x-coordinates: x_new = width - x_old
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target['boxes'] = boxes
        
        # Update sample with augmented data
        sample['img'] = img
        sample['target'] = target
        
        return sample
    
    def get_coco_gt(self):
        """
        Get COCO ground truth object for evaluation.
        
        Returns:
            COCO object with ground truth annotations
        """
        # Convert to COCO format
        return self.convert_to_coco_format()
    
    def convert_to_coco_format(self):
        """
        Convert mixed dataset to COCO format for evaluation.
        
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
        for coco_id, name in coco_names.items():
            coco_dict['categories'].append({
                'id': int(coco_id),
                'name': name,
                'supercategory': 'none'
            })
        
        # Process each dataset
        annotation_id = 0
        image_id_mapping = {}  # To ensure unique image IDs
        
        for dataset_idx, dataset in enumerate(self.datasets):
            # Get COCO ground truth if available
            if hasattr(dataset, 'get_coco_gt') and callable(dataset.get_coco_gt):
                dataset_coco = dataset.get_coco_gt()
                
                # Add images
                for img_id in dataset_coco.imgs:
                    img_info = dataset_coco.imgs[img_id]
                    
                    # Create unique image ID
                    unique_img_id = f"{dataset_idx}_{img_id}"
                    image_id_mapping[img_id] = unique_img_id
                    
                    # Add image entry
                    coco_dict['images'].append({
                        'id': unique_img_id,
                        'width': img_info['width'],
                        'height': img_info['height'],
                        'file_name': img_info['file_name'],
                        'dataset_idx': dataset_idx
                    })
                
                # Add annotations
                for ann_id in dataset_coco.anns:
                    ann = dataset_coco.anns[ann_id]
                    
                    # Map image ID
                    unique_img_id = image_id_mapping[ann['image_id']]
                    
                    # Map category ID to COCO class ID
                    cat_id = ann['category_id']
                    if hasattr(dataset, 'cat_id_to_coco_id'):
                        coco_class_id = dataset.cat_id_to_coco_id.get(cat_id, -1)
                    else:
                        coco_class_id = cat_id
                    
                    # Skip ignored classes
                    if coco_class_id == -1:
                        continue
                    
                    # Add annotation
                    coco_dict['annotations'].append({
                        'id': annotation_id,
                        'image_id': unique_img_id,
                        'category_id': coco_class_id,
                        'bbox': ann['bbox'],
                        'area': ann['area'],
                        'iscrowd': ann['iscrowd'],
                        'segmentation': ann.get('segmentation', [])
                    })
                    
                    annotation_id += 1
        
        # Create COCO object from dictionary
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(coco_dict, f)
            f.flush()
            coco = COCO(f.name)
        
        return coco

def test_trainmultimodels():
    """
    Test function for training MultiModels on mixed datasets.
    Supports training YOLOv8 or HuggingFace models on a combination of COCO and KITTI datasets.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MultiModels on mixed datasets')
    parser.add_argument('--model', type=str, default='yolov8', help='Model type (yolov8, detr, rt-detr, rt-detrv2, vitdet)')
    parser.add_argument('--weights', type=str, default="", help='Path to pretrained weights (optional)')
    parser.add_argument('--hub_model', type=str, default="lkk688/yolov8s-model", help='HF Hub model name (e.g., "facebook/detr-resnet-50")')
    parser.add_argument('--output_dir', type=str, default="output/train", help='Output directory for training results')
    
    # Dataset arguments
    parser.add_argument('--use_coco', action='store_true', default=False, help='Use COCO dataset for training')
    parser.add_argument('--coco_dir', type=str, default="/DATA10T/Datasets/COCO", help='COCO dataset directory')
    parser.add_argument('--coco_weight', type=float, default=1.0, help='Weight for COCO dataset in mixed training')
    
    parser.add_argument('--use_kitti', action='store_true', default=True, help='Use KITTI dataset for training')
    parser.add_argument('--kitti_dir', type=str, default="/DATA10T/Datasets/Kitti", help='KITTI dataset directory')
    parser.add_argument('--kitti_weight', type=float, default=1.0, help='Weight for KITTI dataset in mixed training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type (sgd, adam, adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler type (cosine, step, linear, constant)')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='Use mixed precision training')
    parser.add_argument('--resume', type=str, default="", help='Resume training from checkpoint')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_coco and not args.use_kitti:
        print("Error: At least one dataset (COCO or KITTI) must be selected for training.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing MultiModels with model type: {args.model}")
    if args.hub_model:
        # Load model from Hugging Face Hub
        print(f"Loading model from Hugging Face Hub: {args.hub_model}")
        model = MultiModels(model_name=args.hub_model)
    else:
        # Create model and load weights if provided
        model = MultiModels(model_type=args.model, scale='s')
        if args.weights and os.path.exists(args.weights):
            print(f"Loading pretrained weights from: {args.weights}")
            model.load_weights(args.weights)
    
    # Create datasets
    datasets = []
    dataset_weights = []
    
    # Class mapping dictionaries
    # COCO already uses 80 classes (0-79)
    # KITTI needs mapping to COCO classes
    kitti_to_coco = {
        'Car': 2,         # car in COCO
        'Van': 2,         # car in COCO
        'Truck': 7,       # truck in COCO
        'Pedestrian': 0,  # person in COCO
        'Person_sitting': 0,  # person in COCO
        'Cyclist': 1,     # bicycle in COCO
        'Tram': 6,        # train in COCO
        'Misc': -1,       # ignored
        'DontCare': -1    # ignored
    }
    
    # Add COCO dataset if selected
    if args.use_coco:
        print(f"Creating COCO dataset from: {args.coco_dir}")
        coco_dataset = DetectionDataset(
            dataset_type='coco',
            data_dir=args.coco_dir,
            split='train',
            target_size=(args.img_size, args.img_size),
            augment=False
        )
        datasets.append(coco_dataset)
        dataset_weights.append(args.coco_weight)
    
    # Add KITTI dataset if selected
    if args.use_kitti:
        print(f"Creating KITTI dataset from: {args.kitti_dir}")
        kitti_dataset = DetectionDataset(
            dataset_type='kitti',
            data_dir=args.kitti_dir,
            split='train',
            target_size=(args.img_size, args.img_size),
            augment=False,
            class_map=kitti_to_coco  # Apply class mapping
        )
        datasets.append(kitti_dataset)
        dataset_weights.append(args.kitti_weight)
    
    # Create mixed dataset
    if len(datasets) > 1:
        print("Creating mixed dataset with the following weights:")
        for i, (dataset, weight) in enumerate(zip(datasets, dataset_weights)):
            print(f"  Dataset {i+1}: weight = {weight}")
        
        mixed_dataset = MixedDetectionDataset(
            datasets=datasets,
            weights=dataset_weights,
            num_classes=80  # Fixed to 80 COCO classes
        )
    else:
        # Only one dataset, no need for mixing
        mixed_dataset = datasets[0]
    
    # Create validation dataset
    if args.val_split > 0:
        # Split the dataset into training and validation
        dataset_size = len(mixed_dataset)
        val_size = int(dataset_size * args.val_split)
        train_size = dataset_size - val_size
        
        # Use PyTorch's random_split
        train_dataset, val_dataset = random_split(
            mixed_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    else:
        train_dataset = mixed_dataset
        val_dataset = None
    
    # Train the model
    print(f"\n--- Starting training for {args.epochs} epochs ---")
    training_results = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        resume=args.resume,
        save_interval=max(1, args.epochs // 10),  # Save approximately 10 checkpoints
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision
    )
    
    # Print training summary
    print("\nTraining completed!")
    print(f"Final model saved to: {args.output_dir}/last.pt")
    if val_dataset:
        print(f"Best model saved to: {args.output_dir}/best.pt")
    
    # Optionally evaluate the trained model on test sets
    if args.use_coco:
        print("\nEvaluating on COCO validation set...")
        coco_val_dataset = DetectionDataset(
            dataset_type='coco',
            data_dir=args.coco_dir,
            split='val',
            target_size=(args.img_size, args.img_size)
        )
        
        coco_results = model.evaluate_coco(
            dataset=coco_val_dataset,
            output_dir=os.path.join(args.output_dir, "coco_eval"),
            batch_size=args.batch_size,
            conf_thres=0.25,
            iou_thres=0.45
        )
        
        print("\nCOCO Evaluation Results:")
        if isinstance(coco_results, dict):
            for metric, value in coco_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    if args.use_kitti:
        print("\nEvaluating on KITTI validation set...")
        kitti_val_dataset = DetectionDataset(
            dataset_type='kitti',
            data_dir=args.kitti_dir,
            split='val',
            target_size=(args.img_size, args.img_size)
        )
        
        kitti_results = model.evaluate_kitti(
            kitti_dataset=kitti_val_dataset,
            output_dir=os.path.join(args.output_dir, "kitti_eval"),
            batch_size=args.batch_size,
            conf_thres=0.25,
            iou_thres=0.45
        )
        
        print("\nKITTI Evaluation Results:")
        if isinstance(kitti_results, dict):
            for metric, value in kitti_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

if __name__ == "__main__":
    # test_multimodels()
    test_trainmultimodels()