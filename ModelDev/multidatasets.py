"""Multiple Datasets"""
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

# Set up COCO category ID mapping (model index -> COCO ID)
original_coco_id_mapping = {
    0: 1,   # person
    1: 2,   # bicycle
    2: 3,   # car
    3: 4,   # motorcycle
    4: 5,   # airplane
    5: 6,   # bus
    6: 7,   # train
    7: 8,   # truck
    8: 9,   # boat
    9: 10,  # traffic light
    10: 11, # fire hydrant
    11: 13, # stop sign
    12: 14, # parking meter
    13: 15, # bench
    14: 16, # bird
    15: 17, # cat
    16: 18, # dog
    17: 19, # horse
    18: 20, # sheep
    19: 21, # cow
    20: 22, # elephant
    21: 23, # bear
    22: 24, # zebra
    23: 25, # giraffe
    24: 27, # backpack
    25: 28, # umbrella
    26: 31, # handbag
    27: 32, # tie
    28: 33, # suitcase
    29: 34, # frisbee
    30: 35, # skis
    31: 36, # snowboard
    32: 37, # sports ball
    33: 38, # kite
    34: 39, # baseball bat
    35: 40, # baseball glove
    36: 41, # skateboard
    37: 42, # surfboard
    38: 43, # tennis racket
    39: 44, # bottle
    40: 46, # wine glass
    41: 47, # cup
    42: 48, # fork
    43: 49, # knife
    44: 50, # spoon
    45: 51, # bowl
    46: 52, # banana
    47: 53, # apple
    48: 54, # sandwich
    49: 55, # orange
    50: 56, # broccoli
    51: 57, # carrot
    52: 58, # hot dog
    53: 59, # pizza
    54: 60, # donut
    55: 61, # cake
    56: 62, # chair
    57: 63, # couch
    58: 64, # potted plant
    59: 65, # bed
    60: 67, # dining table
    61: 70, # toilet
    62: 72, # tv
    63: 73, # laptop
    64: 74, # mouse
    65: 75, # remote
    66: 76, # keyboard
    67: 77, # cell phone
    68: 78, # microwave
    69: 79, # oven
    70: 80, # toaster
    71: 81, # sink
    72: 82, # refrigerator
    73: 84, # book
    74: 85, # clock
    75: 86, # vase
    76: 87, # scissors
    77: 88, # teddy bear
    78: 89, # hair drier
    79: 90  # toothbrush
}

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

def convertpred2kitti(img_id, pred_boxes, pred_labels, pred_scores, threshold=0.5, output_dir="output/kittiformat"):
    #pred_boxes format is xyxy
    # Map COCO class ID to KITTI class name
    id_to_kittiname = {
        0: 'Pedestrian',  # person
        1: 'Cyclist',     # bicycle
        2: 'Car',         # car
        3: 'Cyclist',     # motorcycle
        5: 'Car',         # bus
        7: 'Truck',       # truck
        9: 'Misc'         # traffic light
    }

    # Convert each detection to KITTI format
    image_results = []
    
    for box_idx in range(len(pred_boxes)):
        x1, y1, x2, y2 = pred_boxes[box_idx]
        # Get category ID and map to KITTI class
        label = pred_labels[box_idx]
        category_id = int(label)
    
        kitti_class = id_to_kittiname.get(category_id, 'DontCare')
        
        # Skip classes that don't map to KITTI
        if kitti_class == 'DontCare' and pred_scores[box_idx] < threshold:
            continue
        
        # Format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y score
        kitti_line = f"{kitti_class} 0.0 0 0.0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 {pred_scores[box_idx]:.6f}"
        image_results.append(kitti_line)
    
    # Save detections to file if output directory is provided
    if output_dir:
        # Use original ID for file naming
        result_file = os.path.join(output_dir, f"{img_id}.txt")
        with open(result_file, 'w') as f:
            for line in image_results:
                f.write(line + '\n')

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
                 preprocess=True,
                 target_size=(640, 640),
                 cache_images=False,
                 class_map=None,
                 augment=False):
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
            augment: Whether to apply data augmentation
        """
        self.dataset_type = dataset_type.lower()
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.split = split
        self.transforms = transforms #perform transform to the images
        self.preprocess = preprocess
        self.target_size = target_size
        self.cache_images = cache_images
        self.augment = augment
        # Initialize class mapping
        self.class_map = class_map or self._get_default_class_map(dataset_type)     
        # Initialize COCO class names (standard 80 classes)
        self.coco_names = coco_names
        
        # Initialize COCO attribute in __init__
        self._coco = None
        self.image_ids = []
        self.image_paths = []
        # Store image paths for each image ID
        self.image_id_to_path = {}
        # Create category ID to COCO class ID mapping
        self.cat_id_to_coco_id = {}
        
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
        self._coco = COCO(self.annotation_file)
        
        # Get image IDs
        self.image_ids = list(sorted(self._coco.imgs.keys()))
        
        # Create category ID to COCO class ID mapping
        self.cat_id_to_coco_id = {}
        for cat_id in self._coco.cats.keys():
            # Map COCO category ID to standard COCO class ID (0-79)
            # This is needed because COCO category IDs are not sequential
            cat_name = self._coco.cats[cat_id]['name']
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
    
    def _apply_augmentation(self, img, target):
        """
        Apply data augmentation for object detection, optimized for autonomous driving scenarios.
        
        Args:
            img: Input image (numpy array in BGR or RGB format)
            target: Target dictionary with boxes, labels, etc.
            
        Returns:
            Tuple of (augmented_image, augmented_target)
        """
        # Skip augmentation if not enabled
        if not self.augment:
            return img, target
        # Make a copy of the target to avoid modifying the original
        target = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() if isinstance(v, np.ndarray) else v 
                 for k, v in target.items()}
        
        # Get image dimensions
        height, width = img.shape[:2]
        boxes = target['boxes']
        
        # Skip augmentation if no boxes
        if len(boxes) == 0:
            return img, target
        
        # Convert boxes to numpy if they are tensors
        is_tensor = isinstance(boxes, torch.Tensor)
        if is_tensor:
            boxes_np = boxes.numpy()
        else:
            boxes_np = boxes
            
        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)  # 1 for horizontal flip
            
            # Flip bounding boxes
            if 'boxes' in target:
                width = img.shape[1]
                boxes = target['boxes'].copy()
                boxes[:, 0] = width - boxes[:, 0] - boxes[:, 2]  # Flip x coordinates
                target['boxes'] = boxes
        
        # Random brightness and contrast adjustment
        if random.random() < 0.5:
            # Ensure img is not None and has valid shape
            if img is None or img.size == 0:
                return img, target
                
            # Safer contrast adjustment
            try:
                # Get a random contrast factor between 0.5 and 1.5
                contrast_factor = random.uniform(0.5, 1.5)
                
                # Calculate mean value for contrast adjustment
                mean_val = np.mean(img)
                
                # Apply contrast adjustment with error handling
                if mean_val > 0 and contrast_factor > 0:
                    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=(1-contrast_factor)*mean_val)
                else:
                    # Skip contrast adjustment if parameters are invalid
                    pass
            except Exception as e:
                print(f"Warning: Skipping contrast adjustment due to error: {e}")
                # Continue with original image
                
        # 3. Random saturation and hue adjustment (20% probability)
        if random.random() < 0.2:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Adjust saturation
            saturation_factor = random.uniform(0.8, 1.2)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_factor, 0, 255)
            
            # Adjust hue
            hue_factor = random.uniform(-10, 10)
            img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + hue_factor, 0, 179)
            
            # Convert back to RGB
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # 4. Random shadow (15% probability) - simulates shadows on the road
        if random.random() < 0.15:
            # Create a random shadow mask
            shadow_factor = random.uniform(0.6, 0.9)
            
            # Choose random points for shadow polygon
            x1, x2 = random.randint(0, width), random.randint(0, width)
            y1, y2 = 0, height
            
            # Create mask
            mask = np.zeros_like(img[:, :, 0])
            shadow_points = np.array([[(x1, y1), (x2, y2), (width, y2), (width, y1)]], dtype=np.int32)
            cv2.fillPoly(mask, shadow_points, 255)
            
            # Apply shadow
            for c in range(3):
                img[:, :, c] = np.where(mask == 255, 
                                        img[:, :, c] * shadow_factor, 
                                        img[:, :, c])
        
        # 5. Random noise (10% probability)
        if random.random() < 0.1:
            noise_factor = random.uniform(5, 20)
            noise = np.random.randn(*img.shape) * noise_factor
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # 6. Random scaling (20% probability)
        if random.random() < 0.2:
            scale_factor = random.uniform(0.8, 1.2)
            
            # Compute new dimensions
            new_height, new_width = int(height * scale_factor), int(width * scale_factor)
            
            # Resize image
            img = cv2.resize(img, (new_width, new_height))
            
            # Scale boxes
            if len(boxes_np) > 0:
                boxes_np[:, [0, 2]] *= (new_width / width)
                boxes_np[:, [1, 3]] *= (new_height / height)
            
            # Update dimensions
            height, width = new_height, new_width
        
        # 7. Random translation (30% probability)
        if random.random() < 0.3:
            # Maximum translation is 10% of image dimensions
            max_dx = width * 0.1
            max_dy = height * 0.1
            
            dx = random.uniform(-max_dx, max_dx)
            dy = random.uniform(-max_dy, max_dy)
            
            # Create transformation matrix
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # Apply translation to image
            img = cv2.warpAffine(img, M, (width, height))
            
            # Translate boxes
            if len(boxes_np) > 0:
                boxes_np[:, [0, 2]] += dx
                boxes_np[:, [1, 3]] += dy
                
                # Clip boxes to image boundaries
                boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, width)
                boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, height)
        
        # 8. Random rotation (10% probability) - small angles only for driving scenes
        if random.random() < 0.1:
            # Small rotation angles for driving scenes
            angle = random.uniform(-5, 5)
            
            # Get rotation matrix
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to image
            img = cv2.warpAffine(img, M, (width, height))
            
            # Rotate boxes - this is an approximation for small angles
            if len(boxes_np) > 0:
                # Convert boxes to corners
                corners = np.zeros((len(boxes_np), 4, 2))
                for i, box in enumerate(boxes_np):
                    x1, y1, x2, y2 = box
                    corners[i] = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                
                # Rotate corners
                corners = corners.reshape(-1, 2)
                corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
                corners = np.matmul(corners, M.T).reshape(-1, 4, 2)
                
                # Get new bounding boxes from rotated corners
                for i, corner in enumerate(corners):
                    x_min = np.min(corner[:, 0])
                    y_min = np.min(corner[:, 1])
                    x_max = np.max(corner[:, 0])
                    y_max = np.max(corner[:, 1])
                    
                    boxes_np[i] = [x_min, y_min, x_max, y_max]
                
                # Clip boxes to image boundaries
                boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, width)
                boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, height)
        
        # 9. Weather simulation (20% probability)
        if random.random() < 0.2:
            weather_type = random.choice(['rain', 'fog', 'snow'])
            
            if weather_type == 'rain':
                # Simulate rain
                rain_drops = random.randint(500, 1000)
                for _ in range(rain_drops):
                    x = random.randint(0, width-1)
                    y = random.randint(0, height-1)
                    length = random.randint(1, 5)
                    angle = random.uniform(-30, 30)
                    
                    # Calculate end point
                    x2 = int(x + length * np.cos(np.radians(angle)))
                    y2 = int(y + length * np.sin(np.radians(angle)))
                    
                    # Draw rain drop
                    cv2.line(img, (x, y), (x2, y2), (200, 200, 255), 1)
            
            elif weather_type == 'fog':
                # Simulate fog
                fog_factor = random.uniform(0.4, 0.8)
                fog_color = np.array([200, 200, 200], dtype=np.uint8)
                
                # Create fog mask
                fog_mask = np.random.uniform(0, 1, size=(height, width))
                fog_mask = np.stack([fog_mask] * 3, axis=2)
                
                # Apply fog
                img = img * (1 - fog_factor * fog_mask) + fog_color * (fog_factor * fog_mask)
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            elif weather_type == 'snow':
                # Simulate snow
                snow_drops = random.randint(300, 800)
                for _ in range(snow_drops):
                    x = random.randint(0, width-1)
                    y = random.randint(0, height-1)
                    size = random.randint(1, 3)
                    
                    # Draw snow flake
                    cv2.circle(img, (x, y), size, (255, 255, 255), -1)
        
        # 10. Time of day simulation (15% probability)
        if random.random() < 0.15:
            time_of_day = random.choice(['night', 'dawn_dusk'])
            
            if time_of_day == 'night':
                # Simulate night time
                brightness_factor = random.uniform(0.3, 0.6)
                img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                
                # Add slight blue tint for night
                img = cv2.addWeighted(img, 1, np.full_like(img, (20, 0, 0)), 0.1, 0)
            
            elif time_of_day == 'dawn_dusk':
                # Simulate dawn/dusk with orange/red tint
                img = cv2.addWeighted(img, 1, np.full_like(img, (0, 30, 50)), 0.1, 0)
        
        # Convert boxes back to tensor if they were tensors
        if is_tensor:
            target['boxes'] = torch.from_numpy(boxes_np)
        else:
            target['boxes'] = boxes_np
        
        # Filter out invalid boxes (those with zero width or height)
        valid_boxes = []
        valid_labels = []
        valid_indices = []
        
        for i, box in enumerate(target['boxes']):
            if isinstance(box, torch.Tensor):
                box = box.tolist()
            
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                valid_boxes.append(box)
                valid_labels.append(target['labels'][i])
                valid_indices.append(i)
        
        # Update target with valid boxes and labels
        if is_tensor:
            target['boxes'] = torch.tensor(valid_boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(valid_labels, dtype=torch.int64)
        else:
            target['boxes'] = np.array(valid_boxes)
            target['labels'] = np.array(valid_labels)
        
        # Update other target fields if they exist
        if 'area' in target and len(target['area']) > 0:
            if is_tensor:
                target['area'] = torch.tensor([target['area'][i] for i in valid_indices], dtype=torch.float32)
            else:
                target['area'] = np.array([target['area'][i] for i in valid_indices])
        
        if 'iscrowd' in target and len(target['iscrowd']) > 0:
            if is_tensor:
                target['iscrowd'] = torch.tensor([target['iscrowd'][i] for i in valid_indices], dtype=torch.int64)
            else:
                target['iscrowd'] = np.array([target['iscrowd'][i] for i in valid_indices])
        
        return img, target
    
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
        # Check if image_ids is empty
        if not self.image_ids:
            raise ValueError("No images found in the dataset. Check if _load_coco_dataset was called successfully.")
        
        # Get image ID
        img_id = self.image_ids[idx]
        
        # Check if image is cached
        if self.cache_images and self.img_cache and img_id in self.img_cache.keys():
            img = self.img_cache.get(img_id)
        else:
            # Load image
            img_info = self._coco.loadImgs(img_id)[0]
            img_path = os.path.join(str(self.image_dir), str(img_info['file_name']))
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found: {img_path}")
                # Return a dummy image and empty target
                img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'image_id': torch.tensor([img_id]),
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.int64),
                    'orig_size': torch.as_tensor([self.target_size[0], self.target_size[1]], dtype=torch.int64)
                }
                return {'img': torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0, 'target': target, 'image_id': img_id}
            
            # Store the image path for this image ID
            self.image_id_to_path[img_id] = img_path
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Cache image if enabled
            if self.cache_images:
                if self.img_cache is not None:
                    self.img_cache[img_id] = img
        
        # Get annotations
        ann_ids = self._coco.getAnnIds(imgIds=img_id)
        anns = self._coco.loadAnns(ann_ids)
        
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
        
        # Apply augmentation if enabled (moved outside of preprocess condition)
        if self.augment:
            img, target = self._apply_augmentation(img, target)
        
        # Apply transformations if provided
        if self.transforms:
            img, target = self.transforms(img, target)
        elif self.preprocess:
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
        # Store the image path for this image ID
        self.image_id_to_path[img_id] = img_path
        # Also store with numeric ID for compatibility
        if img_id.isdigit():
            self.image_id_to_path[int(img_id)] = img_path
        
        # Check if image is cached
        if self.cache_images and self.img_cache is not None and img_id in self.img_cache.keys():
            img = self.img_cache.get(img_id)
        else:
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Cache image if enabled
            if self.cache_images:
                if self.img_cache is not None:
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
        
        # Apply augmentation if enabled (moved outside of preprocess condition)
        if self.augment:
            img, target = self._apply_augmentation(img, target)
            
        # Apply transformations if provided
        if self.transforms:
            img, target = self.transforms(img, target)
        elif self.preprocess:
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
        if self.cache_images and self.img_cache is not None and img_id in self.img_cache.keys():
            img = self.img_cache.get(img_id)
        elif self.preprocess:
            # Load image
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Cache image if enabled
            if self.cache_images:
                if self.img_cache is not None:
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
        
        # Apply augmentation if enabled (moved outside of preprocess condition)
        if self.augment:
            img, target = self._apply_augmentation(img, target)
        
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
            return self._coco
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

def test_detection_dataset(dataset_type='coco', data_dir=None, num_samples=5, visualize=True, output_dir='output'):
    """
    Test the DetectionDataset class with different dataset types.
    
    Args:
        dataset_type: Type of dataset to test ('coco', 'kitti', 'voc')
        data_dir: Path to dataset directory
        num_samples: Number of samples to visualize
        visualize: Whether to visualize the samples
        output_dir: Directory to save visualizations
    """
    
    # Create output directory if it doesn't exist
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    print(f"\nTesting {dataset_type.upper()} dataset...")
    
    # Create dataset
    dataset = DetectionDataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        split='val',
        target_size=(640, 640),
        cache_images=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get random samples
    if len(dataset) > 0:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        for i, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx]
            img = sample['img'] #tensor [3, 640, 640]
            target = sample['target']
            image_id = sample['image_id'] #'006630'
            
            # Convert tensor to numpy for visualization
            if isinstance(img, torch.Tensor):
                # Convert (C, H, W) -> (H, W, C)
                img_np = img.permute(1, 2, 0).numpy()
                
                # If normalized to [0, 1], convert to [0, 255]
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img
            
            # Print sample info
            print(f"\nSample {i+1}/{len(indices)} (ID: {image_id}):")
            print(f"  Image shape: {img_np.shape}")
            print(f"  Number of objects: {len(target['boxes'])}")
            
            # Print object details
            if len(target['boxes']) > 0:
                print("  Objects:")
                for j, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
                    if isinstance(box, torch.Tensor):
                        box = box.tolist()
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    
                    # Get class name
                    class_name = coco_names[label] if label in coco_names else f"class_{label}"
                    
                    print(f"    {j+1}. {class_name}: {box}")
            
            # Visualize sample
            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(img_np)
                
                # Draw bounding boxes
                for box, label in zip(target['boxes'], target['labels']):
                    if isinstance(box, torch.Tensor):
                        box = box.tolist()
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    
                    # Get class name and color
                    class_name = coco_names.get(label, f"class_{label}")
                    color = plt.cm.hsv(label % 20 / 20.)
                    
                    # Create rectangle patch
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    
                    # Add rectangle to plot
                    plt.gca().add_patch(rect)
                    
                    # Add label
                    plt.text(
                        box[0], box[1] - 5,
                        class_name,
                        color='white',
                        bbox=dict(facecolor=color, alpha=0.8, pad=2)
                    )
                
                plt.title(f"{dataset_type.upper()} Dataset - Sample {i+1} (ID: {image_id})")
                plt.axis('off')
                
                # Save figure
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f"{dataset_type}_sample_{i+1}.jpg"))
                    print(f"  Visualization saved to {os.path.join(output_dir, f'{dataset_type}_sample_{i+1}.jpg')}")
                
                plt.close()
    else:
        print(f"No samples found in {dataset_type.upper()} dataset!")

def test_all_datasets():
    """
    Test all supported dataset types.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test detection datasets')
    parser.add_argument('--coco_dir', type=str, default=None, help='Path to COCO dataset')
    parser.add_argument('--kitti_dir', type=str, default=None, help='Path to KITTI dataset')
    parser.add_argument('--voc_dir', type=str, default=None, help='Path to VOC dataset')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='output/dataset_test', help='Output directory')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    # Test COCO dataset
    if args.coco_dir or os.path.exists('/home/lkk/Developer/VisionLangAnnotate/datasets/coco'):
        test_detection_dataset(
            dataset_type='coco',
            data_dir=args.coco_dir,
            num_samples=args.num_samples,
            visualize=not args.no_vis,
            output_dir=args.output_dir
        )
    
    # Test KITTI dataset
    if args.kitti_dir or os.path.exists('/home/lkk/Developer/VisionLangAnnotate/datasets/kitti'):
        test_detection_dataset(
            dataset_type='kitti',
            data_dir=args.kitti_dir,
            num_samples=args.num_samples,
            visualize=not args.no_vis,
            output_dir=args.output_dir
        )
    
    # Test VOC dataset
    if args.voc_dir or os.path.exists('/home/lkk/Developer/VisionLangAnnotate/datasets/voc'):
        test_detection_dataset(
            dataset_type='voc',
            data_dir=args.voc_dir,
            num_samples=args.num_samples,
            visualize=not args.no_vis,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    test_detection_dataset(
            dataset_type='kitti',
            data_dir="/DATA10T/Datasets/Kitti/",
            num_samples=10,
            visualize=True,
            output_dir="output"
        )
    #test_all_datasets()