import os
import json
import re
import time
import cv2
import datetime
import subprocess
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

# SAM imports
try:
    from transformers import SamProcessor, SamModel
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM (Segment Anything Model) not available. Install transformers with SAM support for segmentation.")

# Import the VLM classifier
try:
    from vlm_classifierv4 import HuggingFaceVLM
except ImportError:
    # Try alternative import paths
    try:
        from .vlm_classifierv4 import HuggingFaceVLM
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from vlm_classifierv4 import HuggingFaceVLM

class QwenObjectDetectionPipeline:
    """
    A robust object detection pipeline using Qwen2.5-VL model.
    
    This pipeline:
    1. Uses Qwen2.5-VL to detect objects in images
    2. Parses the natural language response to extract bounding box coordinates
    3. Converts results to Label Studio compatible JSON format
    4. Visualizes detection results with bounding boxes overlaid on images
    5. Saves all results (JSON and visualizations) to organized output directories
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda",
                 output_dir: str = "./object_detection_results",
                 enable_sam: bool = False):
        """Initialize the object detection pipeline.
        
        Args:
            model_name: Qwen2.5-VL model name
            device: Device to run the model on (cuda/cpu)
            output_dir: Directory to save results
            enable_sam: Whether to enable SAM segmentation capabilities
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.enable_sam = enable_sam and SAM_AVAILABLE
        
        # Initialize the VLM model
        print(f"Initializing {model_name}...")
        self.vlm = HuggingFaceVLM(model_name=model_name, device=device)
        
        # Initialize SAM if enabled
        self.sam_processor = None
        self.sam_model = None
        if self.enable_sam:
            try:
                print("Loading Segment Anything Model...")
                self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
                self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
                print("SAM model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load SAM model: {e}")
                self.enable_sam = False
        
        # Create output directories
        self._setup_output_directories()
        
        # Object detection prompt template
        self.detection_prompt = (
            "Detect objects in this image and provide ONLY the structured output format below. "
            "Do NOT provide explanations, analysis, or verbose descriptions.\n\n"
            "REQUIRED FORMAT (one object per line):\n"
            "Object_name: (x1,y1,x2,y2) confidence description\n\n"
            "EXAMPLES:\n"
            "Car: (100,50,200,150) 0.95 red sedan parked\n"
            "Tree: (300,20,350,180) 0.88 large oak tree\n"
            "Person: (150,80,180,200) 0.92 pedestrian walking\n\n"
            "DETECT: Vehicles (Car, Truck, Bus, Bicycle), People (Pedestrian, Worker), "
            "Infrastructure (Street sign, Traffic light, Utility pole), "
            "Issues (Trash, Pothole), Nature (Tree).\n\n"
            "IMPORTANT: If no clear objects are visible or image is too blurry, respond with: 'No objects detected'\n"
            "IMPORTANT: Use ONLY the exact format shown above. Do NOT add explanatory text."
        )
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        self.json_dir = os.path.join(self.output_dir, "json_annotations")
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        self.raw_dir = os.path.join(self.output_dir, "raw_responses")
        self.seg_dir = os.path.join(self.output_dir, "segmentations")
        
        for dir_path in [self.json_dir, self.viz_dir, self.raw_dir, self.seg_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Generate a timestamp string for file naming."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _parse_detection_response(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Robust parser for Qwen2.5-VL's natural language response to extract object detection results.
        Handles multiple response formats including simple structured format and verbose text.
        
        Args:
            response: Raw response from Qwen2.5-VL
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            List of detected objects with bounding boxes
        """
        objects = []
        
        # Clean up the response
        response = response.strip()
        
        # Method 1: Try simple structured format (Object: (x,y,x,y) confidence description)
        objects = self._parse_simple_format(response, image_width, image_height)
        if objects:
            return objects
        
        # Method 2: Try numbered format (1. **Object** ... **Bounding Box:** (x,y,x,y))
        objects = self._parse_numbered_format(response, image_width, image_height)
        if objects:
            return objects
        
        # Method 3: Try general pattern matching for any coordinate format
        objects = self._parse_general_format(response, image_width, image_height)
        if objects:
            return objects
        
        # Method 4: Handle verbose text responses with no structured data
        # Check if response contains analysis but no coordinates
        if self._is_verbose_analysis(response):
            print("Warning: VLM returned verbose analysis without structured detection data")
            return []  # Return empty list for verbose responses without coordinates
        
        print(f"Warning: Could not parse detection response: {response[:100]}...")
        return objects
    
    def _parse_simple_format(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse simple format: Object: (x,y,x,y) confidence description
        """
        objects = []
        
        # Check for explicit 'no objects' response
        if 'no objects detected' in response.lower():
            return objects  # Return empty list
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Pattern: Object: (x,y,x,y) confidence description
            match = re.match(r'^([^:]+):\s*\((\d+),(\d+),(\d+),(\d+)\)\s*([\d.]+)?\s*(.*)$', line)
            if match:
                label = match.group(1).strip()
                x1, y1, x2, y2 = map(int, match.groups()[1:5])
                confidence = float(match.group(5)) if match.group(5) else 1.0
                description = match.group(6).strip() if match.group(6) else f"A {label.lower()} detected"
                
                if self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                    objects.append({
                        'label': label,
                        'bbox': [x1, y1, x2, y2],
                        'description': description,
                        'confidence': confidence
                    })
        
        return objects
    
    def _parse_numbered_format(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse numbered format: 1. **Object** ... **Bounding Box:** (x,y,x,y)
        """
        objects = []
        
        # Split response into numbered sections
        sections = re.split(r'\n(?=\d+\.\s*\*\*)', response)
        
        for section in sections:
            if not section.strip():
                continue
                
            # Extract object label from numbered format
            label_match = re.search(r'\d+\.\s*\*\*([^*]+)\*\*', section)
            if not label_match:
                continue
                
            object_label = label_match.group(1).strip()
            
            # Extract bounding box coordinates
            bbox_patterns = [
                r'\*\*Bounding Box Coordinates\*\*:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
                r'\*\*Bounding Box:\*\*\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
                r'Bounding Box:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
            ]
            
            for pattern in bbox_patterns:
                bbox_match = re.search(pattern, section)
                if bbox_match:
                    try:
                        x1, y1, x2, y2 = map(int, bbox_match.groups())
                        
                        if self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                            # Extract description
                            description_match = re.search(r'Description:\s*(.+?)(?=\n\n|\n\d+\.|$)', section, re.DOTALL)
                            description = description_match.group(1).strip() if description_match else f"A {object_label.lower()} detected in the image"
                            
                            objects.append({
                                'label': object_label,
                                'bbox': [x1, y1, x2, y2],
                                'description': description,
                                'confidence': 1.0
                            })
                            break
                    except (ValueError, IndexError):
                        continue
        
        return objects
    
    def _parse_general_format(self, response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """
        Parse general format using pattern matching for coordinates
        """
        objects = []
        
        # Find all coordinate patterns in the response
        coord_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        coord_matches = list(re.finditer(coord_pattern, response))
        
        if not coord_matches:
            return objects
        
        # Split response into lines and try to match coordinates with labels
        lines = response.split('\n')
        
        for match in coord_matches:
            x1, y1, x2, y2 = map(int, match.groups())
            
            if not self._validate_bbox(x1, y1, x2, y2, image_width, image_height):
                continue
            
            # Find the line containing this coordinate
            coord_text = match.group(0)
            matching_line = None
            
            for line in lines:
                if coord_text in line:
                    matching_line = line
                    break
            
            if matching_line:
                # Extract label from the line
                label = self._extract_object_label(matching_line)
                description = self._extract_description(matching_line)
                
                objects.append({
                    'label': label,
                    'bbox': [x1, y1, x2, y2],
                    'description': description,
                    'confidence': 1.0
                })
        
        return objects
    
    def _is_verbose_analysis(self, response: str) -> bool:
        """
        Check if response is a verbose analysis without structured detection data
        """
        response_lower = response.lower()
        
        # Check for explicit 'no objects detected' response (this is valid, not verbose)
        if 'no objects detected' in response_lower:
            return False
        
        # Look for indicators of verbose analysis
        verbose_indicators = [
            'the provided image',
            'based on the analysis',
            'however,',
            'there are no',
            'no distinct',
            'no discernible',
            'no clear',
            'too blurry',
            'lacks clear details',
            'appears to be'
        ]
        
        # Check if response contains verbose indicators and no coordinates
        has_verbose_indicators = any(indicator in response_lower for indicator in verbose_indicators)
        has_coordinates = bool(re.search(r'\(\d+,\s*\d+,\s*\d+,\s*\d+\)', response))
        
        return has_verbose_indicators and not has_coordinates
    
    def _extract_object_label(self, section: str) -> str:
        """
        Extract object label from a section of text.
        
        Args:
            section: Text section containing object information
            
        Returns:
            Extracted object label
        """
        # Try to find labels in various formats
        label_patterns = [
            r'\*\*([^*]+)\*\*',  # **Label**
            r'(\d+\.\s*\*\*[^*]+\*\*)',  # 1. **Label**
            r'(\d+\.\s*[^:]+):',  # 1. Label:
            r'^([^:]+):',  # Label: at start of line
            r'(\w+(?:\s+\w+)*?)(?:\s*\(|\s*:)',  # Word(s) before ( or :
        ]
        
        for pattern in label_patterns:
            match = re.search(pattern, section, re.MULTILINE)
            if match:
                label = match.group(1).strip()
                # Clean up the label
                label = re.sub(r'^\d+\.\s*', '', label)  # Remove numbering
                label = re.sub(r'\*\*', '', label)  # Remove markdown
                label = label.strip()
                if label and len(label) > 0:
                    return label
        
        # Fallback: use first few words
        words = section.split()[:3]
        return ' '.join(words) if words else 'Unknown Object'
    
    def _extract_description(self, section: str) -> str:
        """
        Extract object description from a section of text.
        
        Args:
            section: Text section containing object information
            
        Returns:
            Extracted description
        """
        # Look for description patterns
        desc_patterns = [
            r'\*\*Description:\*\*\s*(.+?)(?=\n|$)',
            r'Description:\s*(.+?)(?=\n|$)',
            r'\*\*Bounding Box:\*\*.*?\n(.+?)(?=\n|$)',
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, section, re.DOTALL)
            if match:
                desc = match.group(1).strip()
                if desc:
                    return desc
        
        # Fallback: use the section text, cleaned up
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^\d+\.|\*\*|Bounding Box:', line):
                return line
        
        return "Object detected in image"
    
    def _validate_bbox(self, x1: int, y1: int, x2: int, y2: int, 
                      image_width: int, image_height: int) -> bool:
        """
        Validate bounding box coordinates.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            image_width, image_height: Image dimensions
            
        Returns:
            True if bbox is valid, False otherwise
        """
        # Check if coordinates are within image bounds
        if x1 < 0 or y1 < 0 or x2 >= image_width or y2 >= image_height:
            return False
        
        # Check if x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            return False
        
        # Check if bbox has reasonable size (at least 5x5 pixels)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return False
        
        # Filter out unreasonably large bounding boxes
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        
        # Reject bboxes that cover more than 80% of the image
        if bbox_area > (0.8 * image_area):
            return False
        
        # Reject bboxes that are too wide or too tall relative to image
        if bbox_width > (0.9 * image_width) or bbox_height > (0.9 * image_height):
            return False
        
        # Reject extremely thin or narrow bboxes (aspect ratio checks)
        aspect_ratio = bbox_width / bbox_height
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        
        return True
    
    def _blur_region(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, blur_strength: int = 15) -> np.ndarray:
        """
        Apply Gaussian blur to a specific region of the image.
        
        Args:
            image: Input image as numpy array
            x1, y1, x2, y2: Bounding box coordinates of region to blur
            blur_strength: Strength of the blur effect (higher = more blur)
            
        Returns:
            np.ndarray: Image with blurred region
        """
        # Create a copy of the image
        blurred_image = image.copy()
        
        # Extract the region to blur
        region = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur to the region
        blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
        
        # Replace the original region with the blurred version
        blurred_image[y1:y2, x1:x2] = blurred_region
        
        return blurred_image
    
    def _detect_faces_and_plates(self, objects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Identify faces and license plates from detected objects for privacy protection.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Tuple of (faces, license_plates) lists
        """
        faces = []
        license_plates = []
        
        for obj in objects:
            label = obj.get('label', '').lower()
            description = obj.get('description', '').lower()
            
            # Check for faces
            if ('face' in label or 'person' in label or 'pedestrian' in label or 
                'face' in description or 'facial' in description):
                faces.append(obj)
            
            # Check for license plates
            if ('license' in label or 'plate' in label or 'number plate' in label or
                'license' in description or 'plate' in description or 'registration' in description):
                license_plates.append(obj)
        
        return faces, license_plates
    
    def apply_privacy_protection(self, image_path: str, objects: List[Dict[str, Any]], 
                               output_path: str = None, blur_faces: bool = True, 
                               blur_plates: bool = True) -> str:
        """
        Apply privacy protection by blurring faces and license plates in the image.
        
        Args:
            image_path: Path to the input image
            objects: List of detected objects
            output_path: Path to save the privacy-protected image
            blur_faces: Whether to blur detected faces
            blur_plates: Whether to blur detected license plates
            
        Returns:
            str: Path to the saved privacy-protected image
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect faces and license plates
        faces, license_plates = self._detect_faces_and_plates(objects)
        
        # Apply blurring to faces
        if blur_faces:
            for face in faces:
                bbox = face.get('bbox', {})
                if bbox and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    image = self._blur_region(image, x1, y1, x2, y2, blur_strength=25)
        
        # Apply blurring to license plates
        if blur_plates:
            for plate in license_plates:
                bbox = plate.get('bbox', {})
                if bbox and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    image = self._blur_region(image, x1, y1, x2, y2, blur_strength=20)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.output_dir, "privacy_protected", f"{base_name}_protected.jpg")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the privacy-protected image
        cv2.imwrite(output_path, image)
        
        return output_path
    
    def _convert_to_label_studio_format(self, objects: List[Dict[str, Any]], 
                                       image_path: str, image_width: int, 
                                       image_height: int) -> Dict[str, Any]:
        """
        Convert detection results to Label Studio compatible JSON format.
        
        Args:
            objects: List of detected objects
            image_path: Path to the image file
            image_width, image_height: Image dimensions
            
        Returns:
            Label Studio compatible annotation dictionary
        """
        annotations = []
        
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj['bbox']
            
            # Convert to Label Studio format (percentages)
            x_percent = (x1 / image_width) * 100
            y_percent = (y1 / image_height) * 100
            width_percent = ((x2 - x1) / image_width) * 100
            height_percent = ((y2 - y1) / image_height) * 100
            
            annotation = {
                "id": f"bbox_{i}",
                "type": "rectanglelabels",
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [obj['label']]
                },
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0,
                "original_width": image_width,
                "original_height": image_height
            }
            annotations.append(annotation)
        
        # Create the complete Label Studio task format
        task = {
            "data": {
                "image": image_path
            },
            "annotations": [{
                "id": 1,
                "created_username": "qwen_pipeline",
                "created_ago": "0 minutes",
                "task": 1,
                "result": annotations,
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "lead_time": 0
            }],
            "predictions": [],
            "meta": {
                "model_name": self.model_name,
                "detection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_objects": len(objects)
            }
        }
        
        return task
    
    def _visualize_detections(self, image: Image.Image, objects: List[Dict[str, Any]], 
                            output_path: str) -> None:
        """
        Create visualization of detection results with bounding boxes.
        
        Args:
            image: PIL Image object
            objects: List of detected objects
            output_path: Path to save the visualization
        """
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Define colors for different object types
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj['bbox']
            color = colors[i % len(colors)]
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label text with object name
            label_text = obj['label'] if obj['label'] != 'Unknown Object' else f"Object {i+1}"
            ax.text(
                x1, y1 - 5, label_text,
                fontsize=10, color=color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        plt.title(f"Object Detection Results ({len(objects)} objects detected)", 
                 fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def detect_objects(self, image_path: str, save_results: bool = True, apply_privacy: bool = True, use_sam_segmentation: bool = False) -> Dict[str, Any]:
        """
        Perform object detection on a single image.
        
        Args:
            image_path: Path to the input image
            save_results: Whether to save results to files
            apply_privacy: Whether to apply privacy protection (blur faces/plates)
            use_sam_segmentation: Whether to apply SAM segmentation post-processing
            
        Returns:
            Dictionary containing detection results and file paths
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
        
        # Generate detection prompt and get response
        print("Running object detection...")
        start_time = time.time()
        
        try:
            responses = self.vlm.generate([image], [self.detection_prompt])
            raw_response = responses[0] if responses else ""
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")
        
        detection_time = time.time() - start_time
        print(f"Detection completed in {detection_time:.2f} seconds")
        
        # Parse the response to extract objects
        print("Parsing detection results...")
        objects = self._parse_detection_response(raw_response, image_width, image_height)
        print(f"Found {len(objects)} objects")
        
        # Apply SAM segmentation if requested and available
        segmentation_masks = []
        segmentation_visualization_path = None
        if use_sam_segmentation and self.enable_sam and objects:
            try:
                print("Applying SAM segmentation...")
                segmentation_start_time = time.time()
                segmentation_masks, segmentation_visualization_path = self._apply_sam_segmentation(
                    image, objects, image_path
                )
                segmentation_time = time.time() - segmentation_start_time
                print(f"SAM segmentation completed in {segmentation_time:.2f} seconds")
            except Exception as e:
                print(f"Warning: SAM segmentation failed: {str(e)}")
        
        # Apply privacy protection if requested
        privacy_protected_path = None
        if apply_privacy and objects:
            try:
                privacy_protected_path = self.apply_privacy_protection(image_path, objects)
                print(f"Privacy protection applied, saved to: {privacy_protected_path}")
            except Exception as e:
                print(f"Warning: Privacy protection failed: {str(e)}")
        
        # Prepare results
        results = {
            'image_path': image_path,
            'image_width': image_width,
            'image_height': image_height,
            'objects': objects,
            'raw_response': raw_response,
            'detection_time': detection_time,
            'model_name': self.model_name,
            'privacy_protected_path': privacy_protected_path,
            'segmentation_masks': segmentation_masks,
            'segmentation_visualization_path': segmentation_visualization_path
        }
        
        if save_results:
            # Generate output file names
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save raw response
            raw_file = os.path.join(self.raw_dir, f"{base_name}_{timestamp}_raw.txt")
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Detection Time: {detection_time:.2f}s\n")
                f.write(f"Objects Found: {len(objects)}\n")
                f.write("\n" + "="*50 + "\n")
                f.write(raw_response)
            
            # Convert to Label Studio format and save
            label_studio_data = self._convert_to_label_studio_format(
                objects, image_path, image_width, image_height
            )
            
            json_file = os.path.join(self.json_dir, f"{base_name}_{timestamp}_annotations.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
            
            # Create visualization
            viz_file = os.path.join(self.viz_dir, f"{base_name}_{timestamp}_detection.png")
            self._visualize_detections(image, objects, viz_file)
            
            # Add file paths to results
            results.update({
                'raw_response_file': raw_file,
                'json_annotation_file': json_file,
                'visualization_file': viz_file
            })
            
            print(f"Results saved:")
            print(f"  - Raw response: {raw_file}")
            print(f"  - JSON annotations: {json_file}")
            print(f"  - Visualization: {viz_file}")
            if segmentation_visualization_path:
                print(f"  - SAM Segmentation: {segmentation_visualization_path}")
        
        return results
    
    def detect_objects_batch(self, image_paths: List[str], save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Perform object detection on multiple images.
        
        Args:
            image_paths: List of paths to input images
            save_results: Whether to save results to files
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                result = self.detect_objects(image_path, save_results)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'objects': []
                })
        
        if save_results:
            # Save batch summary
            summary_file = os.path.join(self.output_dir, f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
            summary = {
                'total_images': len(image_paths),
                'successful_detections': len([r for r in results if 'error' not in r]),
                'failed_detections': len([r for r in results if 'error' in r]),
                'total_objects_detected': sum(len(r.get('objects', [])) for r in results),
                'model_name': self.model_name,
                'processing_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'results': results
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\nBatch summary saved: {summary_file}")
        
        return results


    def extract_video_metadata(self, video_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Extract metadata including GPS information from video file using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (metadata, gps_info, creation_time)
        """
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_format', 
            '-show_streams', 
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            metadata = json.loads(result.stdout)
            
            # Extract GPS data if available
            gps_info = {}
            creation_time = None
            
            # Look for creation time and GPS data in format tags
            if 'format' in metadata and 'tags' in metadata['format']:
                tags = metadata['format']['tags']
                
                # Extract creation time
                time_fields = ['creation_time', 'date', 'com.apple.quicktime.creationdate']
                for field in time_fields:
                    if field in tags:
                        creation_time = tags[field]
                        break
                
                # Common GPS metadata fields
                gps_fields = [
                    'location', 'location-eng', 'GPS', 
                    'GPSLatitude', 'GPSLongitude', 'GPSAltitude',
                    'com.apple.quicktime.location.ISO6709'
                ]
                
                for field in gps_fields:
                    if field in tags:
                        gps_info[field] = tags[field]
            
            # Also check stream metadata for creation time if not found
            if creation_time is None and 'streams' in metadata:
                for stream in metadata['streams']:
                    if 'tags' in stream and 'creation_time' in stream['tags']:
                        creation_time = stream['tags']['creation_time']
                        break
            
            # If no creation time found, use file modification time
            if creation_time is None:
                file_mtime = os.path.getmtime(video_path)
                creation_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()
            
            return metadata, gps_info, creation_time
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}, {}, None
    
    def _resize_with_aspect_ratio(self, image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image maintaining aspect ratio.
        
        Args:
            image: PIL Image or numpy array
            target_size: Tuple of (width, height) representing the maximum dimensions
            
        Returns:
            Resized PIL Image
        """
        if isinstance(image, np.ndarray):
            # Convert OpenCV image (numpy array) to PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Handle both integer and tuple inputs for target_size
        if isinstance(target_size, (int, float)):
            # If target_size is a single number, use it as the maximum dimension
            if original_width > original_height:
                target_width = int(target_size)
                target_height = int(target_size * original_height / original_width)
            else:
                target_height = int(target_size)
                target_width = int(target_size * original_width / original_height)
        else:
            # If target_size is a tuple, use it directly
            target_width, target_height = target_size
        
        # Calculate aspect ratios
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        # Determine new dimensions maintaining aspect ratio
        if original_aspect > target_aspect:
            # Width constrained
            new_width = target_width
            new_height = int(target_width / original_aspect)
        else:
            # Height constrained
            new_height = target_height
            new_width = int(target_height * original_aspect)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return resized_image
    
    def process_video_frames(self, video_path: str, target_size: Tuple[int, int] = (640, 480), 
                           extraction_method: str = "scene_change", scenechange_threshold: float = 10.0,
                           save_results: bool = True, use_sam_segmentation: bool = False, 
                           batch_size: int = 4) -> Dict[str, Any]:
        """
        Extract key frames from a video file and perform object detection on each frame.
        
        Args:
            video_path: Path to the input video file
            target_size: Tuple of (width, height) maximum dimensions for resizing
            extraction_method: Method to extract frames ('scene_change', 'interval', or 'both')
            scenechange_threshold: Threshold for scene change detection
            save_results: Whether to save results to files
            use_sam_segmentation: Whether to apply SAM segmentation to detected objects
            batch_size: Number of frames to process in batch for better GPU utilization
            
        Returns:
            Dictionary containing video processing results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ignore video files that are less than 2 seconds long
        if duration < 2.0:
            cap.release()
            raise ValueError(f"Video too short (duration: {duration:.2f} seconds, minimum required: 2.0 seconds)")
        
        # Get video metadata
        metadata, gps_info, creation_time = self.extract_video_metadata(video_path)
        
        print(f"Video Information:")
        print(f"- Frame Rate: {fps} fps")
        print(f"- Frame Count: {frame_count}")
        print(f"- Resolution: {original_width}x{original_height}")
        print(f"- Duration: {duration:.2f} seconds")
        print(f"- Creation Time: {creation_time}")
        
        # Create video-specific output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, f"video_{video_name}_{self._get_timestamp()}")
        
        if save_results:
            os.makedirs(video_output_dir, exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "frames"), exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "raw_responses"), exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "json_annotations"), exist_ok=True)
            os.makedirs(os.path.join(video_output_dir, "visualizations"), exist_ok=True)
            if use_sam_segmentation and self.enable_sam:
                os.makedirs(os.path.join(video_output_dir, "segmentations"), exist_ok=True)
        
        # Initialize variables for frame extraction
        prev_frame = None
        frame_idx = 0
        saved_count = 0
        extracted_frames = []
        all_results = []
        
        # Parameters for scene change detection
        min_scene_change_threshold = scenechange_threshold
        frame_interval = int(fps) * 1  # Save a frame every second as fallback
        
        print(f"Extracting frames using method: {extraction_method}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for scene change detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            should_save = False
            reason = ""
            
            # Method 1: Detect scene changes
            if extraction_method in ["scene_change", "both"]:
                if prev_frame is not None:
                    # Calculate absolute difference between frames
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > min_scene_change_threshold:
                        should_save = True
                        reason = f"scene_change (diff: {mean_diff:.2f})"
            
            # Method 2: Extract frames at regular intervals
            if extraction_method in ["interval", "both"]:
                if frame_idx % frame_interval == 0:
                    should_save = True
                    reason = f"interval ({frame_idx // frame_interval}s)"
            
            # Save frame if criteria met
            if should_save:
                # Calculate timestamp
                timestamp_seconds = frame_idx / fps
                timestamp = f"{int(timestamp_seconds // 60):02d}:{int(timestamp_seconds % 60):02d}.{int((timestamp_seconds % 1) * 1000):03d}"
                
                # Resize frame
                resized_frame_pil = self._resize_with_aspect_ratio(frame, target_size)
                new_width, new_height = resized_frame_pil.size
                
                # Save frame if requested
                frame_filename = None
                if save_results:
                    frame_filename = f"{video_name}_frame_{saved_count:04d}_{timestamp.replace(':', '-').replace('.', '_')}.jpg"
                    frame_path = os.path.join(video_output_dir, "frames", frame_filename)
                    resized_frame_pil.save(frame_path, "JPEG", quality=95)
                
                # Add frame to batch for processing
                frame_data = {
                    'image': resized_frame_pil,
                    'saved_index': saved_count,
                    'timestamp_seconds': timestamp_seconds,
                    'timestamp': timestamp,
                    'extraction_reason': reason,
                    'original_dimensions': {'width': original_width, 'height': original_height},
                    'resized_dimensions': {'width': new_width, 'height': new_height},
                    'frame_filename': frame_filename,
                    'frame_index': frame_idx
                }
                
                # Store frame data for batch processing
                if not hasattr(self, '_frame_batch'):
                    self._frame_batch = []
                
                self._frame_batch.append(frame_data)
                saved_count += 1
                
                print(f"Added frame {saved_count} to batch: {timestamp} - {reason}")
                
                # Process batch when it reaches the specified size
                if len(self._frame_batch) >= batch_size:
                    try:
                        batch_results = self._process_frame_batch(
                            self._frame_batch, video_output_dir, video_name, 
                            use_sam_segmentation, save_results=save_results
                        )
                        
                        # Add results and frame info
                        for i, (frame_data, frame_results) in enumerate(zip(self._frame_batch, batch_results)):
                            frame_info = {
                                'frame_index': frame_data['frame_index'],
                                'saved_index': frame_data['saved_index'],
                                'timestamp_seconds': frame_data['timestamp_seconds'],
                                'timestamp': frame_data['timestamp'],
                                'extraction_reason': frame_data['extraction_reason'],
                                'original_dimensions': frame_data['original_dimensions'],
                                'resized_dimensions': frame_data['resized_dimensions'],
                                'frame_filename': frame_data['frame_filename'],
                                'detection_results': frame_results
                            }
                            
                            extracted_frames.append(frame_info)
                            all_results.append(frame_results)
                            
                            print(f"Processed frame {frame_data['saved_index']}: {frame_data['timestamp']} - {len(frame_results.get('objects', []))} objects detected")
                        
                        # Clear batch
                        self._frame_batch = []
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        # Add empty results for failed batch
                        for frame_data in self._frame_batch:
                            frame_info = {
                                'frame_index': frame_data['frame_index'],
                                'saved_index': frame_data['saved_index'],
                                'timestamp_seconds': frame_data['timestamp_seconds'],
                                'timestamp': frame_data['timestamp'],
                                'extraction_reason': frame_data['extraction_reason'],
                                'original_dimensions': frame_data['original_dimensions'],
                                'resized_dimensions': frame_data['resized_dimensions'],
                                'frame_filename': frame_data['frame_filename'],
                                'detection_results': {
                                    'objects': [],
                                    'raw_response': '',
                                    'inference_time': 0,
                                    'model_name': self.model_name,
                                    'segmentation_masks': [],
                                    'segmentation_visualization_path': None,
                                    'error': str(e)
                                }
                            }
                            extracted_frames.append(frame_info)
                            all_results.append(frame_info['detection_results'])
                        
                        self._frame_batch = []
            
            prev_frame = gray.copy()
            frame_idx += 1
        
        cap.release()
        
        # Process any remaining frames in the batch
        if hasattr(self, '_frame_batch') and self._frame_batch:
            try:
                print(f"Processing final batch of {len(self._frame_batch)} frames...")
                batch_results = self._process_frame_batch(
                    self._frame_batch, video_output_dir, video_name, 
                    use_sam_segmentation, save_results=save_results
                )
                
                # Add results and frame info
                for i, (frame_data, frame_results) in enumerate(zip(self._frame_batch, batch_results)):
                    frame_info = {
                        'frame_index': frame_data['frame_index'],
                        'saved_index': frame_data['saved_index'],
                        'timestamp_seconds': frame_data['timestamp_seconds'],
                        'timestamp': frame_data['timestamp'],
                        'extraction_reason': frame_data['extraction_reason'],
                        'original_dimensions': frame_data['original_dimensions'],
                        'resized_dimensions': frame_data['resized_dimensions'],
                        'frame_filename': frame_data['frame_filename'],
                        'detection_results': frame_results
                    }
                    
                    extracted_frames.append(frame_info)
                    all_results.append(frame_results)
                    
                    print(f"Processed final frame {frame_data['saved_index']}: {frame_data['timestamp']} - {len(frame_results.get('objects', []))} objects detected")
                
                # Clear batch
                self._frame_batch = []
                
            except Exception as e:
                print(f"Error processing final batch: {e}")
                # Add empty results for failed batch
                for frame_data in self._frame_batch:
                    frame_info = {
                        'frame_index': frame_data['frame_index'],
                        'saved_index': frame_data['saved_index'],
                        'timestamp_seconds': frame_data['timestamp_seconds'],
                        'timestamp': frame_data['timestamp'],
                        'extraction_reason': frame_data['extraction_reason'],
                        'original_dimensions': frame_data['original_dimensions'],
                        'resized_dimensions': frame_data['resized_dimensions'],
                        'frame_filename': frame_data['frame_filename'],
                        'detection_results': {
                            'objects': [],
                            'raw_response': '',
                            'inference_time': 0,
                            'model_name': self.model_name,
                            'segmentation_masks': [],
                            'segmentation_visualization_path': None,
                            'error': str(e)
                        }
                    }
                    extracted_frames.append(frame_info)
                    all_results.append(frame_info['detection_results'])
                
                self._frame_batch = []
        
        # Create summary results
        total_objects = sum(len(result.get('objects', [])) for result in all_results)
        
        video_results = {
            'video_path': video_path,
            'video_metadata': {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'original_width': original_width,
                'original_height': original_height,
                'creation_time': creation_time,
                'gps_info': gps_info
            },
            'extraction_settings': {
                'method': extraction_method,
                'scene_change_threshold': scenechange_threshold,
                'target_size': target_size
            },
            'extracted_frames': extracted_frames,
            'summary': {
                'total_frames_processed': frame_idx,
                'frames_extracted': saved_count,
                'total_objects_detected': total_objects,
                'processing_time': time.time()
            }
        }
        
        # Save video summary if requested
        if save_results:
            summary_path = os.path.join(video_output_dir, f"{video_name}_video_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(video_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\nVideo processing complete!")
            print(f"- Processed {frame_idx} total frames")
            print(f"- Extracted {saved_count} key frames")
            print(f"- Detected {total_objects} total objects")
            print(f"- Results saved to: {video_output_dir}")
        
        return video_results
    
    def _process_frame_batch(self, frame_batch: List[Dict[str, Any]], video_output_dir: str, 
                           video_name: str, use_sam_segmentation: bool = False, 
                           save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of frames efficiently with GPU memory optimization.
        
        Args:
            frame_batch: List of frame dictionaries with image data and metadata
            video_output_dir: Output directory for saving results
            video_name: Base name of the video
            use_sam_segmentation: Whether to apply SAM segmentation
            save_results: Whether to save results to files
            
        Returns:
            List of processing results for each frame
        """
        if not frame_batch:
            return []
        
        batch_results = []
        
        try:
            # Extract images from batch
            images = [frame_data['image'] for frame_data in frame_batch]
            prompts = [self.detection_prompt] * len(images)
            
            # Batch inference for better GPU utilization
            print(f"Processing batch of {len(images)} frames...")
            start_time = time.time()
            
            # Process all images in batch
            responses = self.vlm.generate(images, prompts)
            batch_inference_time = time.time() - start_time
            print(f"Batch inference completed in {batch_inference_time:.2f} seconds")
            
            # Process each frame result
            for i, (frame_data, response) in enumerate(zip(frame_batch, responses)):
                try:
                    image = frame_data['image']
                    image_width, image_height = image.size
                    
                    # Parse detection response
                    objects = self._parse_detection_response(response, image_width, image_height)
                    
                    # Apply SAM segmentation if requested
                    segmentation_masks = []
                    segmentation_visualization_path = None
                    if use_sam_segmentation and self.enable_sam and objects:
                        try:
                            segmentation_masks, segmentation_visualization_path = self._apply_sam_segmentation(
                                image, objects, f"frame_{frame_data['saved_index']:04d}"
                            )
                            
                            # Save segmentation to video directory
                            if segmentation_visualization_path and save_results:
                                seg_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_segmentation.png"
                                final_seg_path = os.path.join(video_output_dir, "segmentations", seg_filename)
                                
                                # Copy segmentation result to video directory
                                import shutil
                                shutil.copy2(segmentation_visualization_path, final_seg_path)
                                segmentation_visualization_path = final_seg_path
                                
                        except Exception as e:
                            print(f"Warning: SAM segmentation failed for frame {frame_data['saved_index']}: {str(e)}")
                    
                    # Create frame results
                    frame_results = {
                        'objects': objects,
                        'raw_response': response,
                        'inference_time': batch_inference_time / len(images),  # Approximate per-frame time
                        'model_name': self.model_name,
                        'segmentation_masks': segmentation_masks,
                        'segmentation_visualization_path': segmentation_visualization_path
                    }
                    
                    # Save individual frame results if requested
                    if save_results:
                        # Save raw response
                        raw_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_raw.txt"
                        raw_path = os.path.join(video_output_dir, "raw_responses", raw_filename)
                        with open(raw_path, 'w', encoding='utf-8') as f:
                            f.write(response)
                        
                        # Save JSON annotations
                        json_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_annotations.json"
                        json_path = os.path.join(video_output_dir, "json_annotations", json_filename)
                        
                        # Create Label Studio format for this frame
                        frame_annotation = self._convert_to_label_studio_format(
                            objects, 
                            frame_data.get('frame_filename', f"frame_{frame_data['saved_index']:04d}.jpg"), 
                            image_width, 
                            image_height
                        )
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(frame_annotation, f, indent=2, ensure_ascii=False)
                        
                        # Save visualization
                        if objects:
                            vis_filename = f"{video_name}_frame_{frame_data['saved_index']:04d}_{self._get_timestamp()}_detection.png"
                            vis_path = os.path.join(video_output_dir, "visualizations", vis_filename)
                            self._visualize_detections(image, objects, vis_path)
                    
                    batch_results.append(frame_results)
                    
                    print(f"Processed frame {frame_data['saved_index']}: {frame_data['timestamp']} - {len(objects)} objects detected")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_data['saved_index']}: {e}")
                    # Add empty result to maintain batch consistency
                    batch_results.append({
                        'objects': [],
                        'raw_response': '',
                        'inference_time': 0,
                        'model_name': self.model_name,
                        'segmentation_masks': [],
                        'segmentation_visualization_path': None,
                        'error': str(e)
                    })
            
            # Clear GPU cache after batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Return empty results for the batch
            batch_results = [{
                'objects': [],
                'raw_response': '',
                'inference_time': 0,
                'model_name': self.model_name,
                'segmentation_masks': [],
                'segmentation_visualization_path': None,
                'error': str(e)
            } for _ in frame_batch]
        
        return batch_results
    
    def _apply_sam_segmentation(self, image: Image.Image, objects: List[Dict[str, Any]], 
                               image_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Apply SAM segmentation to detected objects.
        
        Args:
            image: PIL Image object
            objects: List of detected objects with bounding boxes
            image_path: Path to the original image
            
        Returns:
            Tuple of (segmentation_masks, visualization_path)
        """
        if not self.enable_sam or not objects:
            return [], None
        
        try:
            # Convert PIL image to numpy array for processing
            image_np = np.array(image)
            
            # Prepare input points from bounding boxes (center points)
            input_points = []
            input_labels = []
            
            for obj in objects:
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                # Use center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                input_points.append([center_x, center_y])
                input_labels.append(1)  # 1 indicates foreground point
            
            # Process with SAM
            inputs = self.sam_processor(image, input_points=[input_points], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            
            # Get masks
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )
            
            # Process masks and create visualization
            segmentation_masks = []
            mask_overlays = []
            
            # Get the first (and typically only) batch of masks
            batch_masks = masks[0] if masks else []
            
            for i, obj in enumerate(objects):
                if i < len(batch_masks):
                    # Get the mask for this object
                    mask = batch_masks[i]
                    
                    # Convert mask to numpy - handle different possible shapes
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.squeeze().cpu().numpy()
                    else:
                        mask_np = np.array(mask).squeeze()
                    
                    # Ensure mask is 2D
                    if mask_np.ndim > 2:
                        mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.max(axis=0)
                    
                    # Convert to uint8
                    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
                    
                    # Store mask info
                    mask_info = {
                        'object_id': i,
                        'object_label': obj['label'],
                        'mask_shape': mask_np.shape,
                        'bbox': obj['bbox']
                    }
                    segmentation_masks.append(mask_info)
                    
                    # Create colored overlay for visualization
                    color = plt.cm.tab10(i % 10)[:3]  # Get color from colormap
                    colored_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                    mask_indices = mask_np > 0
                    colored_mask[mask_indices] = [int(c * 255) for c in color]
                    mask_overlays.append(colored_mask)
            
            # Create visualization with segmentation, bounding boxes, and labels
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = self._get_timestamp()
            viz_path = os.path.join(self.seg_dir, f"{base_name}_{timestamp}_segmentation.png")
            
            # Combine original image with mask overlays
            combined_image = image_np.copy()
            for mask_overlay in mask_overlays:
                # Alpha blend the mask with the original image
                alpha = 0.5
                mask_indices = np.any(mask_overlay > 0, axis=2)
                combined_image[mask_indices] = (
                    alpha * mask_overlay[mask_indices] + 
                    (1 - alpha) * combined_image[mask_indices]
                ).astype(np.uint8)
            
            # Convert to PIL for drawing bounding boxes and labels
            combined_pil = Image.fromarray(combined_image)
            draw = ImageDraw.Draw(combined_pil)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except (OSError, IOError):
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Define colors for bounding boxes (matching segmentation colors)
            bbox_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
            
            # Draw bounding boxes and labels on top of segmentation
            for i, obj in enumerate(objects):
                if i < len(segmentation_masks):
                    bbox = obj['bbox']
                    x1, y1, x2, y2 = bbox
                    color = bbox_colors[i % len(bbox_colors)]
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Prepare label text
                    label_text = obj['label'] if obj['label'] != 'Unknown Object' else f"Object {i+1}"
                    
                    # Calculate text size and position
                    if font:
                        bbox_text = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        # Estimate text size if font is not available
                        text_width = len(label_text) * 8
                        text_height = 12
                    
                    # Position label above bounding box
                    text_x = x1
                    text_y = max(0, y1 - text_height - 5)
                    
                    # Draw background rectangle for text
                    draw.rectangle(
                        [text_x - 2, text_y - 2, text_x + text_width + 4, text_y + text_height + 2],
                        fill='white', outline=color, width=2
                    )
                    
                    # Draw text
                    if font:
                        draw.text((text_x, text_y), label_text, fill=color, font=font)
                    else:
                        draw.text((text_x, text_y), label_text, fill=color)
            
            # Save enhanced visualization
            combined_pil.save(viz_path)
            
            return segmentation_masks, viz_path
            
        except Exception as e:
            print(f"Error in SAM segmentation: {e}")
            return [], None
    
    def detect_objects_from_image(self, image: Image.Image, save_results: bool = True, use_sam_segmentation: bool = False) -> Dict[str, Any]:
        """
        Perform object detection on a PIL Image object.
        
        Args:
            image: PIL Image object
            save_results: Whether to save results to files
            use_sam_segmentation: Whether to apply SAM segmentation to detected objects
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Get image dimensions
            image_width, image_height = image.size
            
            # Generate detection prompt
            prompt = self.detection_prompt
            
            # Perform inference
            start_time = time.time()
            response = self.vlm.generate([image], [prompt])[0]
            inference_time = time.time() - start_time
            
            # Parse the response
            objects = self._parse_detection_response(response, image_width, image_height)
            
            # Apply SAM segmentation if requested and enabled
            if use_sam_segmentation and self.enable_sam and objects:
                # Create segmentations directory
                seg_dir = os.path.join(self.output_dir, "segmentations")
                os.makedirs(seg_dir, exist_ok=True)
                
                # Apply SAM segmentation to detected objects
                objects = self._apply_sam_segmentation(image, objects, seg_dir)
            
            return {
                'objects': objects,
                'raw_response': response,
                'inference_time': inference_time,
                'image_dimensions': {'width': image_width, 'height': image_height}
            }
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return {
                'objects': [],
                'raw_response': '',
                'inference_time': 0,
                'image_dimensions': {'width': 0, 'height': 0},
                'error': str(e)
            }

def main():
    """
    Example usage of the QwenObjectDetectionPipeline.
    """
    # Initialize the pipeline with SAM segmentation enabled
    pipeline = QwenObjectDetectionPipeline(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda",
        output_dir="./qwen_detection_results",
        enable_sam=True  # Enable SAM segmentation capabilities
    )
    
    # Example: Process a single image with SAM segmentation
    # image_path = "../sampledata/sjsupeople.jpg"
    # results = pipeline.detect_objects(image_path, use_sam_segmentation=True)
    # print(f"Detected {len(results['objects'])} objects with segmentation masks")
    
    # Example: Process a video
    video_path = "output/dashcam_videos/Parking compliance Vantrue dashcam/20250602_065600_00002_T_A.MP4"
    video_results = pipeline.process_video_frames(video_path, extraction_method="scene_change", use_sam_segmentation=True)
    print(f"Extracted {video_results['summary']['frames_extracted']} frames from video")
    
    # Example: Process multiple images
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # batch_results = pipeline.detect_objects_batch(image_paths)
    
    print("QwenObjectDetectionPipeline initialized successfully!")
    print(f"Output directory: {pipeline.output_dir}")
    print(f"SAM segmentation enabled: {pipeline.enable_sam}")
    print("\nTo use the pipeline:")
    print("1. Single image: pipeline.detect_objects('path/to/image.jpg')")
    print("2. With SAM segmentation: pipeline.detect_objects('path/to/image.jpg', use_sam_segmentation=True)")
    print("3. Video processing: pipeline.process_video_frames('path/to/video.mp4')")
    print("4. Multiple images: pipeline.detect_objects_batch(['img1.jpg', 'img2.jpg'])")


if __name__ == "__main__":
    main()