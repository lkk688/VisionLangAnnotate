import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import requests
from io import BytesIO
import argparse
from typing import List, Dict, Tuple, Optional, Union

# Import detector modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VisionLangAnnotateModels.detectors.base_model import BaseMultiModel
from VisionLangAnnotateModels.detectors.ultralyticsyolo import YOLOv8Detector


class RegionCaptioner:
    """
    A class for detecting and captioning regions in images using object detection models and PLM.
    This implementation uses YOLO or Hugging Face models for detection and Perception Models for captioning.
    """
    
    def __init__(self, 
                 caption_model_name: str = "facebook/Perception-LM-3B", 
                 detector_type: str = "yolo",
                 detector_model: str = "yolov8x.pt",
                 device: str = None,
                 confidence_threshold: float = 0.25):
        """
        Initialize the RegionCaptioner with specified models and device.
        
        Args:
            caption_model_name: Name of the PLM model to use for captioning (default: facebook/Perception-LM-3B)
            detector_type: Type of detector to use ('yolo', 'hf_yolo', 'detr', 'rtdetr')
            detector_model: Model name or path for the detector
            device: Device to run the models on ('cuda' or 'cpu'). If None, will use CUDA if available.
            confidence_threshold: Confidence threshold for object detection
        """
        self.caption_model_name = caption_model_name
        self.detector_type = detector_type.lower()
        self.detector_model = detector_model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.caption_model = None
        self.caption_processor = None
        self.detector = None
        
        # Load captioning model (PLM)
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.caption_processor = AutoProcessor.from_pretrained(caption_model_name)
            self.caption_model = AutoModelForVision2Seq.from_pretrained(caption_model_name).to(self.device)
            print(f"Loaded PLM model: {caption_model_name}")
        except Exception as e:
            print(f"Error loading PLM model: {e}")
            print("Please install the required packages:")
            print("pip install transformers torch pillow matplotlib opencv-python requests")
            print("For Perception Models, follow installation at: https://github.com/facebookresearch/perception_models")
            sys.exit(1)
        
        # Initialize detector based on type
        self._initialize_detector()
    
    def _initialize_detector(self):
        """
        Initialize the appropriate object detector based on the specified type.
        """
        try:
            if self.detector_type == "yolo":
                # Initialize Ultralytics YOLO detector
                self.detector = YOLOv8Detector(model_path=self.detector_model)
                print(f"Loaded YOLO detector: {self.detector_model}")
            elif self.detector_type in ["hf_yolo", "detr", "rtdetr"]:
                # Initialize Hugging Face detector
                model_type = "yolo" if self.detector_type == "hf_yolo" else self.detector_type
                self.detector = BaseMultiModel(
                    model_type=model_type,
                    model_name=self.detector_model,
                    device=self.device
                )
                print(f"Loaded {self.detector_type.upper()} detector: {self.detector_model}")
            else:
                print(f"Unsupported detector type: {self.detector_type}. Falling back to YOLO.")
                self.detector = YOLOv8Detector(model_path="yolov8x.pt")
                print("Loaded default YOLO detector: yolov8x.pt")
        except Exception as e:
            print(f"Error initializing detector: {e}")
            print("Please install the required packages:")
            print("pip install ultralytics transformers")
            sys.exit(1)
    
    def detect_objects(self, image: np.ndarray, classes: List[str] = None) -> List[Dict]:
        """
        Detect objects in the image using the initialized detector.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            classes: List of class names to filter detections (if None, return all)
            
        Returns:
            List of detection dictionaries with bbox, class_name, and confidence
        """
        if self.detector is None:
            print("Object detector not available")
            return []
        
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect objects based on detector type
        if isinstance(self.detector, YOLOv8Detector):
            # Ultralytics YOLO detector
            detections = self.detector.detect(rgb_image)
            
            # Convert to standardized format
            results = []
            for det in detections:
                if classes is None or det["label"] in classes:
                    if det["confidence"] >= self.confidence_threshold:
                        # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                        x1, y1, x2, y2 = det["bbox"]
                        results.append({
                            "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            "class_name": det["label"],
                            "confidence": det["confidence"]
                        })
        else:
            # Hugging Face detector
            # Convert numpy array to PIL Image if needed
            if isinstance(rgb_image, np.ndarray):
                pil_image = Image.fromarray(rgb_image)
            else:
                pil_image = rgb_image
                
            # Run inference
            inference_result = self.detector.predict(
                pil_image, 
                conf_thres=self.confidence_threshold,
                visualize=False
            )
            
            # Extract detections
            results = []
            for det in inference_result["detections"]:
                if classes is None or det["class_name"] in classes:
                    # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = det["bbox"]
                    results.append({
                        "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        "class_name": det["class_name"],
                        "confidence": det["score"]
                    })
        
        return results
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the image using the object detector.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of face detection dictionaries with bbox, class_name, and confidence
        """
        # Face classes in COCO and other common datasets
        face_classes = ["person", "face", "human face"]
        detections = self.detect_objects(image, classes=face_classes)
        
        # If no faces detected with object detector, try to use face-specific detection
        if not detections:
            try:
                # Try to use OpenCV's face detector as fallback
                face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Convert to our standard format
                detections = [{
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "class_name": "face",
                    "confidence": 0.9  # Arbitrary confidence for OpenCV detector
                } for (x, y, w, h) in faces]
            except Exception as e:
                print(f"Fallback face detection failed: {e}")
        
        return detections
    
    def detect_license_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates in the image using the object detector.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of license plate detection dictionaries with bbox, class_name, and confidence
        """
        # License plate classes in COCO and other common datasets
        plate_classes = ["license plate", "number plate", "car plate"]
        detections = self.detect_objects(image, classes=plate_classes)
        
        # If no license plates detected with object detector, try to use plate-specific detection
        if not detections:
            try:
                # Try to use OpenCV's license plate detector as fallback
                plate_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                plates = plate_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20))
                
                # Convert to our standard format
                detections = [{
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "class_name": "license plate",
                    "confidence": 0.9  # Arbitrary confidence for OpenCV detector
                } for (x, y, w, h) in plates]
            except Exception as e:
                print(f"Fallback license plate detection failed: {e}")
        
        return detections
        
        return plates
    
    def crop_region(self, image: np.ndarray, detection: Dict, 
                   padding: int = 10) -> np.ndarray:
        """
        Crop a region from the image with optional padding.
        
        Args:
            image: Input image as numpy array
            detection: Detection dictionary with 'bbox' key containing [x, y, w, h]
            padding: Padding to add around the bounding box
            
        Returns:
            Cropped image region
        """
        x, y, w, h = detection["bbox"]
        
        # Add padding with boundary checks
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        return image[y_start:y_end, x_start:x_end]
    
    def caption_region(self, image: np.ndarray, detection: Dict) -> str:
        """
        Generate a caption for the image region based on its class.
        
        Args:
            image: Image region as numpy array
            detection: Detection dictionary with 'class_name' key
            
        Returns:
            Caption for the region
        """
        # Convert from OpenCV BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Determine region type based on class name
        class_name = detection["class_name"].lower()
        
        # Prepare prompt based on detected class
        if "face" in class_name or "person" in class_name:
            prompt = "Describe this person's face in detail, including apparent age, gender, expression, and notable features. Do NOT include any personally identifiable information."
            region_type = "face"
        elif "plate" in class_name or "license" in class_name:
            prompt = "Describe this license plate without reading the actual numbers or letters. Focus on its color, shape, condition, and country/region style if recognizable. Do NOT include the actual plate number."
            region_type = "license_plate"
        else:
            prompt = f"Describe this {class_name} in detail. What do you see in this image region?"
            region_type = class_name
        
        # Process the image and generate caption
        inputs = self.caption_processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.caption_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        
        # Decode the generated caption
        caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption, region_type
    
    def process_image(self, image_input, output_path: str = None, 
                     visualize: bool = True, detect_all: bool = False) -> Dict[str, List[Dict]]:
        """
        Process an image to detect and caption regions of interest.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - URL to image (str starting with 'http')
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            output_path: Path to save the visualization (if None, will not save)
            visualize: Whether to display the visualization
            detect_all: Whether to detect all objects (True) or just faces and license plates (False)
            
        Returns:
            Dictionary with detected regions and their captions
        """
        # Load image
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                response = requests.get(image_input)
                img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            # Assume it's already a numpy array
            img = image_input
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image_input)
            # Convert RGB to BGR if needed (OpenCV uses BGR)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, torch.Tensor):
            # Convert PyTorch tensor to numpy array
            img = image_input.detach().cpu().numpy()
            # If tensor is in [C, H, W] format, convert to [H, W, C]
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            # Convert RGB to BGR if needed
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Create a copy for visualization
        vis_img = img.copy()
        
        # Initialize results dictionary
        results = {}
        
        if detect_all:
            # Detect all objects
            detections = self.detect_objects(img)
        else:
            # Detect only faces and license plates
            face_detections = self.detect_faces(img)
            plate_detections = self.detect_license_plates(img)
            detections = face_detections + plate_detections
        
        # Process each detection
        for i, detection in enumerate(detections):
            # Get bounding box
            x, y, w, h = detection["bbox"]
            
            # Determine color based on class
            class_name = detection["class_name"].lower()
            if "face" in class_name or "person" in class_name:
                color = (0, 255, 0)  # Green for faces
                category = "face"
            elif "plate" in class_name or "license" in class_name:
                color = (0, 0, 255)  # Red for license plates
                category = "license_plate"
            else:
                color = (255, 0, 0)  # Blue for other objects
                category = "object"
            
            # Draw rectangle on visualization image
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            
            # Crop the region
            region_img = self.crop_region(img, detection)
            
            # Caption the region
            caption, region_type = self.caption_region(region_img, detection)
            
            # Add text above the rectangle
            label = f"{detection['class_name']} {i+1}"
            cv2.putText(vis_img, label, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Store results by category
            if region_type not in results:
                results[region_type] = []
                
            results[region_type].append({
                'bbox': detection["bbox"],
                'class_name': detection["class_name"],
                'confidence': detection["confidence"],
                'caption': caption
            })
        
        # Display results
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            # Add captions as text
            y_offset = 0.9
            for region_type, detections in results.items():
                for i, detection in enumerate(detections):
                    caption = detection['caption']
                    plt.figtext(0.1, y_offset - (i * 0.05), 
                              f"{detection['class_name']} {i+1}: {caption[:100]}...", 
                              wrap=True, fontsize=8)
                y_offset -= (len(detections) + 0.5) * 0.05
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            
            plt.show()
        
        return results


def main():
    """
    Main function to run the region captioning from command line.
    """
    parser = argparse.ArgumentParser(description='Detect and caption regions of interest in images')
    parser.add_argument('--image', '--input', '-i', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to save output visualization')
    parser.add_argument('--model', '-m', type=str, default='facebook/Perception-LM-3B', 
                       help='PLM model to use for captioning')
    parser.add_argument('--detector', '-d', type=str, default='yolo', choices=['yolo', 'hf_yolo', 'detr', 'rtdetr'],
                       help='Detector type to use')
    parser.add_argument('--detector-model', type=str, default=None,
                       help='Specific detector model to use (e.g., yolov8n.pt or facebook/detr-resnet-50)')
    parser.add_argument('--detect-all', action='store_true', 
                       help='Detect all objects, not just faces and license plates')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    captioner = RegionCaptioner(
        caption_model_name=args.model,
        detector_type=args.detector,
        detector_model=args.detector_model
    )
    
    results = captioner.process_image(
        image_path=args.image,
        output_path=args.output,
        visualize=not args.no_vis,
        detect_all=args.detect_all
    )
    
    # Print results
    print("\nDetection and Captioning Results:")
    
    total_detections = sum(len(detections) for detections in results.values())
    print(f"Found {total_detections} regions of interest")
    
    for region_type, detections in results.items():
        print(f"\n{region_type.capitalize()} detections ({len(detections)}):") 
        for i, detection in enumerate(detections):
            print(f"\n  {detection['class_name']} {i+1}:")
            print(f"    Confidence: {detection.get('confidence', 'N/A'):.4f}")
            print(f"    Position: {detection['bbox']}")
            print(f"    Caption: {detection['caption']}")


if __name__ == "__main__":
    main()