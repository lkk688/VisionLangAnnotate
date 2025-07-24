#!/usr/bin/env python3
"""
Jetson Object Detection Toolkit
All-in-one object detection with multiple models and TensorRT acceleration

Supported Models:
- Faster R-CNN (Two-stage detector)
- YOLOv8 (One-stage detector with TensorRT support)
- OWL-ViT (Zero-shot vision-language model)
- GroundingDINO (Advanced zero-shot detection)

Usage:
    python3 jetson_object_detection_toolkit.py --model yolo --source camera --tensorrt
    python3 jetson_object_detection_toolkit.py --model owl-vit --source image.jpg --prompts "person,car,dog"
    python3 jetson_object_detection_toolkit.py --model faster-rcnn --source video.mp4
"""

import argparse
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import deque
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from transformers import DetrImageProcessor, DetrForObjectDetection, RTDetrImageProcessor, RTDetrForObjectDetection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseDetector:
    """Base class for all object detectors"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.inference_times = deque(maxlen=30)
        self.fps_history = deque(maxlen=30)
        
    def warmup(self, iterations: int = 10):
        """Warm up the model for consistent performance"""
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy_input)
        logger.info(f"Model warmed up with {iterations} iterations")
    
    def detect(self, image: np.ndarray, **kwargs) -> Dict:
        """Abstract method for detection - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'min_inference_time': np.min(self.inference_times) if self.inference_times else 0,
            'max_inference_time': np.max(self.inference_times) if self.inference_times else 0
        }

class FasterRCNNDetector(BaseDetector):
    """Faster R-CNN detector implementation"""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        
        # COCO class names
        self.classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        logger.info("Faster R-CNN model loaded successfully")
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """Perform object detection using Faster R-CNN"""
        start_time = time.time()
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Post-process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Convert labels to class names
        class_names = [self.classes[label] for label in labels]
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.inference_times.append(inference_time * 1000)
        self.fps_history.append(fps)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'inference_time': inference_time * 1000,
            'fps': fps
        }

class YOLODetector(BaseDetector):
    """YOLOv8 detector with optional TensorRT acceleration"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = 'cuda', use_tensorrt: bool = False):
        super().__init__(device)
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.model.to(device)
        self.use_tensorrt = use_tensorrt
        
        # Convert to TensorRT if requested and on Jetson
        if use_tensorrt:
            self._setup_tensorrt(model_path)
        
        logger.info(f"YOLOv8 model loaded successfully (TensorRT: {use_tensorrt})")
    
    def _setup_tensorrt(self, model_path: str):
        """Setup TensorRT optimization for Jetson"""
        try:
            try:
                import tensorrt as trt
                logger.info(f"TensorRT version: {trt.__version__}")
            except ImportError:
                logger.warning("TensorRT not available. Falling back to PyTorch.")
                self.use_tensorrt = False
                return
            
            # Export to ONNX first
            onnx_path = model_path.replace('.pt', '.onnx')
            if not Path(onnx_path).exists():
                logger.info("Exporting model to ONNX...")
                self.model.export(format='onnx', dynamic=True, simplify=True)
            
            # Convert to TensorRT engine
            engine_path = model_path.replace('.pt', '_fp16.trt')
            if not Path(engine_path).exists():
                logger.info("Converting to TensorRT engine...")
                import subprocess
                cmd = [
                    'trtexec',
                    f'--onnx={onnx_path}',
                    f'--saveEngine={engine_path}',
                    '--fp16',
                    '--workspace=2048',
                    '--verbose'
                ]
                subprocess.run(cmd, check=True)
            
            logger.info(f"TensorRT engine ready: {engine_path}")
            
        except Exception as e:
            logger.warning(f"TensorRT setup failed: {e}. Falling back to PyTorch.")
            self.use_tensorrt = False
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict:
        """Perform object detection using YOLOv8"""
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.device
        )
        
        # Extract results
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            class_names = [results[0].names[int(cls)] for cls in classes]
        else:
            boxes = np.array([])
            scores = np.array([])
            classes = np.array([])
            class_names = []
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.inference_times.append(inference_time * 1000)
        self.fps_history.append(fps)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': classes,
            'class_names': class_names,
            'inference_time': inference_time * 1000,
            'fps': fps
        }

class OWLViTDetector(BaseDetector):
    """OWL-ViT zero-shot object detector"""
    
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = 'cuda'):
        super().__init__(device)
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Enable optimizations if available
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        logger.info(f"OWL-ViT model loaded successfully")
    
    def detect(self, image: np.ndarray, text_prompts: List[str], confidence_threshold: float = 0.1) -> Dict:
        """Detect objects using text prompts"""
        start_time = time.time()
        
        # Convert image format if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Prepare inputs
        inputs = self.processor(
            text=[text_prompts],
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )
        
        # Extract results
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"].cpu().numpy()
        class_names = [text_prompts[label] for label in labels]
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.inference_times.append(inference_time * 1000)
        self.fps_history.append(fps)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'text_prompts': text_prompts,
            'inference_time': inference_time * 1000,
            'fps': fps
        }

class GroundingDINODetector(BaseDetector):
    """GroundingDINO zero-shot object detector"""
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None, device: str = 'cuda'):
        super().__init__(device)
        self.use_huggingface = config_path is None or checkpoint_path is None
        
        if self.use_huggingface:
            # Use Hugging Face transformers implementation
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            
            model_name = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.model.eval().to(device)
            
            logger.info(f"GroundingDINO model loaded from Hugging Face: {model_name}")
        else:
            # Use original GroundingDINO implementation
            try:
                import groundingdino.datasets.transforms as T
                from groundingdino.models import build_model
                from groundingdino.util.utils import clean_state_dict
            except ImportError:
                raise ImportError("Please install GroundingDINO from GitHub")
            
            # Load model
            args = type('Args', (), {
                'config_file': config_path,
                'checkpoint': checkpoint_path,
                'device': device
            })()
            
            self.model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            self.model.eval().to(device)
            
            # Image preprocessing
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            logger.info("GroundingDINO model loaded successfully from local files")
    
    def detect(self, image: np.ndarray, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25) -> Dict:
        """Detect objects using natural language descriptions"""
        start_time = time.time()
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        if self.use_huggingface:
            # Use Hugging Face implementation
            inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)  # (height, width)
            results = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=box_threshold
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"]
            
            # Convert boxes from (x1, y1, x2, y2) format if needed
            if len(boxes) > 0:
                # Ensure boxes are in the correct format
                pass  # Hugging Face already returns in correct format
            
        else:
            # Use original GroundingDINO implementation
            image_tensor, _ = self.transform(pil_image, None)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor[None].to(self.device), captions=[text_prompt])
            
            # Post-process (simplified version)
            prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
            prediction_boxes = outputs["pred_boxes"].cpu()[0]
            
            # Filter predictions
            mask = prediction_logits.max(dim=1)[0] > box_threshold
            boxes = prediction_boxes[mask]
            scores = prediction_logits[mask].max(dim=1)[0]
            
            # Convert to absolute coordinates
            h, w = pil_image.size[1], pil_image.size[0]
            boxes = boxes * torch.tensor([w, h, w, h])
            boxes[:, :2] -= boxes[:, 2:] / 2  # Convert center to top-left
            
            boxes = boxes.numpy()
            scores = scores.numpy()
            labels = np.arange(len(boxes))
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.inference_times.append(inference_time * 1000)
        self.fps_history.append(fps)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': [text_prompt] * len(boxes),
            'text_prompt': text_prompt,
            'inference_time': inference_time * 1000,
            'fps': fps
        }

class DETRDetector(BaseDetector):
    """DETR-based object detector using Hugging Face transformers"""
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50", device: str = 'cuda', threshold: float = 0.9):
        super().__init__(device)
        self.model_name = model_name
        self.threshold = threshold
        
        # Load appropriate processor and model based on model name
        if "rt-detr" in model_name.lower():
            self.processor = RTDetrImageProcessor.from_pretrained(model_name)
            self.model = RTDetrForObjectDetection.from_pretrained(model_name)
        else:
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # COCO class names
        self.class_names = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        logger.info(f"Initialized DETR detector with model: {model_name}")
    
    def detect(self, image: np.ndarray, confidence_threshold: float = None) -> Dict:
        """Detect objects in image using DETR"""
        start_time = time.time()
        
        # Use provided threshold or default
        threshold = confidence_threshold if confidence_threshold is not None else self.threshold
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        # Extract results
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        # Convert labels to class names
        class_names = [self.class_names[label] for label in labels]
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.inference_times.append(inference_time * 1000)
        self.fps_history.append(fps)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'inference_time': inference_time * 1000,
            'fps': fps
        }

class ObjectDetectionToolkit:
    """Main toolkit class for object detection"""
    
    def __init__(self, model_type: str, device: str = 'cuda', **kwargs):
        self.model_type = model_type
        self.device = device
        self.detector = self._create_detector(**kwargs)
        
        # Visualization colors
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
    
    def _create_detector(self, **kwargs) -> BaseDetector:
        """Create detector based on model type"""
        if self.model_type == 'faster-rcnn':
            return FasterRCNNDetector(self.device)
        elif self.model_type == 'yolo':
            return YOLODetector(
                model_path=kwargs.get('model_path', 'yolov8n.pt'),
                device=self.device,
                use_tensorrt=kwargs.get('use_tensorrt', False)
            )
        elif self.model_type == 'owl-vit':
            return OWLViTDetector(
                model_name=kwargs.get('model_name', 'google/owlvit-base-patch32'),
                device=self.device
            )
        elif self.model_type == 'grounding-dino':
            return GroundingDINODetector(
                config_path=kwargs.get('config_path'),
                checkpoint_path=kwargs.get('checkpoint_path'),
                device=self.device
            )
        elif self.model_type == 'detr':
            return DETRDetector(
                model_name=kwargs.get('model_name', 'facebook/detr-resnet-50'),
                device=self.device,
                threshold=kwargs.get('threshold', 0.9)
            )
        elif self.model_type == 'detr-resnet-101':
            return DETRDetector(
                model_name=kwargs.get('model_name', 'facebook/detr-resnet-101'),
                device=self.device,
                threshold=kwargs.get('threshold', 0.9)
            )
        elif self.model_type == 'conditional-detr':
            return DETRDetector(
                model_name=kwargs.get('model_name', 'microsoft/conditional-detr-resnet-50'),
                device=self.device,
                threshold=kwargs.get('threshold', 0.9)
            )
        elif self.model_type == 'rt-detr':
            return DETRDetector(
                model_name=kwargs.get('model_name', 'PekingU/rtdetr_r50vd_coco_o365'),
                device=self.device,
                threshold=kwargs.get('threshold', 0.9)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def detect(self, image: np.ndarray, **kwargs) -> Dict:
        """Perform object detection"""
        return self.detector.detect(image, **kwargs)
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Visualize detection results"""
        image_copy = image.copy()
        
        if len(results['boxes']) == 0:
            return image_copy
        
        for i, (box, score, class_name) in enumerate(zip(
            results['boxes'], results['scores'], results['class_names']
        )):
            x1, y1, x2, y2 = box.astype(int)
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_copy
    
    def run_camera(self, camera_id: int = 0, **detect_kwargs):
        """Run real-time detection on camera feed"""
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up the model
        self.detector.warmup()
        
        logger.info("Starting camera detection. Press 'q' to quit, 's' to save frame.")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.detect(frame, **detect_kwargs)
            
            # Visualize results
            annotated_frame = self.visualize_results(frame, results)
            
            # Display performance metrics
            cv2.putText(annotated_frame, f'FPS: {results["fps"]:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Inference: {results["inference_time"]:.1f}ms', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Model: {self.model_type.upper()}', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f'Object Detection - {self.model_type.upper()}', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'detection_frame_{frame_count:04d}.jpg'
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Frame saved as {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        stats = self.detector.get_performance_stats()
        logger.info(f"Performance Statistics:")
        logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
        logger.info(f"  Average Inference Time: {stats['avg_inference_time']:.2f}ms")
        logger.info(f"  Min/Max Inference Time: {stats['min_inference_time']:.2f}/{stats['max_inference_time']:.2f}ms")
    
    def run_image(self, image_path: str, output_path: Optional[str] = None, **detect_kwargs):
        """Run detection on a single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run detection
        results = self.detect(image, **detect_kwargs)
        
        # Visualize results
        annotated_image = self.visualize_results(image, results)
        
        # Save or display result
        if output_path:
            import os
            # Check if output_path is a directory
            if os.path.isdir(output_path):
                # Extract filename from image_path and create full output path
                image_filename = os.path.basename(image_path)
                name, ext = os.path.splitext(image_filename)
                output_filename = f"{name}_detected{ext}"
                full_output_path = os.path.join(output_path, output_filename)
            else:
                full_output_path = output_path
            
            cv2.imwrite(full_output_path, annotated_image)
            logger.info(f"Result saved to {full_output_path}")
        else:
            cv2.imshow(f'Detection Result - {self.model_type.upper()}', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection results
        logger.info(f"Detection Results:")
        logger.info(f"  Found {len(results['boxes'])} objects")
        logger.info(f"  Inference time: {results['inference_time']:.2f}ms")
        for i, (score, class_name) in enumerate(zip(results['scores'], results['class_names'])):
            logger.info(f"  {i+1}. {class_name}: {score:.3f}")
    
    def run_video(self, video_path: str, output_path: Optional[str] = None, **detect_kwargs):
        """Run detection on a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Warm up the model
        self.detector.warmup()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.detect(frame, **detect_kwargs)
            
            # Visualize results
            annotated_frame = self.visualize_results(frame, results)
            
            # Add frame info
            cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'FPS: {results["fps"]:.1f}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(annotated_frame)
            else:
                cv2.imshow(f'Video Detection - {self.model_type.upper()}', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if output_path:
            out.release()
            logger.info(f"Video saved to {output_path}")
        cv2.destroyAllWindows()
        
        # Print performance statistics
        stats = self.detector.get_performance_stats()
        logger.info(f"Video Processing Complete:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
        logger.info(f"  Average Inference Time: {stats['avg_inference_time']:.2f}ms")

# git clone https://github.com/IDEA-Research/GroundingDINO.git
# cd GroundingDINO/
#When CUDA is installed via conda, it does not get installed in the system-wide default location (like /usr/local/cuda). Instead, it is installed in the conda environment directory.
#export CUDA_HOME=$CONDA_PREFIX
# pip install -e .
def main():
    parser = argparse.ArgumentParser(description='Jetson Object Detection Toolkit')
    parser.add_argument('--model', type=str, default='owl-vit',
                       choices=['faster-rcnn', 'yolo', 'owl-vit', 'grounding-dino', 'detr', 'detr-resnet-101', 'conditional-detr', 'rt-detr'],
                       help='Detection model to use')
    parser.add_argument('--source', type=str, default='VisionLangAnnotateModels/sampledata/sjsupeople.jpg',
                       help='Input source: camera, image path, or video path')
    parser.add_argument('--output', type=str, default='./output', help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--tensorrt', default=True, help='Use TensorRT acceleration (YOLO only)') #action='store_true'
    parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    
    # Model-specific arguments
    parser.add_argument('--model-path', type=str, help='Path to model weights, e.g., ultralytics yolo name/path yolov8n.pt, yolo11n.pt')
    parser.add_argument('--prompts', type=str, default="people", help='Text prompts for zero-shot models (comma-separated)')
    parser.add_argument('--config-path', type=str, help='Config path for GroundingDINO (optional, falls back to Hugging Face if not provided)')
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path for GroundingDINO (optional, falls back to Hugging Face if not provided)')
    
    args = parser.parse_args()
    
    def optimize_prompts_for_model(prompts: str, model_type: str) -> str:
        """Optimize prompts for specific zero-shot detection models"""
        if not prompts:
            return prompts
            
        # Split prompts by comma and clean them
        prompt_list = [p.strip() for p in prompts.split(',')]
        
        if model_type == 'owl-vit':
            # OWL-ViT works better with simple, clear object names
            # Remove articles and make singular
            optimized = []
            for prompt in prompt_list:
                # Remove common articles
                prompt = prompt.lower().replace('a ', '').replace('an ', '').replace('the ', '')
                # Convert plural to singular for common cases
                if prompt.endswith('s') and not prompt.endswith('ss'):
                    prompt = prompt[:-1]
                # Capitalize first letter
                prompt = prompt.capitalize()
                optimized.append(prompt)
            return ','.join(optimized)
            
        elif model_type == 'grounding-dino':
            # GroundingDINO works better with descriptive phrases
            # Join all prompts into a single descriptive phrase
            if len(prompt_list) == 1:
                return prompts.lower()
            else:
                # Create a natural language description
                return ' and '.join(prompt_list).lower()
                
        return prompts
    
    # Optimize prompts for the selected model
    if args.prompts and args.model in ['owl-vit', 'grounding-dino']:
        args.prompts = optimize_prompts_for_model(args.prompts, args.model)
        logger.info(f"Optimized prompts for {args.model}: {args.prompts}")
    
    # Prepare model kwargs
    model_kwargs = {}
    if args.model == 'yolo':
        if args.model_path:
            model_kwargs['model_path'] = args.model_path
        model_kwargs['use_tensorrt'] = args.tensorrt
    elif args.model == 'owl-vit':
        if args.model_path:
            model_kwargs['model_name'] = args.model_path
    elif args.model == 'grounding-dino':
        # Optional config and checkpoint paths - will fall back to Hugging Face if not provided
        if args.config_path:
            model_kwargs['config_path'] = args.config_path
        if args.checkpoint_path:
            model_kwargs['checkpoint_path'] = args.checkpoint_path
    elif args.model in ['detr', 'detr-resnet-101', 'conditional-detr', 'rt-detr']:
        if args.model_path:
            model_kwargs['model_name'] = args.model_path
        model_kwargs['threshold'] = args.confidence
    
    # Create toolkit
    toolkit = ObjectDetectionToolkit(args.model, args.device, **model_kwargs)
    
    # Prepare detection kwargs
    detect_kwargs = {}
    if args.model == 'yolo':
        detect_kwargs['conf_threshold'] = args.confidence
        detect_kwargs['iou_threshold'] = args.iou
    elif args.model in ['owl-vit', 'grounding-dino']:
        if not args.prompts:
            raise ValueError(f"{args.model} requires --prompts argument")
        if args.model == 'owl-vit':
            detect_kwargs['confidence_threshold'] = args.confidence
            detect_kwargs['text_prompts'] = args.prompts.split(',')
        else:  # grounding-dino
            detect_kwargs['box_threshold'] = args.confidence
            detect_kwargs['text_prompt'] = args.prompts
    elif args.model in ['detr', 'detr-resnet-101', 'conditional-detr', 'rt-detr']:
        detect_kwargs['confidence_threshold'] = args.confidence
    
    # Run detection based on source
    if args.source == 'camera':
        toolkit.run_camera(**detect_kwargs)
    elif args.source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        toolkit.run_image(args.source, args.output, **detect_kwargs)
    elif args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        toolkit.run_video(args.source, args.output, **detect_kwargs)
    else:
        # Try as camera ID
        try:
            camera_id = int(args.source)
            toolkit.run_camera(camera_id, **detect_kwargs)
        except ValueError:
            raise ValueError(f"Unsupported source format: {args.source}")

if __name__ == '__main__':
    main()


#python object_detection_toolkit.py --model grounding-dino --source VisionLangAnnotateModels/sampledata/sjsupeople.jpg --prompts "person" --confidence 0.1

# # Use default DETR model
# python object_detection_toolkit.py --model detr --source image.jpg --confidence 0.5

# # Use DETR ResNet-101
# python object_detection_toolkit.py --model detr-resnet-101 --source image.jpg

# # Use Conditional DETR
# python object_detection_toolkit.py --model conditional-detr --source image.jpg

# # Use RT-DETR
# python object_detection_toolkit.py --model rt-detr --source image.jpg

# # Use custom model
# python object_detection_toolkit.py --model detr --model-path facebook/detr-resnet-101 --source image.jpg
