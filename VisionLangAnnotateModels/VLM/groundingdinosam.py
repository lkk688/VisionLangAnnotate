import cv2
import os
import datetime
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fractions import Fraction
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import glob
from transformers import AutoProcessor, AutoImageProcessor, AutoModelForUniversalSegmentation, AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor, AutoModelForObjectDetection
from transformers import SamProcessor, SamModel

def perform_box_segmentation(frames_dir, output_dir, model_name, text_prompt="person, car, bicycle, motorcycle, truck, traffic light", 
                                  box_threshold=0.35, text_threshold=0.25):
    """
    Perform object detection using GroundingDINO and segmentation using Segment Anything Model (SAM).
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_dir: Directory to save segmentation results
    - text_prompt: Text prompt describing objects to detect (comma-separated)
    - box_threshold: Confidence threshold for bounding box predictions
    - text_threshold: Confidence threshold for text-to-box associations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for different visualization types
    bbox_dir = os.path.join(output_dir, "bboxes")
    mask_dir = os.path.join(output_dir, "masks")
    combined_dir = os.path.join(output_dir, "combined")
    
    for directory in [bbox_dir, mask_dir, combined_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if "yolo" in model_name:
        # Load YOLOv12 model
        from ultralytics import YOLO
        yolo_model = YOLO(model_name)  # Use nano model, change to s/m/l/x for different sizes
    else:
        # Load GroundingDINO model and processor
        grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Load SAM model
    print("Loading Segment Anything Model...")

    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base") #"facebook/sam-vit-base"
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device) #"facebook/sam2"
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image and metadata
        image = Image.open(img_path).convert("RGB")
        
        # Step 1: 
        boxes = []
        labels = []
        scores = []
        if "yolo" in model_name:
            confidence=0.25
            yolo_results = yolo_model(image, conf=confidence)
            for result in yolo_results:
                # Convert from YOLO format to list of boxes, labels, scores
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
        else:
            # Object detection with GroundingDINO
            #https://huggingface.co/IDEA-Research/grounding-dino-base
            grounding_inputs = grounding_processor(text=text_prompt, images=image, return_tensors="pt").to(device)
            print(grounding_inputs.keys()) #['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']
            #pixel_values: [1, 3, 750, 1333]
            with torch.no_grad():
                outputs = grounding_model(**grounding_inputs)
            
            # Process Grounding DINO results with the grounded method
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            #https://github.com/huggingface/transformers/blob/main/src/transformers/models/grounding_dino/processing_grounding_dino.py
            results = grounding_processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=grounding_inputs.input_ids,
                target_sizes=target_sizes,
                text_threshold=0.4,
                threshold=0.3  # Adjust this threshold as needed
            )[0]
            print(results.keys()) #['scores', 'boxes', 'text_labels', 'labels']
            
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]  # These are now the actual text labels from the prompt
            
        # Step 2: Generate segmentation masks using SAM
        segmentation_masks = []
        
        if len(boxes) > 0:
            for box in boxes:
                # Process the image and bounding box with SAM
                sam_inputs = sam_processor(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
                
                with torch.no_grad():
                    sam_outputs = sam_model(**sam_inputs)
                
                # Get predicted segmentation mask
                masks = sam_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                segmentation_masks.append(masks[0][0][0].numpy())

        # Visualize results
        boxes_tensor = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes
        result_image = visualize_boxseg(
            image, 
            boxes_tensor, 
            segmentation_masks, 
            labels=labels,
            scores=scores
        )
        result_image.save("output/segmentation_result.png")

        # Save segmentation metadata
        seg_meta = {
            "original_frame": base_filename,
            "segments": segmentation_masks,
            #"frame_metadata": frame_meta,
            "model": "GroundingDINO + SAM",
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")

#pip install ultralytics #for yolo
#pip install transformers #for groundingdino
if __name__ == "__main__":
    frames_dir = ""
    output_dir = ""
    seg_output_dir = f"{output_dir}_segmentation"
    perform_box_segmentation(frames_dir, seg_output_dir, \
                model_name=args.model_name_path, text_prompt="a person. a car. a bicycle. a motorcycle. a truck. traffic light", 
                                  box_threshold=0.35, text_threshold=0.25)