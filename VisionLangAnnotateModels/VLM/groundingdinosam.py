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
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import pil_to_tensor
from skimage import measure

def process_detection_labels(results, categories, empty_label_option):
    """
    Process detection labels based on model results and handle empty labels according to the specified option.
    
    Parameters:
    - results: Dictionary containing detection results from the model
    - categories: List of category names from the text prompt
    - empty_label_option: How to handle empty labels ('ignore', 'unknown', or 'fallback')
    
    Returns:
    - labels: List of processed labels
    - boxes: Filtered bounding boxes
    - scores: Filtered confidence scores
    """
    boxes = results["boxes"]
    scores = results["scores"]
    
    # Check if text_labels are empty and handle accordingly
    if results["text_labels"] and any(label != "" for label in results["text_labels"]):
        # Use text_labels if they're not empty
        labels = results["text_labels"]
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
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        labels = [label for i, label in enumerate(labels) if valid_detections[i]]
    
    return labels, boxes, scores


def filter_large_detections(labels, boxes, scores, image_width, image_height, max_width_ratio=0.9, max_height_ratio=0.9, max_area_ratio=0.8):
    """
    Filter out detections that are too large relative to the image dimensions.
    
    Parameters:
    - labels: List of detection labels
    - boxes: Tensor of bounding boxes in xyxy format
    - scores: List of confidence scores
    - image_width: Width of the image
    - image_height: Height of the image
    - max_width_ratio: Maximum allowed width ratio (default: 0.9)
    - max_height_ratio: Maximum allowed height ratio (default: 0.9)
    - max_area_ratio: Maximum allowed area ratio (default: 0.8)
    
    Returns:
    - filtered_labels: List of labels after filtering
    - filtered_boxes: Tensor of boxes after filtering
    - filtered_scores: List of scores after filtering
    """
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


def visualize_boxseg(image, boxes, masks, labels=None, scores=None, alpha=0.4, mask_bottom=False, bottom_ratio=0.3, show_category_labels=True):
    """
    Visualize bounding boxes and segmentation masks on an image.
    
    Parameters:
    - image: PIL Image or tensor
    - boxes: Tensor of bounding boxes
    - masks: List of segmentation masks
    - labels: List of label strings
    - scores: List of confidence scores
    - alpha: Transparency of the mask overlay
    - mask_bottom: Whether to mask the bottom portion of the image (e.g., to ignore car engine covers)
    - bottom_ratio: Ratio of the image height to mask from the bottom
    - show_category_labels: Whether to display category labels on the image
    """
    # Convert PIL image to tensor if it's not already
    if not isinstance(image, torch.Tensor):
        image_tensor = pil_to_tensor(image)
    else:
        image_tensor = image
    
    # Draw bounding boxes on the image without labels first
    if boxes is not None and len(boxes) > 0:
        boxes = boxes.to(torch.int)
        # Draw boxes without labels first
        image_with_boxes = draw_bounding_boxes(
            image_tensor, 
            boxes=boxes,
            colors="red",
            width=4
        )
    else:
        image_with_boxes = image_tensor
    
    # Convert to PIL for display
    result_image = to_pil_image(image_with_boxes)
    
    # Add labels to bounding boxes manually for better control over appearance
    if boxes is not None and len(boxes) > 0 and labels is not None:
        draw = ImageDraw.Draw(result_image)
        
        # Try to load a font, use default if not available
        try:
            # Use a larger font size for better visibility
            font = ImageFont.truetype("Arial.ttf", 30)  # Larger font for box labels
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 30)
            except IOError:
                font = ImageFont.load_default()
        
        # Add labels to each box
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # Format label with score if available
            if scores is not None:
                # Truncate long labels
                if len(label) > 15:  # Shorter truncation for box labels
                    display_label = f"{label[:15]}...: {scores[i]:.2f}"
                else:
                    display_label = f"{label}: {scores[i]:.2f}"
            else:
                # Truncate long labels
                if len(label) > 15:
                    display_label = f"{label[:15]}..."
                else:
                    display_label = label
            
            # Get text size for background rectangle
            text_bbox = draw.textbbox((0, 0), display_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text at top of bounding box
            x, y = box[0].item(), max(0, box[1].item() - text_height - 5)
            
            # Draw semi-transparent background for better readability
            draw.rectangle(
                [(x, y), (x + text_width, y + text_height)],
                fill=(0, 0, 0, 180)  # Black with some transparency
            )
            
            # Draw text with outline for better visibility
            for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + offset_x, y + offset_y), display_label, fill="black", font=font)
            
            # Draw main text
            draw.text((x, y), display_label, fill="white", font=font)
    result_image_np = np.array(result_image)
    
    # If we have masks, overlay them on the image
    if masks is not None and len(masks) > 0:
        # Create a colormap for the masks
        colors = plt.cm.get_cmap('tab10', len(masks))
        
        # Create a blank mask with the same size as the image
        colored_mask = np.zeros_like(result_image_np)
        
        # Fill in each mask with a different color
        for i, mask in enumerate(masks):
            color = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask == 1, color[c], colored_mask[:, :, c])
        
        # Blend the mask with the original image
        result_image_np = (1 - alpha) * result_image_np + alpha * colored_mask
    
    # Mask the bottom portion of the image if requested
    if mask_bottom:
        height, width = result_image_np.shape[:2]
        bottom_mask_height = int(height * bottom_ratio)
        # Create a black mask for the bottom portion
        result_image_np[height - bottom_mask_height:, :, :] = 0
    
    # Convert back to PIL image
    result_image = Image.fromarray(result_image_np.astype(np.uint8))
    
    # Add category labels to the image if requested
    if show_category_labels and labels is not None and len(labels) > 0:
        draw = ImageDraw.Draw(result_image)
        # Try to load a font, use default if not available
        try:
            # Use a larger font size for better visibility
            font = ImageFont.truetype("Arial.ttf", 36)  # Increased from 20 to 36
        except IOError:
            # If Arial is not available, try another common font
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 36)
            except IOError:
                # Fall back to default font
                font = ImageFont.load_default()
        
        # Add a legend with category labels
        unique_labels = list(set(labels))
        y_position = 20  # Start a bit lower
        max_label_length = 20  # Maximum characters to display
        
        for label in unique_labels:
            # Truncate long labels
            if len(label) > max_label_length:
                display_label = label[:max_label_length] + "..."
            else:
                display_label = label
            
            # Get text size for background rectangle
            text_bbox = draw.textbbox((0, 0), display_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw semi-transparent background for better readability
            draw.rectangle(
                [(5, y_position - 5), (15 + text_width, y_position + text_height + 5)],
                fill=(0, 0, 0, 180)  # Black with some transparency
            )
            
            # Draw text with outline for better visibility
            # Draw black outline
            for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((10 + offset_x, y_position + offset_y), display_label, fill="black", font=font)
            
            # Draw main text
            draw.text((10, y_position), display_label, fill="white", font=font)
            
            # Increase y_position more to accommodate larger font
            y_position += text_height + 15
    
    return result_image

def perform_box_segmentation(frames_dir, output_dir, model_name, text_prompt="person, car, bicycle, motorcycle, truck, traffic light", 
                                  box_threshold=0.35, text_threshold=0.25, enable_segmentation=True, export_label_studio=True,
                                  mask_bottom=False, bottom_ratio=0.3, show_category_labels=True, empty_label_option="fallback"):
    """
    Perform object detection using GroundingDINO and segmentation using Segment Anything Model (SAM).
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_dir: Directory to save segmentation results
    - model_name: Name or path of the model to use (YOLO or GroundingDINO)
    - text_prompt: Text prompt describing objects to detect (comma-separated)
    - box_threshold: Confidence threshold for bounding box predictions
    - text_threshold: Confidence threshold for text-to-box associations
    - enable_segmentation: Whether to enable SAM segmentation (default: True)
    - export_label_studio: Whether to export annotations in Label Studio format (default: True)
    - mask_bottom: Whether to mask the bottom portion of the image (e.g., to ignore car engine covers) (default: False)
    - bottom_ratio: Ratio of the image height to mask from the bottom (default: 0.3)
    - show_category_labels: Whether to display category labels on the image (default: True)
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
    
    # Load SAM model if segmentation is enabled
    sam_processor = None
    sam_model = None
    if enable_segmentation:
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
            # Format text prompt for better detection
            # Split the text prompt into individual categories
            categories = [cat.strip() for cat in text_prompt.split(',')]
            # Format for GroundingDINO: use a period after each category
            formatted_prompt = ". ".join(categories) + "."
            print(f"Using formatted prompt: {formatted_prompt}")
            
            grounding_inputs = grounding_processor(text=formatted_prompt, images=image, return_tensors="pt").to(device)
            #print(grounding_inputs.keys()) #['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']
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
                text_threshold=text_threshold,  # Use the parameter value
                threshold=box_threshold  # Use the parameter value
            )[0]
            #print(results.keys()) #['scores', 'boxes', 'text_labels', 'labels']
            
            # Process labels and apply filters using utility functions
            labels, boxes, scores = process_detection_labels(results, categories, empty_label_option)
            
            # Apply size filter to remove too large detections
            image_width, image_height = image.size
            labels, boxes, scores = filter_large_detections(labels, boxes, scores, image_width, image_height)
            
            # If all detections were filtered out, return early
            if len(boxes) == 0:
                print("All detections were filtered out")
                return [], [], []
            #labels = results["labels"]
        # Step 2: Generate segmentation masks using SAM if enabled
        segmentation_masks = []
        
        if enable_segmentation and len(boxes) > 0:
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
            scores=scores,
            mask_bottom=mask_bottom,
            bottom_ratio=bottom_ratio,
            show_category_labels=show_category_labels
        )
        
        # Save visualization images to respective directories
        bbox_image_path = os.path.join(bbox_dir, f"{base_name}_bbox.jpg")
        result_image.save(bbox_image_path)
        
        if enable_segmentation and segmentation_masks:
            mask_image_path = os.path.join(mask_dir, f"{base_name}_mask.jpg")
            combined_image_path = os.path.join(combined_dir, f"{base_name}_combined.jpg")
            result_image.save(mask_image_path)  # Save the same image for now
            result_image.save(combined_image_path)

        # Save segmentation metadata
        seg_meta = {
            "original_frame": base_filename,
            "segments": "segmentation_masks" if segmentation_masks else [],  # Don't save actual masks in JSON
            "model": "GroundingDINO + SAM" if enable_segmentation else "GroundingDINO",
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        # Prepare Label Studio format annotations if requested
        if export_label_studio:
            # Get image dimensions
            image_width, image_height = image.size
            
            # Create Label Studio annotation entry
            annotation = {
                "id": base_name,
                "data": {
                    "image": base_filename,
                    "width": image_width,
                    "height": image_height,
                    "file_size": os.path.getsize(img_path)
                },
                "predictions": []
            }
            
            # Create prediction entry
            prediction = {
                "model_version": model_name if "yolo" in model_name else "GroundingDINO",
                "score": 1.0,  # Overall prediction score
                "result": []
            }
            
            # Process each detection for Label Studio format
            for i, (box, label, score) in enumerate(zip(boxes_tensor, labels, scores)):
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1
                
                # Create rectangle annotation
                rect_annotation = {
                    "id": f"{base_name}_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": 100 * x1 / image_width,
                        "y": 100 * y1 / image_height,
                        "width": 100 * width / image_width,
                        "height": 100 * height / image_height,
                        "rectanglelabels": [label]
                    },
                    "score": float(score)
                }
                prediction["result"].append(rect_annotation)
                
                # Add polygon annotation if segmentation is enabled
                if enable_segmentation and i < len(segmentation_masks):
                    mask = segmentation_masks[i]
                    
                    # Find contours at a constant value of 0.5
                    contours = measure.find_contours(mask, 0.5)
                    
                    # Select the largest contour
                    if contours:
                        contour = sorted(contours, key=lambda x: len(x))[-1]
                        
                        # Convert to Label Studio polygon format (normalized coordinates)
                        polygon_points = []
                        for point in contour:
                            y, x = point
                            polygon_points.append([
                                100 * x / image_width,
                                100 * y / image_height
                            ])
                        
                        # Add polygon annotation
                        polygon_annotation = {
                            "id": f"{base_name}_{i}_mask",
                            "type": "polygonlabels",
                            "from_name": "polygon",
                            "to_name": "image",
                            "original_width": image_width,
                            "original_height": image_height,
                            "value": {
                                "points": polygon_points,
                                "polygonlabels": [label]
                            },
                            "score": float(score)
                        }
                        prediction["result"].append(polygon_annotation)
            
            annotation["predictions"].append(prediction)
            
            # Save Label Studio format annotation
            label_studio_path = os.path.join(output_dir, f"{base_name}_labelstudio.json")
            with open(label_studio_path, 'w') as f:
                json.dump(annotation, f, indent=2)
        
        print(f"Saved results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")

def get_category_prompts(option="all"):
    """
    Get text prompts from the twostepcategories.md file.
    Returns a dictionary with category names as keys and prompts as values.
    
    Parameters:
    - option: String indicating which categories to return
             "step1" - Return only step1 classes for simple detection
             "all" - Return both step1 and step2 classes (default)
    """
    # Step2 categories (more detailed/specific categories)
    step2_categories = {
        "Poor or damaged road conditions": "potholes, downed bollard, down power line, flooding, faded bike lane",
        "Infrastructure deterioration": "broken street sign, broken traffic light",
        "Illegal parking": "vehicle blocking bike lane, fire hydrant, red curb",
        "Dumped trash/vegetation": "dumped trash, yard waste, glass, residential/commercial trash cans in bike lane",
        "Graffiti": "graffiti",
        "Permissible obstruction": "construction sign, cone, blocked road",
        "Streetlight outage": "streetlight outage",
        "Obstruction to sweepers": "tree overhang",
        "Abandoned/damaged vehicles": "burned, on jacks, shattered windows, missing tires",
        "Other": "street vendors in bike lane"
    }
    
    # Step1 classes (basic object categories)
    step1_classes = {
        "vehicle": "car, van, truck, motorcycle",
        "person": "pedestrian, worker, vendor",
        "trash_can": "residential trash bin, commercial dumpster",
        "road_sign": "street sign, construction sign",
        "traffic_light": "traffic signal light",
        "tree": "tree, overhanging branch",
        "trash": "dumped trash, debris, yard waste",
        "road_surface": "pothole, water, striping",
        "pole/utility": "pole, bollard, cable, cone",
        "other": "graffiti, street vendor, obstruction"
    }
    
    # Return based on the option parameter
    if option.lower() == "step1":
        print("Using only step1 classes for simple detection")
        return step1_classes
    else:  # Default to "all"
        print("Using all categories (step1 and step2) for comprehensive detection")
        return {**step2_categories, **step1_classes}

#pip install ultralytics #for yolo
#pip install transformers #for groundingdino
#pip install scikit-image #for contour detection
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform object detection and segmentation")
    parser.add_argument("--frames_dir", type=str, default="output/gcs_sources/Sweeper 19303/20250210", help="Directory containing frames to process")
    parser.add_argument("--output_dir", type=str, default="output/vlmresult", help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="IDEA-Research/grounding-dino-base", 
                        help="Model name or path (GroundingDINO or YOLO)")
    parser.add_argument("--categories", type=str, nargs="+", default=["all"], 
                        help="Categories to detect (use 'all' for all categories)")
    parser.add_argument("--category_option", type=str, choices=["step1", "all"], default="step1",
                        help="Which category set to use: 'step1' for basic classes only, 'all' for comprehensive detection")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text confidence threshold")
    parser.add_argument("--enable_segmentation", action="store_true", default=True, help="Enable SAM segmentation")
    parser.add_argument("--export_label_studio", action="store_true", default=True, help="Export in Label Studio format")
    parser.add_argument("--mask_bottom", default=True, help="Mask the bottom portion of the image (e.g., to ignore car engine covers)")
    parser.add_argument("--bottom_ratio", type=float, default=0.35, help="Ratio of the image height to mask from the bottom")
    parser.add_argument("--show_category_labels", action="store_true", default=True, help="Display category labels on the image")
    parser.add_argument("--empty_label_option", type=str, choices=["ignore", "unknown", "fallback"], default="fallback",
                        help="How to handle empty labels: 'ignore' to filter out objects, 'unknown' to label them as unknown, or 'fallback' to use intelligent category assignment")
    
    args = parser.parse_args()
    
    # Get category prompts based on the selected option
    all_categories = get_category_prompts(option=args.category_option)
    
    # Build text prompt based on selected categories
    if "all" in args.categories:
        # Use all categories
        text_prompt = ", ".join(item for sublist in all_categories.values() for item in sublist.split(", "))
    else:
        # Use only selected categories
        selected_prompts = []
        for category in args.categories:
            if category in all_categories:
                selected_prompts.append(all_categories[category])
        text_prompt = ", ".join(selected_prompts)
    
    # Create segmentation output directory
    seg_output_dir = os.path.join(args.output_dir, "segmentation")
    
    # Run detection and segmentation
    perform_box_segmentation(
        frames_dir=args.frames_dir,
        output_dir=seg_output_dir,
        model_name=args.model_name,
        text_prompt=text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        enable_segmentation=args.enable_segmentation,
        export_label_studio=args.export_label_studio,
        mask_bottom=args.mask_bottom,
        bottom_ratio=args.bottom_ratio,
        show_category_labels=args.show_category_labels,
        empty_label_option=args.empty_label_option
    )