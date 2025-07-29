import os
import sys
import json
import glob
from PIL import Image
import time
import argparse
import requests
from typing import List, Dict, Any, Tuple, Optional, Union

# Import the HuggingFaceVLM class from vlm_classifierv3.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VisionLangAnnotateModels.VLM.vlm_classifierv3 import HuggingFaceVLM


def load_labelstudio_annotations(annotation_file: str) -> Dict[str, Any]:
    """
    Load Label Studio annotations from a JSON file.
    
    Args:
        annotation_file: Path to the Label Studio annotations JSON file
        
    Returns:
        Dictionary containing the parsed annotations
    """
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations
    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")
        return {}


def extract_bounding_boxes(annotations: Dict[str, Any], image_dir: str) -> List[Dict[str, Any]]:
    """
    Extract bounding boxes from Label Studio annotations.
    
    Args:
        annotations: Dictionary containing Label Studio annotations
        image_dir: Directory containing the images
        
    Returns:
        List of dictionaries, each containing image path, bounding box, and label
    """
    results = []
    
    # Check if annotations have tasks
    if "tasks" in annotations:
        tasks = annotations["tasks"]
    else:
        # If no "tasks" key, assume the annotations are already a list of tasks
        tasks = annotations if isinstance(annotations, list) else [annotations]
    
    for task in tasks:
        # Get image path
        image_filename = task.get("data", {}).get("image", "")
        if not image_filename:
            continue
            
        # Handle different image path formats
        if "/data/local-files/?d=" in image_filename:
            image_filename = image_filename.split("/data/local-files/?d=")[-1]
        
        # Handle path duplication issues, especially with 'blurred' directories
        # First, normalize paths to handle different separators
        norm_image_dir = os.path.normpath(image_dir)
        norm_image_filename = os.path.normpath(image_filename)
        
        # Extract the base filename without any directory structure
        base_filename = os.path.basename(norm_image_filename)
        
        # Check if the image_filename already contains the full image_dir path
        if norm_image_filename.startswith(norm_image_dir):
            # The image filename already contains the complete directory path
            image_path = norm_image_filename
        elif '/blurred/blurred/' in norm_image_filename or '\\blurred\\blurred\\' in norm_image_filename:
            # Handle the specific case where 'blurred' appears twice in the path
            # This fixes paths like 'path/to/video/blurred/blurred/frame.jpg'
            corrected_path = norm_image_filename.replace('/blurred/blurred/', '/blurred/')
            corrected_path = corrected_path.replace('\\blurred\\blurred\\', '\\blurred\\')
            image_path = corrected_path
        else:
            # Normal case - join the directory and filename
            image_path = os.path.join(norm_image_dir, base_filename)
            
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
        
        # Get image dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        
        # Extract bounding boxes from predictions
        if "predictions" in task:
            for prediction in task["predictions"]:
                if "result" in prediction:
                    for result in prediction["result"]:
                        if result.get("type") == "rectanglelabels":
                            value = result.get("value", {})
                            
                            # Get bounding box coordinates (convert from percentages to pixels)
                            x_percent = value.get("x", 0)
                            y_percent = value.get("y", 0)
                            width_percent = value.get("width", 0)
                            height_percent = value.get("height", 0)
                            
                            x1 = int(x_percent * width / 100)
                            y1 = int(y_percent * height / 100)
                            x2 = int(x1 + (width_percent * width / 100))
                            y2 = int(y1 + (height_percent * height / 100))
                            
                            # Get label
                            label = value.get("rectanglelabels", [""])[0]
                            
                            results.append({
                                "image_path": image_path,
                                "bbox": (x1, y1, x2, y2),
                                "label": label,
                                "score": result.get("score", 1.0)
                            })
    
    # Check for any None values in the results list and replace with default values
    for i in range(len(results)):
        if results[i] is None:
            print(f"Warning: Missing result at index {i}. Using default value.")
            results[i] = {
                "class": "Other",
                "confidence": 0.5,
                "reasoning": "No result provided by Ollama processing"
            }
    
    return results


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop an image based on a bounding box.
    
    Args:
        image: PIL Image object
        bbox: Tuple of (x1, y1, x2, y2) coordinates
        
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def generate_prompt(label: str) -> str:
    """
    Generate a prompt for the VLM based on the object label.
    
    Args:
        label: Object label from the first detection step
        
    Returns:
        Prompt string for the VLM
    """
    # Define prompts for different object types
    prompts = {
        "person": "Describe this person in detail. What are they doing? What are they wearing?",
        "car": "Describe this vehicle in detail. What type of car is it? What color is it? Is it parked properly?",
        "truck": "Describe this truck in detail. What type of truck is it? What is its purpose?",
        "bicycle": "Describe this bicycle in detail. What type of bicycle is it? Is it parked properly?",
        "motorcycle": "Describe this motorcycle in detail. What type of motorcycle is it?",
        "bus": "Describe this bus in detail. What type of bus is it? Is it in service?",
        "traffic light": "What color is this traffic light showing? Is it functioning properly?",
        "stop sign": "Describe this stop sign. Is it damaged or vandalized?",
        "other": "Describe this object in detail. What is it and what is its condition?"
    }
    
    # Find the most appropriate prompt
    for key in prompts:
        if key in label.lower():
            return prompts[key]
    
    # Default prompt
    return prompts["other"]

# def generate_prompt_json(label: str) -> str:
#     """
#     Generate a prompt that asks the VLM to return a JSON object with 'class' and 'description' fields.

#     Args:
#         label: Object label from the first detection step (e.g., YOLO output)

#     Returns:
#         Prompt string for the VLM
#     """
#     allowed_classes = [
#         "Car", "Truck", "Vehicle blocking bike lane", "Burned vehicle", "Police car", "Pedestrian", "Worker",
#         "Street vendor", "Residential trash bin", "Commercial dumpster", "Street sign", "Construction sign",
#         "Traffic signal light", "Broken traffic lights", "Tree", "Overhanging branch", "Dumped trash", "Yard waste",
#         "Glass/debris", "Pothole", "Unclear bike lane markings", "Utility pole", "Downed bollard", "Cone",
#         "Streetlight outage", "Graffiti", "Bench", "Vehicle in bike lane", "Bicycle", "Scooter", "Wheelchair",
#         "Bus", "Train", "Ambulance", "Fire truck", "Other"
#     ]

#     class_list_str = ", ".join(allowed_classes)

#     prompt = (
#         f"You are given a cropped image region from a street scene containing a possible object.\n"
#         f"Your task is to analyze the object and return a JSON object with the following two fields:\n"
#         f"1. 'class': the most appropriate label from the following list (case-sensitive). If nothing relevant is detected, return 'no object'.\n"
#         f"2. 'description': a detailed explanation of any city-related issue related to the object (e.g., tree overhangs, debris, potholes, broken signs or signals).\n\n"
#         f"Valid class list: [{class_list_str}]\n\n"
#         f"Example output:\n"
#         f"{{\n"
#         f"  \"class\": \"Dumped trash\",\n"
#         f"  \"description\": \"A pile of household trash is dumped on the street, partially blocking the bike lane.\"\n"
#         f"}}\n\n"
#         f"Now analyze this object and respond with only the JSON object."
#     )

#     return prompt

def generate_prompt_json(label: str) -> str:
    """
    Generate a prompt that asks the VLM to return a JSON object with 'class' and 'description' fields.

    Args:
        label: Object label from the first detection step (e.g., YOLO output)

    Returns:
        Prompt string for the VLM
    """
    allowed_classes = [
        "Car", "Truck", "Vehicle blocking bike lane", "Burned vehicle", "Police car", "Pedestrian", "Worker",
        "Street vendor", "Residential trash bin", "Commercial dumpster", "Street sign", "Construction sign",
        "Traffic signal light", "Broken traffic lights", "Tree", "Overhanging branch", "Dumped trash", "Yard waste",
        "Glass/debris", "Pothole", "Unclear bike lane markings", "Utility pole", "Downed bollard", "Cone",
        "Streetlight outage", "Graffiti", "Bench", "Vehicle in bike lane", "Bicycle", "Scooter", "Wheelchair",
        "Bus", "Train", "Ambulance", "Fire truck", "Other"
    ]

    class_list_str = ", ".join(allowed_classes)

    prompt = (
        "You are given a cropped image of a possible object from a city street.\n\n"
        "Your task is to analyze the object and return a JSON object with exactly two fields:\n"
        "1. \"class\": the most appropriate label from the allowed list below, or \"no object\" if you are unsure or the object is irrelevant.\n"
        "2. \"description\": a short explanation of the object and any related city issue (e.g., trash, road damage, signage problem).\n\n"
        "Rules:\n"
        "- Only select a class if the object is clearly visible and matches the class.\n"
        "- DO NOT assume rare or extreme classes like \"Burned vehicle\" unless the evidence is very clear.\n"
        "- If the object is unclear, damaged beyond recognition, or does not match any label, return:\n"
        "  \"class\": \"Other\"\n"
        "- Be concise and accurate.\n\n"
        f"Allowed class list:\n[{class_list_str}]\n\n"
        "Example outputs:\n"
        "{\n"
        "  \"class\": \"Dumped trash\",\n"
        "  \"description\": \"A pile of household trash is dumped on the street, partially blocking the bike lane.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"Other\",\n"
        "  \"description\": \"The region is too unclear to determine the object type or condition.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"Car\",\n"
        "  \"description\": \"A regular sedan is parked at the curb. No visible issues.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"Broken traffic lights\",\n"
        "  \"description\": \"The traffic signal appears to be damaged with the lights not functioning.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"no object\",\n"
        "  \"description\": \"The region has only background and does not contain any object.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"Vehicle blocking bike lane\",\n"
        "  \"description\": \"A vehicle is parked directly over the painted bike lane, obstructing the path for cyclists.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"Vehicle blocking bike lane\",\n"
        "  \"description\": \"A large SUV is stopped in the bike lane, forcing bicycles into the traffic lane.\"\n"
        "}\n\n"
        "{\n"
        "  \"class\": \"Vehicle blocking bike lane\",\n"
        "  \"description\": \"A delivery truck is occupying the full width of the bike lane near the curb.\"\n"
        "}\n\n"
        "Now respond with ONLY the JSON object."
    )

    return prompt

def process_with_ollama(descriptions: List[str], step1_labels: List[str], image_paths: List[str], ollama_model: str, allowed_classes: List[str] = None) -> List[Dict[str, Any]]:
    """
    Process VLM descriptions using a local Ollama model to standardize outputs.
    Processes all objects from a single image together to save processing time.
    
    Args:
        descriptions: List of VLM descriptions to process
        step1_labels: List of step1 detection labels corresponding to each description
        image_paths: List of image paths corresponding to each description
        ollama_model: Name of the Ollama model to use (e.g., "llama3")
        allowed_classes: List of allowed class names for standardization
        
    Returns:
        List of dictionaries containing processed results
    """
    if allowed_classes is None:
        allowed_classes = [
            "Car", "Truck", "Vehicle blocking bike lane", "Burned vehicle", "Police car", 
            "Pedestrian", "Worker", "Street vendor", "Residential trash bin", "Commercial dumpster", 
            "Street sign", "Construction sign", "Traffic signal light", "Broken traffic lights", 
            "Tree", "Overhanging branch", "Dumped trash", "Yard waste", "Glass/debris", 
            "Pothole", "Unclear bike lane markings", "Utility pole", "Downed bollard", 
            "Cone", "Streetlight outage", "Graffiti", "Bench", "Vehicle in bike lane", 
            "Bicycle", "Scooter", "Wheelchair", "Bus", "Train", "Ambulance", "Fire truck", "Other"
        ]
    
    results = []
    
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    
    # Group descriptions by image path
    image_groups = {}
    for i, (description, step1_label, image_path) in enumerate(zip(descriptions, step1_labels, image_paths)):
        if image_path not in image_groups:
            image_groups[image_path] = []
        image_groups[image_path].append({
            "index": i,
            "description": description,
            "step1_label": step1_label
        })
    
    # Process each image's objects together
    for image_path, objects in image_groups.items():
        # Create a batch prompt for all objects in this image
        objects_text = ""
        for i, obj in enumerate(objects):
            objects_text += f"Object {i+1} (Step1 label: {obj['step1_label']}):\n{obj['description']}\n\n"
        
        prompt = f"""Based on the following descriptions of objects detected in a single image, classify each object into one of these categories: {', '.join(allowed_classes)}.
        
        {objects_text}
        Return a JSON array where each element corresponds to one of the objects above, in the same order. Each element should be a JSON object with the following structure:
        {{
            "class": "The most appropriate class from the allowed list",
            "confidence": A number between 0 and 1 indicating your confidence,
            "reasoning": "Brief explanation for your classification"
        }}
        
        Only return the JSON array, nothing else."""
        
        try:
            # Call Ollama API
            response = requests.post(url, json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json"  # Request JSON format if the model supports it
            }, timeout=60)  # Increased timeout for batch processing
            
            if response.status_code == 200:
                # Parse the response
                response_text = response.json().get("response", "")
                
                # Extract JSON from response (handling potential text before/after JSON)
                try:
                    # First try direct JSON parsing
                    try:
                        processed_results = json.loads(response_text)
                        # Ensure we got an array
                        if not isinstance(processed_results, list):
                            # Try to find array in the response
                            array_start = response_text.find('[')
                            array_end = response_text.rfind(']')
                            if array_start >= 0 and array_end >= 0 and array_end > array_start:
                                json_str = response_text[array_start:array_end+1]
                                processed_results = json.loads(json_str)
                            else:
                                raise ValueError("Response is not a JSON array")
                    except (json.JSONDecodeError, ValueError):
                        # If direct parsing fails, try to extract JSON from text
                        array_start = response_text.find('[')
                        array_end = response_text.rfind(']')
                        
                        if array_start >= 0 and array_end >= 0 and array_end > array_start:
                            json_str = response_text[array_start:array_end+1]
                            processed_results = json.loads(json_str)
                        else:
                            # If no valid JSON array found, create default responses
                            raise ValueError("No valid JSON array found in response")
                    
                    # Validate and process each result
                    if len(processed_results) != len(objects):
                        print(f"Warning: Expected {len(objects)} results but got {len(processed_results)}. Padding or truncating as needed.")
                    
                    # Map results back to original indices
                    for i, obj in enumerate(objects):
                        if i < len(processed_results):
                            result = processed_results[i]
                            # Validate required fields
                            if not all(k in result for k in ["class", "confidence", "reasoning"]):
                                print(f"Warning: Missing required fields for object {i+1}. Using defaults.")
                                result = {
                                    "class": "Other",
                                    "confidence": 0.5,
                                    "reasoning": "Missing fields in Ollama response"
                                }
                            
                            # Ensure the class is in the allowed list
                            if result["class"] not in allowed_classes:
                                closest_match = min(allowed_classes, key=lambda x: abs(len(x) - len(result["class"])))
                                print(f"Warning: Class '{result['class']}' not in allowed list. Using '{closest_match}' instead.")
                                result["class"] = closest_match
                            
                            # Ensure confidence is a float between 0 and 1
                            try:
                                confidence = float(result["confidence"])
                                result["confidence"] = max(0.0, min(1.0, confidence))
                            except (ValueError, TypeError):
                                result["confidence"] = 0.7  # Default confidence
                        else:
                            # Create default result if we don't have enough results
                            result = {
                                "class": "Other",
                                "confidence": 0.5,
                                "reasoning": "No result provided by Ollama"
                            }
                        
                        # Store result at the original index
                        while len(results) <= obj["index"]:
                            results.append(None)
                        results[obj["index"]] = result
                        
                except Exception as e:
                    # Fallback if JSON parsing or validation fails
                    print(f"Error processing Ollama response: {str(e)}")
                    # Create default results for all objects in this image
                    for obj in objects:
                        default_result = {
                            "class": "Other",
                            "confidence": 0.5,
                            "reasoning": f"Failed to parse valid JSON from Ollama response: {str(e)}"
                        }
                        # Store result at the original index
                        while len(results) <= obj["index"]:
                            results.append(None)
                        results[obj["index"]] = default_result
            else:
                # Handle API error
                error_msg = f"Ollama API error: {response.status_code}"
                print(error_msg)
                # Create default results for all objects in this image
                for obj in objects:
                    default_result = {
                        "class": "Other",
                        "confidence": 0.5,
                        "reasoning": error_msg
                    }
                    # Store result at the original index
                    while len(results) <= obj["index"]:
                        results.append(None)
                    results[obj["index"]] = default_result
        except requests.exceptions.Timeout:
            # Handle timeout specifically
            error_msg = f"Timeout while calling Ollama API (60s)"
            print(error_msg)
            # Create default results for all objects in this image
            for obj in objects:
                default_result = {
                    "class": "Other",
                    "confidence": 0.5,
                    "reasoning": error_msg
                }
                # Store result at the original index
                while len(results) <= obj["index"]:
                    results.append(None)
                results[obj["index"]] = default_result
        except Exception as e:
            # Handle connection or other errors
            error_msg = f"Error calling Ollama: {str(e)}"
            print(error_msg)
            # Create default results for all objects in this image
            for obj in objects:
                default_result = {
                    "class": "Other",
                    "confidence": 0.5,
                    "reasoning": error_msg
                }
                # Store result at the original index
                while len(results) <= obj["index"]:
                    results.append(None)
                results[obj["index"]] = default_result
    
    return results


def run_three_step_pipeline(image_dir: str, annotation_file: str, model_name: str = "Salesforce/blip2-opt-2.7b", 
                         device: str = "cuda", output_file: str = "three_step_results.json",
                         ollama_model: Optional[str] = None, allowed_classes: Optional[List[str]] = None):
    """
    Run the three-step pipeline:
    1. Parse Label Studio annotations to get object bounding boxes
    2. Use HuggingFaceVLM to classify each detected object
    3. (Optional) Use Ollama to standardize VLM outputs into predefined classes
    
    Args:
        image_dir: Directory containing the images
        annotation_file: Path to the Label Studio annotations JSON file
        model_name: Name of the HuggingFaceVLM model to use
        device: Device to run the model on ("cuda" or "cpu")
        output_file: Path to save the results
        ollama_model: (Optional) Name of the Ollama model to use for post-processing
        allowed_classes: (Optional) List of allowed class names for standardization
        
    Returns:
        List of dictionaries containing the results
    """
    print(f"Loading annotations from {annotation_file}...")
    annotations = load_labelstudio_annotations(annotation_file)
    
    print(f"Extracting bounding boxes...")
    bbox_results = extract_bounding_boxes(annotations, image_dir)#list of dict 
    print(f"Found {len(bbox_results)} bounding boxes")
    
    if not bbox_results:
        print("No bounding boxes found. Exiting.")
        return []
    
    print(f"Initializing HuggingFaceVLM with model {model_name}...")
    vlm = HuggingFaceVLM(model_name=model_name, device=device)
    
    final_results = []
    
    # Group bounding boxes by image to avoid loading the same image multiple times
    image_to_bboxes = {}
    for result in bbox_results:
        image_path = result["image_path"]
        if image_path not in image_to_bboxes:
            image_to_bboxes[image_path] = []
        image_to_bboxes[image_path].append(result)
    
    # Process each image
    for image_path, bboxes in image_to_bboxes.items():
        print(f"Processing {len(bboxes)} objects in {os.path.basename(image_path)}...")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare crops and prompts
            crops = []
            prompts = []
            
            for bbox_result in bboxes:
                bbox = bbox_result["bbox"]
                label = bbox_result["label"]
                
                # Crop image
                crop = crop_image(image, bbox)
                
                # Generate prompt
                #prompt = generate_prompt(label)
                prompt = generate_prompt_json(label)
                
                crops.append(crop)
                prompts.append(prompt)
            
            # Run VLM inference (Step 2)
            start_time = time.time()
            vlm_descriptions = vlm.generate(crops, prompts)#multiple iteractions?
            end_time = time.time()
            
            print(f"VLM inference completed in {end_time - start_time:.2f} seconds")
            
            # Step 3: Process with Ollama if specified
            ollama_results = None
            if ollama_model:
                print(f"Processing VLM descriptions with Ollama model {ollama_model}...")
                start_time = time.time()
                # Prepare step1 labels and image paths for batch processing
                step1_labels = [bbox_result["label"] for bbox_result in bboxes]
                image_paths = [image_path] * len(vlm_descriptions)
                ollama_results = process_with_ollama(vlm_descriptions, step1_labels, image_paths, ollama_model, allowed_classes)
                end_time = time.time()
                print(f"Ollama processing completed in {end_time - start_time:.2f} seconds")
            
            # Combine results
            for i, (bbox_result, vlm_description) in enumerate(zip(bboxes, vlm_descriptions)):
                result = {
                    "image_path": image_path,
                    "bbox": bbox_result["bbox"],
                    "step1_label": bbox_result["label"],
                    "step1_score": bbox_result["score"],
                    "vlm_description": vlm_description
                }
                
                # Parse vlm_description if it's in JSON format
                try:
                    if isinstance(vlm_description, str) and vlm_description.strip().startswith('{') and vlm_description.strip().endswith('}'): 
                        vlm_json = json.loads(vlm_description)
                        if "class" in vlm_json:
                            result["vlm_class"] = vlm_json["class"]
                        if "description" in vlm_json:
                            result["vlm_detailed_description"] = vlm_json["description"]
                except json.JSONDecodeError:
                    # Not valid JSON, continue with original vlm_description
                    pass
                
                # Add Ollama results if available
                if ollama_results:
                    result["ollama_class"] = ollama_results[i]["class"]
                    result["ollama_confidence"] = ollama_results[i]["confidence"]
                    result["ollama_reasoning"] = ollama_results[i]["reasoning"]
                
                final_results.append(result)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return final_results


def run_two_step_pipeline(image_dir: str, annotation_file: str, model_name: str = "Salesforce/blip2-opt-2.7b", 
                         device: str = "cuda", output_file: str = "two_step_results.json"):
    """
    Run the two-step pipeline:
    1. Parse Label Studio annotations to get object bounding boxes
    2. Use HuggingFaceVLM to classify each detected object
    
    Args:
        image_dir: Directory containing the images
        annotation_file: Path to the Label Studio annotations JSON file
        model_name: Name of the HuggingFaceVLM model to use
        device: Device to run the model on ("cuda" or "cpu")
        output_file: Path to save the results
        
    Returns:
        List of dictionaries containing the results
    """
    # Call the three-step pipeline without Ollama processing
    return run_three_step_pipeline(image_dir, annotation_file, model_name, device, output_file)


def main():
    parser = argparse.ArgumentParser(description="Pipeline for object detection and classification")
    parser.add_argument("--image_dir", default='output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215/blurred', help="Directory containing the images")
    parser.add_argument("--annotation_file", default='output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215/labelstudio_annotations.json', help="Path to the Label Studio annotations JSON file")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the HuggingFaceVLM model to use")
    parser.add_argument("--device", default="cuda", help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--output_file", default="pipeline_results.json", help="Path to save the results")
    parser.add_argument("--ollama_model", help="(Optional) Name of the Ollama model to use for post-processing")
    parser.add_argument("--allowed_classes", help="(Optional) Comma-separated list of allowed class names for standardization")
    
    args = parser.parse_args()
    
    # Parse allowed classes if provided
    allowed_classes = None
    if args.allowed_classes:
        allowed_classes = [cls.strip() for cls in args.allowed_classes.split(',')]
    
    # Determine which pipeline to run based on whether Ollama model is specified
    if args.ollama_model:
        print(f"Running three-step pipeline with Ollama model {args.ollama_model}...")
        run_three_step_pipeline(
            image_dir=args.image_dir,
            annotation_file=args.annotation_file,
            model_name=args.model_name,
            device=args.device,
            output_file=args.output_file,
            ollama_model=args.ollama_model,
            allowed_classes=allowed_classes
        )
    else:
        print("Running two-step pipeline...")
        run_two_step_pipeline(
            image_dir=args.image_dir,
            annotation_file=args.annotation_file,
            model_name=args.model_name,
            device=args.device,
            output_file=args.output_file
        )


if __name__ == "__main__":
    main()