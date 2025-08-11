import os
import sys
import torch
from PIL import Image
import argparse
import json

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from VisionLangAnnotateModels.VLM.vllm_utils import VLLMBackend, VLLM_AVAILABLE
from VisionLangAnnotateModels.VLM.vlm_classifierv3 import VLMClassifier
from VisionLangAnnotateModels.VLM.ollama_utils import process_with_ollama_text

def process_descriptions_with_vllm(descriptions, step1_labels, image_paths, vllm_model, allowed_classes=None):
    """
    Process VLM descriptions using a vLLM model to standardize outputs.
    This is similar to the process_with_ollama function but uses vLLM instead.
    
    Args:
        descriptions: List of VLM descriptions to process
        step1_labels: List of step1 detection labels corresponding to each description
        image_paths: List of image paths corresponding to each description
        vllm_model: Name of the vLLM model to use
        allowed_classes: List of allowed class names for standardization
        
    Returns:
        List of dictionaries containing processed results
    """
    if not VLLM_AVAILABLE:
        print("vLLM is not available. Please install it with 'pip install vllm'.")
        return []
    
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
            objects_text += f"Object {i+1}: {obj['description']}\n"
        
        # Create the system prompt
        system_prompt = f"""You are an AI assistant that standardizes object descriptions from images.
        I will provide descriptions of objects detected in an image.
        For each object, classify it into one of these categories: {', '.join(allowed_classes)}.
        
        Respond in JSON format with an array of objects, each containing:
        1. "object_index": The object number (1, 2, etc.)
        2. "class": The standardized class name (must be from the allowed list)
        3. "confidence": Your confidence level (high, medium, low)
        4. "reasoning": Brief explanation for your classification
        
        Example response format:
        {{"results": [
            {{"object_index": 1, "class": "Car", "confidence": "high", "reasoning": "The description clearly indicates a passenger vehicle."}},
            {{"object_index": 2, "class": "Pothole", "confidence": "medium", "reasoning": "The description mentions a hole in the road."}}
        ]}}
        """
        
        # Create the user prompt
        user_prompt = f"Here are the object descriptions from an image:\n{objects_text}\nPlease standardize these descriptions into the allowed categories."
        
        # Process with vLLM
        from VisionLangAnnotateModels.VLM.vllm_utils import process_with_vllm_text
        response = process_with_vllm_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            vllm_model=vllm_model,
            max_tokens=1000,
            temperature=0.1  # Lower temperature for more deterministic outputs
        )
        
        if response["success"]:
            try:
                # Extract the JSON part from the response
                response_text = response["response"]
                
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{\s*"results"\s*:\s*\[.+?\]\s*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
                
                # Parse the JSON
                parsed_response = json.loads(json_str)
                
                # Map the results back to the original indices
                for result in parsed_response.get("results", []):
                    object_index = result.get("object_index", 0) - 1  # Convert to 0-indexed
                    if 0 <= object_index < len(objects):
                        orig_index = objects[object_index]["index"]
                        results.append({
                            "index": orig_index,
                            "description": descriptions[orig_index],
                            "step1_label": step1_labels[orig_index],
                            "standardized_class": result.get("class", "Other"),
                            "confidence": result.get("confidence", "low"),
                            "reasoning": result.get("reasoning", "")
                        })
            except Exception as e:
                print(f"Error parsing vLLM response: {str(e)}")
                print(f"Response was: {response['response']}")
                
                # Add error results for all objects in this group
                for obj in objects:
                    results.append({
                        "index": obj["index"],
                        "description": descriptions[obj["index"]],
                        "step1_label": step1_labels[obj["index"]],
                        "standardized_class": "Error",
                        "confidence": "low",
                        "reasoning": f"Error processing with vLLM: {str(e)}"
                    })
        else:
            print(f"Error from vLLM: {response.get('error', 'Unknown error')}")
            
            # Add error results for all objects in this group
            for obj in objects:
                results.append({
                    "index": obj["index"],
                    "description": descriptions[obj["index"]],
                    "step1_label": step1_labels[obj["index"]],
                    "standardized_class": "Error",
                    "confidence": "low",
                    "reasoning": f"Error from vLLM: {response.get('error', 'Unknown error')}"
                })
    
    # Sort results by original index
    results.sort(key=lambda x: x["index"])
    return results

def compare_ollama_vllm(descriptions, step1_labels, image_paths, ollama_model, vllm_model, allowed_classes=None):
    """
    Compare the results of processing with Ollama and vLLM.
    
    Args:
        descriptions: List of VLM descriptions to process
        step1_labels: List of step1 detection labels corresponding to each description
        image_paths: List of image paths corresponding to each description
        ollama_model: Name of the Ollama model to use
        vllm_model: Name of the vLLM model to use
        allowed_classes: List of allowed class names for standardization
        
    Returns:
        Tuple of (ollama_results, vllm_results)
    """
    # Process with Ollama
    print(f"Processing with Ollama model: {ollama_model}")
    try:
        from VisionLangAnnotateModels.VLM.ollama_utils import process_with_ollama
        ollama_results = process_with_ollama(
            descriptions=descriptions,
            step1_labels=step1_labels,
            image_paths=image_paths,
            ollama_model=ollama_model,
            allowed_classes=allowed_classes
        )
    except Exception as e:
        print(f"Error processing with Ollama: {str(e)}")
        ollama_results = []
    
    # Process with vLLM
    print(f"Processing with vLLM model: {vllm_model}")
    try:
        vllm_results = process_descriptions_with_vllm(
            descriptions=descriptions,
            step1_labels=step1_labels,
            image_paths=image_paths,
            vllm_model=vllm_model,
            allowed_classes=allowed_classes
        )
    except Exception as e:
        print(f"Error processing with vLLM: {str(e)}")
        vllm_results = []
    
    return ollama_results, vllm_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integration test for vLLM backend")
    parser.add_argument("--ollama_model", type=str, default="llama3",
                        help="Ollama model to use")
    parser.add_argument("--vllm_model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="vLLM model to use")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to an image file")
    args = parser.parse_args()
    
    # Check if vLLM is available
    if not VLLM_AVAILABLE:
        print("vLLM is not available. Please install it with 'pip install vllm'.")
        return
    
    # Create sample data
    descriptions = [
        "A red car parked on the street",
        "A large pothole in the middle of the road",
        "A traffic light that appears to be broken",
        "A bicycle locked to a bike rack"
    ]
    
    step1_labels = ["car", "road_damage", "traffic_light", "bicycle"]
    image_paths = [args.image] * len(descriptions)
    
    # Compare Ollama and vLLM
    ollama_results, vllm_results = compare_ollama_vllm(
        descriptions=descriptions,
        step1_labels=step1_labels,
        image_paths=image_paths,
        ollama_model=args.ollama_model,
        vllm_model=args.vllm_model
    )
    
    # Print results
    print("\nOllama Results:")
    for result in ollama_results:
        print(f"Object: {result['description']}")
        print(f"  Class: {result.get('standardized_class', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        print()
    
    print("\nvLLM Results:")
    for result in vllm_results:
        print(f"Object: {result['description']}")
        print(f"  Class: {result.get('standardized_class', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        print()

if __name__ == "__main__":
    main()