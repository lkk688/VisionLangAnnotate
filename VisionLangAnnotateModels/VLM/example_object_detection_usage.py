#!/usr/bin/env python3
"""
Example usage of the object detection parsing and visualization functionality.
This script demonstrates how to use VLM models to detect objects and generate
structured JSON output with visualizations.
"""

import os
import sys
import json
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vlm_classifierv4 import HuggingFaceVLM
except ImportError:
    print("Warning: vlm_classifierv4 module not found. This example requires the VLM classifier.")
    sys.exit(1)

def example_usage():
    """
    Example of how to use the object detection functionality with VLM models.
    """
    print("Object Detection with VLM Models - Example Usage")
    print("=" * 55)
    
    # Example 1: Using BLIP2 for object detection
    print("\n1. Setting up BLIP2 model for object detection:")
    print("-" * 50)
    
    try:
        # Initialize BLIP2 model
        blip2_model = HuggingFaceVLM(
            model_name="Salesforce/blip2-opt-2.7b",
            model_type="blip2"
        )
        print("✓ BLIP2 model loaded successfully")
        
        # Load an image (you can replace this with your own image)
        if os.path.exists("/tmp/test_detection_image.png"):
            test_image = Image.open("/tmp/test_detection_image.png")
        else:
            print("Creating a test image...")
            test_image = Image.new('RGB', (800, 600), 'white')
            test_image.save("/tmp/test_detection_image.png")
        
        print("✓ Test image loaded")
        
        # Prompt for object detection
        detection_prompt = """
Analyze this image and provide detailed object detection results. 
For each object you detect, please provide:
- Object name
- Bounding box coordinates in format (x1, y1, x2, y2)
- Brief description

Format your response like this:
- **Object Name:**
  **Bounding Box:** (x1, y1, x2, y2)
  **Description:** [description]
"""
        
        print("\n2. Generating object detection results:")
        print("-" * 50)
        
        # Use the new structured output method
        results = blip2_model.generate_with_structured_output(
            images=[test_image],
            prompts=[detection_prompt],
            parse_objects=True,
            visualize=True,
            save_visualizations=True,
            output_dir="./object_detection_results"
        )
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Raw Response: {result['response'][:200]}...")
            
            if result['parsed_objects']:
                print(f"\nParsed Objects ({result['parsed_objects']['total_objects']} found):")
                print(json.dumps(result['parsed_objects'], indent=2))
                
                if result['visualization_path']:
                    print(f"\nVisualization saved to: {result['visualization_path']}")
            else:
                print("No objects detected or parsing failed")
        
    except Exception as e:
        print(f"Error with BLIP2: {e}")
        print("This is expected if the model is not available")
    
    # Example 2: Manual parsing of Qwen-style responses
    print("\n\n3. Manual parsing of Qwen-style object detection responses:")
    print("-" * 60)
    
    # Sample response in Qwen format
    qwen_response = """
I can identify several objects in this image:

- **Person Walking:**
  **Bounding Box:** (120, 50, 220, 280)
  **Description:** A person in casual clothing walking on the sidewalk

- **Red Car:**
  **Bounding Box:** (300, 180, 480, 250)
  **Description:** A red sedan parked along the street

- **Street Lamp:**
  **Bounding Box:** (550, 20, 580, 200)
  **Description:** A tall street lamp providing illumination

- **Building Facade:**
  **Bounding Box:** (50, 80, 250, 300)
  **Description:** The front of a multi-story residential building

- **Traffic Sign:**
  **Bounding Box:** (400, 60, 450, 120)
  **Description:** A circular traffic sign mounted on a pole
"""
    
    # Create a mock VLM for parsing
    class MockVLM(HuggingFaceVLM):
        def __init__(self):
            self.model_name = "qwen-vl"
            self.model_type = "qwen"
    
    mock_vlm = MockVLM()
    
    # Parse the response
    parsed_results = mock_vlm.parse_object_detection_response(qwen_response)
    
    print("Parsed Qwen Response:")
    print(json.dumps(parsed_results, indent=2))
    
    # Create visualization
    if os.path.exists("/tmp/test_detection_image.png"):
        test_image = Image.open("/tmp/test_detection_image.png")
        vis_image = mock_vlm.visualize_detections(
            test_image, 
            parsed_results, 
            "./qwen_detection_visualization.png"
        )
        print("\nVisualization created: ./qwen_detection_visualization.png")
    
    # Example 3: Different response formats
    print("\n\n4. Parsing different response formats:")
    print("-" * 45)
    
    # Simple format
    simple_format = """
Detected objects:
Person: (100, 50, 200, 300)
Car: (250, 200, 400, 280)
Tree: (450, 80, 550, 250)
"""
    
    # List format
    list_format = """
Objects found:
1. Person at coordinates (100, 50, 200, 300)
2. Vehicle at coordinates (250, 200, 400, 280) 
3. Building at coordinates (50, 100, 300, 400)
"""
    
    formats = {
        "Simple Format": simple_format,
        "List Format": list_format
    }
    
    for format_name, response_text in formats.items():
        print(f"\n{format_name}:")
        parsed = mock_vlm.parse_object_detection_response(response_text)
        print(f"Objects detected: {parsed['total_objects']}")
        for obj in parsed['objects']:
            print(f"  - {obj['name']}: {obj['bbox']}")
    
    # Example 4: JSON Schema Documentation
    print("\n\n5. Expected JSON Schema:")
    print("-" * 30)
    
    schema_doc = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Object name/label"},
                        "description": {"type": "string", "description": "Object description"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "Bounding box [x1, y1, x2, y2]"
                        },
                        "confidence": {"type": "number", "description": "Detection confidence (0-1)"}
                    },
                    "required": ["name", "bbox"]
                }
            },
            "total_objects": {"type": "integer"},
            "image_info": {
                "type": "object",
                "properties": {
                    "format": {"type": "string"},
                    "coordinate_system": {"type": "string"}
                }
            }
        }
    }
    
    print("JSON Schema for Object Detection Results:")
    print(json.dumps(schema_doc, indent=2))
    
    print("\n" + "=" * 55)
    print("Example Usage Complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Structured JSON parsing from VLM responses")
    print("✓ Multiple response format support")
    print("✓ Bounding box visualization with labels")
    print("✓ Automatic file saving and organization")
    print("✓ Error handling and fallback parsing")
    
    print("\nFiles created:")
    print("- ./object_detection_results/ (BLIP2 results)")
    print("- ./qwen_detection_visualization.png (Qwen visualization)")

if __name__ == "__main__":
    example_usage()