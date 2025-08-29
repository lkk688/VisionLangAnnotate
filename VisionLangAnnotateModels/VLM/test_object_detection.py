#!/usr/bin/env python3
"""
Test script for object detection parsing and visualization functionality.
Demonstrates how to use the new structured JSON output features.
"""

import os
import sys
import json
import argparse
from PIL import Image, ImageDraw
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vlm_classifierv4 import HuggingFaceVLM
except ImportError:
    print("Warning: vlm_classifierv4 module not found. This test requires the VLM classifier.")
    sys.exit(1)

def create_test_image(width=800, height=600):
    """Create a simple test image with colored rectangles."""
    # Create a white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some colored rectangles to simulate objects
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=2)
    draw.rectangle([300, 150, 450, 300], fill='blue', outline='black', width=2)
    draw.rectangle([500, 200, 650, 400], fill='green', outline='black', width=2)
    draw.rectangle([200, 350, 350, 500], fill='yellow', outline='black', width=2)
    
    return img

def test_object_detection_parsing(output_dir="./test_outputs"):
    """Test the object detection parsing functionality.
    
    Args:
        output_dir (str): Directory to save test outputs. Defaults to './test_outputs'.
    """
    print("Testing Object Detection Parsing and Visualization")
    print("=" * 50)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test image
    test_image = create_test_image()
    test_image_path = os.path.join(output_dir, 'test_detection_image.png')
    test_image.save(test_image_path)
    print(f"Created test image: {test_image_path}")
    
    # Sample Qwen-style response with object detection results
    sample_response = """
I can see several objects in this image:

- **Person 1:**
  **Bounding Box:** (150, 80, 250, 300)
  **Description:** A person standing in the foreground wearing a blue shirt

- **Building on the Left:**
  **Bounding Box:** (50, 120, 200, 400)
  **Description:** A tall residential building with multiple windows

- **Car in Street:**
  **Bounding Box:** (300, 250, 450, 320)
  **Description:** A red sedan parked on the street

- **Tree:**
  **Bounding Box:** (500, 100, 600, 350)
  **Description:** A large oak tree with green foliage
"""
    
    # Alternative simpler format sample
    simple_response = """
Detected objects:
Person: (150, 80, 250, 300)
Building: (50, 120, 200, 400)
Car: (300, 250, 450, 320)
Tree: (500, 100, 600, 350)
"""
    
    # Create a mock VLM instance for testing parsing
    class MockVLM(HuggingFaceVLM):
        def __init__(self):
            # Initialize without loading actual model
            self.model_name = "test-model"
            self.model_type = "test"
    
    mock_vlm = MockVLM()
    
    print("\n1. Testing structured response parsing:")
    print("-" * 40)
    
    # Test parsing of structured response
    parsed_structured = mock_vlm.parse_object_detection_response(sample_response)
    print(f"Parsed {parsed_structured['total_objects']} objects from structured response:")
    print(json.dumps(parsed_structured, indent=2))
    
    print("\n2. Testing simple response parsing:")
    print("-" * 40)
    
    # Test parsing of simple response
    parsed_simple = mock_vlm.parse_object_detection_response(simple_response)
    print(f"Parsed {parsed_simple['total_objects']} objects from simple response:")
    print(json.dumps(parsed_simple, indent=2))
    
    print("\n3. Testing visualization:")
    print("-" * 40)
    
    # Test visualization with structured results
    if parsed_structured['objects']:
        vis_structured_path = os.path.join(output_dir, 'visualization_structured.png')
        vis_image = mock_vlm.visualize_detections(test_image, parsed_structured, vis_structured_path)
        print(f"Created visualization: {vis_structured_path}")
    
    # Test visualization with simple results
    if parsed_simple['objects']:
        vis_simple_path = os.path.join(output_dir, 'visualization_simple.png')
        vis_image = mock_vlm.visualize_detections(test_image, parsed_simple, vis_simple_path)
        print(f"Created visualization: {vis_simple_path}")
    
    print("\n4. Testing generate_with_structured_output method:")
    print("-" * 40)
    
    # Mock the generate method to return our sample response
    def mock_generate(images, prompts):
        return [sample_response for _ in prompts]
    
    mock_vlm.generate = mock_generate
    
    # Test the complete workflow
    detections_dir = os.path.join(output_dir, 'detections')
    results = mock_vlm.generate_with_structured_output(
        images=[test_image],
        prompts=["Detect and describe all objects in this image with bounding boxes."],
        parse_objects=True,
        visualize=True,
        save_visualizations=True,
        output_dir=detections_dir
    )
    
    print(f"Generated {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  - Objects detected: {result['parsed_objects']['total_objects'] if result['parsed_objects'] else 0}")
        print(f"  - Visualization saved: {result['visualization_path']}")
        if result['parsed_objects']:
            for j, obj in enumerate(result['parsed_objects']['objects']):
                print(f"    Object {j+1}: {obj['name']} at {obj['bbox']}")
    
    print("\n5. JSON Schema Example:")
    print("-" * 40)
    
    # Show the JSON schema structure
    schema_example = {
        "objects": [
            {
                "name": "Person",
                "description": "A person standing in the foreground",
                "bbox": [150, 80, 250, 300],
                "confidence": 0.95
            },
            {
                "name": "Building",
                "description": "A tall residential building",
                "bbox": [50, 120, 200, 400],
                "confidence": 0.88
            }
        ],
        "total_objects": 2,
        "image_info": {
            "format": "bbox",
            "coordinate_system": "absolute"
        }
    }
    
    print("Expected JSON schema structure:")
    print(json.dumps(schema_example, indent=2))
    
    print("\n" + "=" * 50)
    print("Object Detection Testing Complete!")
    print(f"\nFiles created in {os.path.abspath(output_dir)}:")
    print(f"- {os.path.join(output_dir, 'test_detection_image.png')} (test image)")
    print(f"- {os.path.join(output_dir, 'visualization_structured.png')} (structured parsing visualization)")
    print(f"- {os.path.join(output_dir, 'visualization_simple.png')} (simple parsing visualization)")
    print(f"- {os.path.join(detections_dir, 'detection_result_0.png')} (complete workflow result)")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test object detection parsing and visualization functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_object_detection.py
  python test_object_detection.py --output-dir ./my_test_results
  python test_object_detection.py -o /home/user/detection_tests
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./test_outputs',
        help='Directory to save test outputs (default: ./test_outputs)'
    )
    
    args = parser.parse_args()
    
    # Get output directory from args or environment variable
    output_dir = args.output_dir
    if 'OBJECT_DETECTION_TEST_OUTPUT' in os.environ:
        output_dir = os.environ['OBJECT_DETECTION_TEST_OUTPUT']
        print(f"Using output directory from environment variable: {output_dir}")
    
    test_object_detection_parsing(output_dir)

if __name__ == "__main__":
    main()