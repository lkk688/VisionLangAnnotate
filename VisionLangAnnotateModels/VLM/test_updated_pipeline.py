#!/usr/bin/env python3
"""
Test script for the updated QwenObjectDetectionPipeline with multi-backend VLM support.

This script tests the integration of both HuggingFaceVLM and VLM_utils backends
to ensure the pipeline works correctly with different VLM configurations.
"""

import os
import sys
import traceback
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_object_detection_pipeline3 import QwenObjectDetectionPipeline

def create_test_image():
    """Create a simple test image for testing."""
    # Create a simple test image with some colored rectangles
    img = Image.new('RGB', (640, 480), color='white')
    pixels = np.array(img)
    
    # Add some colored rectangles to simulate objects
    pixels[100:200, 100:200] = [255, 0, 0]  # Red rectangle
    pixels[300:400, 300:400] = [0, 255, 0]  # Green rectangle
    pixels[200:300, 400:500] = [0, 0, 255]  # Blue rectangle
    
    return Image.fromarray(pixels)

def test_backend(backend_name, backend_config=None):
    """Test a specific VLM backend."""
    print(f"\n{'='*60}")
    print(f"Testing {backend_name} backend")
    print(f"{'='*60}")
    
    try:
        # Initialize pipeline with specific backend
        pipeline = QwenObjectDetectionPipeline(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            device="cuda",
            output_dir=f"./test_results_{backend_name}",
            vlm_backend=backend_name,
            vlm_backend_config=backend_config or {}
        )
        
        print(f"✅ Pipeline initialized successfully")
        
        # Get backend info
        backend_info = pipeline.vlm_backend_info
        print(f"Backend info: {backend_info}")
        
        # Create test image
        test_image = create_test_image()
        test_image_path = f"test_image_{backend_name}.jpg"
        test_image.save(test_image_path)
        
        print(f"Created test image: {test_image_path}")
        
        # Test object detection
        print("Testing object detection...")
        results = pipeline.detect_objects(
            image_path=test_image_path,
            save_results=True
        )
        
        if results:
            print(f"✅ Object detection completed successfully")
            print(f"Detected {len(results.get('objects', []))} objects")
            
            # Print first few objects for verification
            objects = results.get('objects', [])
            for i, obj in enumerate(objects[:3]):
                print(f"  Object {i+1}: {obj.get('label', 'unknown')} - confidence: {obj.get('confidence', 0):.2f}")
        else:
            print("⚠️  No results returned from object detection")
        
        # Test image description
        print("Testing image description...")
        description = pipeline.describe_image(
            image_path=test_image_path,
            prompt="Describe what you see in this image."
        )
        
        if description:
            print(f"✅ Image description completed successfully")
            print(f"Description: {description[:100]}...")
        else:
            print("⚠️  No description returned")
        
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {backend_name} backend: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    print("Testing Updated QwenObjectDetectionPipeline with Multi-Backend VLM Support")
    print("="*80)
    
    # Test configurations for different backends
    test_configs = [
        {
            "name": "auto",
            "config": {}
        },
        {
            "name": "huggingface", 
            "config": {}
        },
        {
            "name": "vllm_api",
            "config": {
                "api_url": "http://localhost:8000",
                "max_tokens": 512
            }
        },
        {
            "name": "vllm_package",
            "config": {
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.25
            }
        },
        {
            "name": "ollama",
            "config": {
                "api_url": "http://localhost:11434",
                "model_name": "llava"
            }
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        backend_name = test_config["name"]
        backend_config = test_config["config"]
        
        success = test_backend(backend_name, backend_config)
        results[backend_name] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for backend_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{backend_name:15} : {status}")
    
    # Overall result
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} backends passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed! The multi-backend integration is working correctly.")
    elif passed_count > 0:
        print("⚠️  Some tests passed. The integration is partially working.")
    else:
        print("❌ All tests failed. There may be issues with the integration.")

if __name__ == "__main__":
    main()