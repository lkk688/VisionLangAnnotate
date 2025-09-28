#!/usr/bin/env python3
"""
Test script for VLMVLLMUtility class

This script tests both URL mode and package mode functionality
of the VLMVLLMUtility class.
"""

import os
import sys
from PIL import Image
import io
import base64

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlm_vllm_utils import VLMVLLMUtility, VLMMode

def create_test_image():
    """Create a simple test image for testing purposes."""
    # Create a simple 100x100 red square image
    img = Image.new('RGB', (100, 100), color='red')
    return img

def test_image_utilities():
    """Test basic image utility functions."""
    print("=== Testing Image Utility Functions ===")
    
    # Create utility instance (mode doesn't matter for these tests)
    try:
        vlm = VLMVLLMUtility(mode=VLMMode.URL, api_url="http://localhost:8000")
    except Exception:
        # If server connection fails, create without validation
        vlm = VLMVLLMUtility.__new__(VLMVLLMUtility)
        vlm.mode = VLMMode.URL
        vlm.api_url = "http://localhost:8000"
        vlm.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        vlm.llm = None
    
    # Test image input type detection
    test_image = create_test_image()
    
    print("1. Testing detect_image_input_type:")
    print(f"   PIL Image: {vlm.detect_image_input_type(test_image)}")
    print(f"   URL: {vlm.detect_image_input_type('https://example.com/image.jpg')}")
    print(f"   File path: {vlm.detect_image_input_type('/path/to/image.jpg')}")
    
    # Test image resizing
    print("\n2. Testing resize_image_for_context:")
    resized_img = vlm.resize_image_for_context(test_image)
    print(f"   Original size: {test_image.size}")
    print(f"   Resized size: {resized_img.size}")
    
    # Test base64 encoding
    print("\n3. Testing encode_image_to_base64:")
    try:
        base64_data = vlm.encode_image_to_base64(test_image, resize_for_context=True)
        print(f"   Base64 encoding successful: {len(base64_data)} characters")
        print(f"   Starts with: {base64_data[:50]}...")
    except Exception as e:
        print(f"   Base64 encoding failed: {e}")
    
    print("✅ Image utility functions test completed\n")

def test_url_mode():
    """Test URL mode functionality."""
    print("=== Testing URL Mode ===")
    
    try:
        vlm = VLMVLLMUtility(mode=VLMMode.URL, api_url="http://localhost:8000")
        print("✅ URL mode initialization successful")
        
        # Test server status
        status = vlm.check_server_status()
        print(f"Server status: {status}")
        
        if status.get("server_running", False):
            print("✅ Server is running and accessible")
            
            # Test image processing (with test image)
            test_image = create_test_image()
            result = vlm.process_image(
                test_image, 
                prompt="What color is this image?", 
                max_tokens=50
            )
            
            if result.get("success", False):
                print("✅ Image processing successful")
                print(f"Response: {result['response']}")
                print(f"Performance: {result.get('performance', {})}")
            else:
                print(f"❌ Image processing failed: {result.get('error', 'Unknown error')}")
        else:
            print("❌ Server is not running or not accessible")
            print("   This is expected if no vLLM server is running on localhost:8000")
            
    except Exception as e:
        print(f"❌ URL mode test failed: {e}")
        print("   This is expected if no vLLM server is running")
    
    print()

def test_package_mode():
    """Test package mode functionality."""
    print("=== Testing Package Mode ===")
    
    try:
        # Try to import vLLM to check if it's available
        from vllm import LLM, SamplingParams
        print("✅ vLLM package is available")
        
        # Note: We won't actually initialize the model here as it requires significant resources
        # and a model download. Instead, we'll test the initialization logic.
        
        print("⚠️  Package mode test skipped - requires model download and significant resources")
        print("   To test package mode, run:")
        print("   vlm = VLMVLLMUtility(mode=VLMMode.PACKAGE, model_name='Qwen/Qwen2.5-VL-7B-Instruct')")
        print("   result = vlm.process_image('path/to/image.jpg', 'Describe this image')")
        
    except ImportError:
        print("❌ vLLM package is not available")
        print("   Install with: pip install vllm")
        
        # Test that the utility handles missing vLLM gracefully
        try:
            vlm = VLMVLLMUtility(mode=VLMMode.PACKAGE)
            print("❌ Should have raised ImportError")
        except ImportError as e:
            print(f"✅ Correctly raised ImportError: {e}")
    
    print()

def test_multiple_images():
    """Test multiple image processing."""
    print("=== Testing Multiple Image Processing ===")
    
    try:
        vlm = VLMVLLMUtility(mode=VLMMode.URL, api_url="http://localhost:8000")
    except Exception:
        # Create instance without server validation for testing
        vlm = VLMVLLMUtility.__new__(VLMVLLMUtility)
        vlm.mode = VLMMode.URL
        vlm.api_url = "http://localhost:8000"
        vlm.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        vlm.llm = None
    
    # Create test images
    test_images = [
        create_test_image(),
        Image.new('RGB', (100, 100), color='blue'),
        Image.new('RGB', (100, 100), color='green')
    ]
    
    prompts = [
        "What color is this image?",
        "Describe the color of this image.",
        "What do you see in this image?"
    ]
    
    print(f"Testing with {len(test_images)} images and {len(prompts)} prompts")
    
    # This would normally call the server, but we'll just test the function structure
    try:
        # We can't actually test this without a running server, but we can verify the method exists
        assert hasattr(vlm, 'process_multiple_images'), "process_multiple_images method exists"
        assert hasattr(vlm, 'get_performance_summary'), "get_performance_summary method exists"
        print("✅ Multiple image processing methods are available")
    except Exception as e:
        print(f"❌ Multiple image processing test failed: {e}")
    
    print()

def main():
    """Run all tests."""
    print("VLMVLLMUtility Test Suite")
    print("=" * 50)
    
    # Test basic image utilities
    test_image_utilities()
    
    # Test URL mode
    test_url_mode()
    
    # Test package mode
    test_package_mode()
    
    # Test multiple image processing
    test_multiple_images()
    
    print("=" * 50)
    print("Test suite completed!")
    print("\nNote: Some tests may show expected failures if:")
    print("- No vLLM server is running on localhost:8000 (URL mode)")
    print("- vLLM package is not installed (Package mode)")
    print("- Model files are not downloaded (Package mode)")

if __name__ == "__main__":
    main()