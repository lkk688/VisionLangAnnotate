#!/usr/bin/env python3
"""
Test script to demonstrate dynamic image input handling in test_vllm_local_image.py
Tests URL, file path, and PIL Image inputs
"""

import sys
import os
from PIL import Image
import requests
from io import BytesIO

# Add the current directory to path to import our functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_vllm_local_image import (
    detect_image_input_type,
    download_image_from_url,
    encode_image_to_base64,
    process_single_image
)

def test_dynamic_image_inputs():
    """Test the dynamic image input functionality"""
    print("🧪 Testing Dynamic Image Input Handling")
    print("=" * 50)
    
    # Test 1: URL input
    print("\n1️⃣ Testing URL input:")
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    try:
        input_type = detect_image_input_type(test_url)
        print(f"   Input type detected: {input_type}")
        
        # Test encoding
        base64_data = encode_image_to_base64(test_url, resize_for_context=True)
        print(f"   Base64 encoding: Success (length: {len(base64_data)} chars)")
        
    except Exception as e:
        print(f"   ❌ URL test failed: {e}")
    
    # Test 2: File path input (if available)
    print("\n2️⃣ Testing file path input:")
    test_files = [
        "VisionLangAnnotateModels/sampledata/bus.jpg",
        "bus.jpg",
        "test_image.jpg"
    ]
    
    existing_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            existing_file = file_path
            break
    
    if existing_file:
        try:
            input_type = detect_image_input_type(existing_file)
            print(f"   Input type detected: {input_type}")
            print(f"   File path: {existing_file}")
            
            # Test encoding
            base64_data = encode_image_to_base64(existing_file, resize_for_context=True)
            print(f"   Base64 encoding: Success (length: {len(base64_data)} chars)")
            
        except Exception as e:
            print(f"   ❌ File path test failed: {e}")
    else:
        print("   ⚠️ No test image files found, skipping file path test")
    
    # Test 3: PIL Image input
    print("\n3️⃣ Testing PIL Image input:")
    try:
        # Create a simple test image
        pil_image = Image.new('RGB', (100, 100), color='red')
        
        input_type = detect_image_input_type(pil_image)
        print(f"   Input type detected: {input_type}")
        print(f"   PIL Image size: {pil_image.size}")
        
        # Test encoding
        base64_data = encode_image_to_base64(pil_image, resize_for_context=True)
        print(f"   Base64 encoding: Success (length: {len(base64_data)} chars)")
        
    except Exception as e:
        print(f"   ❌ PIL Image test failed: {e}")
    
    # Test 4: Full processing with different inputs (if server is available)
    print("\n4️⃣ Testing full processing pipeline:")
    print("   Note: This requires a running vLLM server")
    
    # Test with URL (if available)
    if existing_file:
        try:
            print(f"   Testing with file: {os.path.basename(existing_file)}")
            result = process_single_image(
                existing_file,
                prompt="What do you see in this image?",
                max_tokens=50
            )
            
            if result["success"]:
                print("   ✅ File processing: Success")
            else:
                print(f"   ❌ File processing failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Full processing test failed: {e}")
    
    print("\n✅ Dynamic image input testing completed!")

if __name__ == "__main__":
    test_dynamic_image_inputs()