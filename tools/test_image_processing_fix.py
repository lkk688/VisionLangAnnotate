#!/usr/bin/env python3
"""
Test script for VLMVLLMUtility image processing fix.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vlm_vllm_utils import VLMVLLMUtility

def test_image_processing():
    """Test image processing with the fixed conversation format."""
    print("🧪 Testing VLMVLLMUtility image processing fix...")
    
    try:
        # Initialize utility in package mode
        utility = VLMVLLMUtility(
            mode="package",
            model_name="Qwen/Qwen2.5-VL-7B-Instruct"
        )
        
        print("✅ Package mode initialization successful!")
        
        # Test image processing with a sample image
        image_path = "../VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found at {image_path}, using alternative path...")
            image_path = "VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
            
        if not os.path.exists(image_path):
            print(f"❌ Image not found at {image_path}")
            return False
        
        print(f"🔄 Processing image: {image_path}")
        
        # Process the image
        result = utility.process_image(
            image_path, 
            "Describe this image in detail. What do you see?"
        )
        
        print(f"\n📊 Processing Result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Image Path: {result.get('image_path', 'N/A')}")
        
        if result.get('success'):
            print(f"   Response: {result.get('response', 'No response')[:200]}...")
            print(f"   Performance: {result.get('performance', {})}")
            print("✅ Image processing test completed successfully!")
            return True
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print("❌ Image processing test failed!")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("VLMVLLMUtility Image Processing Fix Test")
    print("=" * 60)
    
    success = test_image_processing()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Image processing fix test passed!")
        sys.exit(0)
    else:
        print("💥 Image processing fix test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()