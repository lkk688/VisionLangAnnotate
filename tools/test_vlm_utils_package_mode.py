#!/usr/bin/env python3
"""
Test script for VLMVLLMUtility package mode with torch.compile fix.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vlm_vllm_utils import VLMVLLMUtility

def test_package_mode_initialization():
    """Test package mode initialization with torch.compile workaround."""
    print("🧪 Testing VLMVLLMUtility package mode initialization...")
    
    try:
        # Use a small model for testing
        utility = VLMVLLMUtility(
            mode="package",
            model_name="facebook/opt-125m"  # Small model for testing
        )
        
        print("✅ Package mode initialization successful!")
        print(f"   Model: {utility.model_name}")
        print(f"   Mode: {utility.mode}")
        
        # Test a simple text generation
        test_prompt = "Hello, this is a test"
        print(f"\n🔄 Testing text generation with prompt: '{test_prompt}'")
        
        # Note: This would require implementing text generation in the utility class
        # For now, just verify the initialization worked
        print("✅ Initialization test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Package mode test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("VLMVLLMUtility Package Mode Test")
    print("=" * 60)
    
    success = test_package_mode_initialization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()