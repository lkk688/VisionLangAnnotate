#!/usr/bin/env python3
"""
Minimal test to isolate vLLM Qwen2.5-VL issues
"""

import os
import sys
from PIL import Image

try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported successfully")
except ImportError as e:
    print(f"❌ Failed to import vLLM: {e}")
    sys.exit(1)

def test_minimal_qwen25vl():
    """Minimal test with different configurations"""
    
    # Test different configurations
    configs = [
        {
            "name": "Basic Config",
            "params": {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "trust_remote_code": True,
                "enforce_eager": True,
                "max_model_len": 2048,
                "limit_mm_per_prompt": {"image": 1}
            }
        },
        {
            "name": "Minimal Config",
            "params": {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "trust_remote_code": True,
                "enforce_eager": True
            }
        }
    ]
    
    # Load test image
    image_path = "../VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    image = Image.open(image_path)
    print(f"🖼️ Loaded image: {image.size}")
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']}...")
        
        try:
            # Initialize LLM
            llm = LLM(**config['params'])
            print("✅ LLM initialized")
            
            # Test simple text generation first
            text_prompt = "Hello, how are you?"
            text_outputs = llm.generate(text_prompt, SamplingParams(max_tokens=10))
            print(f"✅ Text generation works: {text_outputs[0].outputs[0].text[:50]}")
            
            # Test image generation
            prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
            
            outputs = llm.generate({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
            
            print(f"✅ Image generation successful!")
            print(f"Response: {outputs[0].outputs[0].text[:100]}...")
            return True
            
        except Exception as e:
            print(f"❌ Failed with {config['name']}: {e}")
            continue
    
    return False

if __name__ == "__main__":
    print("🧪 Running minimal vLLM Qwen2.5-VL test...")
    success = test_minimal_qwen25vl()
    
    if success:
        print("\n✅ Test passed!")
        sys.exit(0)
    else:
        print("\n❌ All tests failed!")
        sys.exit(1)