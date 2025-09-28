#!/usr/bin/env python3
"""
Test script to verify the exact vLLM format for Qwen2.5-VL
Based on official vLLM documentation examples
"""

import os
import sys
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported successfully")
except ImportError as e:
    print(f"❌ Failed to import vLLM: {e}")
    sys.exit(1)

def test_qwen25vl_exact_format():
    """Test Qwen2.5-VL with exact format from vLLM documentation"""
    
    print("🔧 Initializing vLLM with Qwen2.5-VL...")
    
    # Initialize LLM exactly as shown in vLLM docs
    llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
    
    print("✅ vLLM initialized successfully")
    
    # Load test image
    image_path = "../VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    print(f"🖼️ Loading image: {image_path}")
    image = Image.open(image_path)
    
    # Use exact format from vLLM documentation for Qwen2.5-VL
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
    
    print(f"📝 Using prompt: {prompt}")
    
    try:
        # Generate using exact format from documentation
        outputs = llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        })
        
        print("✅ Generation successful!")
        
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"🤖 Generated response: {generated_text[:200]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing exact vLLM format for Qwen2.5-VL...")
    success = test_qwen25vl_exact_format()
    
    if success:
        print("✅ Test passed!")
        sys.exit(0)
    else:
        print("❌ Test failed!")
        sys.exit(1)