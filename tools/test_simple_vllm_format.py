#!/usr/bin/env python3
"""
Simple test to verify the correct vLLM format for Qwen2.5-VL
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

def test_simple_vllm_format():
    """Test the exact format from vLLM documentation"""
    print("🔄 Testing simple vLLM format for Qwen2.5-VL...")
    
    try:
        # Initialize LLM with minimal parameters
        llm = LLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 1}
        )
        print("✅ vLLM initialized successfully")
        
        # Load image
        image_path = "../VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
            
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image_path}")
        
        # Test exact format from vLLM documentation
        prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
        
        # Create sampling parameters
        sampling_params = SamplingParams(max_tokens=100, temperature=0.1)
        
        # Generate using the exact format from documentation
        outputs = llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }, sampling_params=sampling_params)
        
        if outputs:
            output = outputs[0]
            response = output.outputs[0].text
            print(f"✅ Generation successful!")
            print(f"📝 Response: {response[:200]}...")
            return True
        else:
            print("❌ No outputs generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_vllm_format()
    if success:
        print("\n============================================================")
        print("🎉 Simple vLLM format test passed!")
    else:
        print("\n============================================================")
        print("💥 Simple vLLM format test failed!")
        sys.exit(1)