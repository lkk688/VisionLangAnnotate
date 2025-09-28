#!/usr/bin/env python3
"""
Simple test to isolate vLLM initialization issues with Qwen2.5-VL
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported successfully")
except ImportError as e:
    print(f"❌ Failed to import vLLM: {e}")
    sys.exit(1)

def test_minimal_vllm_init():
    """Test minimal vLLM initialization with reduced parameters."""
    print("🔄 Testing minimal vLLM initialization...")
    
    try:
        # Start with very minimal configuration
        llm = LLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            max_model_len=1024,  # Very small context
            gpu_memory_utilization=0.5,  # Even lower memory usage
            enforce_eager=True,
            disable_log_stats=True,
            disable_custom_all_reduce=True
        )
        print("✅ vLLM initialized successfully with minimal config")
        return llm
        
    except Exception as e:
        print(f"❌ vLLM initialization failed: {e}")
        return None

def test_text_generation(llm):
    """Test basic text generation without images."""
    if llm is None:
        return False
        
    print("🔄 Testing basic text generation...")
    
    try:
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50
        )
        
        prompt = "Hello, how are you?"
        outputs = llm.generate([prompt], sampling_params)
        
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"📝 Generated: {generated_text[:100]}...")
            
        print("✅ Text generation successful")
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 vLLM Simple Initialization Test")
    print("=" * 60)
    
    # Test initialization
    llm = test_minimal_vllm_init()
    
    if llm:
        # Test basic text generation
        success = test_text_generation(llm)
        
        if success:
            print("\n✅ All tests passed! vLLM is working correctly.")
        else:
            print("\n❌ Text generation failed.")
    else:
        print("\n❌ vLLM initialization failed.")
    
    print("=" * 60)