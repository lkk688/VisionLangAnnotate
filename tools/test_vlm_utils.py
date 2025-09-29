#!/usr/bin/env python3
"""
Test script for the new VLM_utils class
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlm_utils import VLMUtils, VLMBackend

def test_vllm_api_backend():
    """Test vLLM API backend specifically."""
    print("=== Testing vLLM API Backend ===")
    
    try:
        # Initialize with vLLM API backend only
        vlm = VLMUtils(
            backend=VLMBackend.VLLM_API,
            api_url="http://localhost:8000",
            model_name="Qwen/Qwen2.5-VL-7B-Instruct"
        )
        
        # Check backend info
        backend_info = vlm.get_backend_info()
        print(f"Backend info: {backend_info}")
        
        if vlm.is_available():
            print("✅ vLLM API backend is available")
            
            # Test with a sample image
            test_image = "../VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
            if os.path.exists(test_image):
                print(f"Testing with image: {test_image}")
                
                result = vlm.process_image(
                    test_image,
                    "What do you see in this image? Describe the people and setting.",
                    max_tokens=512
                )
                
                print(f"Success: {result.get('success', False)}")
                if result.get('success'):
                    print(f"Response: {result.get('response', 'No response')[:200]}...")
                    print(f"Performance: {result.get('performance', {})}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"⚠️  Test image not found: {test_image}")
                
                # Test with a simple URL image instead
                print("Testing with a URL image...")
                result = vlm.process_image(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "Describe this landscape image.",
                    max_tokens=256
                )
                
                print(f"Success: {result.get('success', False)}")
                if result.get('success'):
                    print(f"Response: {result.get('response', 'No response')[:200]}...")
                    print(f"Performance: {result.get('performance', {})}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print("❌ vLLM API backend is not available")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")

def test_backend_detection():
    """Test automatic backend detection."""
    print("\n=== Testing Automatic Backend Detection ===")
    
    try:
        # Let it auto-detect available backends
        vlm = VLMUtils()
        
        backend_info = vlm.get_backend_info()
        print(f"Auto-detected backend info: {backend_info}")
        
        if vlm.is_available():
            print("✅ At least one backend is available")
        else:
            print("❌ No backends available")
            
    except Exception as e:
        print(f"❌ Auto-detection test failed: {str(e)}")

def test_ollama_backend():
    """Test Ollama backend if available."""
    print("\n=== Testing Ollama Backend ===")
    
    try:
        vlm_ollama = VLMUtils(
            backend=VLMBackend.OLLAMA,
            ollama_url="http://localhost:11434",
            ollama_model="llava"
        )
        
        if vlm_ollama.is_available():
            print("✅ Ollama backend is available")
            
            # Simple test
            result = vlm_ollama.process_image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                "What do you see in this image?",
                max_tokens=256
            )
            
            print(f"Ollama Success: {result.get('success', False)}")
            if result.get('success'):
                print(f"Ollama Response: {result.get('response', 'No response')[:200]}...")
            else:
                print(f"Ollama Error: {result.get('error', 'Unknown error')}")
        else:
            print("❌ Ollama backend is not available")
            
    except Exception as e:
        print(f"❌ Ollama test failed: {str(e)}")

if __name__ == "__main__":
    print("Starting VLM_utils tests...\n")
    
    test_vllm_api_backend()
    test_backend_detection()
    test_ollama_backend()
    
    print("\n=== Test Summary ===")
    print("Tests completed. Check the output above for results.")