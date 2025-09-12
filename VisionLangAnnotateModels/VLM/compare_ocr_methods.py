#!/usr/bin/env python3
"""
Compare DotsOCR vs Qwen2.5VL for OCR extraction on math worksheet.
"""

import os
import sys
import time
from PIL import Image
import base64
from io import BytesIO

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dots_ocr_pipeline import DotsOCRPipeline
except ImportError:
    print("DotsOCR pipeline not available")
    DotsOCRPipeline = None

try:
    from huggingfaceVLM_utils import HuggingFaceVLMClient
except ImportError:
    print("HuggingFace VLM utils not available")
    HuggingFaceVLMClient = None

def save_image_from_base64(base64_data, output_path):
    """Save base64 image data to file."""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode and save
        image_data = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def test_dots_ocr(image_path):
    """Test DotsOCR on the image."""
    if DotsOCRPipeline is None:
        return {"error": "DotsOCR not available"}
    
    try:
        print("\n=== Testing DotsOCR ===")
        pipeline = DotsOCRPipeline()
        
        # Load image
        image = Image.open(image_path)
        
        # Perform OCR
        start_time = time.time()
        result = pipeline.perform_ocr_on_page(image, 1)
        end_time = time.time()
        
        return {
            "method": "DotsOCR",
            "time": end_time - start_time,
            "result": result,
            "success": "error" not in result or not result["error"]
        }
    except Exception as e:
        return {
            "method": "DotsOCR",
            "error": str(e),
            "success": False
        }

def test_qwen2vl_ocr(image_path):
    """Test Qwen2.5VL for OCR extraction."""
    if HuggingFaceVLMClient is None:
        return {"error": "HuggingFace VLM not available"}
    
    try:
        print("\n=== Testing Qwen2.5VL ===")
        
        # Initialize Qwen2.5VL client
        client = HuggingFaceVLMClient(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            device="auto"
        )
        
        # Load image
        image = Image.open(image_path)
        
        # OCR prompt for extracting math questions
        ocr_prompt = """Please extract all the text from this math worksheet image. 
        Focus on:
        1. Question numbers and text
        2. Mathematical problems and equations
        3. Answer blanks
        4. Any instructions or labels
        
        Format the output as structured text, preserving the layout and question structure."""
        
        # Perform OCR
        start_time = time.time()
        result = client.process_vision(
            images=[image],
            prompts=[ocr_prompt],
            max_tokens=2000
        )
        end_time = time.time()
        
        return {
            "method": "Qwen2.5VL",
            "time": end_time - start_time,
            "result": result,
            "success": result.get("success", False)
        }
    except Exception as e:
        return {
            "method": "Qwen2.5VL",
            "error": str(e),
            "success": False
        }

def compare_ocr_methods(image_path):
    """Compare both OCR methods on the same image."""
    print(f"Comparing OCR methods on: {image_path}")
    
    # Test DotsOCR
    dots_result = test_dots_ocr(image_path)
    
    # Test Qwen2.5VL
    qwen_result = test_qwen2vl_ocr(image_path)
    
    # Print comparison
    print("\n" + "="*60)
    print("OCR METHOD COMPARISON RESULTS")
    print("="*60)
    
    for result in [dots_result, qwen_result]:
        method = result.get("method", "Unknown")
        success = result.get("success", False)
        time_taken = result.get("time", 0)
        
        print(f"\n{method}:")
        print(f"  Success: {success}")
        print(f"  Time: {time_taken:.2f}s")
        
        if success:
            if method == "DotsOCR":
                ocr_result = result["result"].get("ocr_result", {})
                raw_text = result["result"].get("raw_text", "")
                print(f"  Raw text length: {len(raw_text)}")
                if "elements" in ocr_result:
                    print(f"  Elements found: {len(ocr_result['elements'])}")
            elif method == "Qwen2.5VL":
                responses = result["result"].get("responses", [])
                if responses:
                    print(f"  Response length: {len(responses[0])}")
                    print(f"  Preview: {responses[0][:200]}...")
        else:
            error = result.get("error", "Unknown error")
            print(f"  Error: {error}")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if dots_result.get("success") and qwen_result.get("success"):
        dots_time = dots_result.get("time", float('inf'))
        qwen_time = qwen_result.get("time", float('inf'))
        
        if dots_time < qwen_time:
            print("✅ DotsOCR: Faster processing, good for structured documents")
        else:
            print("✅ Qwen2.5VL: More reliable for general OCR tasks")
    elif qwen_result.get("success"):
        print("✅ Qwen2.5VL: DotsOCR has vision embeddings issues, use Qwen2.5VL")
    elif dots_result.get("success"):
        print("✅ DotsOCR: Working despite some limitations")
    else:
        print("❌ Both methods failed - check image format and model availability")
    
    return dots_result, qwen_result

def main():
    # For testing, create a sample image path
    test_image = "math_worksheet_test.png"
    
    if not os.path.exists(test_image):
        print(f"Please save your math worksheet image as '{test_image}' in this directory")
        print("Or modify the script to use your image path")
        return
    
    compare_ocr_methods(test_image)

if __name__ == "__main__":
    main()