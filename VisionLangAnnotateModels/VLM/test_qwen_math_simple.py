#!/usr/bin/env python3
"""
Simple test of Qwen Math OCR Pipeline with the 3B model
to verify basic functionality.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Handle import with proper error handling
try:
    from qwen_math_ocr_pipeline import QwenMathOCRPipeline
except ImportError:
    # Add current directory to path and try again
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        from qwen_math_ocr_pipeline import QwenMathOCRPipeline
    except ImportError as e:
        print(f"Error: Cannot import QwenMathOCRPipeline: {e}")
        print("Please ensure qwen_math_ocr_pipeline.py is in the same directory.")
        sys.exit(1)

def create_simple_math_image():
    """Create a very simple math problem image."""
    img = Image.new('RGB', (400, 200), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw simple math problems
    draw.text((20, 20), "Question 1: What is 5 + 3?", fill='black', font=font)
    draw.text((20, 60), "Answer: _______", fill='black', font=font)
    
    draw.text((20, 120), "Question 2: What is 10 - 4?", fill='black', font=font)
    draw.text((20, 160), "Answer: _______", fill='black', font=font)
    
    img.save("simple_math.png")
    print("Created simple_math.png")
    return "simple_math.png"

def test_simple_math_ocr():
    """Test the pipeline with a simple math image."""
    print("=" * 50)
    print("SIMPLE QWEN MATH OCR TEST")
    print("=" * 50)
    
    # Create test image
    image_path = create_simple_math_image()
    
    try:
        # Initialize with 3B model (more stable)
        print("\n1. Initializing pipeline with 3B model...")
        pipeline = QwenMathOCRPipeline(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            device="auto"
        )
        
        # Test text extraction (simpler than JSON)
        print("\n2. Testing text extraction...")
        result = pipeline.process_image(
            image_path,
            prompt_type="extraction",
            max_new_tokens=512,
            temperature=0.0
        )
        
        if result["success"]:
            print(f"✅ Success! Processing time: {result['processing_time']:.2f}s")
            print(f"\n📝 Extracted text:")
            print("-" * 30)
            print(result.get('raw_response', 'No response'))
            print("-" * 30)
            
            # Save result
            with open("simple_math_result.txt", 'w') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                f.write(f"Model: {result.get('model_info', {}).get('model_name', 'Unknown')}\n")
                f.write("\nExtracted text:\n")
                f.write(result.get('raw_response', 'No response'))
            
            print("\n💾 Result saved to: simple_math_result.txt")
            
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    
    print("\n✅ Simple test completed successfully!")
    return True

def main():
    """Run the simple test."""
    success = test_simple_math_ocr()
    
    if success:
        print("\n🎉 The Qwen Math OCR Pipeline is working!")
        print("\nYou can now use it for your math worksheets:")
        print("```python")
        print("from qwen_math_ocr_pipeline import QwenMathOCRPipeline")
        print("pipeline = QwenMathOCRPipeline()")
        print("result = pipeline.process_image('your_math_worksheet.png')")
        print("```")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()