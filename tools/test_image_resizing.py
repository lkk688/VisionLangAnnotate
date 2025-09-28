#!/usr/bin/env python3
"""
Test script to demonstrate image resizing for Qwen2.5-VL-7B-Instruct context limits.
Shows how different image sizes affect visual token usage.
"""

import os
import sys
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_vllm_local_image import resize_image_for_context

def analyze_image_for_context(image_path):
    """Analyze an image and show how it would be processed for the model"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"\nAnalyzing: {image_path}")
    print("=" * 50)
    
    # Original image info
    with Image.open(image_path) as img:
        orig_width, orig_height = img.size
        orig_pixels = orig_width * orig_height
        orig_tokens = orig_pixels // (28 * 28)
        
        print(f"Original image:")
        print(f"  Dimensions: {orig_width} x {orig_height}")
        print(f"  Total pixels: {orig_pixels:,}")
        print(f"  Estimated visual tokens: {orig_tokens:,}")
        
        # Check if resizing is needed
        max_pixels = 1280 * 28 * 28  # 1,003,520 pixels
        min_pixels = 256 * 28 * 28   # 200,704 pixels
        
        if orig_pixels > max_pixels:
            print(f"  Status: TOO LARGE (exceeds {max_pixels:,} pixels)")
            print(f"  Action: Will be resized DOWN")
        elif orig_pixels < min_pixels:
            print(f"  Status: TOO SMALL (below {min_pixels:,} pixels)")
            print(f"  Action: Will be resized UP")
        else:
            print(f"  Status: OPTIMAL (within {min_pixels:,} - {max_pixels:,} pixels)")
            print(f"  Action: No resizing needed")
    
    # Resized image info
    resized_img = resize_image_for_context(image_path)
    resized_width, resized_height = resized_img.size
    resized_pixels = resized_width * resized_height
    resized_tokens = resized_pixels // (28 * 28)
    
    print(f"\nAfter processing:")
    print(f"  Dimensions: {resized_width} x {resized_height}")
    print(f"  Total pixels: {resized_pixels:,}")
    print(f"  Estimated visual tokens: {resized_tokens:,}")
    
    # Context usage estimation
    max_context = 32768
    remaining_tokens = max_context - resized_tokens
    print(f"\nContext usage (max-model-len=32768):")
    print(f"  Visual tokens: {resized_tokens:,}")
    print(f"  Remaining for text: {remaining_tokens:,}")
    print(f"  Context utilization: {(resized_tokens/max_context)*100:.1f}%")

def main():
    """Test with different image sizes"""
    print("Image Resolution Analysis for Qwen2.5-VL-7B-Instruct")
    print("Max model length: 32,768 tokens")
    print("Each 28x28 pixel patch = 1 visual token")
    print("Recommended range: 256-1280 visual tokens per image")
    
    # Test with bus.jpg if it exists
    test_images = [
        "../sampledata/bus.jpg",
        "bus.jpg",
        "/home/lkk/Developer/VisionLangAnnotate/sampledata/bus.jpg"
    ]
    
    found_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            found_image = img_path
            break
    
    if found_image:
        analyze_image_for_context(found_image)
    else:
        print("\nNo test image found. Please provide an image path as argument.")
        print("Usage: python test_image_resizing.py <image_path>")
        
        # Show theoretical examples
        print("\nTheoretical examples:")
        print("=" * 50)
        
        examples = [
            ("Small image", 400, 300),
            ("Medium image", 800, 600), 
            ("Large image", 1920, 1080),
            ("Very large image", 4000, 3000),
            ("Ultra high-res", 8000, 6000)
        ]
        
        for name, width, height in examples:
            pixels = width * height
            tokens = pixels // (28 * 28)
            max_pixels = 1280 * 28 * 28
            min_pixels = 256 * 28 * 28
            
            if pixels > max_pixels:
                scale = (max_pixels / pixels) ** 0.5
                new_w, new_h = int(width * scale), int(height * scale)
                new_pixels = new_w * new_h
                new_tokens = new_pixels // (28 * 28)
                status = f"RESIZE DOWN to {new_w}x{new_h} ({new_tokens:,} tokens)"
            elif pixels < min_pixels:
                scale = (min_pixels / pixels) ** 0.5
                new_w, new_h = int(width * scale), int(height * scale)
                new_pixels = new_w * new_h
                new_tokens = new_pixels // (28 * 28)
                status = f"RESIZE UP to {new_w}x{new_h} ({new_tokens:,} tokens)"
            else:
                status = f"NO RESIZE ({tokens:,} tokens)"
            
            print(f"{name:15} {width:4}x{height:4} ({pixels:8,} px) -> {status}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image_for_context(sys.argv[1])
    else:
        main()