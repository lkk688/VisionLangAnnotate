#!/usr/bin/env python3

"""
Test script for verifying the Qwen batch processing optimization.

This script tests the optimized _generate_qwen method in the HuggingFaceVLM class
to ensure it correctly processes multiple image regions in a single batch.
"""

import os
import sys
import time
from PIL import Image
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VisionLangAnnotateModels.VLM.vlm_classifierv3 import HuggingFaceVLM


def create_test_images(num_images=3, size=(224, 224)):
    """
    Create test images for testing batch processing.
    
    Args:
        num_images: Number of test images to create
        size: Size of each test image
        
    Returns:
        List of PIL Image objects
    """
    images = []
    for i in range(num_images):
        # Create a simple colored image with a different color for each image
        color = np.array([i * 80, 255 - i * 80, 128])
        img_array = np.ones((size[0], size[1], 3), dtype=np.uint8) * color.reshape((1, 1, 3))
        images.append(Image.fromarray(img_array))
    return images


def test_qwen_batch_processing():
    """
    Test the Qwen batch processing optimization.
    """
    print("Testing Qwen batch processing optimization...")
    
    # Initialize the VLM with Qwen model
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # Use the appropriate model name
    device = "cuda"  # Use "cpu" if no GPU is available
    
    try:
        vlm = HuggingFaceVLM(model_name=model_name, device=device)
        print(f"Successfully initialized {model_name}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Skipping test due to model initialization failure.")
        return
    
    # Create test images
    images = create_test_images(num_images=3)
    
    # Create test prompts
    prompts = [
        "What color is this image? Describe it briefly.",
        "Is this image mostly red, green, or blue? Explain.",
        "What is the dominant color in this image? Provide details."
    ]
    
    # Test individual processing
    print("\nTesting individual processing...")
    start_time = time.time()
    individual_results = vlm._generate_qwen_single(images, prompts)
    individual_time = time.time() - start_time
    print(f"Individual processing completed in {individual_time:.2f} seconds")
    
    # Test batch processing
    print("\nTesting batch processing...")
    start_time = time.time()
    batch_results = vlm._generate_qwen(images, prompts)
    batch_time = time.time() - start_time
    print(f"Batch processing completed in {batch_time:.2f} seconds")
    
    # Compare results
    print("\nResults comparison:")
    print(f"Time improvement: {individual_time - batch_time:.2f} seconds ({(1 - batch_time/individual_time) * 100:.1f}% faster)")
    
    print("\nIndividual processing results:")
    for i, result in enumerate(individual_results):
        print(f"Image {i+1}: {result[:100]}..." if len(result) > 100 else f"Image {i+1}: {result}")
    
    print("\nBatch processing results:")
    for i, result in enumerate(batch_results):
        print(f"Image {i+1}: {result[:100]}..." if len(result) > 100 else f"Image {i+1}: {result}")


if __name__ == "__main__":
    test_qwen_batch_processing()