#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of the RegionCaptioner class for detecting and captioning
regions of interest in images using YOLO or Hugging Face models.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from region_captioning import RegionCaptioner


def example_with_sample_images():
    """
    Run region captioning on sample images (one with faces, one with license plates).
    """
    # Initialize the captioner with default OpenCV detector
    captioner = RegionCaptioner()
    
    # Example 1: Image with faces (using a sample URL)
    print("\n=== Example 1: Processing image with faces (OpenCV detector) ===")
    face_image_url = "https://github.com/opencv/opencv/raw/master/samples/data/group.jpg"
    
    try:
        results = captioner.process_image(
            image_path=face_image_url,
            output_path="example_faces_output.jpg"
        )
        
        print(f"Found {len(results['faces'])} faces")
        for i, face_data in enumerate(results['faces']):
            print(f"\nFace {i+1}:")
            print(f"  Caption: {face_data['caption']}")
    except Exception as e:
        print(f"Error processing face image: {e}")
    
    # Example 2: Image with license plate using YOLO detector
    print("\n=== Example 2: Processing image with license plate (YOLO detector) ===")
    plate_image_url = "https://github.com/openalpr/openalpr/raw/master/samples/eu-1.jpg"
    
    try:
        # Initialize with YOLO detector
        yolo_captioner = RegionCaptioner(
            caption_model_name="facebook/Perception-LM-3B",
            detector_type="yolo",
            detector_model="yolov8n.pt"
        )
        
        results = yolo_captioner.process_image(
            image_path=plate_image_url,
            output_path="example_plate_yolo_output.jpg",
            detect_all=False  # Only detect faces and license plates
        )
        
        # Print results for all detected regions
        for region_type, detections in results.items():
            print(f"\nFound {len(detections)} {region_type}")
            for i, detection in enumerate(detections):
                print(f"\n{region_type.capitalize()} {i+1}:")
                print(f"  Confidence: {detection.get('confidence', 'N/A'):.4f}")
                print(f"  Caption: {detection['caption']}")
    except Exception as e:
        print(f"Error processing license plate image with YOLO: {e}")


def example_with_custom_image(image_input, detector_type="opencv"):
    """
    Run region captioning on a custom image provided by the user.
    
    Args:
        image_input: Input image in one of the following formats:
            - Path to image file (str)
            - URL to image (str starting with 'http')
            - PIL Image object
            - NumPy array (BGR or RGB format)
            - PyTorch tensor (C,H,W format)
        detector_type: Type of detector to use (opencv, yolo, or detr)
    """
    # Initialize the captioner based on detector type
    if detector_type == "yolo":
        captioner = RegionCaptioner(
            caption_model_name="facebook/Perception-LM-3B",
            detector_type="yolo",
            detector_model="yolov8n.pt"
        )
        detect_all = True  # YOLO can detect many object types
    elif detector_type == "detr":
        captioner = RegionCaptioner(
            caption_model_name="facebook/Perception-LM-3B",
            detector_type="detr",
            detector_model="facebook/detr-resnet-50"
        )
        detect_all = True  # DETR can detect many object types
    else:  # default to opencv
        captioner = RegionCaptioner()
        detect_all = False  # OpenCV only detects faces and license plates
    
    # Display information about the image being processed
    if isinstance(image_input, str):
        image_info = image_input
    elif isinstance(image_input, np.ndarray):
        image_info = f"NumPy array with shape {image_input.shape}"
    elif isinstance(image_input, Image.Image):
        image_info = f"PIL Image with size {image_input.size}"
    elif isinstance(image_input, torch.Tensor):
        image_info = f"PyTorch tensor with shape {image_input.shape}"
    else:
        image_info = "Unknown image format"
    
    print(f"\n=== Processing custom image with {detector_type.upper()} detector: {image_info} ===")
    
    try:
        # Generate a suitable output filename
        if isinstance(image_input, str) and image_input.startswith('http'):
            output_path = f"custom_image_{detector_type}_output.jpg"
        elif isinstance(image_input, str):
            base_name = os.path.splitext(os.path.basename(image_input))[0]
            output_path = f"{base_name}_{detector_type}_captioned.jpg"
        else:
            # For non-string inputs, use a generic filename
            output_path = f"custom_image_{detector_type}_{int(time.time())}_output.jpg"
        
        # Process the image
        results = captioner.process_image(
            image_input=image_input,
            output_path=output_path,
            detect_all=detect_all
        )
        
        # Print results
        print(f"Results saved to {output_path}")
        
        # Count total detections
        total_detections = sum(len(detections) for detections in results.values())
        print(f"Found {total_detections} regions of interest")
        
        # Print details for each region type
        for region_type, detections in results.items():
            print(f"\n{region_type.capitalize()} detections ({len(detections)}):")
            for i, detection in enumerate(detections):
                print(f"  {i+1}. {detection.get('class_name', region_type)}:")
                print(f"     Position: {detection['bbox']}")
                print(f"     Confidence: {detection.get('confidence', 'N/A')}")
                print(f"     Caption: {detection['caption']}")
            
    except Exception as e:
        print(f"Error processing image: {e}")


def main():
    """
    Main function to parse arguments and run examples.
    """
    parser = argparse.ArgumentParser(description='Region Captioning Examples')
    parser.add_argument('--image', type=str, help='Path to custom image or URL')
    parser.add_argument('--detector', type=str, choices=['opencv', 'yolo', 'detr'], 
                        default='opencv', help='Detector type to use')
    parser.add_argument('--samples', action='store_true', help='Run with sample images')
    args = parser.parse_args()
    
    if not args.image and not args.samples:
        print("Please provide either --image or --samples flag")
        parser.print_help()
        return
    
    if args.samples:
        example_with_sample_images()
    
    if args.image:
        example_with_custom_image(args.image, args.detector)


if __name__ == "__main__":
    main()