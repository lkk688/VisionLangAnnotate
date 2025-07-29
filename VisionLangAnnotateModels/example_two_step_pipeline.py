#!/usr/bin/env python3

"""
Example script for running the multi-step pipeline on a sample dataset.

This script demonstrates how to use the two_step_pipeline.py to process
Label Studio annotations and generate detailed descriptions using VLMs,
with an optional third step using Ollama for standardizing outputs.

The third step (Ollama) processes all objects from a single image together
in batch mode for improved efficiency, reducing processing time and API calls.
"""

import os
import argparse
import json
from VisionLangAnnotateModels.two_step_pipeline import run_two_step_pipeline, run_three_step_pipeline


def main():
    parser = argparse.ArgumentParser(description="Example for running the multi-step pipeline")
    parser.add_argument(
        "--data_dir", 
        default="/home/lkk/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215",
        help="Directory containing images and labelstudio_annotations.json"
    )
    parser.add_argument(
        "--model_name", 
        default="Salesforce/blip2-opt-2.7b",
        help="Name of the HuggingFaceVLM model to use"
    )
    parser.add_argument(
        "--device", 
        default="cuda",
        help="Device to run the model on (cuda or cpu)"
    )
    parser.add_argument(
        "--output_file", 
        default="example_pipeline_results.json",
        help="Path to save the results"
    )
    parser.add_argument(
        "--ollama_model",
        help="(Optional) Name of the Ollama model to use for post-processing (e.g., 'llama2')"
    )
    parser.add_argument(
        "--allowed_classes",
        help="(Optional) Comma-separated list of allowed class names for standardization"
    )
    
    args = parser.parse_args()
    
    # Construct the annotation file path
    annotation_file = os.path.join(args.data_dir, "labelstudio_annotations.json")
    
    # Check if the annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file {annotation_file} not found")
        return
    
    # Parse allowed classes if provided
    allowed_classes = None
    if args.allowed_classes:
        allowed_classes = [cls.strip() for cls in args.allowed_classes.split(',')]
    
    # Determine which pipeline to run
    if args.ollama_model:
        print(f"Running three-step pipeline on {args.data_dir}...")
        print(f"Using VLM model: {args.model_name}")
        print(f"Using Ollama model: {args.ollama_model}")
        print(f"Device: {args.device}")
        print(f"Output file: {args.output_file}")
        
        # Run the three-step pipeline
        results = run_three_step_pipeline(
            image_dir=args.data_dir,
            annotation_file=annotation_file,
            model_name=args.model_name,
            device=args.device,
            output_file=args.output_file,
            ollama_model=args.ollama_model,
            allowed_classes=allowed_classes
        )
    else:
        print(f"Running two-step pipeline on {args.data_dir}...")
        print(f"Using VLM model: {args.model_name}")
        print(f"Device: {args.device}")
        print(f"Output file: {args.output_file}")
        
        # Run the two-step pipeline
        results = run_two_step_pipeline(
            image_dir=args.data_dir,
            annotation_file=annotation_file,
            model_name=args.model_name,
            device=args.device,
            output_file=args.output_file
        )
    
    # Print summary
    print(f"\nProcessed {len(results)} objects across {len(set(r['image_path'] for r in results))} images")
    
    # Print a few example results
    if results:
        print("\nExample results:")
        for i, result in enumerate(results[:3]):
            print(f"\nObject {i+1}:")
            print(f"  Image: {os.path.basename(result['image_path'])}")
            print(f"  First-step label: {result['step1_label']}")
            print(f"  Bounding box: {result['bbox']}")
            print(f"  VLM description: {result['vlm_description']}")
            if 'ollama_class' in result:
                print(f"  Ollama class: {result['ollama_class']}")
                print(f"  Ollama confidence: {result['ollama_confidence']}")
                print(f"  Ollama reasoning: {result['ollama_reasoning']}")
        
        if len(results) > 3:
            print(f"\n... and {len(results) - 3} more objects")


if __name__ == "__main__":
    main()