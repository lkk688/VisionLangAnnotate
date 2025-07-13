#!/usr/bin/env python3
"""
Command-line interface for the Region Captioning module.

This script provides a user-friendly command-line interface for detecting
and captioning faces and license plates in images using the RegionCaptioner.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional

# Import the RegionCaptioner
from region_captioning import RegionCaptioner


def process_single_image(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process a single image with the RegionCaptioner.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with detection and captioning results
    """
    # Initialize the captioner
    captioner = RegionCaptioner(model_name=args.model)
    
    # Process the image
    results = captioner.process_image(
        image_path=args.image,
        output_path=args.output,
        visualize=not args.no_vis
    )
    
    # Print results
    print("\nDetection and Captioning Results:")
    print(f"Found {len(results['faces'])} faces and {len(results['license_plates'])} license plates")
    
    for i, face_data in enumerate(results['faces']):
        print(f"\nFace {i+1}:")
        print(f"  Position: {face_data['bbox']}")
        print(f"  Caption: {face_data['caption']}")
    
    for i, plate_data in enumerate(results['license_plates']):
        print(f"\nLicense Plate {i+1}:")
        print(f"  Position: {plate_data['bbox']}")
        print(f"  Caption: {plate_data['caption']}")
    
    # Save results to JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")
    
    return results


def process_batch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Process a batch of images from a directory.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of dictionaries with detection and captioning results for each image
    """
    if not os.path.isdir(args.batch):
        print(f"Error: {args.batch} is not a directory")
        sys.exit(1)
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [
        os.path.join(args.batch, f) for f in os.listdir(args.batch)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in {args.batch}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files to process")
    
    # Initialize the captioner
    captioner = RegionCaptioner(model_name=args.model)
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each image
    all_results = []
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {image_path}")
        
        # Determine output paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        output_path = None
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{base_name}_captioned.jpg")
        
        json_path = None
        if args.json_dir:
            os.makedirs(args.json_dir, exist_ok=True)
            json_path = os.path.join(args.json_dir, f"{base_name}_results.json")
        
        # Process the image
        try:
            results = captioner.process_image(
                image_path=image_path,
                output_path=output_path,
                visualize=not args.no_vis
            )
            
            # Add filename to results
            results['filename'] = image_path
            
            # Save results to JSON if requested
            if json_path:
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Print summary
            print(f"  Found {len(results['faces'])} faces and {len(results['license_plates'])} license plates")
            if output_path:
                print(f"  Visualization saved to {output_path}")
            if json_path:
                print(f"  Results saved to {json_path}")
            
            all_results.append(results)
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    # Save batch summary if requested
    if args.batch_summary:
        with open(args.batch_summary, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBatch summary saved to {args.batch_summary}")
    
    return all_results


def main():
    """
    Main function to parse arguments and run the appropriate processing mode.
    """
    parser = argparse.ArgumentParser(
        description='Region Captioning CLI for faces and license plates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')
    
    # Single image mode
    single_parser = subparsers.add_parser('single', help='Process a single image')
    single_parser.add_argument('--image', type=str, required=True, 
                             help='Path to input image or URL')
    single_parser.add_argument('--output', type=str, default=None, 
                             help='Path to save output visualization')
    single_parser.add_argument('--json', type=str, default=None, 
                             help='Path to save results as JSON')
    single_parser.add_argument('--model', type=str, default='facebook/Perception-LM-3B', 
                             help='PLM model to use for captioning')
    single_parser.add_argument('--no-vis', action='store_true', 
                             help='Disable visualization')
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Process a batch of images')
    batch_parser.add_argument('--batch', type=str, required=True, 
                            help='Directory containing images to process')
    batch_parser.add_argument('--output-dir', type=str, default=None, 
                            help='Directory to save output visualizations')
    batch_parser.add_argument('--json-dir', type=str, default=None, 
                            help='Directory to save individual JSON results')
    batch_parser.add_argument('--batch-summary', type=str, default=None, 
                            help='Path to save batch summary JSON')
    batch_parser.add_argument('--model', type=str, default='facebook/Perception-LM-3B', 
                            help='PLM model to use for captioning')
    batch_parser.add_argument('--no-vis', action='store_true', 
                            help='Disable visualization')
    
    args = parser.parse_args()
    
    # Check if a mode was specified
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Run the appropriate mode
    if args.mode == 'single':
        process_single_image(args)
    elif args.mode == 'batch':
        process_batch(args)


if __name__ == "__main__":
    main()