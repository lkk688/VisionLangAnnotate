#!/usr/bin/env python3
"""
Integration module for connecting the RegionCaptioner with the VisionLangAnnotate project.

This module provides a wrapper class that adapts the RegionCaptioner to work within
the VisionLangAnnotate framework, allowing for seamless integration with the existing
project structure and UI.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the parent directory to the path to import from VisionLangAnnotateModels
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the RegionCaptioner
from VLM.region_captioning import RegionCaptioner


class RegionCaptioningModel:
    """
    Wrapper class for integrating RegionCaptioner with VisionLangAnnotate.
    
    This class adapts the RegionCaptioner to conform to the expected interface
    of the VisionLangAnnotate project, allowing it to be used alongside other
    models in the system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RegionCaptioningModel with the given configuration.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - model_name: Name of the PLM model to use
                - device: Device to run the model on ('cuda' or 'cpu')
                - face_detection: Whether to enable face detection
                - license_plate_detection: Whether to enable license plate detection
        """
        self.config = config or {}
        
        # Extract configuration options
        model_name = self.config.get('model_name', 'facebook/Perception-LM-3B')
        device = self.config.get('device', None)
        
        # Initialize the underlying captioner
        self.captioner = RegionCaptioner(model_name=model_name, device=device)
        
        # Configure which detections to perform
        self.detect_faces = self.config.get('face_detection', True)
        self.detect_license_plates = self.config.get('license_plate_detection', True)
        
        # Set default visualization options
        self.visualize = self.config.get('visualize', True)
        
        print(f"Initialized RegionCaptioningModel with model: {model_name}")
        print(f"Face detection: {self.detect_faces}, License plate detection: {self.detect_license_plates}")
    
    def process_image(self, image_input, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image to detect and caption regions.
        
        Args:
            image_input: Input image in one of the following formats:
                - Path to image file (str)
                - URL to image (str starting with 'http')
                - PIL Image object
                - NumPy array (BGR or RGB format)
                - PyTorch tensor (C,H,W format)
            output_path: Path to save the visualization (if None, will not save)
            
        Returns:
            Dictionary with detection results in the format expected by VisionLangAnnotate
        """
        # Use the underlying captioner to process the image
        raw_results = self.captioner.process_image(
            image_input=image_input,
            output_path=output_path,
            visualize=self.visualize
        )
        
        # Convert results to the format expected by VisionLangAnnotate
        formatted_results = self._format_results(raw_results)
        
        return formatted_results
    
    def _format_results(self, raw_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Format the raw results from RegionCaptioner to match the expected format
        of VisionLangAnnotate.
        
        Args:
            raw_results: Raw results from RegionCaptioner
            
        Returns:
            Formatted results dictionary
        """
        # Initialize the formatted results
        formatted_results = {
            'regions': [],
            'metadata': {
                'model': self.config.get('model_name', 'facebook/Perception-LM-3B'),
                'total_regions': len(raw_results['faces']) + len(raw_results['license_plates'])
            }
        }
        
        # Add face regions
        for i, face_data in enumerate(raw_results['faces']):
            x, y, w, h = face_data['bbox']
            
            region = {
                'id': f"face_{i+1}",
                'type': 'face',
                'bbox': [x, y, x+w, y+h],  # Convert to [x1, y1, x2, y2] format
                'caption': face_data['caption'],
                'confidence': 1.0,  # Placeholder, could be replaced with actual confidence
                'attributes': {
                    'region_type': 'face'
                }
            }
            
            formatted_results['regions'].append(region)
        
        # Add license plate regions
        for i, plate_data in enumerate(raw_results['license_plates']):
            x, y, w, h = plate_data['bbox']
            
            region = {
                'id': f"license_plate_{i+1}",
                'type': 'license_plate',
                'bbox': [x, y, x+w, y+h],  # Convert to [x1, y1, x2, y2] format
                'caption': plate_data['caption'],
                'confidence': 1.0,  # Placeholder, could be replaced with actual confidence
                'attributes': {
                    'region_type': 'license_plate'
                }
            }
            
            formatted_results['regions'].append(region)
        
        return formatted_results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save the detection and captioning results to a JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save the results JSON
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration for the RegionCaptioningModel.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model_name': 'facebook/Perception-LM-3B',
            'device': None,  # Will use CUDA if available
            'face_detection': True,
            'license_plate_detection': True,
            'visualize': True
        }


def main():
    """
    Example usage of the RegionCaptioningModel.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Region Captioning Integration Example')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output visualization')
    parser.add_argument('--json', type=str, default=None, help='Path to save results JSON')
    parser.add_argument('--no-faces', action='store_true', help='Disable face detection')
    parser.add_argument('--no-plates', action='store_true', help='Disable license plate detection')
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'face_detection': not args.no_faces,
        'license_plate_detection': not args.no_plates
    }
    
    # Initialize the model
    model = RegionCaptioningModel(config)
    
    # Process the image
    results = model.process_image(
        image_path=args.image,
        output_path=args.output
    )
    
    # Print results
    print(f"\nDetected {len(results['regions'])} regions:")
    for region in results['regions']:
        print(f"\n{region['type'].capitalize()} {region['id']}:")
        print(f"  Position: {region['bbox']}")
        print(f"  Caption: {region['caption']}")
    
    # Save results to JSON if specified
    if args.json:
        model.save_results(results, args.json)


if __name__ == "__main__":
    main()