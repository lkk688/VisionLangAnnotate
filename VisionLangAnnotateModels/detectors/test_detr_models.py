#!/usr/bin/env python3
"""
Test script to verify DETR models integration in object_detection_toolkit.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from object_detection_toolkit import ObjectDetectionToolkit
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detr_models():
    """Test all DETR model variants"""
    
    # Test image path
    test_image_path = "../sampledata/sjsupeople.jpg"
    
    if not os.path.exists(test_image_path):
        logger.error(f"Test image not found: {test_image_path}")
        return False
    
    # Load test image
    image = cv2.imread(test_image_path)
    if image is None:
        logger.error(f"Could not load image: {test_image_path}")
        return False
    
    # DETR models to test
    detr_models = [
        ('detr', 'facebook/detr-resnet-50'),
        ('detr-resnet-101', 'facebook/detr-resnet-101'),
        ('conditional-detr', 'microsoft/conditional-detr-resnet-50'),
        ('rt-detr', 'PekingU/rtdetr_r50vd_coco_o365')
    ]
    
    results = {}
    
    for model_type, default_model_name in detr_models:
        try:
            logger.info(f"\n=== Testing {model_type} ===")
            
            # Create toolkit
            toolkit = ObjectDetectionToolkit(model_type, device='cuda')
            
            # Run detection
            detection_results = toolkit.detect(image, confidence_threshold=0.5)
            
            # Store results
            results[model_type] = {
                'success': True,
                'num_detections': len(detection_results['boxes']),
                'inference_time': detection_results['inference_time'],
                'fps': detection_results['fps']
            }
            
            logger.info(f"✓ {model_type}: {len(detection_results['boxes'])} objects detected")
            logger.info(f"  Inference time: {detection_results['inference_time']:.2f}ms")
            logger.info(f"  FPS: {detection_results['fps']:.2f}")
            
        except Exception as e:
            logger.error(f"✗ {model_type} failed: {str(e)}")
            results[model_type] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    logger.info("\n=== DETR Models Test Summary ===")
    successful_models = 0
    for model_type, result in results.items():
        if result['success']:
            successful_models += 1
            logger.info(f"✓ {model_type}: {result['num_detections']} detections, {result['inference_time']:.2f}ms")
        else:
            logger.error(f"✗ {model_type}: {result['error']}")
    
    logger.info(f"\nTotal: {successful_models}/{len(detr_models)} models working successfully")
    
    return successful_models == len(detr_models)

if __name__ == '__main__':
    success = test_detr_models()
    sys.exit(0 if success else 1)