#!/usr/bin/env python3
"""
Test script for the RegionCaptioner module.

This script runs a series of tests to verify that the RegionCaptioner
module is working correctly, including detection and captioning of
faces and license plates in sample images.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

# Import the modules to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VLM.region_captioning import RegionCaptioner
from VLM.region_captioning_integration import RegionCaptioningModel


class TestRegionCaptioner(unittest.TestCase):
    """
    Test cases for the RegionCaptioner class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock for the model and processor to avoid loading actual models
        self.model_patcher = patch('VLM.region_captioning.AutoModelForVision2Seq')
        self.processor_patcher = patch('VLM.region_captioning.AutoProcessor')
        self.face_detector_patcher = patch('VLM.region_captioning.cv2.CascadeClassifier')
        self.yolo_patcher = patch('VLM.region_captioning.YOLOv8Detector')
        self.base_model_patcher = patch('VLM.region_captioning.BaseMultiModel')
        
        # Start the patchers
        self.mock_model = self.model_patcher.start()
        self.mock_processor = self.processor_patcher.start()
        self.mock_face_detector = self.face_detector_patcher.start()
        self.mock_yolo = self.yolo_patcher.start()
        self.mock_base_model = self.base_model_patcher.start()
        
        # Configure the mocks
        self.mock_model.from_pretrained.return_value = MagicMock()
        self.mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        self.mock_model.from_pretrained.return_value.to.return_value.generate.return_value = [MagicMock()]
        
        self.mock_processor.from_pretrained.return_value = MagicMock()
        self.mock_processor.from_pretrained.return_value.decode.return_value = "This is a mock caption."
        
        self.mock_face_detector.return_value = MagicMock()
        self.mock_face_detector.return_value.detectMultiScale.return_value = [(10, 20, 100, 150)]
        
        # Mock YOLO detector
        self.mock_yolo.return_value = MagicMock()
        self.mock_yolo.return_value.detect.return_value = [
            {"bbox": (10, 20, 100, 150), "class_name": "person", "confidence": 0.95}
        ]
        
        # Mock Hugging Face detector
        self.mock_base_model.return_value = MagicMock()
        self.mock_base_model.return_value.predict.return_value = [
            {"bbox": (10, 20, 100, 150), "class_name": "person", "confidence": 0.95}
        ]
        
        # Create a test instance with the mocked dependencies
        self.captioner = RegionCaptioner(caption_model_name="test_model", detector_type="opencv")
        
        # Create a mock image for testing
        self.test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    def tearDown(self):
        """
        Clean up test fixtures.
        """
        # Stop the patchers
        self.model_patcher.stop()
        self.processor_patcher.stop()
        self.face_detector_patcher.stop()
        self.yolo_patcher.stop()
        self.base_model_patcher.stop()
    
    def test_init(self):
        """
        Test that the RegionCaptioner initializes correctly.
        """
        self.assertEqual(self.captioner.caption_model_name, "test_model")
        self.assertEqual(self.captioner.detector_type, "opencv")
        self.assertIsNotNone(self.captioner.model)
        self.assertIsNotNone(self.captioner.processor)
        self.assertIsNotNone(self.captioner.face_detector)
        
    def test_init_with_yolo(self):
        """
        Test initialization of RegionCaptioner with YOLO detector.
        """
        captioner = RegionCaptioner(caption_model_name="test_model", detector_type="yolo", detector_model="yolov8n.pt")
        self.assertIsNotNone(captioner)
        self.assertEqual(captioner.detector_type, "yolo")
        self.assertEqual(captioner.detector_model, "yolov8n.pt")
        self.mock_yolo.assert_called_once()
        
    def test_init_with_hf(self):
        """
        Test initialization of RegionCaptioner with Hugging Face detector.
        """
        captioner = RegionCaptioner(caption_model_name="test_model", detector_type="hf", detector_model="facebook/detr-resnet-50")
        self.assertIsNotNone(captioner)
        self.assertEqual(captioner.detector_type, "hf")
        self.assertEqual(captioner.detector_model, "facebook/detr-resnet-50")
        self.mock_base_model.assert_called_once()
    
    def test_detect_faces(self):
        """
        Test face detection functionality.
        """
        faces = self.captioner.detect_faces(self.test_image)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["bbox"], (10, 20, 100, 150))
        self.assertEqual(faces[0]["class_name"], "face")
        
        # Verify the correct method was called
        self.captioner.face_detector.detectMultiScale.assert_called_once()
        
    def test_detect_license_plates(self):
        """
        Test license plate detection functionality.
        """
        # Mock the license plate detector
        self.captioner.license_plate_detector = MagicMock()
        self.captioner.license_plate_detector.detectMultiScale.return_value = [(30, 40, 60, 70)]
        
        # Mock the detect_objects method to return no license plates
        self.captioner.detect_objects = MagicMock(return_value=[])
        
        # Call the method
        license_plates = self.captioner.detect_license_plates(self.test_image)
        
        # Verify the result
        self.assertEqual(len(license_plates), 1)
        self.assertEqual(license_plates[0]["bbox"], (30, 40, 60, 70))
        self.assertEqual(license_plates[0]["class_name"], "license_plate")
        
        # Verify the correct methods were called
        self.captioner.detect_objects.assert_called_once()
        self.captioner.license_plate_detector.detectMultiScale.assert_called_once()
        
    def test_detect_license_plates_with_detector(self):
        """
        Test license plate detection with object detector.
        """
        # Mock the detect_objects method to return license plates
        self.captioner.detect_objects = MagicMock(return_value=[
            {"bbox": (30, 40, 60, 70), "class_name": "license_plate", "confidence": 0.95}
        ])
        
        # Call the method
        license_plates = self.captioner.detect_license_plates(self.test_image)
        
        # Verify the result
        self.assertEqual(len(license_plates), 1)
        self.assertEqual(license_plates[0]["bbox"], (30, 40, 60, 70))
        self.assertEqual(license_plates[0]["class_name"], "license_plate")
        self.assertEqual(license_plates[0]["confidence"], 0.95)
        
        # Verify the correct method was called
        self.captioner.detect_objects.assert_called_once()
        # License plate detector should not be called when objects are found
        self.captioner.license_plate_detector.detectMultiScale.assert_not_called()
        
    def test_detect_objects_with_opencv(self):
        """
        Test object detection with OpenCV detector.
        """
        # Test with default OpenCV detector
        objects = self.captioner.detect_objects(self.test_image)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["bbox"], (10, 20, 100, 150))
        self.assertEqual(objects[0]["class_name"], "face")
        
    def test_detect_objects_with_yolo(self):
        """
        Test object detection with YOLO detector.
        """
        # Create a captioner with YOLO detector
        captioner = RegionCaptioner(caption_model_name="test_model", detector_type="yolo", detector_model="yolov8n.pt")
        objects = captioner.detect_objects(self.test_image)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["bbox"], (10, 20, 100, 150))
        self.assertEqual(objects[0]["class_name"], "person")
        self.assertEqual(objects[0]["confidence"], 0.95)
        
    def test_detect_objects_with_hf(self):
        """
        Test object detection with Hugging Face detector.
        """
        # Create a captioner with HF detector
        captioner = RegionCaptioner(caption_model_name="test_model", detector_type="hf", detector_model="facebook/detr-resnet-50")
        objects = captioner.detect_objects(self.test_image)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["bbox"], (10, 20, 100, 150))
        self.assertEqual(objects[0]["class_name"], "person")
        self.assertEqual(objects[0]["confidence"], 0.95)
    
    def test_crop_region(self):
        """
        Test region cropping functionality.
        """
        detection = {"bbox": (50, 60, 70, 80), "class_name": "face", "confidence": 0.95}
        cropped = self.captioner.crop_region(self.test_image, detection, padding=5)
        
        # Check dimensions of cropped region
        expected_height = 80 + 10  # height + 2*padding
        expected_width = 70 + 10   # width + 2*padding
        self.assertEqual(cropped.shape[0], expected_height)
        self.assertEqual(cropped.shape[1], expected_width)
    
    @patch('VLM.region_captioning.Image')
    @patch('VLM.region_captioning.cv2.cvtColor')
    def test_caption_region(self, mock_cvtcolor, mock_image):
        """
        Test region captioning functionality.
        """
        # Configure mocks
        mock_cvtcolor.return_value = self.test_image
        mock_image.fromarray.return_value = MagicMock()
        
        # Test captioning a face region
        detection = {"bbox": (10, 20, 100, 150), "class_name": "face", "confidence": 0.95}
        caption, region_type = self.captioner.caption_region(self.test_image, detection)
        self.assertEqual(caption, "This is a mock caption.")
        self.assertEqual(region_type, "face")
        
        # Verify the model was called with the right prompt
        self.captioner.processor.assert_called_once()
        
        # Test captioning a license plate region
        self.captioner.processor.reset_mock()
        detection = {"bbox": (10, 20, 100, 150), "class_name": "license_plate", "confidence": 0.95}
        caption, region_type = self.captioner.caption_region(self.test_image, detection)
        self.assertEqual(caption, "This is a mock caption.")
        self.assertEqual(region_type, "license_plate")
        
        # Verify the model was called with the right prompt
        self.captioner.processor.assert_called_once()


class TestRegionCaptioningIntegration(unittest.TestCase):
    """
    Test cases for the RegionCaptioningModel integration class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Patch the RegionCaptioner to avoid loading actual models
        self.captioner_patcher = patch('VLM.region_captioning_integration.RegionCaptioner')
        self.mock_captioner_class = self.captioner_patcher.start()
        
        # Configure the mock
        self.mock_captioner = MagicMock()
        self.mock_captioner_class.return_value = self.mock_captioner
        
        # Set up mock results
        self.mock_results = {
            'faces': [
                {'bbox': (10, 20, 100, 150), 'caption': 'A person with glasses', 'class_name': 'face', 'confidence': 0.95}
            ],
            'license_plates': [
                {'bbox': (200, 300, 80, 40), 'caption': 'A blue license plate', 'class_name': 'license_plate', 'confidence': 0.90}
            ]
        }
        self.mock_captioner.process_image.return_value = self.mock_results
        
        # Create a test instance
        self.model = RegionCaptioningModel(caption_model_name="test_model", detector_type="opencv")
    
    def tearDown(self):
        """
        Clean up test fixtures.
        """
        self.captioner_patcher.stop()
    
    def test_init(self):
        """
        Test that the RegionCaptioningModel initializes correctly.
        """
        self.assertIsNotNone(self.model.captioner)
        self.assertTrue(self.model.detect_faces)
        self.assertTrue(self.model.detect_license_plates)
    
    def test_process_image(self):
        """
        Test image processing functionality.
        """
        results = self.model.process_image("test_image.jpg")
        
        # Verify the captioner was called
        self.mock_captioner.process_image.assert_called_once_with(
            image_path="test_image.jpg",
            output_path=None,
            visualize=True,
            detect_all=False
        )
        
        # Check the results format
        self.assertIn('regions', results)
        self.assertIn('metadata', results)
        self.assertEqual(len(results['regions']), 2)  # 1 face + 1 license plate
        
    def test_process_image_with_detect_all(self):
        """
        Test image processing with detect_all=True.
        """
        # Configure mock to return different results for detect_all
        self.mock_captioner.process_image.return_value = {
            'faces': [
                {'bbox': (10, 20, 100, 150), 'caption': 'A person with glasses'}
            ],
            'license_plates': [
                {'bbox': (200, 300, 80, 40), 'caption': 'A blue license plate'}
            ],
            'other_objects': [
                {'bbox': (150, 160, 70, 80), 'caption': 'A dog', 'class_name': 'dog'}
            ]
        }
        
        results = self.model.process_image("test_image.jpg", detect_all=True)
        
        # Verify the captioner was called with detect_all=True
        self.mock_captioner.process_image.assert_called_once_with(
            image_path="test_image.jpg",
            output_path=None,
            visualize=True,
            detect_all=True
        )
        
        # Check the results format
        self.assertIn('regions', results)
        self.assertIn('metadata', results)
        self.assertEqual(len(results['regions']), 3)  # 1 face + 1 license plate + 1 other object
    
    def test_format_results(self):
        """
        Test result formatting functionality.
        """
        formatted = self.model._format_results(self.mock_results)
        
        # Check the formatted results
        self.assertIn('regions', formatted)
        self.assertIn('metadata', formatted)
        
        # Check that regions were properly formatted
        self.assertEqual(len(formatted['regions']), 2)
        
        # Check face region
        face_region = formatted['regions'][0]
        self.assertEqual(face_region['type'], 'face')
        self.assertEqual(face_region['bbox'], [10, 20, 110, 170])  # [x, y, x+w, y+h]
        self.assertEqual(face_region['caption'], 'A person with glasses')
        self.assertEqual(face_region['class_name'], 'face')
        self.assertEqual(face_region['confidence'], 0.95)
        
        # Check license plate region
        plate_region = formatted['regions'][1]
        self.assertEqual(plate_region['type'], 'license_plate')
        self.assertEqual(plate_region['bbox'], [200, 300, 280, 340])  # [x, y, x+w, y+h]
        self.assertEqual(plate_region['caption'], 'A blue license plate')
        self.assertEqual(plate_region['class_name'], 'license_plate')
        self.assertEqual(plate_region['confidence'], 0.90)
    
    def test_save_results(self):
        """
        Test saving results to a JSON file.
        """
        results = {
            'regions': [
                {'type': 'face', 'bbox': [10, 20, 110, 170], 'caption': 'A person with glasses', 'class_name': 'face', 'confidence': 0.95},
                {'type': 'license_plate', 'bbox': [200, 300, 280, 340], 'caption': 'A blue license plate', 'class_name': 'license_plate', 'confidence': 0.90},
                {'type': 'object', 'bbox': [150, 160, 220, 240], 'caption': 'A dog', 'class_name': 'dog', 'confidence': 0.85}
            ],
            'metadata': {
                'image_path': 'test_image.jpg',
                'detector_type': 'yolo',
                'detector_model': 'yolov8n.pt'
            }
        }
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the results
            self.model.save_results(results, tmp_path)
            
            # Check that the file was created
            self.assertTrue(os.path.exists(tmp_path))
            
            # Check the file contents
            import json
            with open(tmp_path, 'r') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded, results)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestMainFunction(unittest.TestCase):
    """
    Test the main function with different detector options.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock for the RegionCaptioner
        self.captioner_patcher = patch('VLM.region_captioning.RegionCaptioner')
        self.mock_captioner_class = self.captioner_patcher.start()
        
        # Configure the mock
        self.mock_captioner = MagicMock()
        self.mock_captioner_class.return_value = self.mock_captioner
        
        # Mock the process_image method
        self.mock_captioner.process_image.return_value = {
            'image': np.zeros((300, 400, 3), dtype=np.uint8),
            'regions': [
                {'type': 'face', 'bbox': (10, 20, 100, 150), 'caption': 'A person with glasses', 'class_name': 'face', 'confidence': 0.95}
            ]
        }
        
        # Mock argparse
        self.argparse_patcher = patch('VLM.region_captioning.argparse.ArgumentParser')
        self.mock_argparse = self.argparse_patcher.start()
        self.mock_parser = MagicMock()
        self.mock_argparse.return_value = self.mock_parser
        self.mock_args = MagicMock()
        self.mock_parser.parse_args.return_value = self.mock_args
        
        # Mock plt.show
        self.plt_patcher = patch('VLM.region_captioning.plt.show')
        self.mock_plt_show = self.plt_patcher.start()
    
    def tearDown(self):
        """
        Clean up test fixtures.
        """
        self.captioner_patcher.stop()
        self.argparse_patcher.stop()
        self.plt_patcher.stop()
    
    def test_main_with_opencv(self):
        """
        Test main function with OpenCV detector.
        """
        # Configure mock args
        self.mock_args.input = "test_image.jpg"
        self.mock_args.output = None
        self.mock_args.model = "test_model"
        self.mock_args.detector = "opencv"
        self.mock_args.detector_model = None
        self.mock_args.detect_all = False
        
        # Call main function
        with patch('VLM.region_captioning.main') as mock_main:
            from VLM.region_captioning import main
            main()
        
        # Verify RegionCaptioner was initialized with correct parameters
        self.mock_captioner_class.assert_called_once_with(
            caption_model_name="test_model",
            detector_type="opencv",
            detector_model=None
        )
        
        # Verify process_image was called
        self.mock_captioner.process_image.assert_called_once_with(
            image_path="test_image.jpg",
            output_path=None,
            visualize=True,
            detect_all=False
        )
    
    def test_main_with_yolo(self):
        """
        Test main function with YOLO detector.
        """
        # Configure mock args
        self.mock_args.input = "test_image.jpg"
        self.mock_args.output = "output.jpg"
        self.mock_args.model = "test_model"
        self.mock_args.detector = "yolo"
        self.mock_args.detector_model = "yolov8n.pt"
        self.mock_args.detect_all = True
        
        # Call main function
        with patch('VLM.region_captioning.main') as mock_main:
            from VLM.region_captioning import main
            main()
        
        # Verify RegionCaptioner was initialized with correct parameters
        self.mock_captioner_class.assert_called_once_with(
            caption_model_name="test_model",
            detector_type="yolo",
            detector_model="yolov8n.pt"
        )
        
        # Verify process_image was called
        self.mock_captioner.process_image.assert_called_once_with(
            image_path="test_image.jpg",
            output_path="output.jpg",
            visualize=True,
            detect_all=True
        )

if __name__ == '__main__':
    unittest.main()