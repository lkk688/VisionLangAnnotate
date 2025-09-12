#!/usr/bin/env python3
"""
Test script for Qwen Object Detection Pipeline Video Processing

This script demonstrates the video processing capabilities of the QwenObjectDetectionPipeline,
including frame extraction via scene changes and object detection using Qwen2.5-VL.
"""

import os
import sys
import cv2
import numpy as np

# Import the pipeline class
try:
    from qwen_object_detection_pipeline import QwenObjectDetectionPipeline
except ImportError:
    # Add current directory to path if needed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen_object_detection_pipeline import QwenObjectDetectionPipeline

def create_test_video(output_path: str, duration: int = 10, fps: int = 30):
    """
    Create a simple test video with moving objects for demonstration.
    
    Args:
        output_path: Path where the test video will be saved
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background color that changes over time
        bg_color = int(50 + 30 * np.sin(frame_num * 0.1))
        frame[:, :] = (bg_color, bg_color//2, bg_color//3)
        
        # Add moving rectangle (simulating a car)
        rect_x = int(50 + 400 * (frame_num / total_frames))
        rect_y = height // 2 - 25
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 50), (0, 255, 0), -1)
        
        # Add moving circle (simulating a person)
        circle_x = int(100 + 300 * np.sin(frame_num * 0.05))
        circle_y = int(height // 3 + 50 * np.cos(frame_num * 0.03))
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 0, 0), -1)
        
        # Add scene change at specific intervals
        if frame_num % (fps * 3) == 0 and frame_num > 0:  # Scene change every 3 seconds
            # Add a flash effect to trigger scene change detection
            frame = cv2.addWeighted(frame, 0.3, np.ones_like(frame) * 255, 0.7, 0)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = f"Time: {frame_num/fps:.2f}s"
        cv2.putText(frame, timestamp, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created successfully: {output_path}")

def test_video_processing():
    """
    Test the video processing functionality of QwenObjectDetectionPipeline.
    """
    print("=" * 60)
    print("Testing Qwen Object Detection Pipeline - Video Processing")
    print("=" * 60)
    
    # Create test video
    test_video_path = "./test_video.mp4"
    create_test_video(test_video_path, duration=8, fps=15)  # Short video for testing
    
    try:
        # Initialize the pipeline
        print("\nInitializing Qwen Object Detection Pipeline...")
        pipeline = QwenObjectDetectionPipeline(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            device="cuda",
            output_dir="./qwen_video_test_results"
        )
        
        print("\n" + "="*50)
        print("TEST 1: Scene Change Detection")
        print("="*50)
        
        # Test scene change detection
        results_scene = pipeline.process_video_frames(
            video_path=test_video_path,
            target_size=(640, 480),
            extraction_method="scene_change",
            scenechange_threshold=10.0,  # Lower threshold for more sensitivity
            save_results=True
        )
        
        print(f"\nScene Change Results:")
        print(f"- Total frames processed: {results_scene['summary']['total_frames_processed']}")
        print(f"- Frames extracted: {results_scene['summary']['frames_extracted']}")
        print(f"- Total objects detected: {results_scene['summary']['total_objects_detected']}")
        
        print("\n" + "="*50)
        print("TEST 2: Interval-based Extraction")
        print("="*50)
        
        # Test interval-based extraction
        results_interval = pipeline.process_video_frames(
            video_path=test_video_path,
            target_size=(640, 480),
            extraction_method="interval",
            scenechange_threshold=5.0,
            save_results=True
        )
        
        print(f"\nInterval-based Results:")
        print(f"- Total frames processed: {results_interval['summary']['total_frames_processed']}")
        print(f"- Frames extracted: {results_interval['summary']['frames_extracted']}")
        print(f"- Total objects detected: {results_interval['summary']['total_objects_detected']}")
        
        print("\n" + "="*50)
        print("TEST 3: Combined Method (Both)")
        print("="*50)
        
        # Test combined method
        results_both = pipeline.process_video_frames(
            video_path=test_video_path,
            target_size=(640, 480),
            extraction_method="both",
            scenechange_threshold=8.0,
            save_results=True
        )
        
        print(f"\nCombined Method Results:")
        print(f"- Total frames processed: {results_both['summary']['total_frames_processed']}")
        print(f"- Frames extracted: {results_both['summary']['frames_extracted']}")
        print(f"- Total objects detected: {results_both['summary']['total_objects_detected']}")
        
        # Display detailed frame information
        print("\n" + "="*50)
        print("DETAILED FRAME ANALYSIS (Combined Method)")
        print("="*50)
        
        for i, frame_info in enumerate(results_both['extracted_frames'][:5]):  # Show first 5 frames
            print(f"\nFrame {i+1}:")
            print(f"  - Timestamp: {frame_info['timestamp']}")
            print(f"  - Extraction reason: {frame_info['extraction_reason']}")
            print(f"  - Objects detected: {len(frame_info['detection_results'].get('objects', []))}")
            
            # Show detected objects
            objects = frame_info['detection_results'].get('objects', [])
            if objects:
                print(f"  - Detected objects:")
                for obj in objects[:3]:  # Show first 3 objects
                    print(f"    * {obj['label']} (confidence: {obj['confidence']:.2f})")
        
        print("\n" + "="*60)
        print("VIDEO PROCESSING TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nResults saved to: {pipeline.output_dir}")
        print("\nCheck the following directories for outputs:")
        print("- frames/: Extracted video frames")
        print("- raw_responses/: Raw VLM responses")
        print("- json_annotations/: Label Studio compatible annotations")
        print("- visualizations/: Object detection visualizations")
        
        return True
        
    except Exception as e:
        print(f"\nError during video processing test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"\nCleaned up test video: {test_video_path}")

def test_with_existing_video(video_path: str):
    """
    Test video processing with an existing video file.
    
    Args:
        video_path: Path to an existing video file
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"\nTesting with existing video: {video_path}")
    
    try:
        # Initialize the pipeline
        pipeline = QwenObjectDetectionPipeline(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            device="cuda",
            output_dir="./qwen_existing_video_results"
        )
        
        # Process the video
        results = pipeline.process_video_frames(
            video_path=video_path,
            target_size=(640, 480),
            extraction_method="scene_change",
            scenechange_threshold=5.0,
            save_results=True
        )
        
        print(f"\nProcessing Results:")
        print(f"- Video duration: {results['video_metadata']['duration']:.2f} seconds")
        print(f"- Original resolution: {results['video_metadata']['original_width']}x{results['video_metadata']['original_height']}")
        print(f"- FPS: {results['video_metadata']['fps']:.2f}")
        print(f"- Total frames processed: {results['summary']['total_frames_processed']}")
        print(f"- Frames extracted: {results['summary']['frames_extracted']}")
        print(f"- Total objects detected: {results['summary']['total_objects_detected']}")
        
        return True
        
    except Exception as e:
        print(f"Error processing existing video: {e}")
        return False

if __name__ == "__main__":
    print("Qwen Object Detection Pipeline - Video Processing Test")
    print("This script tests the video processing capabilities with Qwen2.5-VL")
    
    # Check if a video path is provided as argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"\nUsing provided video: {video_path}")
        success = test_with_existing_video(video_path)
    else:
        print("\nNo video provided, creating test video...")
        success = test_video_processing()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("\nThe Qwen Object Detection Pipeline now supports:")
        print("1. Video frame extraction via scene change detection")
        print("2. Interval-based frame extraction")
        print("3. Combined extraction methods")
        print("4. Object detection on extracted frames using Qwen2.5-VL")
        print("5. Label Studio compatible JSON output")
        print("6. Visualization of detection results")
        print("7. Video metadata extraction (GPS, creation time, etc.)")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)