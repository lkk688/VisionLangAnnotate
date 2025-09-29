#!/usr/bin/env python3
"""
Simplified Gradio VLM Interface
Uses only QwenObjectDetectionPipeline with its built-in multiple backend support.
"""

import os
import sys
import shutil
import glob
from argparse import ArgumentParser
from typing import List, Dict, Any

import gradio as gr

# Import QwenObjectDetectionPipeline
try:
    # Add the VLM directory to path
    vlm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VisionLangAnnotateModels', 'VLM')
    if vlm_path not in sys.path:
        sys.path.append(vlm_path)
    
    from qwen_object_detection_pipeline3 import QwenObjectDetectionPipeline
    OBJECT_DETECTION_AVAILABLE = True
    print("QwenObjectDetectionPipeline loaded successfully")
except ImportError as e:
    OBJECT_DETECTION_AVAILABLE = False
    print(f"Warning: QwenObjectDetectionPipeline not available: {e}")
    
    # Create a placeholder class
    class QwenObjectDetectionPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def detect_objects(self, *args, **kwargs):
            return {"error": "Object detection not available"}
        def describe_image(self, *args, **kwargs):
            return "Object detection not available"


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--backend',
                        type=str,
                        choices=['huggingface', 'vllm', 'ollama'],
                        default='huggingface',
                        help='Backend to use: huggingface, vllm, or ollama')
    parser.add_argument('--detection-output-dir',
                        type=str,
                        default='./detection_results',
                        help='Output directory for detection results')

    args = parser.parse_args()
    return args


def _load_pipeline(args):
    """Initialize the QwenObjectDetectionPipeline"""
    if not OBJECT_DETECTION_AVAILABLE:
        print("Error: QwenObjectDetectionPipeline not available")
        return None
    
    try:
        pipeline = QwenObjectDetectionPipeline(
            model_name=args.checkpoint_path,
            device="cuda" if not args.cpu_only else "cpu",
            output_dir=args.detection_output_dir,
            enable_sam=True,  # Enable SAM segmentation capabilities
            enable_traditional_detectors=True,  # Enable traditional object detectors
            traditional_detectors=['yolo', 'detr'],  # Use YOLO and DETR detectors
            vlm_backend=args.backend
        )
        print("Object detection pipeline initialized successfully")
        return pipeline
    except Exception as e:
        print(f"Error: Failed to initialize object detection pipeline: {e}")
        return None


def _is_video_file(filename):
    """Check if the file is a video file"""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _launch_demo(args, pipeline):
    """Launch the Gradio demo interface"""
    
    def call_image_description(image_path, prompt=""):
        """Call the describe_image function for image description"""
        if not pipeline:
            return "Pipeline not available. Please check initialization."
        
        try:
            if prompt.strip():
                description = pipeline.describe_image(image_path, prompt)
            else:
                description = pipeline.describe_image(image_path)
            return f"Image Description:\n{description}"
        except Exception as e:
            return f"Error generating image description: {str(e)}"

    def call_object_detection(image_path, detection_method="Hybrid Mode"):
        """Call object detection pipeline for object detection with selected method"""
        if not OBJECT_DETECTION_AVAILABLE:
            return "Object detection not available. Please check pipeline initialization.", None
        
        if image_path is None:
            return "Please upload an image first.", None
        
        if pipeline is None:
            return "Pipeline not available. Please check initialization.", None
        
        # Clear detection output directory before each detection call
        try:
            detection_output_dir = args.detection_output_dir
            if os.path.exists(detection_output_dir):
                # Clear all subdirectories
                for subdir in ['json_annotations', 'raw_responses', 'segmentations', 'visualizations']:
                    subdir_path = os.path.join(detection_output_dir, subdir)
                    if os.path.exists(subdir_path):
                        shutil.rmtree(subdir_path)
                        os.makedirs(subdir_path, exist_ok=True)
                        print(f"Cleared {subdir_path}")
        except Exception as e:
            print(f"Warning: Could not clear detection output directory: {e}")
        
        try:
            # Choose detection method based on dropdown selection
            if detection_method == "VLM Only":
                # VLM-only detection
                result = pipeline.detect_objects(image_path, use_sam_segmentation=True)
                print(f"VLM-only detection: {len(result['objects'])} objects")
            elif detection_method == "Hybrid-Sequential":
                # Hybrid-Sequential detection (sequential_mode=True, cropped_sequential_mode=False)
                result = pipeline.detect_objects_hybrid(image_path, use_sam_segmentation=True, sequential_mode=True, cropped_sequential_mode=False, save_results=True)
                print(f"Hybrid-Sequential detection: {len(result['objects'])} objects")
                print(f"Traditional detectors used: {result['traditional_detectors_used']}")
            else:  # Hybrid Mode
                # Hybrid detection (sequential_mode=False, cropped_sequential_mode=False)
                result = pipeline.detect_objects_hybrid(image_path, use_sam_segmentation=True, sequential_mode=False, cropped_sequential_mode=False, save_results=True)
                print(f"Hybrid detection: {len(result['objects'])} objects")
                print(f"Traditional detectors used: {result['traditional_detectors_used']}")
            
            objects = result.get('objects', [])
            raw_response = result.get('raw_response', '')
            visualization_path = result.get('visualization_path', None)
            
            response = f"Detection Method: {detection_method}\n"
            response += f"Raw Model Response:\n{raw_response}\n\n"
            
            if objects:
                response += f"Detected Objects ({len(objects)}):\n"
                for i, obj in enumerate(objects, 1):
                    label = obj.get('label', 'Unknown')
                    bbox = obj.get('bbox', [])
                    description = obj.get('description', 'No description')
                    confidence = obj.get('confidence', 0.0)
                    
                    response += f"{i}. {label}"
                    if bbox and len(bbox) >= 4:
                        response += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                    response += f" (confidence: {confidence:.2f})\n"
                    response += f"   Description: {description}\n"
            else:
                response += "No objects detected with bounding boxes."
            
            # Get the saved visualization and segmentation images if not provided in result
            visualization_images = []
            segmentation_images = []
            
            if not visualization_path and hasattr(pipeline, 'output_dir') and pipeline.output_dir:
                import glob
                
                # Look for saved visualization images
                viz_pattern = os.path.join(pipeline.output_dir, "visualizations", "*.jpg")
                viz_images = glob.glob(viz_pattern)
                viz_pattern_png = os.path.join(pipeline.output_dir, "visualizations", "*.png")
                viz_images.extend(glob.glob(viz_pattern_png))
                
                # Look for saved segmentation images
                seg_pattern = os.path.join(pipeline.output_dir, "segmentations", "*.jpg")
                seg_images = glob.glob(seg_pattern)
                seg_pattern_png = os.path.join(pipeline.output_dir, "segmentations", "*.png")
                seg_images.extend(glob.glob(seg_pattern_png))
                
                # Sort by modification time (newest first)
                if viz_images:
                    viz_images.sort(key=os.path.getmtime, reverse=True)
                    visualization_images = viz_images[:5]  # Get up to 5 most recent images
                
                if seg_images:
                    seg_images.sort(key=os.path.getmtime, reverse=True)
                    segmentation_images = seg_images[:5]  # Get up to 5 most recent images
                
                # Combine all images for gallery display
                all_images = []
                for img in visualization_images:
                    all_images.append(img)  # Just the path for gallery
                for img in segmentation_images:
                    all_images.append(img)  # Just the path for gallery
                
                # Sort combined list by modification time
                if all_images:
                    all_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    visualization_path = all_images  # Return list of paths for gallery
                else:
                    visualization_path = []
            
            # Find and return JSON annotation file
            json_file_path = None
            json_content = None
            
            if hasattr(pipeline, 'output_dir') and pipeline.output_dir:
                json_pattern = os.path.join(pipeline.output_dir, "json_annotations", "*.json")
                json_files = glob.glob(json_pattern)
                
                if json_files:
                    # Get the most recent JSON file
                    json_files.sort(key=os.path.getmtime, reverse=True)
                    json_file_path = json_files[0]
                    
                    # Read JSON content for display
                    try:
                        import json
                        with open(json_file_path, 'r') as f:
                            json_content = json.load(f)
                    except Exception as e:
                        print(f"Error reading JSON file: {e}")
                        json_content = {"error": f"Could not read JSON file: {e}"}
            
            return response, visualization_path, json_file_path, json_content
                
        except Exception as e:
            return f"Error in object detection pipeline: {str(e)}", None, None, None

    def call_video_analysis(video_path, task_type="description"):
        """Call pipeline for video analysis"""
        if not pipeline:
            return "Pipeline not available. Please check initialization."
        
        try:
            result = pipeline.detect_objects(
                image_path=video_path,
                save_results=True,
                apply_privacy=False,
                use_sam_segmentation=False
            )
            
            if isinstance(result, list):
                # Multiple frames processed
                response = f"Video Analysis ({len(result)} frames processed):\n\n"
                for i, frame_result in enumerate(result):
                    response += f"Frame {i+1}:\n"
                    if task_type == "description":
                        if 'objects' in frame_result and frame_result['objects']:
                            descriptions = []
                            for obj in frame_result['objects']:
                                if 'description' in obj:
                                    descriptions.append(obj['description'])
                            if descriptions:
                                response += "\n".join(descriptions) + "\n\n"
                        else:
                            response += frame_result.get('raw_response', 'No description available') + "\n\n"
                    else:  # detection
                        objects = frame_result.get('objects', [])
                        if objects:
                            response += f"Detected Objects ({len(objects)}):\n"
                            for j, obj in enumerate(objects, 1):
                                label = obj.get('label', 'Unknown')
                                bbox = obj.get('bbox', [])
                                confidence = obj.get('confidence', 0.0)
                                
                                response += f"  {j}. {label}"
                                if bbox and len(bbox) >= 4:
                                    response += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                                response += f" (confidence: {confidence:.2f})\n"
                        else:
                            response += "No objects detected in this frame.\n"
                        response += "\n"
                return response
            else:
                # Single result
                if task_type == "description":
                    if 'objects' in result and result['objects']:
                        descriptions = []
                        for obj in result['objects']:
                            if 'description' in obj:
                                descriptions.append(obj['description'])
                        if descriptions:
                            return f"Video Description:\n" + "\n".join(descriptions)
                    return result.get('raw_response', 'No video description available.')
                else:  # detection
                    objects = result.get('objects', [])
                    raw_response = result.get('raw_response', '')
                    
                    response = f"Video Object Detection:\n"
                    if raw_response:
                        response += f"Model Response: {raw_response}\n\n"
                    
                    if objects:
                        response += f"Detected Objects ({len(objects)}):\n"
                        for i, obj in enumerate(objects, 1):
                            label = obj.get('label', 'Unknown')
                            bbox = obj.get('bbox', [])
                            confidence = obj.get('confidence', 0.0)
                            
                            response += f"{i}. {label}"
                            if bbox and len(bbox) >= 4:
                                response += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                            response += f" (confidence: {confidence:.2f})\n"
                    else:
                        response += "No objects detected in the video."
                    
                    return response
                    
        except Exception as e:
            return f"Error in video analysis pipeline: {str(e)}"

    def process_file(file_path, task_type, custom_prompt=""):
        """Process uploaded file based on task type"""
        if not file_path:
            return "Please upload a file first."
        
        if _is_video_file(file_path):
            return call_video_analysis(file_path, task_type)
        else:
            if task_type == "description":
                return call_image_description(file_path, custom_prompt)
            else:  # detection
                return call_object_detection(file_path)

    # Create Gradio interface with three tabs
    with gr.Blocks(title="Qwen VL Object Detection & Description") as demo:
        gr.Markdown("# Qwen VL Object Detection & Description Interface")
        gr.Markdown(f"Backend: {args.backend} | Model: {args.checkpoint_path}")
        
        with gr.Tabs():
            # Tab 1: Image Description
            with gr.TabItem("Image Description"):
                with gr.Row():
                    with gr.Column():
                        desc_image_input = gr.Image(
                            label="Upload Image",
                            type="filepath"
                        )
                        
                        desc_prompt = gr.Textbox(
                            label="Custom Prompt",
                            placeholder="Enter custom prompt for image description...",
                            lines=3
                        )
                        
                        desc_submit_btn = gr.Button("Generate Description", variant="primary")
                        
                    with gr.Column():
                        desc_output = gr.Textbox(
                            label="Image Description",
                            lines=15,
                            max_lines=25
                        )
                
                desc_submit_btn.click(
                    fn=call_image_description,
                    inputs=[desc_image_input, desc_prompt],
                    outputs=desc_output
                )
                
                gr.Examples(
                    examples=[
                        ["Describe what you see in this image"],
                        ["What objects are visible and where are they located?"],
                        ["Describe the scene in detail"],
                        ["What is the main subject of this image?"]
                    ],
                    inputs=[desc_prompt],
                    label="Example Prompts"
                )
            
            # Tab 2: Object Detection
            with gr.TabItem("Object Detection"):
                with gr.Row():
                    with gr.Column():
                        det_image_input = gr.Image(
                            label="Upload Image",
                            type="filepath"
                        )
                        
                        det_method_dropdown = gr.Dropdown(
                            choices=["VLM Only", "Hybrid Mode", "Hybrid-Sequential"],
                            value="Hybrid Mode",
                            label="Detection Method",
                            info="VLM Only: Uses only vision-language model. Hybrid Mode: Combines VLM with traditional detectors. Hybrid-Sequential: Sequential hybrid detection mode."
                        )
                        
                        det_submit_btn = gr.Button("Detect Objects", variant="primary")
                        
                        det_output = gr.Textbox(
                            label="Detection Results",
                            lines=10,
                            max_lines=15
                        )
                        
                    with gr.Column():
                        det_visualization = gr.Gallery(
                            label="Visualization & Segmentation Results",
                            show_label=True,
                            elem_id="detection_gallery",
                            columns=2,
                            rows=2,
                            height=500,
                            allow_preview=True,
                            preview=True
                        )
                        
                        # JSON file display and download
                        det_json_file = gr.File(
                            label="Detection Results JSON (Download)",
                            visible=True,
                            interactive=False
                        )
                        
                        det_json_display = gr.JSON(
                            label="JSON Annotations (Preview)",
                            visible=True
                        )
                
                det_submit_btn.click(
                    fn=call_object_detection,
                    inputs=[det_image_input, det_method_dropdown],
                    outputs=[det_output, det_visualization, det_json_file, det_json_display]
                )
            
            # Tab 3: Video Processing
            with gr.TabItem("Video Processing"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Upload Video"
                        )
                        
                        video_task_type = gr.Radio(
                            choices=["description", "detection"],
                            value="description",
                            label="Analysis Type"
                        )
                        
                        video_submit_btn = gr.Button("Process Video", variant="primary")
                        
                    with gr.Column():
                        video_output = gr.Textbox(
                            label="Video Analysis Results",
                            lines=20,
                            max_lines=30
                        )
                
                video_submit_btn.click(
                    fn=call_video_analysis,
                    inputs=[video_input, video_task_type],
                    outputs=video_output
                )

    # Launch the demo
    demo.launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name
    )


def main():
    """Main function"""
    args = _get_args()
    pipeline = _load_pipeline(args)
    
    if pipeline is None:
        print("Failed to initialize pipeline. Exiting.")
        sys.exit(1)
    
    _launch_demo(args, pipeline)


if __name__ == '__main__':
    main()