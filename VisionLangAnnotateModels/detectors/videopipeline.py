import cv2
import os
import datetime
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fractions import Fraction
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import glob
import gc
from transformers import AutoProcessor, AutoImageProcessor, AutoModelForUniversalSegmentation, AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor, AutoModelForObjectDetection
from transformers import SamProcessor, SamModel

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# Use visualize_results for different visualization types
from VisionLangAnnotateModels.detectors.inference import ModelInference
import torch
from PIL import Image
import numpy as np
import os
import time
import random

def extract_metadata(video_path):
    """Extract metadata including GPS information from video file using ffprobe."""
    cmd = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams', 
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        
        # Extract GPS data if available
        gps_info = {}
        creation_time = None
        
        # Look for creation time and GPS data in format tags
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            
            # Extract creation time
            time_fields = ['creation_time', 'date', 'com.apple.quicktime.creationdate']
            for field in time_fields:
                if field in tags:
                    creation_time = tags[field]
                    break
            
            # Common GPS metadata fields
            gps_fields = [
                'location', 'location-eng', 'GPS', 
                'GPSLatitude', 'GPSLongitude', 'GPSAltitude',
                'com.apple.quicktime.location.ISO6709'
            ]
            
            for field in gps_fields:
                if field in tags:
                    gps_info[field] = tags[field]
        
        # Also check stream metadata for creation time if not found
        if creation_time is None and 'streams' in metadata:
            for stream in metadata['streams']:
                if 'tags' in stream and 'creation_time' in stream['tags']:
                    creation_time = stream['tags']['creation_time']
                    break
        
        # If no creation time found, use file modification time
        if creation_time is None:
            file_mtime = os.path.getmtime(video_path)
            creation_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()
        
        return metadata, gps_info, creation_time
    
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, {}, None

def resize_with_aspect_ratio(image, target_size):
    """
    Resize image maintaining aspect ratio.
    
    Parameters:
    - image: PIL Image or numpy array
    - target_size: Tuple of (width, height) representing the maximum dimensions
    
    Returns:
    - Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        # Convert OpenCV image (numpy array) to PIL
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Get original dimensions
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate aspect ratios
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    # Determine new dimensions maintaining aspect ratio
    if original_aspect > target_aspect:
        # Width constrained
        new_width = target_width
        new_height = int(target_width / original_aspect)
    else:
        # Height constrained
        new_height = target_height
        new_width = int(target_height * original_aspect)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

def finalize_labelstudio_data(labelstudio_data, output_dir, saved_count, extraction_method, privacy_blur):
    """
    Finalize and save Label Studio format data.
    
    Parameters:
    - labelstudio_data: Dictionary containing Label Studio data
    - output_dir: Directory to save the Label Studio JSON file
    - saved_count: Number of frames saved
    - extraction_method: Method used for frame extraction
    - privacy_blur: Whether privacy blur was applied
    
    Returns:
    - Path to the saved Label Studio JSON file
    """
    # Add project metadata
    labelstudio_data["project_metadata"].update({
        "frames_count": saved_count,
        "extraction_time": datetime.datetime.now().isoformat(),
        "extraction_method": extraction_method,
        "privacy_blur_applied": privacy_blur
    })
    
    # Create Label Studio config
    label_config = '''
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="bbox" toName="image">
        <Label value="person" background="green"/>
        <Label value="car" background="blue"/>
        <Label value="truck" background="orange"/>
        <Label value="bicycle" background="purple"/>
        <Label value="motorcycle" background="yellow"/>
        <Label value="bus" background="red"/>
        <Label value="traffic light" background="cyan"/>
        <Label value="stop sign" background="magenta"/>
      </RectangleLabels>
    </View>
    '''
    
    # Add label config to the data
    labelstudio_data["label_config"] = label_config
    
    # Save Label Studio format JSON
    labelstudio_file = os.path.join(output_dir, "labelstudio_annotations.json")
    with open(labelstudio_file, 'w') as f:
        json.dump(labelstudio_data, f, indent=4)
    print(f"Label Studio format annotations saved to: {labelstudio_file}")
    
    return labelstudio_file

def create_labelstudio_task(output_path, output_dir, video_path, timestamp, frame_idx, detection_results, new_width, new_height, saved_count, detection_model):
    """
    Create a Label Studio task for an image with object detection results.
    
    Parameters:
    - output_path: Path where the frame is saved
    - output_dir: Directory where frames are saved
    - video_path: Path to the source video
    - timestamp: Formatted timestamp string
    - frame_idx: Index of the frame in the video
    - detection_results: Object detection results
    - new_width: Width after resizing
    - new_height: Height after resizing
    - saved_count: Counter for saved frames
    - detection_model: Name of the detection model used
    
    Returns:
    - Dictionary containing a Label Studio task
    """
    # Create image entry
    image_entry = {
        "image": os.path.relpath(output_path, os.path.dirname(output_dir)),
        "video_source": os.path.basename(video_path),
        "timestamp": timestamp,
        "frame_index": frame_idx
    }
    
    # Create annotations for this image
    annotations = []
    
    # Add bounding box annotations
    for i, (box, label, score) in enumerate(zip(
        detection_results["boxes"], 
        detection_results["labels"], 
        detection_results["scores"]
    )):
        # Convert box coordinates to Label Studio format (percentages)
        x1, y1, x2, y2 = box if isinstance(box, list) else box.tolist()
        width_percent = 100 * (x2 - x1) / new_width
        height_percent = 100 * (y2 - y1) / new_height
        x_percent = 100 * x1 / new_width
        y_percent = 100 * y1 / new_height
        
        # Create annotation entry
        bbox_annotation = {
            "id": f"bbox_{saved_count}_{i}",
            "type": "rectanglelabels",
            "value": {
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rotation": 0,
                "rectanglelabels": [label]
            },
            "score": float(score),
            "to_name": "image",
            "from_name": "bbox"
        }
        
        annotations.append(bbox_annotation)
    
    # Create a new task for this image
    task = {
        "id": saved_count,
        "data": image_entry,
        "predictions": [
            {
                "model_version": detection_model,
                "score": float(np.mean([float(s) for s in detection_results["scores"]])) if detection_results["scores"] else 0.0,
                "result": annotations
            }
        ]
    }
    
    return task

def create_frame_metadata(frame_idx, timestamp_seconds, timestamp, reason, original_width, original_height, 
                       new_width, new_height, creation_time, frame_creation_time, output_path, 
                       detection_results=None, gps_info=None):
    """
    Create metadata for an extracted video frame.
    
    Parameters:
    - frame_idx: Index of the frame in the video
    - timestamp_seconds: Timestamp in seconds
    - timestamp: Formatted timestamp string
    - reason: Reason for extracting this frame
    - original_width: Original video width
    - original_height: Original video height
    - new_width: Width after resizing
    - new_height: Height after resizing
    - creation_time: Video creation time
    - frame_creation_time: Calculated frame creation time
    - output_path: Path where the frame is saved
    - detection_results: Object detection results (optional)
    - gps_info: GPS information (optional)
    
    Returns:
    - Dictionary containing frame metadata
    """
    # Create base metadata
    frame_meta = {
        "frame_index": frame_idx,
        "timestamp_seconds": timestamp_seconds,
        "timestamp": timestamp,
        "extraction_reason": reason,
        "original_dimensions": {
            "width": original_width,
            "height": original_height
        },
        "resized_dimensions": {
            "width": new_width,
            "height": new_height
        },
        "video_creation_time": creation_time,
        "frame_creation_time": frame_creation_time,
        "extraction_time": datetime.datetime.now().isoformat(),
        "frame_path": output_path
    }
    
    # Add detection results to metadata if available
    if detection_results:
        frame_meta["detection_results"] = {
            "boxes": [box.tolist() if hasattr(box, 'tolist') else box for box in detection_results["boxes"]],
            "labels": detection_results["labels"],
            "scores": [float(score) for score in detection_results["scores"]]
        }
    
    # Add GPS data to frame metadata
    if gps_info:
        frame_meta["gps_info"] = gps_info
    
    return frame_meta

def process_frame(frame, target_size, perform_detection=True, detector=None, privacy_blur=False, detector_face=None, detector_plate=None, device=None, confidence_threshold=0.35, visualize_detections=False, output_dir=None, free_memory=False):
    """
    Process a video frame: resize, perform object detection, and apply privacy blur.
    
    Parameters:
    - frame: Original frame from video (numpy array)
    - target_size: Tuple of (width, height) for resizing
    - perform_detection: Whether to perform object detection
    - detector: Instance of a detector class
    - privacy_blur: Whether to apply privacy blur
    - detector_face: Pre-loaded face detection model (ModelInference instance)
    - detector_plate: Pre-loaded license plate detection model (ModelInference instance)
    - device: Torch device to use
    - confidence_threshold: Confidence threshold for detection
    - visualize_detections: Whether to save visualization of detection results
    - output_dir: Directory to save visualization images
    - free_memory: Whether to free memory after inference (useful for batch processing)
    
    Returns:
    - processed_frame: Resized (and optionally blurred) frame
    - detection_results: Detection results if perform_detection is True, else None
    - new_width, new_height: Dimensions of the processed frame
    """
    # First resize the frame maintaining aspect ratio
    pil_img = resize_with_aspect_ratio(frame, target_size) #nparray(1080, 1920, 3)=>pil(800-width, 450)
    new_width, new_height = pil_img.size
    
    # Convert PIL image back to numpy array for processing
    resized_frame = np.array(pil_img)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
    
    # Perform object detection if requested (on the resized frame)
    detection_results = None
    if perform_detection:
        detection_results = perform_object_detection(
            resized_frame,
            detector=detector,
            confidence_threshold=confidence_threshold,
            visualize=visualize_detections,
            output_dir=output_dir,
            free_memory=free_memory
        )
    
    # Create a copy of the resized frame for output
    processed_frame = resized_frame.copy() #(450, 800, 3)
    
    # Apply privacy blur if requested (on the resized frame)
    if privacy_blur:
        processed_frame = perform_privacyblur(
            resized_frame, 
            detector_face=detector_face,
            detector_plate=detector_plate,
            device=device,
            detection_results=detection_results,
            free_memory=free_memory
        )
    
    # Convert back to PIL image for saving
    processed_pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    
    return processed_pil_img, detection_results, new_width, new_height

def extract_key_frames(video_path, output_dir, target_size=(640, 480), extraction_method="scene_change", scenechange_threshold=5.0, privacy_blur=False, perform_detection=True, output_labelstudio=True, detector=None, detector_face=None, detector_plate=None, detection_model="PekingU/rtdetr_v2_r18vd", blurred_frames_dir=None, visualize_detections=False, free_memory=True):
    """
    Extract key frames from a video file, perform object detection, apply privacy blur, and save them with timestamp names.
    
    Parameters:
    - video_path: Path to the input video file
    - output_dir: Directory to save extracted frames
    - target_size: Tuple of (width, height) maximum dimensions for resizing
    - extraction_method: Method to extract frames ('scene_change', 'interval', or 'both')
    - privacy_blur: Whether to apply privacy blur to faces and license plates
    - perform_detection: Whether to perform object detection on extracted frames
    - output_labelstudio: Whether to output results in Label Studio format
    - detection_model: Model to use for object detection
    - blurred_frames_dir: Directory to save privacy-blurred frames (if None, will use output_dir/blurred)
    - visualize_detections: Whether to save visualization of detection results in a subfolder
    - free_memory: Whether to free memory after each frame processing (default: True)
    
    Returns:
    - Path to the Label Studio format JSON file if output_labelstudio is True
    """

        # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ignore video files that are less than 2 seconds long
    if duration < 2.0:
        print(f"Ignoring video {video_path} as it is too short (duration: {duration:.2f} seconds, minimum required: 2.0 seconds)")
        cap.release()
        return
    
    # Get video metadata
    metadata, gps_info, creation_time = extract_metadata(video_path)
    
    # Calculate video creation time based on metadata or file timestamp
    video_creation_datetime = None
    if creation_time:
        try:
            # Try different time formats
            for time_format in [
                "%Y-%m-%dT%H:%M:%S.%fZ", 
                "%Y-%m-%dT%H:%M:%SZ", 
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]:
                try:
                    video_creation_datetime = datetime.datetime.strptime(creation_time, time_format)
                    break
                except ValueError:
                    continue
        except:
            pass  # Use None if parsing fails
    
    print(f"Video Information:")
    print(f"- Frame Rate: {fps} fps")
    print(f"- Frame Count: {frame_count}")
    print(f"- Resolution: {original_width}x{original_height}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Creation Time: {creation_time}")
    print(f"- GPS Info: {gps_info}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create blurred frames directory if needed
    if privacy_blur and blurred_frames_dir is None:
        blurred_frames_dir = os.path.join(output_dir, "blurred")
    
    if privacy_blur and not os.path.exists(blurred_frames_dir):
        os.makedirs(blurred_frames_dir)
    
    if visualize_detections:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    metadata_path = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_path, exist_ok=True)
    
    # Save metadata to a JSON file with video name included in the filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    metadata_file = os.path.join(output_dir, f"{video_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'video_name': video_name,
            'video_path': video_path,
            'video_metadata': metadata, 
            'gps_info': gps_info, 
            'creation_time': creation_time,
            'fps': fps,
            'frame_count': frame_count,
            'original_width': original_width,
            'original_height': original_height,
            'duration': duration
        }, f, indent=4)
    
    # Initialize Label Studio format data
    #follow Label Studio's expected format with tasks , project_metadata , and label_config
    #Each task now contains the image data and predictions with bounding box annotations
    #if output_labelstudio:
    labelstudio_data = {
        "tasks": [],
        "project_metadata": {
            "video_path": video_path,
            "video_metadata": {
                "creation_time": creation_time,
                "gps_info": gps_info
            }
        }
    }
    
    # Initialize variables
    prev_frame = None
    frame_idx = 0
    saved_count = 0
    
    # Parameters for scene change detection
    min_scene_change_threshold = scenechange_threshold #30.0  # Minimum threshold for scene change
    frame_interval = int(fps) * 1  # Save a frame every second as fallback
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if perform_detection ==True and detector == None and detection_model is not None:
        detector = ModelInference(model_type="auto", model_name=detection_model, device=device)
    
    if privacy_blur ==True and (detector_face == None or detector_plate == None):
        # Load detection models
        print("Loading privacy models...")

        # Initialize face detector using YOLOv11n face detection model
        detector_face = ModelInference(model_type="yolo", model_name="/DATA10T/models/yolov11face/model.pt", device=device)
        
        # Initialize license plate detector using YOLOv11 license plate detection model
        detector_plate = ModelInference(model_type="yolo", model_name="/DATA10T/models/yolov11licenseplate/license-plate-finetune-v1l.pt", device=device)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for scene change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        should_save = False
        reason = ""
        
        # Method 1: Detect scene changes
        if extraction_method in ["scene_change", "both"]:
            if prev_frame is not None:
                # Calculate mean absolute difference between current and previous frame
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                
                if mean_diff > min_scene_change_threshold:
                    should_save = True
                    reason = f"scene_change (diff={mean_diff:.2f})"
            else:
                should_save = True
                reason = f"save first scene"
        
        # Method 2: Save frames at regular intervals
        if extraction_method in ["interval", "both"]:
            if frame_idx % frame_interval == 0:
                should_save = True
                reason = "interval"
        
        # Save the frame if needed
        if should_save:
            # Calculate timestamp in the video
            timestamp_seconds = frame_idx / fps
            timestamp = str(datetime.timedelta(seconds=int(timestamp_seconds)))
            milliseconds = int((timestamp_seconds - int(timestamp_seconds)) * 1000)
            timestamp = f"{timestamp}.{milliseconds:03d}"
            
            # Calculate frame creation time if video creation time is available
            frame_creation_time = None
            if video_creation_datetime:
                frame_creation_time = (video_creation_datetime + 
                                      datetime.timedelta(seconds=timestamp_seconds)).isoformat()
            
            # Process the frame: resize, detect objects, apply privacy blur
            original_frame = frame.copy()  # Keep a copy of the original unprocessed frame (1080, 1920, 3)
            pil_img, detection_results, new_width, new_height = process_frame(
                frame=frame,
                target_size=target_size,
                perform_detection=perform_detection,
                detector=detector,
                privacy_blur=privacy_blur,
                detector_face=detector_face if privacy_blur else None,
                detector_plate=detector_plate if privacy_blur else None,
                device=device,
                confidence_threshold=0.35,
                visualize_detections = visualize_detections,
                output_dir = vis_dir,
                free_memory=free_memory
            )
            
            # Generate filename
            filename = f"frame_{timestamp.replace(':', '-')}_{reason}.jpg"
            
            # Save the original or blurred frame
            if privacy_blur:
                # Save blurred frame to blurred directory
                blurred_path = os.path.join(blurred_frames_dir, filename)
                pil_img.save(blurred_path, quality=95)
                output_path = blurred_path
            else:
                # Save original frame
                output_path = os.path.join(output_dir, filename)
                pil_img.save(output_path, quality=95)
            
            # Create and save frame metadata
            frame_meta = create_frame_metadata(
                frame_idx=frame_idx,
                timestamp_seconds=timestamp_seconds,
                timestamp=timestamp,
                reason=reason,
                original_width=original_width,
                original_height=original_height,
                new_width=new_width,
                new_height=new_height,
                creation_time=creation_time,
                frame_creation_time=frame_creation_time,
                output_path=metadata_path,
                detection_results=detection_results,
                gps_info=gps_info
            )
            
            # Save frame metadata
            frame_meta_file = os.path.join(metadata_path, f"{filename.replace('.jpg', '.json')}")
            with open(frame_meta_file, 'w') as f:
                json.dump(frame_meta, f, indent=4)
            
            # Add to Label Studio format if requested
                if output_labelstudio and detection_results:
                    # Create and add Label Studio task for this image
                    task = create_labelstudio_task(
                        output_path=output_path,
                        output_dir=output_dir,
                        video_path=video_path,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        detection_results=detection_results,
                        new_width=new_width,
                        new_height=new_height,
                        saved_count=saved_count,
                        detection_model=detection_model
                    )
                    
                    labelstudio_data["tasks"].append(task)
            
            saved_count += 1
            print(f"Saved frame {saved_count}: {filename} ({reason})")
        
        # Update variables for next iteration
        prev_frame = gray.copy()
        frame_idx += 1
    
    # Release resources
    cap.release()
    print(f"Extraction complete. Saved {saved_count} key frames to {output_dir}")
    
    # Finalize Label Studio format data if requested
    if output_labelstudio:
        labelstudio_file = finalize_labelstudio_data(
            labelstudio_data=labelstudio_data,
            output_dir=output_dir,
            saved_count=saved_count,
            extraction_method=extraction_method,
            privacy_blur=privacy_blur
        )
        return labelstudio_file
    
    return output_dir

def perform_object_detection(frame, detector=None, model_name=None, device=None, confidence_threshold=0.35, visualize=False, output_dir=None, free_memory=False):
    """
    Perform object detection on a video frame using a detection model.
    
    Args:
        frame: numpy array image frame from video (already resized)
        detector: initialized detection model
        model_name: name of the detection model to use
        device: torch device to use (if None, will use GPU if available)
        confidence_threshold: minimum confidence score for detections
        visualize: whether to save visualization of detection results
        output_dir: directory to save visualization images (required if visualize=True)
        free_memory: whether to free memory after inference (useful for batch processing)
    
    Returns:
        Dictionary containing detection results with boxes, labels, and scores
    """
    
    # Convert numpy array to PIL Image
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        pil_image = frame
    
    # Initialize the model
    if detector is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector = ModelInference(model_type="auto", model_name=model_name, device=device)
    
    # Prepare visualization path if needed
    save_path = None
    if visualize and output_dir:
        # Create visualization subfolder
        # vis_dir = os.path.join(output_dir, "visualizations")
        # os.makedirs(vis_dir, exist_ok=True)
        
        # Generate a unique filename based on timestamp if frame is not from a file
        if isinstance(frame, np.ndarray):
            image_name = f"detection_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
        else:
            # Use the original image name if available
            image_name = f"detection_{os.path.splitext(os.path.basename(getattr(pil_image, 'filename', 'image')))[0]}.jpg"
        
        save_path = os.path.join(output_dir, image_name)
    
    # Perform detection
    results = detector.predict(
        image_input=pil_image,
        conf_thres=confidence_threshold,
        visualize=visualize,
        save_path=save_path, #image file name
        free_memory=free_memory
    )
    
    # Extract detections
    detections = results['detections']
    
    # Format results
    detection_results = {
        'boxes': [],
        'labels': [],
        'scores': [],
        'image_size': results['image_size']
    }
    
    for det in detections:
        detection_results['boxes'].append(det['bbox'])
        detection_results['labels'].append(det['class_name'])
        detection_results['scores'].append(det['score'])
    
    return detection_results

def perform_privacyblur(frame, detector_face=None, detector_plate=None, device=None, confidence_threshold=0.5, detection_results=None, free_memory=False):
    """
    Detect and blur faces and license plates in a video frame for privacy compliance.
    
    Args:
        frame: numpy array image frame from video (already resized)
        detector_face: pre-loaded face detection model (ModelInference instance)
        detector_plate: pre-loaded license plate detection model (ModelInference instance)
        device: torch device to use (if None, will use GPU if available)
        confidence_threshold: minimum confidence score for detections
        detection_results: optional pre-computed detection results
        free_memory: whether to free memory after inference (useful for batch processing)
    
    Returns:
        numpy array of the frame with faces and license plates blurred
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models if not provided
    if detector_face is None:
        print("Loading face detection model...")
        detector_face = ModelInference(model_type="yolo", model_name="AdamCodd/YOLOv11n-face-detection", device=device)
    
    if detector_plate is None:
        print("Loading license plate detection model...")
        detector_plate = ModelInference(model_type="yolo", model_name="morsetechlab/yolov11-license-plate-detection", device=device)
    
    # Make a copy of the original frame
    blurred_frame = frame.copy()
    
    # Detect faces
    face_results = detector_face.predict(
        image_input=frame,
        conf_thres=confidence_threshold,
        visualize=False,
        free_memory=free_memory
    )
    
    # Blur faces
    if 'detections' in face_results:
        for detection in face_results['detections']:
            # Extract face coordinates
            x1, y1, x2, y2 = detection['bbox']
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Calculate region width and height
            region_width = x2 - x1
            region_height = y2 - y1
            
            # Ignore regions that are too small (less than 20x20 pixels)
            if region_width < 20 or region_height < 20:
                print(f"Ignoring small face region: {region_width}x{region_height} pixels")
                continue
            
            # Extract the region and apply blur
            face_region = blurred_frame[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            blurred_frame[y1:y2, x1:x2] = blurred_face
    
    # Detect license plates
    plate_results = detector_plate.predict(
        image_input=frame,
        conf_thres=confidence_threshold,
        visualize=False,
        free_memory=free_memory
    )
    
    # Blur license plates
    if 'detections' in plate_results:
        for detection in plate_results['detections']:
            # Extract license plate coordinates
            x1, y1, x2, y2 = detection['bbox']
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Calculate region width and height
            region_width = x2 - x1
            region_height = y2 - y1
            
            # Ignore regions that are too small (less than 20x20 pixels)
            if region_width < 20 or region_height < 20:
                print(f"Ignoring small license plate region: {region_width}x{region_height} pixels")
                continue
            
            # Extract the region and apply blur
            plate_region = blurred_frame[y1:y2, x1:x2]
            blurred_plate = cv2.GaussianBlur(plate_region, (99, 99), 30)
            blurred_frame[y1:y2, x1:x2] = blurred_plate
    
    return blurred_frame

def get_category_prompts(option="all"):
    """
    Get text prompts from the twostepcategories.md file.
    Returns a dictionary with category names as keys and prompts as values.
    
    Parameters:
    - option: String indicating which categories to return
             "step1" - Return only step1 classes for simple detection
             "all" - Return both step1 and step2 classes (default)
    """
    # Step2 categories (more detailed/specific categories)
    step2_categories = {
        "Poor or damaged road conditions": "potholes, downed bollard, down power line, flooding, faded bike lane",
        "Infrastructure deterioration": "broken street sign, broken traffic light",
        "Illegal parking": "vehicle blocking bike lane, fire hydrant, red curb",
        "Dumped trash/vegetation": "dumped trash, yard waste, glass, residential/commercial trash cans in bike lane",
        "Graffiti": "graffiti",
        "Permissible obstruction": "construction sign, cone, blocked road",
        "Streetlight outage": "streetlight outage",
        "Obstruction to sweepers": "tree overhang",
        "Abandoned/damaged vehicles": "burned, on jacks, shattered windows, missing tires",
        "Other": "street vendors in bike lane"
    }
    
    # Step1 classes (basic object categories)
    step1_classes = {
        "vehicle": "car, van, truck, motorcycle",
        "person": "pedestrian, worker, vendor",
        "trash_can": "residential trash bin, commercial dumpster",
        "road_sign": "street sign, construction sign",
        "traffic_light": "traffic signal light",
        "tree": "tree, overhanging branch",
        "trash": "dumped trash, debris, yard waste",
        "road_surface": "pothole, water, striping",
        "pole/utility": "pole, bollard, cable, cone",
        "other": "graffiti, street vendor, obstruction"
    }
    
    # Return based on the option parameter
    if option.lower() == "step1":
        print("Using only step1 classes for simple detection")
        return step1_classes
    else:  # Default to "all"
        print("Using all categories (step1 and step2) for comprehensive detection")
        return {**step2_categories, **step1_classes}
#pip install ultralytics #for yolo
#pip install transformers #for groundingdino
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract key frames from videos with object detection and privacy blur')
    parser.add_argument('--video_path', type=str, help='Path to the input video file')
    parser.add_argument('--video_folder', type=str, default='./output/dashcam_videos/Parking compliance Vantrue dashcam', help='Path to folder containing videos to process')
    parser.add_argument('--output_dir', type=str, default='./output/dashcam_videos/onevideo_yolo11l_v2', help='Directory to save extracted frames')
    parser.add_argument('--visualize_detections', action='store_true', default=True, 
                        help='Save visualization of detection results in a sub-folder')
    #parser.add_argument('--vis_dir', type=str, default='./output/dashcam_videos/parking_frames_vis', help='Directory to save visualization frames')
    parser.add_argument('--blurred_dir', type=str, help='Directory to save privacy-blurred frames (default: output_dir/blurred)')
    parser.add_argument("--categories", type=str, nargs="+", default=["all"], 
                        help="Categories to detect (use 'all' for all categories)")
    parser.add_argument("--category_option", type=str, choices=["step1", "all"], default="step1",
                        help="Which category set to use: 'step1' for basic classes only, 'all' for comprehensive detection")
    parser.add_argument('--target_width', type=int, default=1024, help='Target width for resizing frames, 1920')
    parser.add_argument('--target_height', type=int, default=1024, help='Target height for resizing frames, 1080')
    parser.add_argument('--method', type=str, default='scene_change', choices=['scene_change', 'interval', 'both'], 
                        help='Method to extract frames: scene_change, interval, or both')
    parser.add_argument('--scenechange_threshold', type=float, default=3.0, help='Scene change threshold')
    parser.add_argument('--skip_extraction', action='store_true', help='Skip video extraction and use existing frames')
    parser.add_argument('--privacy_blur', action='store_true', default=True, help='Apply privacy blur to faces and license plates')
    parser.add_argument('--perform_detection', action='store_true', default=True, help='Perform object detection on extracted frames')
    parser.add_argument('--detection_model', type=str, default="/DATA10T/models/yolo11l.pt", 
                        help='Model to use for object detection, PekingU/rtdetr_v2_r18vd, groundingdino')
    parser.add_argument('--output_labelstudio', action='store_true', default=True, 
                        help='Generate Label Studio format JSON file')
    parser.add_argument('--confidence_threshold', type=float, default=0.35, 
                        help='Confidence threshold for object detection')
    
    args = parser.parse_args()
    
    # Create timestamp for output directory naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Get category prompts based on the selected option
    all_categories = get_category_prompts(option=args.category_option)
    
    # Build text prompt based on selected categories
    if "all" in args.categories:
        # Use all categories
        text_prompt = ", ".join(item for sublist in all_categories.values() for item in sublist.split(", "))
    else:
        # Use only selected categories
        selected_prompts = []
        for category in args.categories:
            if category in all_categories:
                selected_prompts.append(all_categories[category])
        text_prompt = ", ".join(selected_prompts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector=None
    detector_plate=None
    detector_face=None
    if args.perform_detection ==True:
        detector = ModelInference(model_type="auto", model_name=args.detection_model, device=device)
    
    if args.privacy_blur ==True:
        # Load detection models
        print("Loading privacy models...")
        # Initialize face detector using YOLOv11n face detection model
        detector_face = ModelInference(model_type="yolo", model_name="/DATA10T/models/yolov11face/model.pt", device=device)
        
        # Initialize license plate detector using YOLOv11 license plate detection model
        detector_plate = ModelInference(model_type="yolo", model_name="/DATA10T/models/yolov11licenseplate/license-plate-finetune-v1l.pt", device=device)
        
    # Process a single video if provided
    if args.video_path and not args.skip_extraction:
        output_dir = f"{args.output_dir}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Processing video: {args.video_path}")
        extract_key_frames(
            video_path=args.video_path, 
            output_dir=output_dir, 
            target_size=(args.target_width, args.target_height),
            extraction_method=args.method,
            scenechange_threshold=args.scenechange_threshold,
            privacy_blur=args.privacy_blur,
            perform_detection=args.perform_detection,
            output_labelstudio=args.output_labelstudio,
            detector=detector,
            detector_face=detector_face,
            detector_plate=detector_plate,
            detection_model=args.detection_model,
            blurred_frames_dir=args.blurred_dir,
            visualize_detections=args.visualize_detections
        )
        
        # Label Studio format JSON is saved in the extract_key_frames function
    
    # Process all videos in a folder
    elif args.video_folder and not args.skip_extraction:
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(args.video_folder, ext)))
        
        if not video_files:
            print(f"No video files found in {args.video_folder}")
            exit(1)
            
        print(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(args.output_dir, f"{video_name}_{timestamp}")
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print(f"Processing video: {video_path}")
            extract_key_frames(
                video_path=video_path, 
                output_dir=output_dir, 
                target_size=(args.target_width, args.target_height),
                extraction_method=args.method,
                scenechange_threshold=args.scenechange_threshold,
                privacy_blur=args.privacy_blur,
                detector=detector,
                detector_face=detector_face,
                detector_plate=detector_plate,
                perform_detection=args.perform_detection,
                output_labelstudio=args.output_labelstudio,
                detection_model=args.detection_model,
                blurred_frames_dir=args.blurred_dir,
                visualize_detections=args.visualize_detections
            )
            
            # Label Studio format JSON is saved in the extract_key_frames function
    
    else:
        print("Please provide either --video_path or --video_folder to process videos.")
        parser.print_help()
    
    