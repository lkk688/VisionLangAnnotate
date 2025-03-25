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
from transformers import AutoProcessor, AutoImageProcessor, AutoModelForUniversalSegmentation, AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor, AutoModelForObjectDetection
from transformers import SamProcessor, SamModel

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# Use visualize_results for different visualization types

def visualize_results(
    image,
    boxes=None, #boxes (np.ndarray or torch.Tensor, optional): Bounding boxes in format [x1, y1, x2, y2]
    labels=None,
    scores=None,
    semantic_seg=None,
    instance_seg=None,
    panoptic_seg=None,
    draw_boxes=True, #for panoptic 
    draw_masks=True,
    depth_map=None,
    class_names=None,
    colors=None,
    output_path=None,
    alpha=0.5,
    box_thickness=2,
    text_size=12,
    depth_cmap='plasma',
    show_legend=False,
    label_segments=False,  # New parameter to enable segment labeling
    label_font_size=10     # New parameter for segment label font size
):
    """
    Visualize detection and segmentation results on an image.
    
    Args:
        image (PIL.Image or np.ndarray): The original image
        boxes (np.ndarray or torch.Tensor, optional): Bounding boxes in format [x1, y1, x2, y2]
        labels (np.ndarray or torch.Tensor, optional): Class labels for each box
        scores (np.ndarray or torch.Tensor, optional): Confidence scores for each box
        semantic_seg (np.ndarray or torch.Tensor, optional): Semantic segmentation map
        instance_seg (np.ndarray or torch.Tensor, optional): Instance segmentation map
        panoptic_seg (dict, optional): Dict with 'segments_info' and 'panoptic_seg' keys
        depth_map (np.ndarray or torch.Tensor, optional): Depth map
        class_names (list, optional): List of class names
        colors (dict, optional): Dict mapping class IDs to RGB tuples
        output_path (str, optional): Path to save the visualization
        alpha (float, optional): Transparency of segmentation overlay
        box_thickness (int, optional): Thickness of bounding box lines
        text_size (int, optional): Size of text for labels
        depth_cmap (str, optional): Matplotlib colormap for depth visualization
        show_legend (bool, optional): Whether to show a legend for classes
        label_segments (bool, optional): Whether to add labels to segment centroids
        label_font_size (int, optional): Font size for segment labels
        
    Returns:
        PIL.Image: The visualization result
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype('uint8'))
    
    # Create a copy for drawing
    result_img = image.copy()
    
    # Convert tensors to numpy if needed
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if semantic_seg is not None and isinstance(semantic_seg, torch.Tensor):
        semantic_seg = semantic_seg.cpu().numpy()
    if instance_seg is not None and isinstance(instance_seg, torch.Tensor):
        instance_seg = instance_seg.cpu().numpy()
    if depth_map is not None and isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if panoptic_seg is not None and 'panoptic_seg' in panoptic_seg and isinstance(panoptic_seg['panoptic_seg'], torch.Tensor):
        panoptic_seg['panoptic_seg'] = panoptic_seg['panoptic_seg'].cpu().numpy()
    
    # Generate colors if not provided
    if colors is None and (labels is not None or semantic_seg is not None or instance_seg is not None or panoptic_seg is not None):
        max_classes = 0
        if labels is not None:
            max_classes = max(max_classes, len(labels) + 1)
        if semantic_seg is not None:
            max_classes = max(max_classes, np.max(semantic_seg) + 1)
        if panoptic_seg is not None and 'segments_info' in panoptic_seg:
            #max_classes = max(max_classes, max([s['category_id'] for s in panoptic_seg['segments_info']]) + 1)
            max_classes = max(max_classes, max([s['label_id'] for s in panoptic_seg['segments_info']]) + 1)
        
        # Ensure we have at least 10 colors even if max_classes is smaller
        max_classes = max(10, max_classes)
        
        # colors = {}
        # for i in range(max_classes):
        #     colors[i] = (
        #         int((i * 37 + 142) % 255),
        #         int((i * 91 + 89) % 255),
        #         int((i * 173 + 127) % 255)
        #     )
        # Generate random colors for segmentation masks
        def get_random_colors(n):
            colors = []
            for i in range(n):
                # Use HSV color space for better visual distinction
                hue = i / n
                saturation = 0.9
                value = 0.9
                rgb = hsv_to_rgb((hue, saturation, value))
                colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
            return colors
        
        colors = get_random_colors(max_classes)
    
    # Handle different segmentation types
    segmentation_overlay = None
    segment_centroids = []  # Store centroids for labeling
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", text_size)
        segment_font = ImageFont.truetype("arial.ttf", label_font_size)
    except IOError:
        font = ImageFont.load_default()
        segment_font = ImageFont.load_default()
    
    # Semantic Segmentation
    if semantic_seg is not None:
        # Create a colored segmentation map
        seg_colored = np.zeros((semantic_seg.shape[0], semantic_seg.shape[1], 3), dtype=np.uint8)
        
        # Store class centroids for labeling
        if label_segments:
            for class_id in np.unique(semantic_seg):
                if class_id == 0 and len(np.unique(semantic_seg)) > 1:  # Skip background if it's 0
                    continue
                
                mask = semantic_seg == class_id
                # Calculate centroid
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # Get class name
                    if class_names is not None and class_id < len(class_names):
                        label_text = class_names[class_id]
                    else:
                        label_text = f"Class {class_id}"
                    
                    segment_centroids.append({
                        'x': center_x,
                        'y': center_y,
                        'label': label_text,
                        'color': colors.get(class_id, (255, 255, 255))
                    })
                
                # Color the mask
                color = colors.get(class_id, (255, 255, 255))  # Default to white if no color
                seg_colored[mask] = color
        else:
            # Just color the segments without calculating centroids
            for class_id in np.unique(semantic_seg):
                if class_id == 0 and len(np.unique(semantic_seg)) > 1:  # Skip background if it's 0
                    continue
                mask = semantic_seg == class_id
                color = colors.get(class_id, (255, 255, 255))  # Default to white if no color
                seg_colored[mask] = color
        
        segmentation_overlay = Image.fromarray(seg_colored)
    
    # Instance Segmentation
    if instance_seg is not None:
        # Create a colored instance map
        instance_colored = np.zeros((instance_seg.shape[0], instance_seg.shape[1], 3), dtype=np.uint8)
        
        # Process each instance
        for instance_id in np.unique(instance_seg):
            if instance_id == 0:  # Skip background
                continue
                
            mask = instance_seg == instance_id
            
            # If labels are provided, use the class color, otherwise random color based on instance_id
            class_id = labels[instance_id - 1] if labels is not None and instance_id <= len(labels) else instance_id
            color = colors.get(class_id, colors.get(instance_id % len(colors), (255, 255, 255)))
            
            # Color the mask
            instance_colored[mask] = color
            
            # Store instance centroid for labeling
            if label_segments:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # Get instance label
                    if class_names is not None and class_id < len(class_names):
                        label_text = f"{class_names[class_id]} {instance_id}"
                    else:
                        label_text = f"Instance {instance_id}"
                    
                    segment_centroids.append({
                        'x': center_x,
                        'y': center_y,
                        'label': label_text,
                        'color': color
                    })
        
        # If we already have semantic seg, blend them
        if segmentation_overlay is not None:
            instance_overlay = Image.fromarray(instance_colored)
            # Blend semantic and instance with equal weight
            segmentation_overlay = Image.blend(segmentation_overlay, instance_overlay, 0.5)
        else:
            segmentation_overlay = Image.fromarray(instance_colored)
    
    # Panoptic Segmentation
    # if panoptic_seg is not None and 'panoptic_seg' in panoptic_seg and 'segments_info' in panoptic_seg:
    #     panoptic_colored = np.zeros((panoptic_seg['panoptic_seg'].shape[0], panoptic_seg['panoptic_seg'].shape[1], 3), dtype=np.uint8)
        
    #     for segment in panoptic_seg['segments_info']:
    #         segment_id = segment['id']
    #         category_id = segment['category_id']
    #         mask = panoptic_seg['panoptic_seg'] == segment_id
            
    #         # Get color for this category
    #         color = colors.get(category_id, (255, 255, 255))
            
    #         # Color the mask
    #         panoptic_colored[mask] = color
            
    #         # Store segment centroid for labeling
    #         if label_segments:
    #             y_indices, x_indices = np.where(mask)
    #             if len(y_indices) > 0:
    #                 center_x = int(np.mean(x_indices))
    #                 center_y = int(np.mean(y_indices))
                    
    #                 # Get category name
    #                 if class_names is not None and category_id < len(class_names):
    #                     label_text = class_names[category_id]
    #                 else:
    #                     label_text = f"Category {category_id}"
                    
    #                 # Add instance ID for things (not stuff)
    #                 if 'isthing' in segment and segment['isthing']:
    #                     label_text = f"{label_text} {segment_id}"
                    
    #                 segment_centroids.append({
    #                     'x': center_x,
    #                     'y': center_y,
    #                     'label': label_text,
    #                     'color': color
    #                 })
        
    #     panoptic_overlay = Image.fromarray(panoptic_colored)
        
    #     # If we already have other segmentation, blend them
    #     if segmentation_overlay is not None:
    #         segmentation_overlay = Image.blend(segmentation_overlay, panoptic_overlay, 0.5)
    #     else:
    #         segmentation_overlay = panoptic_overlay
    # Process panoptic segmentation
        # Process panoptic segmentation
    if panoptic_seg is not None and 'panoptic_seg' in panoptic_seg and 'segments_info' in panoptic_seg:
        # Process all segments
        panoptic_seg_np = panoptic_seg['panoptic_seg'] #(450, 800)
        segments_info = panoptic_seg['segments_info'] #list
        id2label = {i: label for i, label in enumerate(labels)} if labels is not None else None
        segments_data = [
            _process_panoptic_segment(
                segment_info, 
                panoptic_seg_np, 
                (result_img.height, result_img.width),
                id2label
            )
            for segment_info in segments_info
        ] #get bbox
        
        # Create overlay if masks should be drawn
        if draw_masks:
            overlay = _create_panoptic_overlay(
                (result_img.height, result_img.width),
                segments_data,
                colors
            )
            # Convert overlay to PIL Image and store as segmentation_overlay
            if segmentation_overlay is not None:
                # Blend with existing overlay
                panoptic_overlay = overlay
                segmentation_overlay = Image.blend(segmentation_overlay, panoptic_overlay, 0.5)
            else:
                segmentation_overlay = overlay
        #segmentation_overlay.save("output/testseg.png")
        # Draw bounding boxes and labels if requested
        if draw_boxes:
            draw = ImageDraw.Draw(result_img)
            for segment, color in zip(segments_data, colors):
                if segment is not None:
                    _draw_panoptic_segment(draw, segment, color, font, alpha)


    # Apply segmentation overlay
    if segmentation_overlay is not None:
        # Resize if needed
        if segmentation_overlay.size != result_img.size:
            segmentation_overlay = segmentation_overlay.resize(result_img.size, Image.NEAREST)
        
        # Blend with original image
        result_img = Image.blend(result_img, segmentation_overlay, alpha)
        
        # Add segment labels at centroids if requested
        if label_segments and segment_centroids:
            draw = ImageDraw.Draw(result_img)
            
            for centroid in segment_centroids:
                # Draw text with outline for better visibility
                # First draw black outline
                for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text(
                        (centroid['x'] + offset_x, centroid['y'] + offset_y),
                        centroid['label'],
                        font=segment_font,
                        fill=(0, 0, 0)
                    )
                
                # Then draw text in white or contrasting color
                # Choose white or black text based on background color brightness
                r, g, b = centroid['color']
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
                
                draw.text(
                    (centroid['x'], centroid['y']),
                    centroid['label'],
                    font=segment_font,
                    fill=text_color
                )
    
    # Depth map visualization
    if depth_map is not None:
        # Normalize the depth map
        depth_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
        
        # Apply colormap
        depth_colored = (cm.get_cmap(depth_cmap)(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth_overlay = Image.fromarray(depth_colored)
        
        # Resize if needed
        if depth_overlay.size != result_img.size:
            depth_overlay = depth_overlay.resize(result_img.size, Image.NEAREST)
        
        # Create a composite image with depth map
        if segmentation_overlay is not None:
            # We already have segmentation, create a separate depth visualization
            # Create a new figure for side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(np.array(result_img))
            ax1.set_title("Detection & Segmentation")
            ax1.axis('off')
            
            ax2.imshow(np.array(depth_overlay))
            ax2.set_title("Depth Map")
            ax2.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                # Save the figure
                directory = os.path.dirname(output_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # Return the first visualization as the primary result
            # We'll continue with result_img unchanged
        else:
            # No segmentation, we can blend depth with the original
            result_img = Image.blend(result_img, depth_overlay, alpha)
    
    # Draw bounding boxes and labels
    if boxes is not None:
        draw = ImageDraw.Draw(result_img)
        
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes[i]
            x1, y1, x2, y2 = box.astype(int) if isinstance(box, np.ndarray) else map(int, box)
            
            # Get class info
            class_id = int(labels[i]) if labels is not None else 0
            score = scores[i] if scores is not None else None
            color = colors.get(class_id, (255, 0, 0))
            
            # Prepare label text
            if class_names is not None and class_id < len(class_names):
                label_text = class_names[class_id]
            else:
                label_text = f"Class {class_id}"
                
            if score is not None:
                label_text = f"{label_text}: {score:.2f}"
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)
            
            # Draw label background
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:]
            draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)
            
            # Draw label text
            draw.text((x1, y1), label_text, fill=(255, 255, 255), font=font)
    
    # Add legend if requested
    if show_legend and class_names is not None and (labels is not None or semantic_seg is not None or panoptic_seg is not None):
        legend_width = 200
        legend_height = len(class_names) * 25
        legend_img = Image.new('RGB', (legend_width, legend_height), (255, 255, 255))
        legend_draw = ImageDraw.Draw(legend_img)
        
        for i, name in enumerate(class_names):
            if i >= len(class_names):
                break
                
            color = colors.get(i, (255, 0, 0))
            y_pos = i * 25 + 5
            legend_draw.rectangle([5, y_pos, 20, y_pos + 15], fill=color)
            legend_draw.text((30, y_pos), name, fill=(0, 0, 0), font=font)
        
        # Create a composite image with the legend
        composite_width = result_img.width + legend_img.width
        composite_height = max(result_img.height, legend_img.height)
        composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
        composite.paste(result_img, (0, 0))
        composite.paste(legend_img, (result_img.width, 0))
        result_img = composite
    
    # Save the result if output path is provided
    if output_path and depth_map is None:  # If depth map is present, we've already saved
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        result_img.save(output_path)
    
    return result_img

#draw_panoptic_segmentation(**panoptic_segmentation)
def draw_panoptic_segmentation(segmentation, segments_info, id2label):
    segmentation = segmentation.cpu()
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation) #[450, 800]
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id'] #1
        segment_label_id = segment['label_id'] #
        segment_label = id2label[segment_label_id] #'car'
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    ax.legend(handles=handles)
    plt.savefig('output/panoptic_test1.png')

def draw_semantic_segmentation(segmentation, id2label):
    segmentation = segmentation.cpu()
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    # get all the unique numbers
    labels_ids = torch.unique(segmentation).tolist()
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    handles = []
    for label_id in labels_ids:
        label = model.config.id2label[label_id]
        color = viridis(label_id)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles)
    plt.savefig('output/semantic_test1.png')

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

def extract_key_frames(video_path, output_dir, target_size=(640, 480), extraction_method="scene_change", privacy_blur=False):
    """
    Extract key frames from a video file and save them with timestamp names.
    
    Parameters:
    - video_path: Path to the input video file
    - output_dir: Directory to save extracted frames
    - target_size: Tuple of (width, height) maximum dimensions for resizing
    - extraction_method: Method to extract frames ('scene_change', 'interval', or 'both')
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get video metadata
    metadata, gps_info, creation_time = extract_metadata(video_path)
    
    # Save metadata to a JSON file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'video_metadata': metadata, 
            'gps_info': gps_info, 
            'creation_time': creation_time
        }, f, indent=4)
    
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
    
    # Initialize variables
    prev_frame = None
    frame_idx = 0
    saved_count = 0
    
    # Parameters for scene change detection
    min_scene_change_threshold = 30.0  # Minimum threshold for scene change
    frame_interval = int(fps) * 1  # Save a frame every second as fallback
    
    if privacy_blur ==True:
        # Load detection models
        print("Loading privacy models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        face_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
        face_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small").to(device)
        
        plate_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        plate_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device)
        
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
            
            if privacy_blur == True:
                frame = perform_privacyblur(
                        frame, 
                        face_model=face_model, 
                        plate_model=plate_model,
                        device=device
                    )
                
            # Resize the frame maintaining aspect ratio
            pil_img = resize_with_aspect_ratio(frame, target_size)
            new_width, new_height = pil_img.size
            
            # Save the frame
            filename = f"frame_{timestamp.replace(':', '-')}_{reason}.jpg"
            output_path = os.path.join(output_dir, filename)
            pil_img.save(output_path, quality=95)
            
            # Save frame metadata
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
                "extraction_time": datetime.datetime.now().isoformat()
            }
            
            # Add GPS data to frame metadata
            if gps_info:
                frame_meta["gps_info"] = gps_info
            
            # Save frame metadata
            frame_meta_file = os.path.join(output_dir, f"{filename.replace('.jpg', '.json')}")
            with open(frame_meta_file, 'w') as f:
                json.dump(frame_meta, f, indent=4)
            
            saved_count += 1
            print(f"Saved frame {saved_count}: {filename} ({reason})")
        
        # Update variables for next iteration
        prev_frame = gray.copy()
        frame_idx += 1
    
    # Release resources
    cap.release()
    print(f"Extraction complete. Saved {saved_count} key frames to {output_dir}")

def perform_privacyblur(frame, face_model=None, plate_model=None, device=None, confidence_threshold=0.5):
    """
    Detect and blur faces and license plates in a video frame for privacy compliance.
    
    Args:
        frame: numpy array image frame from video
        face_model: pre-loaded face detection model (if None, will load default)
        plate_model: pre-loaded license plate detection model (if None, will load default)
        device: torch device to use (if None, will use GPU if available)
        confidence_threshold: minimum confidence score for detections
    
    Returns:
        numpy array of the frame with faces and license plates blurred
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models if not provided
    if face_model is None:
        print("Loading face detection model...")
        face_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
        face_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small").to(device)
    
    if plate_model is None:
        print("Loading license plate detection model...")
        # Using a general object detection model that can detect license plates
        # You might want to fine-tune this for better license plate detection
        plate_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        plate_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device)
    
    # Make a copy of the original frame
    blurred_frame = frame.copy()
    
    # Detect faces
    face_inputs = face_processor(images=frame, return_tensors="pt").to(device)
    face_outputs = face_model(**face_inputs)
    face_results = face_processor.post_process_object_detection(face_outputs, threshold=confidence_threshold)[0]
    
    # Blur faces
    for score, label, box in zip(face_results["scores"], face_results["labels"], face_results["boxes"]):
        if face_model.config.id2label[label.item()] == "person":
            # Extract face coordinates and add some margin
            x1, y1, x2, y2 = box.int().tolist()
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Extract the region and apply blur
            face_region = blurred_frame[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            blurred_frame[y1:y2, x1:x2] = blurred_face
    
    # Detect license plates
    plate_inputs = plate_processor(images=frame, return_tensors="pt").to(device)
    plate_outputs = plate_model(**plate_inputs)
    plate_results = plate_processor.post_process_object_detection(plate_outputs, threshold=confidence_threshold)[0]
    
    # Blur license plates
    for score, label, box in zip(plate_results["scores"], plate_results["labels"], plate_results["boxes"]):
        # For general object detection models, look for relevant categories
        label_name = plate_model.config.id2label[label.item()]
        if any(keyword in label_name.lower() for keyword in ["car", "vehicle", "truck", "plate"]):
            # Extract license plate coordinates
            x1, y1, x2, y2 = box.int().tolist()
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Extract the region and apply blur
            plate_region = blurred_frame[y1:y2, x1:x2]
            blurred_plate = cv2.GaussianBlur(plate_region, (99, 99), 30)
            blurred_frame[y1:y2, x1:x2] = blurred_plate
    
    return blurred_frame

def perform_panoptic_segmentation(frames_dir, output_dir, model_name="facebook/mask2former-swin-large-coco-panoptic"):
    """
    Perform panoptic segmentation on extracted frames and save visualizations.
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_dir: Directory to save segmentation results
    - model_name: Name or path of the segmentation model to use
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for different visualization types
    bbox_dir = os.path.join(output_dir, "bboxes")
    mask_dir = os.path.join(output_dir, "masks")
    combined_dir = os.path.join(output_dir, "combined")
    
    for directory in [bbox_dir, mask_dir, combined_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    # Get all image files in the input directory
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    print(f"Found {len(image_files)} images to process")
    
    # Generate random colors for segmentation masks
    def get_random_colors(n):
        colors = []
        for i in range(n):
            # Use HSV color space for better visual distinction
            hue = i / n
            saturation = 0.9
            value = 0.9
            rgb = hsv_to_rgb((hue, saturation, value))
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        return colors
    
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image
        image = Image.open(img_path)
        input_image = image.copy()
        
        # Load associated metadata
        json_path = os.path.join(frames_dir, base_name + ".json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                frame_meta = json.load(f)
        else:
            frame_meta = {}
        
        # Preprocess the image
        inputs = processor(images=input_image, return_tensors="pt").to(device)
        #pixel_values: [1, 3, 384, 384], pixel_mask: [1, 384, 384]
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
        #class_queries_logits [1, 200, 134],  masks_queries_logits [1, 200, 96, 96]
        
        # Post-process results
        result = processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[input_image.size[::-1]]
        )[0]
        
        # Extract segmentation mask and metadata
        panoptic_seg = result["segmentation"] #[450, 800]
        segments_info = result["segments_info"] #list of 15,
        print(segments_info[0].keys()) #['id', 'label_id', 'was_fused', 'score']
        
        # Convert panoptic segmentation to numpy array for easier manipulation
        panoptic_seg_np = panoptic_seg.cpu().numpy() #(450, 800) all 12
        
        # Get unique segments and class information
        segments = []
        random_colors = get_random_colors(len(segments_info))
        
        # Create visualization images
        bbox_image = input_image.copy()
        mask_image = Image.new("RGB", input_image.size, (0, 0, 0))
        combined_image = input_image.copy()
        
        draw_bbox = ImageDraw.Draw(bbox_image)
        draw_combined = ImageDraw.Draw(combined_image)
        
        # Try to load font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw each segment
        for i, segment_info in enumerate(segments_info):
            segment_id = segment_info["id"]
            label_id = segment_info["label_id"]#["category_id"] 2
            score = segment_info.get("score", 1.0)
            was_fused = segment_info.get("was_fused", False)
            
            # Get label from processor's id2label mapping
            if hasattr(processor, 'id2label') and label_id in processor.id2label:
                label = processor.id2label[label_id]
            else:
                label = f"Class {label_id}"
            
            # Get binary mask for this segment and calculate area
            binary_mask = (panoptic_seg_np == segment_id) #True/False (450, 800)
            segment_area = np.sum(binary_mask)
            
            # Calculate bounding box
            y_indices, x_indices = np.where(binary_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            else:
                # Skip segments with no visible pixels
                continue
            
            # Determine if this is a "thing" (object) or "stuff" (background)
            # This is a heuristic based on common datasets like COCO
            # Objects typically have smaller areas and higher scores
            is_thing = True
            if segment_area > (input_image.width * input_image.height * 0.4):
                # Large segments are likely background "stuff"
                is_thing = False
            
            # Store segment information
            segment_data = {
                "id": segment_id,
                "label_id": label_id,
                "label": label,
                "score": float(score) if isinstance(score, (int, float, np.number)) else None,
                "area": int(segment_area),
                "bbox": bbox,
                "is_thing": is_thing,
                "was_fused": was_fused
            }
            segments.append(segment_data)
            
            # Get color for this segment
            color = random_colors[i]
            
            # Draw bounding box
            draw_bbox.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            if is_thing:
                label_text = f"{label}"
                if score is not None and isinstance(score, (int, float, np.number)):
                    label_text += f" {score:.2f}"
                draw_bbox.text((x_min, y_min - 12), label_text, fill=color, font=font)
            
            # Draw segmentation mask
            mask_data = np.zeros((input_image.height, input_image.width, 3), dtype=np.uint8)
            mask_data[binary_mask] = color #(450, 800, 3)
            mask_img = Image.fromarray(mask_data)
            mask_image = Image.alpha_composite(
                mask_image.convert('RGBA'), 
                Image.blend(Image.new('RGBA', mask_image.size, (0, 0, 0, 0)), 
                           mask_img.convert('RGBA'), 
                           alpha=1)
            ).convert('RGB')
            
            # Draw combined visualization (transparent mask over image)
            overlay = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            mask_color = color + (64,)  # Add alpha value
            
            # Create mask from binary array
            for y in range(binary_mask.shape[0]):
                for x in range(binary_mask.shape[1]):
                    if binary_mask[y, x]:
                        overlay_draw.point((x, y), fill=mask_color)
            
            # Composite the overlay onto the combined image
            combined_image = Image.alpha_composite(
                combined_image.convert('RGBA'),
                overlay
            ).convert('RGB')
            
            # Draw bounding box on combined image
            draw_combined = ImageDraw.Draw(combined_image)
            draw_combined.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            if is_thing:
                label_text = f"{label}"
                if score is not None and isinstance(score, (int, float, np.number)):
                    label_text += f" {score:.2f}"
                draw_combined.text((x_min, y_min - 12), label_text, fill=color, font=font)
        
        # Save segmentation results
        bbox_image.save(os.path.join(bbox_dir, f"{base_name}_bbox.jpg"))
        mask_image.save(os.path.join(mask_dir, f"{base_name}_mask.jpg"))
        combined_image.save(os.path.join(combined_dir, f"{base_name}_combined.jpg"))
        
        # Save segmentation metadata
        seg_meta = {
            "original_frame": base_filename,
            "segments": segments,
            "frame_metadata": frame_meta,
            "model": model_name,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")


def perform_panoptic_segmentation2(frames_dir, output_dir, model_name="facebook/mask2former-swin-large-coco-panoptic", task='panoptic'):
    """
    Perform panoptic segmentation on extracted frames and save visualizations.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    #processor = AutoImageProcessor.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    # Print the id2label mapping
    id2label = None
    labels = None
    if hasattr(processor, 'id2label'):
        print("ID to Label Mapping:")
        id2label = processor.id2label
    elif hasattr(model.config, 'id2label'):
        id2label=model.config.id2label
    else:
        print("No id2label mapping found in processor")
    if id2label is not None:
        labels = list(id2label.values())
        for id, label in id2label.items():
            print(f"  {id}: {label}")
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image and metadata
        image = Image.open(img_path)
        input_image = np.array(image)  # Convert to numpy array for visualization
        
        json_path = os.path.join(frames_dir, base_name + ".json")
        frame_meta = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                frame_meta = json.load(f)
        
        # Preprocess and generate predictions
        if "oneformer" in model_name:
            inputs = processor(images=image, task_inputs=[task], return_tensors="pt").to(device)
            #decode the task inputs back to text:
            checktask=processor.tokenizer.batch_decode(inputs.task_inputs) #task_inputs: [1, 77]
            print("Decode task is: ", checktask)
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)
        
        for k,v in inputs.items():
            print(k,v.shape) 
        #pixel_values:[1, 3, 749, 1333], 
        # pixel_mask: [1, 749, 1333], 
        # task_inputs: [1, 77]
  
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        target_size=[image.size[::-1]]
        if task == "panoptic":
            result = processor.post_process_panoptic_segmentation(
                outputs, 
                target_sizes=target_size
            )[0]
            print(result.keys()) #['segmentation', 'segments_info']
        elif task == "semantic":
            result = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_size
            )[0]
            #[450, 800] tensor
        elif task == "instance":
            result = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_size
            )[0]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        draw_panoptic_segmentation(result["segmentation"], result["segments_info"], id2label)
        # Prepare visualization inputs
        panoptic_result = {
            "panoptic_seg": result["segmentation"].cpu().numpy(), #(450, 800)
            "segments_info": result["segments_info"] #list of dict
        }
        
        # Create output paths
        bbox_path = os.path.join(output_dir, f"{base_name}_bbox.jpg")
        mask_path = os.path.join(output_dir, f"{base_name}_mask.jpg")
        combined_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
        
        # Visualize bounding boxes only
        bbox_img = visualize_results(
            image=input_image.copy(),  # Use a copy to prevent modifications
            panoptic_seg=panoptic_result,
            draw_boxes=True,
            draw_masks=False,
            labels=labels,
            alpha=0.5,
            output_path=bbox_path
        )
        
        # Visualize masks only
        mask_img = visualize_results(
            image=input_image.copy(),
            panoptic_seg=panoptic_result,
            draw_boxes=False,
            draw_masks=True,
            labels=labels,
            alpha=0.7,  # Increased alpha for better mask visibility
            output_path=mask_path
        )
        
        # Visualize combined (boxes + masks)
        combined_img = visualize_results(
            image=input_image.copy(),
            panoptic_seg=panoptic_result,
            draw_boxes=True,
            draw_masks=True,
            labels=labels,
            alpha=0.5,
            output_path=combined_path
        )
        
        # Save segmentation metadata
        segments = []
        for segment_info in result["segments_info"]:
            segment_data = {
                "id": segment_info["id"],
                "label_id": segment_info["label_id"],
                "label": id2label[segment_info["label_id"]],
                "score": float(segment_info.get("score", 1.0)),
                "was_fused": segment_info.get("was_fused", False)
            }
            segments.append(segment_data)
        
        seg_meta = {
            "original_frame": base_filename,
            "segments": segments,
            "frame_metadata": frame_meta,
            "model": model_name,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
# Function to visualize the results with bounding boxes and segmentation masks
def visualize_boxseg(image, boxes, masks, labels=None, scores=None, alpha=0.4):
    # Convert PIL image to tensor if it's not already
    if not isinstance(image, torch.Tensor):
        image_tensor = pil_to_tensor(image)
    else:
        image_tensor = image
    
    # Draw bounding boxes on the image
    if boxes is not None and len(boxes) > 0:
        boxes = boxes.to(torch.int)
        if labels is not None:
            image_with_boxes = draw_bounding_boxes(
                image_tensor, 
                boxes=boxes,
                labels=[f"{l}: {s:.2f}" for l, s in zip(labels, scores)] if scores is not None else labels,
                colors="red",
                width=4
            )
        else:
            image_with_boxes = draw_bounding_boxes(image_tensor, boxes=boxes, colors="red", width=4)
    else:
        image_with_boxes = image_tensor
    
    # Convert to PIL for display
    result_image = to_pil_image(image_with_boxes)
    
    # If we have masks, overlay them on the image
    if masks is not None and len(masks) > 0:
        result_image_np = np.array(result_image)
        
        # Create a colormap for the masks
        colors = plt.cm.get_cmap('tab10', len(masks))
        
        # Create a blank mask with the same size as the image
        colored_mask = np.zeros_like(result_image_np)
        
        # Fill in each mask with a different color
        for i, mask in enumerate(masks):
            color = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask == 1, color[c], colored_mask[:, :, c])
        
        # Blend the mask with the original image
        result_image_np = (1 - alpha) * result_image_np + alpha * colored_mask
        result_image = Image.fromarray(result_image_np.astype(np.uint8))
    
    return result_image

def perform_box_segmentation(frames_dir, output_dir, model_name, text_prompt="person, car, bicycle, motorcycle, truck, traffic light", 
                                  box_threshold=0.35, text_threshold=0.25):
    """
    Perform object detection using GroundingDINO and segmentation using Segment Anything Model (SAM).
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_dir: Directory to save segmentation results
    - text_prompt: Text prompt describing objects to detect (comma-separated)
    - box_threshold: Confidence threshold for bounding box predictions
    - text_threshold: Confidence threshold for text-to-box associations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for different visualization types
    bbox_dir = os.path.join(output_dir, "bboxes")
    mask_dir = os.path.join(output_dir, "masks")
    combined_dir = os.path.join(output_dir, "combined")
    
    for directory in [bbox_dir, mask_dir, combined_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if "yolo" in model_name:
        # Load YOLOv12 model
        from ultralytics import YOLO
        yolo_model = YOLO(model_name)  # Use nano model, change to s/m/l/x for different sizes
    else:
        # Load GroundingDINO model and processor
        grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Load SAM model
    print("Loading Segment Anything Model...")

    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base") #"facebook/sam-vit-base"
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device) #"facebook/sam2"
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image and metadata
        image = Image.open(img_path).convert("RGB")
        
        # Step 1: 
        boxes = []
        labels = []
        scores = []
        if "yolo" in model_name:
            confidence=0.25
            yolo_results = yolo_model(image, conf=confidence)
            for result in yolo_results:
                # Convert from YOLO format to list of boxes, labels, scores
                if len(result.boxes) > 0:
                    # Convert boxes from xywh to xyxy format
                    boxes_tensor = result.boxes.xyxy
                    boxes.append(boxes_tensor)
                    
                    # Get class labels
                    cls_indices = result.boxes.cls.cpu().numpy()
                    cls_names = [result.names[int(idx)] for idx in cls_indices]
                    labels.extend(cls_names)
                    
                    # Get confidence scores
                    scores.extend(result.boxes.conf.cpu().numpy())
            # Convert list of tensors to single tensor
            if boxes:
                boxes = torch.cat(boxes, dim=0)
            else:
                boxes = torch.zeros((0, 4))  # Empty tensor if no detections
        else:
            # Object detection with GroundingDINO
            #https://huggingface.co/IDEA-Research/grounding-dino-base
            grounding_inputs = grounding_processor(text=text_prompt, images=image, return_tensors="pt").to(device)
            print(grounding_inputs.keys()) #['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']
            #pixel_values: [1, 3, 750, 1333]
            with torch.no_grad():
                outputs = grounding_model(**grounding_inputs)
            
            # Process Grounding DINO results with the grounded method
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            #https://github.com/huggingface/transformers/blob/main/src/transformers/models/grounding_dino/processing_grounding_dino.py
            results = grounding_processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=grounding_inputs.input_ids,
                target_sizes=target_sizes,
                text_threshold=0.4,
                threshold=0.3  # Adjust this threshold as needed
            )[0]
            print(results.keys()) #['scores', 'boxes', 'text_labels', 'labels']
            
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]  # These are now the actual text labels from the prompt
            
        # Step 2: Generate segmentation masks using SAM
        segmentation_masks = []
        
        if len(boxes) > 0:
            for box in boxes:
                # Process the image and bounding box with SAM
                sam_inputs = sam_processor(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
                
                with torch.no_grad():
                    sam_outputs = sam_model(**sam_inputs)
                
                # Get predicted segmentation mask
                masks = sam_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                segmentation_masks.append(masks[0][0][0].numpy())

        # Visualize results
        boxes_tensor = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes
        result_image = visualize_boxseg(
            image, 
            boxes_tensor, 
            segmentation_masks, 
            labels=labels,
            scores=scores
        )
        result_image.save("output/segmentation_result.png")

        # Save segmentation metadata
        seg_meta = {
            "original_frame": base_filename,
            "segments": segmentation_masks,
            #"frame_metadata": frame_meta,
            "model": "GroundingDINO + SAM",
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")

def generate_label_studio_predictions(frames_dir, output_file, model_name, text_prompt="person, car, bicycle, motorcycle, truck, traffic light", 
                                     confidence_threshold=0.25, include_masks=False):
    """
    Generate pre-annotations in Label Studio format from object detection results. https://labelstud.io/guide/predictions
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_file: Path to save the Label Studio predictions JSON file
    - model_name: Name of the model to use for detection (e.g., 'yolov8n.pt')
    - text_prompt: Text prompt for GroundingDINO (if used)
    - confidence_threshold: Minimum confidence score for detections
    - include_masks: Whether to include segmentation masks (if available)
    """
    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if "yolo" in model_name:
        # Load YOLO model
        from ultralytics import YOLO
        model = YOLO(model_name)
    else:
        # Load GroundingDINO model and processor
        grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Load SAM model if including masks
    if include_masks:
        print("Loading Segment Anything Model...")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(image_files)} images to process")
    
    # Prepare Label Studio predictions
    predictions = []
    
    # Process each image
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        print(f"Processing {base_filename}...")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_width, image_height = image.size
        
        # Get image file size
        file_size = os.path.getsize(img_path)
        
        # Create task entry for this image
        task = {
            "data": {
                "image": base_filename,
                "width": image_width,
                "height": image_height,
                "file_size": file_size
            },
            "predictions": []
        }
        
        # Detect objects
        boxes = []
        labels = []
        scores = []
        
        if "yolo" in model_name:
            # YOLO detection
            yolo_results = model(image, conf=confidence_threshold)
            
            # Create prediction entry
            prediction = {
                "model_version": model_name,
                "score": 1.0,  # Overall prediction score
                "result": []
            }
            
            for result in yolo_results:
                if len(result.boxes) > 0:
                    # Get boxes, labels, and scores
                    boxes_tensor = result.boxes.xyxy.cpu()
                    cls_indices = result.boxes.cls.cpu().numpy()
                    cls_names = [result.names[int(idx)] for idx in cls_indices]
                    confidence_scores = result.boxes.conf.cpu().numpy()
                    
                    # Process each detection
                    for i, (box, label, score) in enumerate(zip(boxes_tensor, cls_names, confidence_scores)):
                        if score >= confidence_threshold:
                            x1, y1, x2, y2 = box.tolist()
                            
                            # Calculate normalized coordinates for Label Studio
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Create annotation result
                            annotation = {
                                "id": f"{base_filename}_{i}",
                                "type": "rectanglelabels",
                                "from_name": "label",
                                "to_name": "image",
                                "original_width": image_width,
                                "original_height": image_height,
                                "image_rotation": 0,
                                "value": {
                                    "rotation": 0,
                                    "x": 100 * x1 / image_width,
                                    "y": 100 * y1 / image_height,
                                    "width": 100 * width / image_width,
                                    "height": 100 * height / image_height,
                                    "rectanglelabels": [label]
                                },
                                "score": float(score)
                            }
                            
                            # Add segmentation mask if requested
                            if include_masks:
                                # Process the image and bounding box with SAM
                                sam_inputs = sam_processor(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
                                
                                with torch.no_grad():
                                    sam_outputs = sam_model(**sam_inputs)
                                
                                # Get predicted segmentation mask
                                masks = sam_processor.post_process_masks(
                                    sam_outputs.pred_masks.cpu(),
                                    sam_inputs["original_sizes"].cpu(),
                                    sam_inputs["reshaped_input_sizes"].cpu()
                                )
                                
                                # Convert mask to polygon points for Label Studio
                                mask = masks[0][0][0].numpy()
                                from skimage import measure
                                
                                # Find contours at a constant value of 0.5
                                contours = measure.find_contours(mask, 0.5)
                                
                                # Select the largest contour
                                if contours:
                                    contour = sorted(contours, key=lambda x: len(x))[-1]
                                    
                                    # Convert to Label Studio polygon format (normalized coordinates)
                                    polygon_points = []
                                    for point in contour:
                                        y, x = point
                                        polygon_points.append([
                                            100 * x / image_width,
                                            100 * y / image_height
                                        ])
                                    
                                    # Add polygon annotation
                                    polygon_annotation = {
                                        "id": f"{base_filename}_{i}_mask",
                                        "type": "polygonlabels",
                                        "from_name": "polygon",
                                        "to_name": "image",
                                        "original_width": image_width,
                                        "original_height": image_height,
                                        "value": {
                                            "points": polygon_points,
                                            "polygonlabels": [label]
                                        },
                                        "score": float(score)
                                    }
                                    
                                    prediction["result"].append(polygon_annotation)
                            
                            prediction["result"].append(annotation)
            
            task["predictions"].append(prediction)
            
        else:
            # GroundingDINO detection
            grounding_inputs = grounding_processor(text=text_prompt, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**grounding_inputs)
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            results = grounding_processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=grounding_inputs.input_ids,
                target_sizes=target_sizes,
                text_threshold=confidence_threshold,
                threshold=confidence_threshold
            )[0]
            
            # Create prediction entry
            prediction = {
                "model_version": "GroundingDINO",
                "score": 1.0,
                "result": []
            }
            
            # Process each detection
            for i, (box, label, score) in enumerate(zip(results["boxes"], results["labels"], results["scores"])):
                if score >= confidence_threshold:
                    x1, y1, x2, y2 = box.cpu().tolist()
                    
                    # Calculate normalized coordinates for Label Studio
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create annotation result
                    annotation = {
                        "id": f"{base_filename}_{i}",
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": image_width,
                        "original_height": image_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": 100 * x1 / image_width,
                            "y": 100 * y1 / image_height,
                            "width": 100 * width / image_width,
                            "height": 100 * height / image_height,
                            "rectanglelabels": [label]
                        },
                        "score": float(score)
                    }
                    
                    prediction["result"].append(annotation)
                    
                    # Add segmentation mask if requested
                    if include_masks:
                        # Similar SAM processing as in YOLO case
                        sam_inputs = sam_processor(image, input_boxes=[[box.cpu().tolist()]], return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            sam_outputs = sam_model(**sam_inputs)
                        
                        masks = sam_processor.post_process_masks(
                            sam_outputs.pred_masks.cpu(),
                            sam_inputs["original_sizes"].cpu(),
                            sam_inputs["reshaped_input_sizes"].cpu()
                        )
                        
                        # Convert mask to polygon points
                        mask = masks[0][0][0].numpy()
                        from skimage import measure
                        
                        contours = measure.find_contours(mask, 0.5)
                        
                        if contours:
                            contour = sorted(contours, key=lambda x: len(x))[-1]
                            
                            polygon_points = []
                            for point in contour:
                                y, x = point
                                polygon_points.append([
                                    100 * x / image_width,
                                    100 * y / image_height
                                ])
                            
                            polygon_annotation = {
                                "id": f"{base_filename}_{i}_mask",
                                "type": "polygonlabels",
                                "from_name": "polygon",
                                "to_name": "image",
                                "original_width": image_width,
                                "original_height": image_height,
                                "value": {
                                    "points": polygon_points,
                                    "polygonlabels": [label]
                                },
                                "score": float(score)
                            }
                            
                            prediction["result"].append(polygon_annotation)
            
            task["predictions"].append(prediction)
        
        predictions.append(task)
    
    # Save predictions to file
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Saved Label Studio predictions for {len(predictions)} images to {output_file}")
    
    # Print example Label Studio configuration
    print("\nExample Label Studio configuration:")
    print("""
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="person" background="#FF0000"/>
        <Label value="car" background="#00FF00"/>
        <Label value="bicycle" background="#0000FF"/>
        <Label value="motorcycle" background="#FFFF00"/>
        <Label value="truck" background="#00FFFF"/>
        <Label value="traffic light" background="#FF00FF"/>
      </RectangleLabels>
      <PolygonLabels name="polygon" toName="image">
        <Label value="person" background="#FF0000"/>
        <Label value="car" background="#00FF00"/>
        <Label value="bicycle" background="#0000FF"/>
        <Label value="motorcycle" background="#FFFF00"/>
        <Label value="truck" background="#00FFFF"/>
        <Label value="traffic light" background="#FF00FF"/>
      </PolygonLabels>
    </View>
    """)
    
    return predictions

#pip install ultralytics #for yolo
#pip install transformers #for groundingdino
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract key frames and perform panoptic segmentation')
    parser.add_argument('--video_path', type=str, default='data/SJSU_Sample_Video.mp4', help='Path to the input video file')
    parser.add_argument('--frames_dir', type=str, default='output/extracted', help='Directory containing already extracted frames (skip video extraction)')
    parser.add_argument('--output_dir', type=str, default='output/output_frames', help='Directory to save extracted frames')
    parser.add_argument('--max_width', type=int, default=800, help='Maximum width to resize frames to')
    parser.add_argument('--max_height', type=int, default=800, help='Maximum height to resize frames to')
    parser.add_argument('--method', type=str, default='both', choices=['scene_change', 'interval', 'both'], 
                        help='Method to extract frames: scene_change, interval, or both')
    parser.add_argument('--model_name_path', type=str, default="yolov8n.pt",
                        help='Model to use for object detection, panoptic segmentation: yolo, shi-labs/oneformer_coco_swin_large, facebook/mask2former-swin-large-cityscapes-panoptic, facebook/mask2former-swin-large-coco-panoptic')
    parser.add_argument('--segmentation_type', type=str, default='grounding', 
                    choices=['universal', 'grounding'], 
                    help='Type of segmentation to perform')
    parser.add_argument('--task', type=str, default='semantic', 
                    choices=['semantic', 'instance', 'panoptic'], 
                    help='Type of tasks to perform')#task = "semantic" #task_inputs=["panoptic"], "instance" semantic
    parser.add_argument('--text_prompt', type=str, 
                    default='person, car, bicycle, motorcycle, truck, traffic light', 
                    help='Text prompt for GroundingDINO (comma-separated objects)')
    parser.add_argument('--skip_extraction', default=False, help='Skip video extraction and use existing frames') #action='store_true'
    parser.add_argument('--privacy_blur', default=True, help='Blur face and license plate number')
    parser.add_argument('--output_labelstudio', default=True, action='store_true', help='output results to label studio')
    parser.add_argument('--include_masks', default=False, action='store_true', help='Skip segmentation and only extract frames')
    
    args = parser.parse_args()
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process video if provided and not skipped
    if args.video_path and not args.skip_extraction:
        frames_dir = f"{args.output_dir}_frames_{timestamp}"
        extract_key_frames(
            args.video_path, 
            frames_dir, 
            target_size=(args.max_width, args.max_height),
            extraction_method=args.method
        )
    else:
        frames_dir = args.frames_dir
    
    if args.output_labelstudio:
        labelstudio_output_dir = f"{frames_dir}/labelstudio_predictions.json"
        generate_label_studio_predictions(
            frames_dir=frames_dir, 
            output_file=labelstudio_output_dir, #"output/label_studio_predictions.json",
            model_name=args.model_name_path, #"yolov8n.pt",
            confidence_threshold=0.3,
            include_masks=args.include_masks
        )
    else:
        seg_output_dir = f"{args.output_dir}_segmentation_{timestamp}"
        if args.segmentation_type == "universal":
            perform_panoptic_segmentation2(
                frames_dir,
                seg_output_dir,
                model_name=args.model_name_path,
                task=args.task
            )
        else:
            perform_box_segmentation(frames_dir, seg_output_dir, \
                model_name=args.model_name_path, text_prompt="a person. a car. a bicycle. a motorcycle. a truck. traffic light", 
                                  box_threshold=0.35, text_threshold=0.25)