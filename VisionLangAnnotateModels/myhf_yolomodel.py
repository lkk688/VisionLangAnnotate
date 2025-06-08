import os
import cv2 #pip install opencv-python
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from tqdm import tqdm

def test_huggingface_model(repo_id, image_path, output_dir=None, confidence_threshold=0.25):
    """
    Download a YOLO model from HuggingFace, run inference on images, and save results.
    
    Args:
        repo_id (str): HuggingFace repository ID (e.g., 'lkk688/yolov8s-model')
        image_path (str): Path to an image or directory of images
        output_dir (str, optional): Directory to save results. If None, will use './output'
        confidence_threshold (float): Confidence threshold for detections (0.0 to 1.0)
    
    Returns:
        list: List of paths to saved result images
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    register_yolo_architecture()
    print(f"Downloading model from {repo_id}...")
    # Load model and processor from Hugging Face
    processor = AutoImageProcessor.from_pretrained(repo_id)#("facebook/detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained(repo_id)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded and moved to {device}")
    
    # Get class names from model config
    id2label = model.config.id2label
    
    # Process single image or directory
    if os.path.isdir(image_path):
        # Get all image files in directory
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        print(f"Found {len(image_files)} images in directory")
    else:
        # Single image file
        image_files = [image_path]
    
    result_paths = []
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Get base filename for output
            base_name = os.path.basename(img_file)
            output_path = os.path.join(output_dir, f"detected_{base_name}")
            
            # Load image with PIL for the processor
            pil_image = Image.open(img_file).convert("RGB")
            
            # Also load with OpenCV for visualization
            cv_image = cv2.imread(img_file)
            
            # Process image and run inference
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs, 
                threshold=confidence_threshold,
                target_sizes=target_sizes
            )[0]
            
            # Draw bounding boxes on the image
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [int(i) for i in box.tolist()]
                
                # Generate a color based on the class label
                color_factor = (int(label) * 50) % 255
                color = (color_factor, 255 - color_factor, 128)
                
                # Draw bounding box
                cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Get class name
                class_name = id2label[str(label.item())]
                
                # Create label with class name and score
                label_text = f"{class_name}: {score.item():.2f}"
                
                # Add a filled rectangle behind text for better visibility
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(cv_image, (box[0], box[1] - text_size[1] - 10), 
                             (box[0] + text_size[0], box[1]), color, -1)
                
                # Add text with white color for better contrast
                cv2.putText(cv_image, label_text, (box[0], box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save the result
            cv2.imwrite(output_path, cv_image)
            result_paths.append(output_path)
            
            print(f"Processed {img_file} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Processed {len(result_paths)} images. Results saved to {output_dir}")
    return result_paths

def test_multiple_models():
    """
    Test multiple YOLOv8 models from Hugging Face on the same images.
    """
    # Directory containing test images
    image_dir = "/SSD250G/MyRepo/DeepDataMiningLearning/sampledata"
    
    # List of models to test (different scales)
    models = [
        "lkk688/yolov8n-model",  # nano
        "lkk688/yolov8s-model",  # small
        "lkk688/yolov8m-model",  # medium
        # "lkk688/yolov8l-model",  # large
        # "lkk688/yolov8x-model",  # xlarge
    ]
    
    # Test each model
    for model_repo in models:
        scale = model_repo.split('yolov8')[1].split('-')[0]
        print(f"\n=== Testing YOLOv8{scale} model ===\n")
        
        # Create a scale-specific output directory
        output_dir = f"/SSD250G/MyRepo/DeepDataMiningLearning/output/yolov8{scale}"
        
        # Run inference
        test_huggingface_model(
            repo_id=model_repo,
            image_path=image_dir,
            output_dir=output_dir,
            confidence_threshold=0.25
        )

def test_single_image():
    """
    Test a single YOLOv8 model on a single image.
    """
    # Model to use
    model_repo = "lkk688/yolov8s-model"  # small model
    
    # Single image path
    image_path = "ModelDev/sampledata/bus.jpg"
    
    # Output directory
    output_dir = "output/single_test"
    
    # Run inference
    test_huggingface_model(
        repo_id=model_repo,
        image_path=image_path,
        output_dir=output_dir,
        confidence_threshold=0.25
    )

import torch
from transformers import AutoModelForObjectDetection, AutoConfig

# Import your model and registration function
from DeepDataMiningLearning.detection.modeling_yolohf import YoloDetectionModel, YoloConfig, register_yolo_architecture

def test():
    # Register the architecture first
    register_yolo_architecture()

    # Now load the model
    model_id = "lkk688/yolov8s-model"
    model = AutoModelForObjectDetection.from_pretrained(model_id)
    print(f"Successfully loaded model from {model_id}")

if __name__ == "__main__":
    #test()
    # Test a single image with one model
    test_single_image()
    
    # Uncomment to test multiple models on multiple images
    # test_multiple_models()