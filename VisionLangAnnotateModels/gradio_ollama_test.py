import os
import sys
import json
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Try to import gradio, but handle the case where it's not installed
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio is not installed. To use the GUI interface, please install it with:")
    print("pip install gradio")
    print("Falling back to command-line interface.")

import numpy as np
from PIL import Image, ImageDraw

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Ollama utilities
from VisionLangAnnotateModels.VLM.ollama_utils import (
    OllamaClient,
    process_with_ollama_text,
    process_with_ollama_vision,
    process_with_ollama
)


def test_text_interface():
    """
    Test the text-only interface for Ollama.
    """
    print("\n=== Testing Text-Only Interface ===")
    
    # Example prompt for text processing
    prompt = "What is the capital of France?"
    model = "qwen2.5vl"  # Using the available model on the Ollama instance
    
    print(f"Sending prompt to {model}: '{prompt}'")
    
    # Call the text interface
    result = process_with_ollama_text(prompt, model)
    
    # Print the result
    print("Success:", result["success"])
    if result["success"]:
        print("Response:", result["response"])
    else:
        print("Error:", result["error"])


def test_vision_interface():
    """
    Test the vision interface for Ollama.
    """
    print("\n=== Testing Vision Interface ===")
    
    # Create a simple test image
    width, height = 224, 224
    image = Image.new('RGB', (width, height), color='white')
    
    # Draw a simple shape on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle([(50, 50), (150, 150)], fill='blue')
    
    # Example prompt for vision processing
    prompt = "What do you see in this image?"
    model = "qwen2.5-vl"  # Replace with an available vision model on your Ollama instance
    
    print(f"Sending image and prompt to {model}: '{prompt}'")
    
    # Call the vision interface
    result = process_with_ollama_vision([image], [prompt], model)
    
    # Print the result
    print("Success:", result["success"])
    if result["success"]:
        print("Response:", result["response"])
    else:
        print("Error:", result["error"])


def test_batch_vision_interface():
    """
    Test the batch vision interface for Ollama.
    """
    print("\n=== Testing Batch Vision Interface ===")
    
    # Create two simple test images
    images = []
    prompts = []
    
    # Image 1: Red circle
    img1 = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img1)
    draw.ellipse([(50, 50), (150, 150)], fill='red')
    images.append(img1)
    prompts.append("What shape and color do you see in this image?")
    
    # Image 2: Green triangle
    img2 = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img2)
    draw.polygon([(100, 50), (50, 150), (150, 150)], fill='green')
    images.append(img2)
    prompts.append("What shape and color do you see in this image?")
    
    # Combined prompt for batch processing
    combined_prompt = "Describe each of these images, focusing on the shapes and colors present."
    model = "qwen2.5-vl"  # Replace with an available vision model on your Ollama instance
    
    print(f"Sending {len(images)} images to {model} with combined prompt")
    
    # Call the vision interface with combined prompt
    result = process_with_ollama_vision(images, prompts, model, combined_prompt)
    
    # Print the result
    print("Success:", result["success"])
    if result["success"]:
        print("Response:", result["response"])
    else:
        print("Error:", result["error"])


def test_original_interface():
    """
    Test the original process_with_ollama interface.
    """
    print("\n=== Testing Original Interface ===")
    
    # Example data for the original interface
    descriptions = [
        "A blue car parked on the street",
        "A person walking on the sidewalk",
        "A traffic light showing red signal"
    ]
    
    step1_labels = ["car", "person", "traffic light"]
    image_paths = ["image1.jpg", "image1.jpg", "image2.jpg"]
    model = "qwen2.5vl"  # Using the available model on the Ollama instance
    
    print(f"Processing {len(descriptions)} descriptions with {model}")
    
    # Call the original interface
    results = process_with_ollama(descriptions, step1_labels, image_paths, model)
    
    # Print the results
    print(f"Received {len(results)} results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Class: {result.get('class', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")


# Global variables to store image paths and current index
image_paths = []
current_image_index = 0


def load_images_from_folder(folder_path: str) -> List[str]:
    """
    Load all image files from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return []
    
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_files = []
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file_path)
    
    return sorted(image_files)


def select_folder(folder_path: str) -> Tuple[str, gr.Dropdown, gr.Image, str]:
    """
    Select a folder and load all images from it.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        Status message, updated dropdown, first image, and image path
    """
    global image_paths, current_image_index
    
    if not folder_path:
        return "Please select a folder", gr.Dropdown(choices=[], value=None), None, ""
    
    image_paths = load_images_from_folder(folder_path)
    current_image_index = 0
    
    if not image_paths:
        return "No images found in the selected folder", gr.Dropdown(choices=[], value=None), None, ""
    
    # Create dropdown with image filenames (not full paths)
    image_filenames = [os.path.basename(path) for path in image_paths]
    dropdown = gr.Dropdown(choices=image_filenames, value=image_filenames[0] if image_filenames else None)
    
    # Load the first image
    first_image = Image.open(image_paths[0]) if image_paths else None
    
    return f"Loaded {len(image_paths)} images from {folder_path}", dropdown, first_image, image_paths[0]


def select_image_from_dropdown(image_filename: str) -> Tuple[gr.Image, str]:
    """
    Select an image from the dropdown.
    
    Args:
        image_filename: Filename of the selected image
        
    Returns:
        Selected image and its path
    """
    global image_paths, current_image_index
    
    if not image_filename or not image_paths:
        return None, ""
    
    # Find the index of the selected image
    for i, path in enumerate(image_paths):
        if os.path.basename(path) == image_filename:
            current_image_index = i
            selected_image = Image.open(path)
            return selected_image, path
    
    return None, ""


def navigate_images(direction: str, current_image_path: str) -> Tuple[gr.Image, gr.Dropdown, str]:
    """
    Navigate to the next or previous image.
    
    Args:
        direction: "next", "prev", or "random"
        current_image_path: Path of the current image
        
    Returns:
        New image, updated dropdown, and new image path
    """
    global image_paths, current_image_index
    
    if not image_paths:
        return None, gr.Dropdown(choices=[], value=None), ""
    
    if direction == "next":
        current_image_index = (current_image_index + 1) % len(image_paths)
    elif direction == "prev":
        current_image_index = (current_image_index - 1) % len(image_paths)
    elif direction == "random":
        current_image_index = random.randint(0, len(image_paths) - 1)
    
    new_image_path = image_paths[current_image_index]
    new_image = Image.open(new_image_path)
    
    # Update dropdown to match the current image
    image_filenames = [os.path.basename(path) for path in image_paths]
    dropdown = gr.Dropdown(choices=image_filenames, value=os.path.basename(new_image_path))
    
    return new_image, dropdown, new_image_path


def process_text_query(model: str, prompt: str) -> str:
    """
    Process a text query using the selected Ollama model.
    
    Args:
        model: Name of the Ollama model to use
        prompt: Text prompt to send to the model
        
    Returns:
        Model response or error message
    """
    if not model or not prompt:
        return "Please provide both a model name and a prompt"
    
    result = process_with_ollama_text(prompt, model)
    
    if result["success"]:
        return result["response"]
    else:
        return f"Error: {result.get('error', 'Unknown error')}"


def process_vision_query(model: str, prompt: str, image, image_path: str) -> str:
    """
    Process a vision query using the selected Ollama model.
    
    Args:
        model: Name of the Ollama vision model to use
        prompt: Text prompt to send with the image
        image: PIL Image object
        image_path: Path to the image file
        
    Returns:
        Model response or error message
    """
    if not model or not prompt or image is None:
        return "Please provide a model name, prompt, and image"
    
    # Convert image to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    result = process_with_ollama_vision([image], [prompt], model)
    
    if result["success"]:
        return result["response"]
    else:
        return f"Error: {result.get('error', 'Unknown error')}"


def save_results(image, image_path: str, prompt: str, response: str) -> str:
    """
    Save the image and results to a file.
    
    Args:
        image: PIL Image object
        image_path: Path to the original image file
        prompt: Text prompt used for the query
        response: Model response
        
    Returns:
        Status message
    """
    if image is None or not image_path or not response:
        return "Nothing to save"
    
    # Create a results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ollama_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the original image filename without extension
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the image
    image_save_path = os.path.join(results_dir, f"{image_filename}_{timestamp}.png")
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(image_save_path)
    else:
        image.save(image_save_path)
    
    # Save the results as JSON
    results_save_path = os.path.join(results_dir, f"{image_filename}_{timestamp}.json")
    results_data = {
        "original_image": image_path,
        "prompt": prompt,
        "response": response,
        "timestamp": timestamp,
        "saved_image": image_save_path
    }
    
    with open(results_save_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    return f"Results saved to {results_dir}"


def create_gradio_interface():
    """
    Create a Gradio interface for testing Ollama models.
    """
    if not GRADIO_AVAILABLE:
        print("Cannot create Gradio interface: Gradio is not installed.")
        return None
        
    with gr.Blocks(title="Ollama Model Tester") as demo:
        gr.Markdown("# Ollama Model Tester")
        
        with gr.Tab("Text Model"):
            with gr.Row():
                text_model = gr.Textbox(label="Model Name", placeholder="llama3", value="llama3")
            
            with gr.Row():
                text_prompt = gr.Textbox(label="Prompt", placeholder="What is the capital of France?", 
                                        lines=5)
            
            with gr.Row():
                text_submit = gr.Button("Submit")
            
            with gr.Row():
                text_response = gr.Textbox(label="Response", lines=10)
            
            text_submit.click(
                fn=process_text_query,
                inputs=[text_model, text_prompt],
                outputs=text_response
            )
        
        with gr.Tab("Vision Model"):
            with gr.Row():
                vision_model = gr.Textbox(label="Model Name", placeholder="qwen2.5-vl", value="qwen2.5-vl")
            
            with gr.Row():
                folder_input = gr.Textbox(label="Image Folder Path", placeholder="/path/to/images")
                folder_button = gr.Button("Load Folder")
            
            with gr.Row():
                folder_status = gr.Textbox(label="Folder Status")
            
            with gr.Row():
                image_dropdown = gr.Dropdown(label="Select Image", choices=[])
            
            with gr.Row():
                prev_button = gr.Button("Previous Image")
                next_button = gr.Button("Next Image")
                random_button = gr.Button("Random Image")
            
            with gr.Row():
                vision_image = gr.Image(label="Image", type="pil")
                vision_image_path = gr.Textbox(label="Image Path", visible=False)
            
            with gr.Row():
                vision_prompt = gr.Textbox(label="Prompt", placeholder="What do you see in this image?", 
                                          lines=3, value="Describe what you see in this image.")
            
            with gr.Row():
                vision_submit = gr.Button("Submit")
            
            with gr.Row():
                vision_response = gr.Textbox(label="Response", lines=10)
            
            with gr.Row():
                save_button = gr.Button("Save Results")
                save_status = gr.Textbox(label="Save Status")
            
            # Connect components
            folder_button.click(
                fn=select_folder,
                inputs=[folder_input],
                outputs=[folder_status, image_dropdown, vision_image, vision_image_path]
            )
            
            image_dropdown.change(
                fn=select_image_from_dropdown,
                inputs=[image_dropdown],
                outputs=[vision_image, vision_image_path]
            )
            
            prev_button.click(
                fn=lambda: navigate_images("prev", vision_image_path.value),
                inputs=[],
                outputs=[vision_image, image_dropdown, vision_image_path]
            )
            
            next_button.click(
                fn=lambda: navigate_images("next", vision_image_path.value),
                inputs=[],
                outputs=[vision_image, image_dropdown, vision_image_path]
            )
            
            random_button.click(
                fn=lambda: navigate_images("random", vision_image_path.value),
                inputs=[],
                outputs=[vision_image, image_dropdown, vision_image_path]
            )
            
            vision_submit.click(
                fn=process_vision_query,
                inputs=[vision_model, vision_prompt, vision_image, vision_image_path],
                outputs=vision_response
            )
            
            save_button.click(
                fn=save_results,
                inputs=[vision_image, vision_image_path, vision_prompt, vision_response],
                outputs=save_status
            )
    
    return demo


def run_cli_tests():
    """
    Run the command-line tests.
    """
    print("Running command-line tests")
    print("=======================")
    
    test_text_interface()
    # test_vision_interface()  # Uncomment if you have a vision model available
    # test_batch_vision_interface()  # Uncomment if you have a vision model available
    test_original_interface()

#python test_ollama_utils.py --cli
def main():
    """
    Run the Gradio interface or command-line tests.
    """
    # Check if --cli flag is provided or if Gradio is not available
    if len(sys.argv) > 1 and sys.argv[1] == "--cli" or not GRADIO_AVAILABLE:
        run_cli_tests()
    else:
        # Run the Gradio interface
        demo = create_gradio_interface()
        if demo:
            print("Starting Gradio interface. Access it in your web browser.")
            demo.launch(server_name="0.0.0.0", server_port=7860)
        else:
            print("Failed to create Gradio interface. Running command-line tests instead.")
            run_cli_tests()


if __name__ == "__main__":
    main()