import os
import sys
import torch
from PIL import Image
import argparse

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from VisionLangAnnotateModels.VLM.vllm_utils import VLLMBackend, VLLM_AVAILABLE
from VisionLangAnnotateModels.VLM.vlm_classifierv3 import VLMClassifier

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Example of using vLLM backend with VLMClassifier")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="vLLM model to use")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to an image file")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for the image")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    args = parser.parse_args()
    
    # Check if vLLM is available
    if not VLLM_AVAILABLE:
        print("vLLM is not available. Please install it with 'pip install vllm'.")
        return
    
    # Load the image
    try:
        image = Image.open(args.image).convert("RGB")
        print(f"Loaded image: {args.image}")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return
    
    # Create the vLLM backend
    try:
        print(f"Creating VLLMBackend with model: {args.model}")
        backend = VLLMBackend(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size
        )
    except Exception as e:
        print(f"Error creating vLLM backend: {str(e)}")
        return
    
    # Create the classifier
    classifier = VLMClassifier(backend=backend)
    
    # Prepare the input
    image_prompt_pair = [(image, args.prompt)]
    
    # Run classification
    print(f"Running classification with prompt: '{args.prompt}'")
    results = classifier.classify(image_prompt_pair)
    
    # Print the result
    print("\nClassification Result:")
    print(results[0])
    
    # Try other classification methods (for demonstration)
    print("\nTrying other classification methods (note: vLLM ignores the images):\n")
    
    # Relative position classification
    print("1. Relative Position Classification:")
    rel_results = classifier.classify_relativepos(
        object_crops=[image],
        full_context_images=[image],
        prompts=[args.prompt]
    )
    print(rel_results[0])
    
    # Overlay classification
    print("\n2. Overlay Classification:")
    overlay_results = classifier.classify_overlay(
        full_context_images_with_bbox=[image],
        prompts=[args.prompt]
    )
    print(overlay_results[0])
    
    # Two-view classification
    print("\n3. Two-view Classification:")
    twoview_results = classifier.classify_twoview(
        crops=[image],
        full_contexts=[image],
        prompts=[args.prompt]
    )
    print(twoview_results[0])

if __name__ == "__main__":
    main()