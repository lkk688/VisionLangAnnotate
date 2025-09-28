#!/usr/bin/env python3
"""
VLM vLLM Utility Class

A utility class that wraps common vLLM functionality with two modes:
1. URL Mode: Uses existing vLLM server via HTTP API
2. vLLM Package Mode: Uses vLLM Python package directly

This class provides a unified interface for vision-language model operations
regardless of the underlying implementation.
"""

import requests
import json
import base64
import os
from PIL import Image
import io
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Literal
import statistics
import urllib.request
import urllib.parse
from enum import Enum

try:
    from vllm import LLM, SamplingParams
    #from vllm.utils import is_hip
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

class VLMMode(Enum):
    """Enumeration for VLM operation modes."""
    URL = "url"
    PACKAGE = "package"

class VLMVLLMUtility:
    """
    A utility class for Vision-Language Model operations using vLLM.
    
    Supports two modes:
    1. URL Mode: Communicates with a running vLLM server via HTTP API
    2. Package Mode: Uses vLLM Python package directly for inference
    """
    
    def __init__(self, 
                 mode: Union[VLMMode, str] = VLMMode.URL,
                 api_url: str = "http://localhost:8000",
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 **kwargs):
        """
        Initialize the VLM utility.
        
        Args:
            mode: Operation mode - either VLMMode.URL or VLMMode.PACKAGE
            api_url: Base URL for vLLM server (URL mode only)
            model_name: Model name/path to use
            **kwargs: Additional arguments for vLLM initialization (Package mode only)
        """
        if isinstance(mode, str):
            mode = VLMMode(mode)
        
        self.mode = mode
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.llm = None
        
        if self.mode == VLMMode.PACKAGE:
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM package is not available. Install it with: pip install vllm"
                )
            self._initialize_package_mode(**kwargs)
        elif self.mode == VLMMode.URL:
            self._validate_server_connection()
    
    def _initialize_package_mode(self, **kwargs):
        """Initialize vLLM in package mode."""
        default_kwargs = {
            "trust_remote_code": True,
            "max_model_len": 1024,  # Very small context to minimize memory usage
            "limit_mm_per_prompt": {"image": 1},
            "enforce_eager": True,  # Add this to avoid engine core issues
            "gpu_memory_utilization": 0.25,  # Very conservative memory usage
            "allowed_local_media_path": "/home/lkk/Developer/VisionLangAnnotate",  # Allow local image loading
            "disable_log_stats": True,  # Reduce overhead
            "disable_custom_all_reduce": True  # Reduce memory overhead
        }
        default_kwargs.update(kwargs)
        
        try:
            self.llm = LLM(model=self.model_name, **default_kwargs)
            print(f"✅ vLLM package mode initialized with model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM package mode: {str(e)}")
    
    def _validate_server_connection(self):
        """Validate connection to vLLM server in URL mode."""
        try:
            health_url = f"{self.api_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to vLLM server at {self.api_url}")
            else:
                raise ConnectionError(f"Server returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to vLLM server: {str(e)}")
    
    def detect_image_input_type(self, image_input: Union[str, Image.Image]) -> str:
        """
        Detect the type of image input: URL, file path, or PIL Image.
        
        Args:
            image_input: Can be a URL string, file path string, or PIL Image object
            
        Returns:
            String indicating the input type: 'url', 'file_path', or 'pil_image'
        """
        if isinstance(image_input, Image.Image):
            return 'pil_image'
        elif isinstance(image_input, str):
            # Check if it's a URL
            parsed = urllib.parse.urlparse(image_input)
            if parsed.scheme in ('http', 'https'):
                return 'url'
            else:
                return 'file_path'
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def download_image_from_url(self, url: str) -> Image.Image:
        """
        Download an image from a URL and return as PIL Image.
        
        Args:
            url: HTTP/HTTPS URL to the image
            
        Returns:
            PIL Image object
        """
        try:
            with urllib.request.urlopen(url) as response:
                image_data = response.read()
                return Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to download image from URL {url}: {str(e)}")

    def resize_image_for_context(self, 
                                image_input: Union[str, Image.Image], 
                                max_pixels: int = 1280*28*28, 
                                min_pixels: int = 256*28*28) -> Image.Image:
        """
        Resize image to fit within token constraints for Qwen2.5-VL-7B-Instruct.
        
        With max-model-len=32768:
        - Each 28x28 pixel patch = 1 visual token
        - Default range: 4-16384 visual tokens per image
        - Recommended range: 256-1280 tokens (256*28*28 to 1280*28*28 pixels)
        - This leaves room for text tokens in the 32K context window
        
        Args:
            image_input: Can be a file path (str), URL (str), or PIL Image object
            max_pixels: Maximum pixels (default: 1280*28*28 = 1,003,520 pixels)
            min_pixels: Minimum pixels (default: 256*28*28 = 200,704 pixels)
        
        Returns:
            PIL Image object resized to fit constraints
        """
        # Handle different input types
        input_type = self.detect_image_input_type(image_input)
        
        if input_type == 'pil_image':
            img = image_input.copy()
        elif input_type == 'url':
            img = self.download_image_from_url(image_input)
        else:  # file_path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = Image.open(image_input)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        current_pixels = width * height

        # If image is within bounds, return as-is
        if min_pixels <= current_pixels <= max_pixels:
            return img.copy()
        
        # Calculate scaling factor
        if current_pixels > max_pixels:
            scale_factor = (max_pixels / current_pixels) ** 0.5
        else:  # current_pixels < min_pixels
            scale_factor = (min_pixels / current_pixels) ** 0.5
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img

    def encode_image_to_base64(self, 
                              image_input: Union[str, Image.Image], 
                              resize_for_context: bool = True) -> str:
        """Convert image input (URL, file path, or PIL Image) to base64 data URL with optional resizing"""
        
        # Handle different input types
        input_type = self.detect_image_input_type(image_input)
        
        if resize_for_context:
            # Resize image to fit context constraints
            img = self.resize_image_for_context(image_input)
            
            # Convert PIL image to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = 'image/jpeg'
        else:
            # Handle different input types without resizing
            if input_type == 'pil_image':
                # Convert PIL image to base64
                buffer = io.BytesIO()
                image_input.save(buffer, format='JPEG', quality=95)
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                mime_type = 'image/jpeg'
            elif input_type == 'url':
                # Download and convert URL image
                img = self.download_image_from_url(image_input)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95)
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                mime_type = 'image/jpeg'
            else:  # file_path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                
                # Use original image file
                with open(image_input, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Determine MIME type based on file extension
                ext = os.path.splitext(image_input)[1].lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg', 
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp'
                }.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{encoded_string}"

    def check_server_status(self) -> Dict[str, Any]:
        """
        Check vLLM server status and retrieve server information (URL mode only).
        
        Returns:
            Dictionary containing server status and information
        """
        if self.mode != VLMMode.URL:
            return {"error": "Server status check only available in URL mode"}
        
        result = {
            "server_running": False,
            "models": [],
            "server_info": {},
            "error": None
        }
        
        try:
            # Check if server is running with health endpoint
            health_url = f"{self.api_url}/health"
            health_response = requests.get(health_url, timeout=5)
            
            if health_response.status_code == 200:
                result["server_running"] = True
            else:
                result["error"] = f"Server returned status code: {health_response.status_code}"
                return result
                
        except requests.exceptions.RequestException as e:
            result["error"] = f"Cannot connect to server: {str(e)}"
            return result
        
        try:
            # Get model information
            models_url = f"{self.api_url}/v1/models"
            models_response = requests.get(models_url, timeout=10)
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                result["models"] = models_data.get("data", [])
            else:
                result["error"] = f"Failed to get models: {models_response.status_code}"
                
        except requests.exceptions.RequestException as e:
            result["error"] = f"Failed to get model info: {str(e)}"
        
        return result

    def process_image_url_mode(self, 
                              image_input: Union[str, Image.Image], 
                              prompt: str = "Describe this image in detail. What objects do you see?",
                              max_tokens: int = 1024, 
                              resize_for_context: bool = True,
                              **kwargs) -> Dict[str, Any]:
        """
        Process a single image using URL mode (vLLM server).
        
        Args:
            image_input: Image input (URL, file path, or PIL Image)
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            resize_for_context: Whether to resize image for context limits
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dictionary containing response and performance metrics
        """
        result = {
            "image_path": str(image_input) if not isinstance(image_input, Image.Image) else "PIL_Image",
            "success": False,
            "response": None,
            "performance": {},
            "error": None
        }
        
        try:
            # Start timing
            start_time = time.time()
            
            # Process image
            image_data_url = self.encode_image_to_base64(image_input, resize_for_context=resize_for_context)
            
            # Create payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}}
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add any additional parameters
            payload.update(kwargs)
            
            # Make API request
            response = requests.post(f"{self.api_url}/v1/chat/completions", 
                                   json=payload, 
                                   timeout=120)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response
                if 'choices' in data and data['choices']:
                    result["response"] = data['choices'][0]['message']['content']
                    result["success"] = True
                
                # Extract performance metrics
                if 'usage' in data:
                    usage = data['usage']
                    result["performance"] = {
                        "duration_seconds": end_time - start_time,
                        "prompt_tokens": usage.get('prompt_tokens', 0),
                        "completion_tokens": usage.get('completion_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0)
                    }
                    
                    # Calculate tokens per second
                    if result["performance"]["total_tokens"] > 0:
                        result["performance"]["tokens_per_second"] = (
                            result["performance"]["total_tokens"] / result["performance"]["duration_seconds"]
                        )
                
            else:
                result["error"] = f"API request failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result

    def process_image_package_mode(self, 
                                  image_input: Union[str, Image.Image], 
                                  prompt: str = "Describe this image in detail. What objects do you see?",
                                  max_tokens: int = 1024, 
                                  resize_for_context: bool = True,
                                  **kwargs) -> Dict[str, Any]:
        """
        Process a single image using package mode (vLLM Python package).
        
        Args:
            image_input: Image input (URL, file path, or PIL Image)
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            resize_for_context: Whether to resize image for context limits
            **kwargs: Additional parameters for sampling
            
        Returns:
            Dictionary containing response and performance metrics
        """
        if self.llm is None:
            return {"error": "vLLM not initialized in package mode", "success": False}
        
        result = {
            "image_path": str(image_input) if not isinstance(image_input, Image.Image) else "PIL_Image",
            "success": False,
            "response": None,
            "performance": {},
            "error": None
        }
        
        try:
            # Start timing
            start_time = time.time()
            
            # Process image
            if resize_for_context:
                img = self.resize_image_for_context(image_input)
            else:
                input_type = self.detect_image_input_type(image_input)
                if input_type == 'pil_image':
                    img = image_input
                elif input_type == 'url':
                    img = self.download_image_from_url(image_input)
                else:  # file_path
                    if not os.path.exists(image_input):
                        raise FileNotFoundError(f"Image file not found: {image_input}")
                    img = Image.open(image_input)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Use generate format with proper Qwen2.5-VL prompt format
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # Create prompt input with multi_modal_data
            prompt_input = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": img}
            }
            
            # Generate response
            outputs = self.llm.generate(
                prompt_input,
                sampling_params=sampling_params
            )
            
            end_time = time.time()
            
            if outputs:
                output = outputs[0]
                result["response"] = output.outputs[0].text
                result["success"] = True
                
                # Calculate performance metrics
                result["performance"] = {
                    "duration_seconds": end_time - start_time,
                    "prompt_tokens": len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0,
                    "completion_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
                    "total_tokens": 0
                }
                
                result["performance"]["total_tokens"] = (
                    result["performance"]["prompt_tokens"] + result["performance"]["completion_tokens"]
                )
                
                # Calculate tokens per second
                if result["performance"]["total_tokens"] > 0:
                    result["performance"]["tokens_per_second"] = (
                        result["performance"]["total_tokens"] / result["performance"]["duration_seconds"]
                    )
            else:
                result["error"] = "No output generated"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result

    def process_image(self, 
                     image_input: Union[str, Image.Image], 
                     prompt: str = "Describe this image in detail. What objects do you see?",
                     max_tokens: int = 1024, 
                     resize_for_context: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Process a single image with the configured mode.
        
        Args:
            image_input: Image input (URL, file path, or PIL Image)
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            resize_for_context: Whether to resize image for context limits
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing response and performance metrics
        """
        if self.mode == VLMMode.URL:
            return self.process_image_url_mode(
                image_input, prompt, max_tokens, resize_for_context, **kwargs
            )
        elif self.mode == VLMMode.PACKAGE:
            return self.process_image_package_mode(
                image_input, prompt, max_tokens, resize_for_context, **kwargs
            )
        else:
            return {"error": f"Unsupported mode: {self.mode}", "success": False}

    def process_multiple_images(self, 
                               image_inputs: List[Union[str, Image.Image]], 
                               prompts: Union[str, List[str]] = "Describe this image in detail. What objects do you see?",
                               max_tokens: int = 1024, 
                               resize_for_context: bool = True,
                               **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple images.
        
        Args:
            image_inputs: List of image inputs
            prompts: Single prompt or list of prompts (one per image)
            max_tokens: Maximum tokens to generate per image
            resize_for_context: Whether to resize images for context limits
            **kwargs: Additional parameters
            
        Returns:
            List of dictionaries containing responses and performance metrics
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_inputs)
        elif len(prompts) != len(image_inputs):
            raise ValueError("Number of prompts must match number of images")
        
        results = []
        for image_input, prompt in zip(image_inputs, prompts):
            result = self.process_image(
                image_input, prompt, max_tokens, resize_for_context, **kwargs
            )
            results.append(result)
        
        return results

    def get_performance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance summary from multiple results.
        
        Args:
            results: List of result dictionaries from process_image calls
            
        Returns:
            Dictionary containing performance summary statistics
        """
        successful_results = [r for r in results if r.get('success', False) and 'performance' in r]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        durations = [r['performance']['duration_seconds'] for r in successful_results]
        tokens_per_second = [r['performance'].get('tokens_per_second', 0) for r in successful_results]
        total_tokens = [r['performance']['total_tokens'] for r in successful_results]
        
        summary = {
            "total_images": len(results),
            "successful_images": len(successful_results),
            "failed_images": len(results) - len(successful_results),
            "total_duration": sum(durations),
            "average_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "total_tokens": sum(total_tokens),
            "average_tokens_per_second": statistics.mean([tps for tps in tokens_per_second if tps > 0]),
            "median_tokens_per_second": statistics.median([tps for tps in tokens_per_second if tps > 0])
        }
        
        return summary


# Example usage and testing functions
def example_usage():
    """Example usage of the VLMVLLMUtility class."""
    
    # Example 1: URL Mode
    print("=== URL Mode Example ===")
    try:
        vlm_url = VLMVLLMUtility(mode=VLMMode.URL, api_url="http://localhost:8000")
        
        # Check server status
        status = vlm_url.check_server_status()
        print(f"Server status: {status}")
        
        # Process an image (you'll need to provide a valid image path)
        result = vlm_url.process_image("VisionLangAnnotateModels/sampledata/sjsupeople.jpg", "What do you see in this image?")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"URL mode error: {e}")
    
    # Example 2: Package Mode (requires vLLM package installed)
    print("\n=== Package Mode Example ===")
    try:
        vlm_package = VLMVLLMUtility(
            mode=VLMMode.PACKAGE, 
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        # Process an image (you'll need to provide a valid image path)
        result = vlm_package.process_image("VisionLangAnnotateModels/sampledata/sjsupeople.jpg", "Describe this image.")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Package mode error: {e}")


if __name__ == "__main__":
    example_usage()