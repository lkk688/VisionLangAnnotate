import torch
import time
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

# Try to import vllm, but don't fail if it's not available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM package not available. VLLMBackend will not work.")
    
    # Define placeholder classes for when vLLM is not available
    class SamplingParams:
        def __init__(self, *args, **kwargs):
            pass
    
    class LLM:
        def __init__(self, *args, **kwargs):
            pass

from VisionLangAnnotateModels.VLM.vlm_classifierv3 import VLMBackend

class VLLMClient:
    """
    Client for interacting with vLLM models.
    
    This client handles text-only processing using vLLM for efficient inference.
    It does not support vision models directly, but can be used to process text
    descriptions of images or to standardize outputs from vision models.
    """
    
    def __init__(self, model_name: str, 
                 device: Optional[str] = None,
                 tensor_parallel_size: int = 1,
                 stream_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the vLLM client.
        
        Args:
            model_name: Name of the model to use (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: Device to run the model on ("cuda", "cpu", etc.)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            stream_callback: Optional callback function for streaming responses
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM package is required to use VLLMClient")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stream_callback = stream_callback
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the vLLM model.
        """
        try:
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.8,
                trust_remote_code=True
            )
            print(f"Successfully loaded vLLM model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM model: {str(e)}")
    
    def process_text(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None,
                     max_tokens: int = 1000, 
                     temperature: float = 0.7, 
                     top_k: int = 50, 
                     top_p: float = 0.95,
                     stream: bool = False) -> Dict[str, Any]:
        """
        Process text using the vLLM model.
        
        Args:
            prompt: The user prompt to process
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Combine system and user prompts if system prompt is provided
            if system_prompt:
                # Format depends on the model, this is a common format
                full_prompt = f"<s>[INST] {system_prompt} [/INST]\n\n{prompt}</s>"
            else:
                # Simple format for user-only prompt
                full_prompt = f"<s>[INST] {prompt} [/INST]</s>"
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            if stream:
                # Stream the response
                response_generator = self._stream_generate(full_prompt, sampling_params)
                response_text = ""
                
                for chunk in response_generator:
                    response_text += chunk
                    if self.stream_callback:
                        self.stream_callback(chunk)
                
                return {
                    "success": True,
                    "response": response_text
                }
            else:
                # Generate the full response at once
                outputs = self.model.generate([full_prompt], sampling_params)
                response_text = outputs[0].outputs[0].text
                
                return {
                    "success": True,
                    "response": response_text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _stream_generate(self, prompt: str, sampling_params: SamplingParams):
        """
        Generate tokens one by one and yield them.
        
        Args:
            prompt: The prompt to generate from
            sampling_params: Sampling parameters for generation
            
        Yields:
            Generated text chunks
        """
        # vLLM has built-in streaming support
        outputs_generator = self.model.generate_iterator(
            prompts=[prompt],
            sampling_params=sampling_params
        )
        
        previous_text = ""
        for outputs in outputs_generator:
            output = outputs[0].outputs[0].text
            new_text = output[len(previous_text):]
            previous_text = output
            yield new_text


class VLLMBackend(VLMBackend):
    """
    vLLM-based backend for text processing.
    
    This backend does not support vision models directly, but can be used to process
    text descriptions of images or to standardize outputs from vision models.
    """
    
    def __init__(self, model_name: str, 
                 device: Optional[str] = None,
                 tensor_parallel_size: int = 1):
        """
        Initialize the vLLM backend.
        
        Args:
            model_name: Name of the model to use (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: Device to run the model on ("cuda", "cpu", etc.)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM package is required to use VLLMBackend")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize the client
        self.client = VLLMClient(
            model_name=model_name, 
            device=self.device,
            tensor_parallel_size=tensor_parallel_size
        )
    
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """
        Process text prompts about images.
        
        Note: This method doesn't actually use the images, as vLLM doesn't support
        vision models directly. It's included to conform to the VLMBackend interface.
        
        Args:
            images: List of PIL Image objects (not used)
            prompts: List of text prompts corresponding to each image
            
        Returns:
            List of generated responses
        """
        results = []
        
        # Process each prompt (ignoring images)
        for prompt in prompts:
            result = self.client.process_text(prompt)
            
            if result["success"]:
                results.append(result["response"])
            else:
                # Error case
                error_msg = result.get("error", "Unknown error")
                results.append(f"Error: {error_msg}")
        
        return results
    
    def get_name(self) -> str:
        """
        Get the name of the backend model.
        
        Returns:
            String representation of the model name
        """
        return f"vLLM-{self.model_name.split('/')[-1]}"


def process_with_vllm_text(prompt: str, 
                         vllm_model: str,
                         system_prompt: Optional[str] = None,
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         top_k: int = 50,
                         top_p: float = 0.95,
                         stream: bool = False,
                         stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process text using a vLLM model.
    
    Args:
        prompt: The user prompt to process
        vllm_model: Name of the vLLM model to use
        system_prompt: Optional system prompt to prepend
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed result or error information
    """
    if not VLLM_AVAILABLE:
        return {
            "success": False,
            "error": "vLLM package is not available. Please install it with 'pip install vllm'."
        }
    
    try:
        client = VLLMClient(model_name=vllm_model, stream_callback=stream_callback)
        return client.process_text(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stream=stream
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def text_stream_callback(chunk):
    """
    Simple callback function for streaming text responses.
    
    Args:
        chunk: Text chunk to print
    """
    print(chunk, end="", flush=True)


# Example usage
if __name__ == "__main__":
    # Example of using the vLLM backend
    if VLLM_AVAILABLE:
        # Process text with streaming
        result = process_with_vllm_text(
            prompt="Explain the concept of deep learning in simple terms.",
            vllm_model="meta-llama/Llama-2-7b-chat-hf",
            stream=True,
            stream_callback=text_stream_callback
        )
        
        print("\n\nSuccess:", result["success"])
        
        # Create a backend and use it
        backend = VLLMBackend(model_name="meta-llama/Llama-2-7b-chat-hf")
        responses = backend.generate(
            images=[Image.new("RGB", (100, 100))],  # Dummy image
            prompts=["What is machine learning?"]
        )
        
        print("\nResponse:", responses[0])
    else:
        print("vLLM is not available. Please install it with 'pip install vllm'.")