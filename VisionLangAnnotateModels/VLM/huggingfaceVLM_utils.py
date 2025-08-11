import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from PIL import Image
import json

# Check if transformers is available
try:
    import transformers
    from transformers import (
        AutoProcessor, AutoModelForCausalLM, AutoTokenizer,
        Blip2Processor, Blip2ForConditionalGeneration,
        LlavaProcessor, LlavaForConditionalGeneration,
        AutoImageProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not found. HuggingFace models will not be available.")

# Constants for model types
MODEL_TYPE_BLIP2 = "blip2"
MODEL_TYPE_LLAVA = "llava"
MODEL_TYPE_SMOLVLM = "smolvlm"
MODEL_TYPE_MINIGPT4 = "minigpt4"
MODEL_TYPE_QWEN = "qwen"
MODEL_TYPE_GLIP = "glip"

class HuggingFaceClient:
    """
    Client for interacting with HuggingFace models for both text-only and vision LLMs.
    
    This class provides interfaces for:
    1. Text-only LLMs (process_text)
    2. Vision LLMs (process_vision)
    
    It handles model loading, inference, and response parsing.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None, stream_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the HuggingFace client.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ("cuda", "cpu", etc.)
            stream_callback: Optional callback function for streaming responses
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required to use HuggingFaceClient")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stream_callback = stream_callback
        
        # Determine model type based on model name
        self.model_type = self._determine_model_type(model_name)
        
        # Initialize model, processor, and tokenizer
        self._initialize_model()
    
    def _determine_model_type(self, model_name: str) -> str:
        """
        Determine the type of model based on the model name.
        
        Args:
            model_name: Name of the HuggingFace model
            
        Returns:
            String indicating the model type
        """
        model_name_lower = model_name.lower()
        
        if "blip" in model_name_lower:
            return MODEL_TYPE_BLIP2
        elif "llava" in model_name_lower:
            return MODEL_TYPE_LLAVA
        elif "smolvlm" in model_name_lower:
            return MODEL_TYPE_SMOLVLM
        elif "minigpt" in model_name_lower:
            return MODEL_TYPE_MINIGPT4
        elif "qwen" in model_name_lower and "vl" in model_name_lower:
            return MODEL_TYPE_QWEN
        elif "glip" in model_name_lower:
            return MODEL_TYPE_GLIP
        else:
            # Default to a text-only model
            return "text"
    
    def _initialize_model(self):
        """
        Initialize the model, processor, and tokenizer based on the model type.
        """
        try:
            # Set default dtype for efficiency
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            if self.model_type == MODEL_TYPE_BLIP2:
                self.processor = Blip2Processor.from_pretrained(self.model_name)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None
                )
                
            elif self.model_type == MODEL_TYPE_LLAVA:
                from transformers import AutoProcessor, LlavaForConditionalGeneration
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                
                # Special handling for LLaVA 1.6 Mistral to avoid Flash Attention 2.0 initialization issue
                if "llava-v1.6-mistral" in self.model_name.lower():
                    self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
                    # First initialize on CPU, then move to GPU to avoid Flash Attention 2.0 issues
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        use_flash_attention_2=True if self.device == "cuda" else False,
                        device_map=None  # Initialize on CPU first
                    )
                    self.model = self.model.to(self.device)  # Then move to GPU
                    self.is_llava_mistral = True
                else:
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    # For other LLaVA models, use standard initialization
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    self.is_llava_mistral = False
                
                self.tokenizer = self.processor.tokenizer
                    
            elif self.model_type == MODEL_TYPE_SMOLVLM:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
                )
                self.model = self.model.to(self.device)
                self.tokenizer = self.processor.tokenizer
                
            elif self.model_type == MODEL_TYPE_QWEN:
                # Import Qwen-specific modules
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    
                    # Initialize Qwen model with flash attention if available
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
                    )
                    self.model = self.model.to(self.device)
                    self.tokenizer = self.processor.tokenizer
                except ImportError:
                    raise ImportError("Please install the latest transformers version for Qwen2.5-VL support: "
                                    "pip install git+https://github.com/huggingface/transformers accelerate")
                except Exception as e:
                    raise Exception(f"Error loading Qwen model: {str(e)}. Make sure you have the latest transformers version.")
                
            elif self.model_type == MODEL_TYPE_MINIGPT4:
                # MiniGPT-4 requires special handling
                raise NotImplementedError("MiniGPT-4 support not yet implemented in HuggingFaceClient")
                
            elif self.model_type == MODEL_TYPE_GLIP:
                # GLIP requires special handling
                raise NotImplementedError("GLIP support not yet implemented in HuggingFaceClient")
                
            else:  # Text-only model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Error initializing model {self.model_name}: {str(e)}")
    
    def process_text(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None,
                     max_tokens: int = 1000,
                     stream: bool = False) -> Dict[str, Any]:
        """
        Process text using a text-only HuggingFace LLM.
        
        Args:
            prompt: The text prompt to send to the model
            system_prompt: Optional system prompt to set context
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
            
        try:
            # Prepare the input based on model type
            if system_prompt:
                # Combine system prompt and user prompt
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Tokenize the input
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                if stream:
                    # Streaming generation
                    generated_text = ""
                    for token in self._stream_generate(inputs, max_tokens):
                        generated_text += token
                        if self.stream_callback:
                            self.stream_callback(token)
                    
                    return {
                        "success": True,
                        "response": generated_text,
                        "raw_response": {"response": generated_text}
                    }
                else:
                    # Non-streaming generation
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the generated part (remove the input prompt)
                    if output_text.startswith(full_prompt):
                        output_text = output_text[len(full_prompt):].strip()
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                    
        except Exception as e:
            # Handle errors
            error_msg = f"Error processing text with HuggingFace model: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _stream_generate(self, inputs, max_tokens):
        """
        Stream tokens from the model one by one.
        
        Args:
            inputs: Tokenized inputs
            max_tokens: Maximum number of tokens to generate
            
        Yields:
            Generated tokens one by one
        """
        # Get the input token length
        input_length = inputs.input_ids.shape[1]
        
        # Generate the first token
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        
        # Get the first token
        next_token_id = outputs.sequences[0, input_length:input_length+1]
        yield self.tokenizer.decode(next_token_id, skip_special_tokens=True)
        
        # Update inputs with the new token
        current_output = outputs.sequences
        
        # Generate remaining tokens one by one
        for _ in range(1, max_tokens):
            outputs = self.model.generate(
                input_ids=current_output,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )
            
            # Get the next token
            next_token_id = outputs.sequences[0, -1:]
            next_token = self.tokenizer.decode(next_token_id, skip_special_tokens=True)
            
            # Check if we've reached the end of generation
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
            yield next_token
            
            # Update current output
            current_output = outputs.sequences
    
    def process_vision(self,
                      images: List[Image.Image],
                      prompts: List[str],
                      combined_prompt: Optional[str] = None,
                      max_tokens: int = 1000,
                      stream: bool = False) -> Dict[str, Any]:
        """
        Process images using a vision-capable HuggingFace LLM.
        
        This method supports both single and multiple images with corresponding prompts.
        For multiple images, it can either process them individually or as a batch with a
        combined prompt.
        
        Args:
            images: List of PIL Image objects
            prompts: List of text prompts corresponding to each image
            combined_prompt: Optional combined prompt for batch processing
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed results or error information
        """
        # Validate inputs
        if not images or len(images) != len(prompts):
            return {
                "success": False,
                "error": "Number of images must match number of prompts"
            }
        
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
        
        try:
            # For single image processing
            if len(images) == 1:
                return self._process_single_vision(images[0], prompts[0], max_tokens, stream)
            
            # For multiple images
            if combined_prompt:
                # Process as batch with combined prompt
                return self._process_batch_vision(images, combined_prompt, max_tokens, stream)
            else:
                # Process each image individually and combine results
                results = []
                for i, (image, prompt) in enumerate(zip(images, prompts)):
                    result = self._process_single_vision(image, prompt, max_tokens, stream)
                    results.append(result)
                
                # Check if all were successful
                all_successful = all(r.get("success", False) for r in results)
                
                if all_successful:
                    return {
                        "success": True,
                        "responses": [r.get("response", "") for r in results],
                        "raw_responses": [r.get("raw_response", {}) for r in results]
                    }
                else:
                    # Return the first error encountered
                    for result in results:
                        if not result.get("success", False):
                            return result
                    
                    # Fallback error if none found but success is still False
                    return {
                        "success": False,
                        "error": "Unknown error in batch processing"
                    }
                
        except Exception as e:
            # Handle other errors
            error_msg = f"Error in vision processing: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_single_vision(self, image: Image.Image, prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Process a single image with a vision-capable HuggingFace model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Process based on model type
            if self.model_type == MODEL_TYPE_BLIP2:
                return self._generate_blip2(image, prompt, max_tokens, stream)
            elif self.model_type == MODEL_TYPE_LLAVA:
                return self._generate_llava(image, prompt, max_tokens, stream)
            elif self.model_type == MODEL_TYPE_SMOLVLM:
                return self._generate_smolvlm(image, prompt, max_tokens, stream)
            elif self.model_type == MODEL_TYPE_QWEN:
                return self._generate_qwen_single(image, prompt, max_tokens, stream)
            elif self.model_type == MODEL_TYPE_MINIGPT4:
                return self._generate_minigpt4(image, prompt, max_tokens, stream)
            elif self.model_type == MODEL_TYPE_GLIP:
                return self._generate_glip(image, prompt)
            else:
                return {
                    "success": False,
                    "error": f"Model type {self.model_type} does not support vision processing"
                }
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error processing vision with HuggingFace model: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_batch_vision(self, images: List[Image.Image], combined_prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Process multiple images as a batch with a combined prompt.
        
        Args:
            images: List of PIL Image objects
            combined_prompt: Combined prompt for all images
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Currently, only Qwen models support efficient batch processing
            if self.model_type == MODEL_TYPE_QWEN:
                return self._generate_qwen(images, combined_prompt, max_tokens, stream)
            else:
                # For other models, process images individually and combine results
                results = []
                for image in images:
                    result = self._process_single_vision(image, combined_prompt, max_tokens, stream)
                    results.append(result)
                
                # Check if all were successful
                all_successful = all(r.get("success", False) for r in results)
                
                if all_successful:
                    # Combine responses
                    combined_response = "\n\n".join([f"Image {i+1}: {r.get('response', '')}" for i, r in enumerate(results)])
                    
                    return {
                        "success": True,
                        "response": combined_response,
                        "raw_responses": [r.get("raw_response", {}) for r in results]
                    }
                else:
                    # Return the first error encountered
                    for result in results:
                        if not result.get("success", False):
                            return result
                    
                    # Fallback error if none found but success is still False
                    return {
                        "success": False,
                        "error": "Unknown error in batch processing"
                    }
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error in batch vision processing: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_blip2(self, image: Image.Image, prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using a BLIP2 model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Prepare inputs
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            
            # Move inputs to the correct device and convert to the right dtype
            for k, v in inputs.items():
                if v.dtype == torch.float:
                    inputs[k] = v.to(self.model.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
                else:
                    inputs[k] = v.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                if stream:
                    # BLIP2 doesn't support true streaming, so we'll simulate it
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Simulate streaming by sending chunks
                    if self.stream_callback:
                        chunk_size = max(1, len(output_text) // 10)  # Divide into ~10 chunks
                        for i in range(0, len(output_text), chunk_size):
                            chunk = output_text[i:i+chunk_size]
                            self.stream_callback(chunk)
                            time.sleep(0.1)  # Small delay to simulate streaming
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                else:
                    # Non-streaming generation
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                    
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating with BLIP2: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_llava(self, image: Image.Image, prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using a LLaVA model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Special handling for LLaVA 1.6 Mistral
            if hasattr(self, 'is_llava_mistral') and self.is_llava_mistral:
                # LLaVA 1.6 Mistral uses a different prompt format
                prompt = f"<image>\n{prompt}"
            
            # Prepare inputs
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            # Move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                if stream:
                    # LLaVA doesn't support true streaming, so we'll simulate it
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "assistant:" in output_text.lower():
                        output_text = output_text.split("assistant:", 1)[1].strip()
                    
                    # Simulate streaming by sending chunks
                    if self.stream_callback:
                        chunk_size = max(1, len(output_text) // 10)  # Divide into ~10 chunks
                        for i in range(0, len(output_text), chunk_size):
                            chunk = output_text[i:i+chunk_size]
                            self.stream_callback(chunk)
                            time.sleep(0.1)  # Small delay to simulate streaming
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                else:
                    # Non-streaming generation
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "assistant:" in output_text.lower():
                        output_text = output_text.split("assistant:", 1)[1].strip()
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                    
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating with LLaVA: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_smolvlm(self, image: Image.Image, prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using a SmolVLM model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Prepare inputs
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            # Move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                if stream:
                    # SmolVLM doesn't support true streaming, so we'll simulate it
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "assistant:" in output_text.lower():
                        output_text = output_text.split("assistant:", 1)[1].strip()
                    
                    # Simulate streaming by sending chunks
                    if self.stream_callback:
                        chunk_size = max(1, len(output_text) // 10)  # Divide into ~10 chunks
                        for i in range(0, len(output_text), chunk_size):
                            chunk = output_text[i:i+chunk_size]
                            self.stream_callback(chunk)
                            time.sleep(0.1)  # Small delay to simulate streaming
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                else:
                    # Non-streaming generation
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "assistant:" in output_text.lower():
                        output_text = output_text.split("assistant:", 1)[1].strip()
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                    
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating with SmolVLM: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_minigpt4(self, image: Image.Image, prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using a MiniGPT-4 model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        # MiniGPT-4 requires special handling
        return {
            "success": False,
            "error": "MiniGPT-4 support not yet implemented in HuggingFaceClient"
        }
    
    def _generate_qwen_single(self, image: Image.Image, prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using a Qwen model for a single image.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Resize image if needed (Qwen models require dimensions to be multiples of 28)
            width, height = image.size
            if width < 28 or height < 28 or width % 28 != 0 or height % 28 != 0:
                # Calculate new dimensions that are multiples of 28
                new_width = max(28, ((width + 27) // 28) * 28)
                new_height = max(28, ((height + 27) // 28) * 28)
                
                # Resize the image using a high-quality resampling method
                try:
                    # For newer PIL versions
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # For older PIL versions
                    image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Prepare inputs
            # Apply chat template to the prompt
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback for older transformers versions
                text = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            
            # Process image and text
            inputs = self.processor(text=text, images=image, return_tensors="pt")
            
            # Move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                if stream:
                    # Qwen doesn't support true streaming, so we'll simulate it
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "<|im_start|>assistant" in output_text:
                        output_text = output_text.split("<|im_start|>assistant", 1)[1].strip()
                        if "<|im_end|>" in output_text:
                            output_text = output_text.split("<|im_end|>", 1)[0].strip()
                    
                    # Simulate streaming by sending chunks
                    if self.stream_callback:
                        chunk_size = max(1, len(output_text) // 10)  # Divide into ~10 chunks
                        for i in range(0, len(output_text), chunk_size):
                            chunk = output_text[i:i+chunk_size]
                            self.stream_callback(chunk)
                            time.sleep(0.1)  # Small delay to simulate streaming
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                else:
                    # Non-streaming generation
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "<|im_start|>assistant" in output_text:
                        output_text = output_text.split("<|im_start|>assistant", 1)[1].strip()
                        if "<|im_end|>" in output_text:
                            output_text = output_text.split("<|im_end|>", 1)[0].strip()
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                    
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating with Qwen: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_qwen(self, images: List[Image.Image], combined_prompt: str, max_tokens: int = 1000, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using a Qwen model for multiple images.
        
        Args:
            images: List of PIL Image objects
            combined_prompt: Combined prompt for all images
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Resize images if needed (Qwen models require dimensions to be multiples of 28)
            processed_images = []
            for image in images:
                width, height = image.size
                if width < 28 or height < 28 or width % 28 != 0 or height % 28 != 0:
                    # Calculate new dimensions that are multiples of 28
                    new_width = max(28, ((width + 27) // 28) * 28)
                    new_height = max(28, ((height + 27) // 28) * 28)
                    
                    # Resize the image using a high-quality resampling method
                    try:
                        # For newer PIL versions
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # For older PIL versions
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                processed_images.append(image)
            
            # Prepare inputs
            # Apply chat template to the prompt
            messages = [
                {"role": "user", "content": combined_prompt}
            ]
            
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback for older transformers versions
                text = f"<|im_start|>user\n{combined_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            
            # Process images and text
            inputs = self.processor(text=text, images=processed_images, return_tensors="pt")
            
            # Move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                if stream:
                    # Qwen doesn't support true streaming, so we'll simulate it
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "<|im_start|>assistant" in output_text:
                        output_text = output_text.split("<|im_start|>assistant", 1)[1].strip()
                        if "<|im_end|>" in output_text:
                            output_text = output_text.split("<|im_end|>", 1)[0].strip()
                    
                    # Simulate streaming by sending chunks
                    if self.stream_callback:
                        chunk_size = max(1, len(output_text) // 10)  # Divide into ~10 chunks
                        for i in range(0, len(output_text), chunk_size):
                            chunk = output_text[i:i+chunk_size]
                            self.stream_callback(chunk)
                            time.sleep(0.1)  # Small delay to simulate streaming
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                else:
                    # Non-streaming generation
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract only the assistant's response
                    if "<|im_start|>assistant" in output_text:
                        output_text = output_text.split("<|im_start|>assistant", 1)[1].strip()
                        if "<|im_end|>" in output_text:
                            output_text = output_text.split("<|im_end|>", 1)[0].strip()
                    
                    return {
                        "success": True,
                        "response": output_text,
                        "raw_response": {"response": output_text}
                    }
                    
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating with Qwen batch: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_glip(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """
        Generate object detection results using a GLIP model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image (contains object categories to detect)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        # GLIP requires special handling
        return {
            "success": False,
            "error": "GLIP support not yet implemented in HuggingFaceClient"
        }


class HuggingFaceVLM:
    """
    HuggingFace Vision-Language Model implementation.
    
    This class provides a unified interface for various HuggingFace vision-language models,
    including BLIP2, LLaVA, SmolVLM, MiniGPT-4, Qwen, and GLIP.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the HuggingFace VLM.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ("cuda", "cpu", etc.)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the client
        self.client = HuggingFaceClient(model_name=model_name, device=self.device)
    
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """
        Generate responses for a list of images and prompts.
        
        Args:
            images: List of PIL Image objects
            prompts: List of text prompts corresponding to each image
            
        Returns:
            List of generated responses
        """
        results = []
        
        # Process each image-prompt pair
        for image, prompt in zip(images, prompts):
            result = self.client.process_vision([image], [prompt])
            
            if result["success"]:
                if "responses" in result:
                    # Multiple responses (should not happen for single image)
                    results.append(result["responses"][0])
                else:
                    # Single response
                    results.append(result["response"])
            else:
                # Error case
                error_msg = result.get("error", "Unknown error")
                results.append(f"Error: {error_msg}")
        
        return results
    
    def get_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            String with the model name
        """
        return f"HuggingFace-{self.model_name}"


class HuggingFaceLLM:
    """
    HuggingFace Language Model implementation.
    
    This class provides a unified interface for HuggingFace text-only language models.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the HuggingFace LLM.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ("cuda", "cpu", etc.)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the client
        self.client = HuggingFaceClient(model_name=model_name, device=self.device)
    
    def generate(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        Generate responses for a list of prompts.
        
        Args:
            prompts: List of text prompts
            system_prompt: Optional system prompt to set context
            
        Returns:
            List of generated responses
        """
        results = []
        
        # Process each prompt
        for prompt in prompts:
            result = self.client.process_text(prompt, system_prompt=system_prompt)
            
            if result["success"]:
                results.append(result["response"])
            else:
                # Error case
                error_msg = result.get("error", "Unknown error")
                results.append(f"Error: {error_msg}")
        
        return results
    
    def get_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            String with the model name
        """
        return f"HuggingFace-{self.model_name}"


def process_with_huggingface_text(prompt: str, 
                                model_name: str,
                                system_prompt: Optional[str] = None,
                                max_tokens: int = 1000,
                                device: Optional[str] = None,
                                stream: bool = False,
                                stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process text using a HuggingFace text-only LLM.
    
    Args:
        prompt: The text prompt to send to the model
        model_name: Name of the HuggingFace model to use
        system_prompt: Optional system prompt to set context
        max_tokens: Maximum number of tokens to generate
        device: Device to run the model on ("cuda", "cpu", etc.)
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed result or error information
    """
    client = HuggingFaceClient(model_name=model_name, device=device, stream_callback=stream_callback)
    return client.process_text(prompt, system_prompt=system_prompt, max_tokens=max_tokens, stream=stream)


def process_with_huggingface_vision(images: List[Image.Image], 
                                  prompts: List[str], 
                                  model_name: str,
                                  combined_prompt: Optional[str] = None,
                                  max_tokens: int = 1000,
                                  device: Optional[str] = None,
                                  stream: bool = False,
                                  stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process images using a HuggingFace vision-capable LLM.
    
    Args:
        images: List of PIL Image objects
        prompts: List of text prompts corresponding to each image
        model_name: Name of the HuggingFace vision model to use
        combined_prompt: Optional combined prompt for batch processing
        max_tokens: Maximum number of tokens to generate
        device: Device to run the model on ("cuda", "cpu", etc.)
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed results or error information
    """
    client = HuggingFaceClient(model_name=model_name, device=device, stream_callback=stream_callback)
    return client.process_vision(images, prompts, combined_prompt=combined_prompt, max_tokens=max_tokens, stream=stream)


if __name__ == "__main__":
    # Example usage
    import os
    from PIL import Image
    
    # Example 1: Text processing
    def text_stream_callback(chunk):
        print(chunk, end="", flush=True)
    
    print("\nText processing example:")
    try:
        text_result = process_with_huggingface_text(
            prompt="Explain quantum computing in simple terms.",
            model_name="google/gemma-2b",  # Use a smaller model for testing
            max_tokens=100,
            stream=False
        )
        
        if text_result["success"]:
            print(f"Response: {text_result['response']}")
        else:
            print(f"Error: {text_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error running text example: {str(e)}")
    
    # Example 2: Vision processing
    # Try to find a sample image in the project
    sample_image_paths = [
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/bus.jpg",
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/064224_000100original.jpg",
        "/home/lkk/Developer/VisionLangAnnotate/static/uploads/test.jpg"
    ]
    
    image_path = None
    for path in sample_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path:
        print(f"\nVision processing example using image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        try:
            vision_result = process_with_huggingface_vision(
                images=[image],
                prompts=["Describe what you see in this image."],
                model_name="Salesforce/blip2-opt-2.7b",  # Use BLIP2 for testing
                max_tokens=100,
                stream=False
            )
            
            if vision_result["success"]:
                print(f"Response: {vision_result['response']}")
            else:
                print(f"Error: {vision_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error running vision example: {str(e)}")
    else:
        print("No sample image found. Skipping vision examples.")