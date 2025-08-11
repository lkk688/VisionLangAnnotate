import os
import json
import base64
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

# Default API URLs
DEFAULT_LLAMACPP_API_URL = "http://localhost:8000/v1/completions"
DEFAULT_LLAMACPP_CHAT_API_URL = "http://localhost:8000/v1/chat/completions"

def extract_llamacpp_response_text(response: Dict[str, Any]) -> str:
    """
    Extract text from a llama-cpp-python API response.
    
    Args:
        response: The API response from llama-cpp-python server.
        
    Returns:
        The extracted text from the response.
    """
    try:
        if "choices" in response and len(response["choices"]) > 0:
            # Handle chat API format
            if "message" in response["choices"][0]:
                return response["choices"][0]["message"]["content"]
            # Handle completions API format
            elif "text" in response["choices"][0]:
                return response["choices"][0]["text"]
    except Exception as e:
        print(f"Error extracting text from response: {e}")
    
    return ""

class LlamaCppClient:
    """
    Client for interacting with llama-cpp-python server.
    """
    
    def __init__(
        self,
        api_url: str = DEFAULT_LLAMACPP_API_URL,
        chat_api_url: str = DEFAULT_LLAMACPP_CHAT_API_URL,
        timeout: int = 120,
        stream_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the LlamaCppClient.
        
        Args:
            api_url: URL for the llama-cpp-python completions API.
            chat_api_url: URL for the llama-cpp-python chat API.
            timeout: Timeout in seconds for API requests.
            stream_callback: Optional callback function for streaming responses.
        """
        self.api_url = api_url
        self.chat_api_url = chat_api_url
        self.timeout = timeout
        self.stream_callback = stream_callback
    
    def process_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        json_format: bool = False,
        **kwargs
    ) -> str:
        """
        Process text using llama-cpp-python server.
        
        Args:
            prompt: The text prompt to process.
            model: The model to use for processing.
            system_prompt: Optional system prompt for chat models.
            temperature: Temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            json_format: Whether to format the response as JSON.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text response.
        """
        # Try chat API first
        try:
            return self._process_text_chat_api(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                json_format=json_format,
                **kwargs
            )
        except Exception as e:
            print(f"Chat API failed: {e}. Falling back to completions API.")
            
            # Fall back to completions API
            try:
                return self._process_text_completions_api(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    **kwargs
                )
            except Exception as e:
                print(f"Completions API failed: {e}")
                return ""
    
    def _process_text_chat_api(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        json_format: bool = False,
        **kwargs
    ) -> str:
        """
        Process text using the llama-cpp-python chat API.
        
        Args:
            prompt: The text prompt to process.
            model: The model to use for processing.
            system_prompt: Optional system prompt.
            temperature: Temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            json_format: Whether to format the response as JSON.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text response.
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add response format for JSON if requested
        if json_format:
            payload["response_format"] = {"type": "json_object"}
        
        # Add any additional parameters
        payload.update(kwargs)
        
        if stream:
            return self._handle_streaming_request(self.chat_api_url, payload)
        else:
            # Make the API request
            response = requests.post(
                self.chat_api_url,
                json=payload,
                timeout=self.timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                return extract_llamacpp_response_text(response.json())
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def _process_text_completions_api(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Process text using the llama-cpp-python completions API.
        
        Args:
            prompt: The text prompt to process.
            model: The model to use for processing.
            temperature: Temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text response.
        """
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        if stream:
            return self._handle_streaming_request(self.api_url, payload)
        else:
            # Make the API request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                return extract_llamacpp_response_text(response.json())
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def _handle_streaming_request(self, api_url: str, payload: Dict[str, Any]) -> str:
        """
        Handle streaming API requests.
        
        Args:
            api_url: The API URL to use.
            payload: The request payload.
            
        Returns:
            The concatenated response text.
        """
        if not self.stream_callback:
            raise ValueError("Stream callback must be provided for streaming requests")
        
        full_response = ""
        
        with requests.post(api_url, json=payload, stream=True, timeout=self.timeout) as response:
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    # Skip the data: prefix if present (SSE format)
                    if line_text.startswith('data: '):
                        line_text = line_text[6:]
                    
                    # Skip empty lines or [DONE] marker
                    if line_text.strip() and line_text != '[DONE]':
                        try:
                            chunk = json.loads(line_text)
                            
                            # Extract text from the chunk based on API format
                            chunk_text = ""
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                
                                # Chat API format
                                if "delta" in choice and "content" in choice["delta"]:
                                    chunk_text = choice["delta"]["content"]
                                # Completions API format
                                elif "text" in choice:
                                    chunk_text = choice["text"]
                            
                            if chunk_text:
                                full_response += chunk_text
                                self.stream_callback(chunk_text)
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            pass
        
        return full_response
    
    def process_vision(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        combined_prompt: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Process images using a vision-capable model.
        
        Args:
            images: Path(s) to image file(s) or PIL Image object(s).
            prompt: The text prompt to process with each image.
            model: The vision model to use.
            system_prompt: Optional system prompt.
            temperature: Temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            combined_prompt: Optional prompt to use when processing multiple images together.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text response(s).
        """
        # Handle single image case
        if isinstance(images, (str, Image.Image)):
            return self._process_single_vision(images, prompt, model, system_prompt, temperature, max_tokens, stream, **kwargs)
        
        # Handle multiple images case
        if combined_prompt:
            # Process all images together with a combined prompt
            return self._process_batch_vision(images, combined_prompt, model, system_prompt, temperature, max_tokens, stream, **kwargs)
        else:
            # Process each image individually
            results = []
            for image in images:
                result = self._process_single_vision(image, prompt, model, system_prompt, temperature, max_tokens, stream, **kwargs)
                results.append(result)
            return results
    
    def _process_single_vision(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Process a single image using a vision-capable model.
        
        Args:
            image: Path to image file or PIL Image object.
            prompt: The text prompt to process with the image.
            model: The vision model to use.
            system_prompt: Optional system prompt.
            temperature: Temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text response.
        """
        # Load and prepare the image
        if isinstance(image, str):
            # Load image from file path
            img = Image.open(image)
        else:
            # Use the provided PIL Image object
            img = image
        
        # Convert image to base64
        base64_image = self._prepare_image_for_vision(img)
        
        # Prepare messages for the chat API
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message with image and prompt
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        })
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            if stream:
                return self._handle_streaming_request(self.chat_api_url, payload)
            else:
                # Make the API request
                response = requests.post(
                    self.chat_api_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check if the request was successful
                if response.status_code == 200:
                    return extract_llamacpp_response_text(response.json())
                else:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Vision processing failed: {e}")
            return ""
    
    def _process_batch_vision(
        self,
        images: List[Union[str, Image.Image]],
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Process multiple images together using a vision-capable model.
        
        Args:
            images: List of paths to image files or PIL Image objects.
            prompt: The text prompt to process with the images.
            model: The vision model to use.
            system_prompt: Optional system prompt.
            temperature: Temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text response.
        """
        # Prepare messages for the chat API
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Prepare content with all images and the prompt
        content = []
        
        # Add each image to the content
        for image in images:
            # Load and prepare the image
            if isinstance(image, str):
                # Load image from file path
                img = Image.open(image)
            else:
                # Use the provided PIL Image object
                img = image
            
            # Convert image to base64
            base64_image = self._prepare_image_for_vision(img)
            
            # Add image to content
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
        # Add the prompt text
        content.append({"type": "text", "text": prompt})
        
        # Add user message with all images and prompt
        messages.append({"role": "user", "content": content})
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            if stream:
                return self._handle_streaming_request(self.chat_api_url, payload)
            else:
                # Make the API request
                response = requests.post(
                    self.chat_api_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check if the request was successful
                if response.status_code == 200:
                    return extract_llamacpp_response_text(response.json())
                else:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Batch vision processing failed: {e}")
            return ""
    
    def _prepare_image_for_vision(self, image: Image.Image) -> str:
        """
        Prepare an image for vision processing.
        
        Args:
            image: The PIL Image object to prepare.
            
        Returns:
            Base64 encoded image data.
        """
        # Resize image to be a multiple of 28 (common requirement for vision models)
        width, height = image.size
        new_width = (width // 28) * 28
        new_height = (height // 28) * 28
        
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height))
        
        # Convert image to RGB if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Utility functions for easier usage

def process_with_llamacpp_text(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None,
    api_url: str = DEFAULT_LLAMACPP_API_URL,
    chat_api_url: str = DEFAULT_LLAMACPP_CHAT_API_URL,
    timeout: int = 120,
    **kwargs
) -> str:
    """
    Process text with llama-cpp-python server.
    
    Args:
        prompt: The text prompt to process.
        model: The model to use for processing.
        system_prompt: Optional system prompt.
        temperature: Temperature for text generation.
        max_tokens: Maximum number of tokens to generate.
        stream: Whether to stream the response.
        stream_callback: Optional callback function for streaming responses.
        api_url: URL for the llama-cpp-python completions API.
        chat_api_url: URL for the llama-cpp-python chat API.
        timeout: Timeout in seconds for API requests.
        **kwargs: Additional parameters to pass to the API.
        
    Returns:
        The generated text response.
    """
    client = LlamaCppClient(
        api_url=api_url,
        chat_api_url=chat_api_url,
        timeout=timeout,
        stream_callback=stream_callback
    )
    
    return client.process_text(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs
    )

def process_with_llamacpp_vision(
    images: Union[str, List[str], Image.Image, List[Image.Image]],
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None,
    combined_prompt: Optional[str] = None,
    api_url: str = DEFAULT_LLAMACPP_API_URL,
    chat_api_url: str = DEFAULT_LLAMACPP_CHAT_API_URL,
    timeout: int = 120,
    **kwargs
) -> Union[str, List[str]]:
    """
    Process images with a vision-capable model using llama-cpp-python server.
    
    Args:
        images: Path(s) to image file(s) or PIL Image object(s).
        prompt: The text prompt to process with each image.
        model: The vision model to use.
        system_prompt: Optional system prompt.
        temperature: Temperature for text generation.
        max_tokens: Maximum number of tokens to generate.
        stream: Whether to stream the response.
        stream_callback: Optional callback function for streaming responses.
        combined_prompt: Optional prompt to use when processing multiple images together.
        api_url: URL for the llama-cpp-python completions API.
        chat_api_url: URL for the llama-cpp-python chat API.
        timeout: Timeout in seconds for API requests.
        **kwargs: Additional parameters to pass to the API.
        
    Returns:
        The generated text response(s).
    """
    client = LlamaCppClient(
        api_url=api_url,
        chat_api_url=chat_api_url,
        timeout=timeout,
        stream_callback=stream_callback
    )
    
    return client.process_vision(
        images=images,
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        combined_prompt=combined_prompt,
        **kwargs
    )

def process_with_llamacpp(
    vlm_descriptions: List[Dict[str, Any]],
    allowed_classes: List[str],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    api_url: str = DEFAULT_LLAMACPP_API_URL,
    chat_api_url: str = DEFAULT_LLAMACPP_CHAT_API_URL,
    timeout: int = 120,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process VLM descriptions with llama-cpp-python server.
    
    Args:
        vlm_descriptions: List of VLM descriptions to process.
        allowed_classes: List of allowed classes for classification.
        model: The model to use for processing.
        system_prompt: Optional system prompt.
        temperature: Temperature for text generation.
        max_tokens: Maximum number of tokens to generate.
        api_url: URL for the llama-cpp-python completions API.
        chat_api_url: URL for the llama-cpp-python chat API.
        timeout: Timeout in seconds for API requests.
        **kwargs: Additional parameters to pass to the API.
        
    Returns:
        List of processed results with class, confidence, and reasoning.
    """
    client = LlamaCppClient(
        api_url=api_url,
        chat_api_url=chat_api_url,
        timeout=timeout
    )
    
    results = []
    
    # Group descriptions by image_id
    grouped_descriptions = {}
    for desc in vlm_descriptions:
        image_id = desc.get("image_id")
        if image_id not in grouped_descriptions:
            grouped_descriptions[image_id] = []
        grouped_descriptions[image_id].append(desc)
    
    # Process each group of descriptions
    for image_id, descriptions in grouped_descriptions.items():
        # Prepare the prompt
        prompt = f"""Classify the following objects into one of these categories: {', '.join(allowed_classes)}.
        
        For each object, provide:
        1. The most appropriate class from the allowed categories
        2. A confidence score between 0 and 1
        3. A brief reasoning for your classification
        
        Objects to classify:
        """
        
        for i, desc in enumerate(descriptions):
            prompt += f"\n{i+1}. {desc.get('description', '')}\n"
        
        prompt += "\nRespond in JSON format as a list of objects with 'class', 'confidence', and 'reasoning' fields."
        
        # Process the prompt
        response = client.process_text(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_format=True,
            **kwargs
        )
        
        # Parse the response
        try:
            parsed_response = json.loads(response)
            
            # Ensure the response is a list
            if not isinstance(parsed_response, list):
                if isinstance(parsed_response, dict) and "results" in parsed_response:
                    parsed_response = parsed_response["results"]
                else:
                    parsed_response = []
            
            # Process each result
            for i, result in enumerate(parsed_response):
                if i < len(descriptions):
                    # Get the original description
                    desc = descriptions[i]
                    
                    # Extract class, confidence, and reasoning
                    class_name = result.get("class", "")
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", "")
                    
                    # Ensure class is in allowed classes
                    if class_name not in allowed_classes:
                        class_name = allowed_classes[0] if allowed_classes else ""
                    
                    # Ensure confidence is between 0 and 1
                    confidence = max(0.0, min(1.0, float(confidence)))
                    
                    # Create the result
                    result_item = {
                        "image_id": image_id,
                        "object_id": desc.get("object_id", i),
                        "class": class_name,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "description": desc.get("description", "")
                    }
                    
                    results.append(result_item)
        except json.JSONDecodeError:
            # Handle JSON parsing error
            print(f"Failed to parse JSON response for image {image_id}")
            
            # Create default results
            for i, desc in enumerate(descriptions):
                result_item = {
                    "image_id": image_id,
                    "object_id": desc.get("object_id", i),
                    "class": allowed_classes[0] if allowed_classes else "",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse response",
                    "description": desc.get("description", "")
                }
                
                results.append(result_item)
    
    return results

# Example usage
def py_restapi_vlmodel():
    """
    Example of using the REST API for vision models.
    """
    # Initialize the client
    client = LlamaCppClient()
    
    # Process a single image
    image_path = "path/to/image.jpg"
    prompt = "Describe this image in detail."
    model = "llava-1.5-7b-q4"
    
    response = client.process_vision(
        images=image_path,
        prompt=prompt,
        model=model
    )
    
    print(response)

def py_chatapi_vlmodel():
    """
    Example of using the Chat API for vision models.
    """
    # Initialize the client
    client = LlamaCppClient()
    
    # Process multiple images
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    prompt = "Compare these images and describe their differences."
    model = "llava-1.5-13b-q4"
    
    response = client.process_vision(
        images=image_paths,
        prompt=prompt,
        model=model,
        combined_prompt=prompt  # Process all images together
    )
    
    print(response)

# Main execution
if __name__ == "__main__":
    # Example of using the client for text processing
    client = LlamaCppClient()
    
    # Process text
    text_response = client.process_text(
        prompt="What is the capital of France?",
        model="llama-3-8b-q4",
        system_prompt="You are a helpful assistant."
    )
    
    print("Text Response:")
    print(text_response)
    
    # Process a single image
    image_path = "path/to/image.jpg"
    vision_response = client.process_vision(
        images=image_path,
        prompt="Describe this image in detail.",
        model="llava-1.5-7b-q4"
    )
    
    print("\nVision Response (Single Image):")
    print(vision_response)
    
    # Process multiple images
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    batch_vision_response = client.process_vision(
        images=image_paths,
        prompt="Compare these images and describe their differences.",
        model="llava-1.5-13b-q4",
        combined_prompt="Compare these images and describe their differences."  # Process all images together
    )
    
    print("\nVision Response (Multiple Images):")
    print(batch_vision_response)
    
    # Example of streaming text
    def print_stream(text):
        print(text, end="")
    
    streaming_client = LlamaCppClient(stream_callback=print_stream)
    
    print("\nStreaming Text Response:")
    streaming_client.process_text(
        prompt="Tell me a short story about a robot.",
        model="llama-3-8b-q4",
        stream=True
    )
    
    # Example of streaming vision
    print("\nStreaming Vision Response:")
    streaming_client.process_vision(
        images=image_path,
        prompt="Describe this image in detail.",
        model="llava-1.5-7b-q4",
        stream=True
    )