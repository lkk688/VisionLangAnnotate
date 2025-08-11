# VLM module initialization

# Import VLMBackend abstract class
from .vlm_classifierv3 import VLMBackend, VLMClassifier

# Import backend implementations
try:
    from .huggingfaceVLM_utils import HuggingFaceVLM, HuggingFaceLLM, process_with_huggingface_text, process_with_huggingface_vision
except ImportError:
    print("Warning: HuggingFace VLM utilities could not be imported. HuggingFace backends will not be available.")

try:
    from .ollama_utils import OllamaVLM, process_with_ollama_text, process_with_ollama_vision
except ImportError:
    print("Warning: Ollama utilities could not be imported. Ollama backends will not be available.")

try:
    from .vllm_utils import VLLMBackend, process_with_vllm_text
except ImportError:
    print("Warning: vLLM utilities could not be imported. vLLM backend will not be available.")

# Define available backends
__all__ = [
    # Base classes
    'VLMBackend',
    'VLMClassifier',
    
    # HuggingFace backends
    'HuggingFaceVLM',
    'HuggingFaceLLM',
    'process_with_huggingface_text',
    'process_with_huggingface_vision',
    
    # Ollama backends
    'OllamaVLM',
    'process_with_ollama_text',
    'process_with_ollama_vision',
    
    # vLLM backend
    'VLLMBackend',
    'process_with_vllm_text',
]