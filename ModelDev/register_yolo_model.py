import os
import sys
import torch
from transformers import AutoConfig, AutoModel, AutoModelForObjectDetection
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
from transformers import PretrainedConfig
# Add the parent directory to the path to import your module
#install our DeepDataMiningLearning package
#pip install pycocotools pandas scipy
from DeepDataMiningLearning.detection.modeling_yolohf import YoloDetectionModel

from transformers import PretrainedConfig
class YoloConfig(PretrainedConfig):
    """Configuration class for YOLOv8 models."""
    model_type = "yolov8"
    
    def __init__(
        self,
        scale="s",
        nc=80,
        ch=3,
        min_size=640,
        max_size=640,
        use_fp16=False,
        id2label=None,
        label2id=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.nc = nc  # number of classes
        self.ch = ch  # number of channels
        self.min_size = min_size
        self.max_size = max_size
        self.use_fp16 = use_fp16
        
        # Set up id2label and label2id mappings
        if id2label is None:
            id2label = {str(i): f"class_{i}" for i in range(nc)}
        self.id2label = id2label
        
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        self.label2id = label2id
        
        # Add model architecture info
        self.architectures = ["YoloDetectionModel"]

def register_yolo_architecture():
    """
    Register the YOLOv8 model architecture with the Hugging Face transformers library
    for full integration with the transformers ecosystem.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForObjectDetection
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
    
    # Register the config
    CONFIG_MAPPING.register("yolov8", YoloConfig)
    
    # Register the model architecture
    MODEL_MAPPING.register(YoloConfig, YoloDetectionModel)
    MODEL_FOR_OBJECT_DETECTION_MAPPING.register(YoloConfig, YoloDetectionModel)
    
    print("YOLOv8 architecture registered successfully with Hugging Face transformers")
    
def test_model_loading(repo_id):
    """Test loading a YOLOv8 model from Hugging Face Hub."""
    print(f"Testing model loading from {repo_id}...")
    
    # Register the model first
    register_yolo_architecture()
    
    # Try to load the model
    try:
        model = AutoModelForObjectDetection.from_pretrained(repo_id)
        print(f"Successfully loaded model from {repo_id}")
        print(f"Model type: {type(model).__name__}")
        print(f"Model scale: {model.scale}")
        print(f"Number of classes: {model.config.num_classes}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    # Register the model
    register_yolo_architecture()
    
    # Test with a specific model repository
    repo_id = "lkk688/yolov8s-model"
    test_model_loading(repo_id)