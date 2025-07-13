import torch
import os
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from transformers import DetrImageProcessor, DetrConfig
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from transformers import AutoConfig
from VisionLangAnnotateModels.detectors.modeling_yolohf import YoloDetectionModel, YoloConfig, YoloImageProcessor, register_yolo_architecture
from VisionLangAnnotateModels.detectors.multidatasets import coco_names

class BaseMultiModel:
    """
    Base class for MultiModels with core functionality for model creation,
    loading, and configuration management.
    """
    
    def __init__(self, model=None, config=None, model_type="myyolohf", model_name=None, scale='s', device=None):
        """
        Initialize BaseMultiModel with a model or create a new one.
        
        Args:
            model: Existing detection model or None to create a new one
            config: Model config object or None to create a default one
            model_type: Type of model to use ('myyolohf', 'detr', 'rtdetr', 'yolo', 'groundingdino')
            model_name: Specific model name/checkpoint from Hugging Face Hub
            scale: Model scale ('n', 's', 'm', 'l', 'x') if creating a new YOLO model
            device: Device to use (None for auto-detection)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        
        # Initialize special models as None (will be loaded separately)
        self.yolo_model = None
        self.grounding_model = None
        self.grounding_processor = None
        
        # For special model types, we handle them differently
        if self.model_type == 'yolo':
            if model_name:
                # YOLO models will be loaded using load_yolo_model
                self.model = None
                # We'll load the YOLO model after initializing other attributes
            else:
                raise ValueError("model_name must be provided for YOLO models")
        elif self.model_type == 'groundingdino':
            if model_name:
                # GroundingDINO models will be loaded using load_groundingdino_model
                self.model = None
                # We'll load the GroundingDINO model after initializing other attributes
            else:
                # Use default model name for GroundingDINO
                model_name = "IDEA-Research/grounding-dino-base"
        else:
            # Create or use provided model for non-YOLO types
            if model is None:
                if model_name:
                    self.model = self._load_from_hub(model_name)
                    self.model_type = self._detect_model_type(self.model, model_name)
                else: # myyolohf model
                    self.model = self._create_yolohf_model(model_type, config, scale, model_name)
            else:
                self.model = model
                if model_type == "auto":
                    self.model_type = self._detect_model_type(self.model)
                
            if self.model is not None:
                self.model = self.model.to(self.device)
        
        # Store or create config
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        elif config is not None:
            self.config = config
        else:
            self.config = self._create_default_config(model_type, scale)
        
        # Initialize appropriate image processor based on model type
        self.processor = self._create_processor()
        
        # class names
        if self.config and hasattr(self.config, 'id2label'):
            num_labels = len(self.config.id2label)
            self.class_names = self.config.id2label
        else:
            self.class_names = coco_names
            num_labels = len(self.class_names)
        print(f"model has {num_labels} classes")
        
        # Update model config with COCO class names if needed
        self._update_model_config_with_class_names()
        self.dataset = None
        
        # Load special models based on model_type
        if self.model_type == 'yolo' and model_name is not None:
            self.load_yolo_model(model_name)
        elif self.model_type == 'groundingdino' and model_name is not None:
            self.load_groundingdino_model(model_name)
    
    def to(self, device):
        """Move the model to the specified device."""
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def _update_model_config_with_class_names(self):
        """Update the model's configuration with COCO class names if needed."""
        if hasattr(self.model, 'config'):
            if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
                self.model.config.id2label = {str(k): v for k, v in self.class_names.items()}
                self.model.config.label2id = {v: str(k) for k, v in self.class_names.items()}
            elif all(isinstance(k, int) for k in self.model.config.id2label.keys()):
                self.model.config.id2label = {str(k): v for k, v in self.model.config.id2label.items()}
                self.model.config.label2id = {v: str(k) for k, v in self.model.config.id2label.items()}
    
    def _create_yolohf_model(self, model_type, config=None, scale='s', model_name=None):
        """Create a new model based on the specified type."""
        if config is None:
            config = YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if (isinstance(self.device, str) and 'cuda' in self.device) or \
                          (hasattr(self.device, 'type') and self.device.type == 'cuda') else False
            )
            
        if model_name and 'yolo' in model_name.lower():
            register_yolo_architecture()
            try:
                print(f"Loading YOLO model from Hugging Face Hub: {model_name}")
                model = AutoModelForObjectDetection.from_pretrained(model_name)
                if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                    if not model.config.id2label or len(model.config.id2label) == 0:
                        model.config.id2label = {str(k): v for k, v in coco_names.items()}
                        model.config.label2id = {v: str(k) for k, v in coco_names.items()}
                return model
            except Exception as e:
                print(f"Error loading YOLO model from HF Hub: {e}")
        
        return YoloDetectionModel(cfg=config, device=self.device)
    
    def _load_from_hub(self, model_name):
        """Load a model from Hugging Face Hub."""
        
        try:
            if 'yolo' in model_name.lower():
                register_yolo_architecture()
                
            print(f"Loading model from Hugging Face Hub: {model_name}")
            if 'rt-detr' in model_name.lower():
                model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
            else:
                model = AutoModelForObjectDetection.from_pretrained(model_name)
            
            if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                if not model.config.id2label or len(model.config.id2label) == 0:
                    model.config.id2label = {str(k): v for k, v in coco_names.items()}
                    model.config.label2id = {v: str(k) for k, v in coco_names.items()}
            
            return model
        except Exception as e:
            print(f"Error loading model from hub: {e}")
    
    def _detect_model_type(self, model, model_name=None):
        """Detect the type of model based on its architecture or name."""
        if model_name:
            model_name_lower = model_name.lower()
            if 'yolo' in model_name_lower:
                return 'yolo'
            elif 'detr-' in model_name_lower:
                return 'detr'
            elif 'rtdetr' in model_name_lower:
                return 'rt-detr'
            elif 'vit-det' in model_name_lower:
                return 'vitdet'
            elif 'grounding-dino' in model_name_lower or 'groundingdino' in model_name_lower:
                return 'groundingdino'
        
        model_class_name = model.__class__.__name__
        if 'Yolo' in model_class_name:
            return 'yolo'
        elif 'Detr' in model_class_name:
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                if 'rt-detr' in model.config.model_type.lower():
                    return 'rt-detr'
            return 'detr'
        elif 'ViT' in model_class_name or 'Vit' in model_class_name:
            return 'vitdet'
        elif 'GroundingDino' in model_class_name or 'ZeroShotObjectDetection' in model_class_name:
            return 'groundingdino'
        
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            model_type = model.config.model_type.lower()
            if 'yolo' in model_type:
                return 'yolo'
            elif 'rtdetr' in model_type:
                return 'rt-detr'
            elif 'detr' in model_type:
                return 'detr'
            elif 'vit' in model_type and 'det' in model_type:
                return 'vitdet'
            elif 'groundingdino' in model_type or 'grounding_dino' in model_type:
                return 'groundingdino'
        
        print(f"Could not determine model type, defaulting to 'yolo'")
        return 'yolo'
        
    # Caches for models to avoid reloading
    _yolo_model_cache = {}
    _grounding_model_cache = {}
    _grounding_processor_cache = {}
    
    def load_yolo_model(self, model_name):
        """
        Load a YOLO model from Ultralytics.
        
        Args:
            model_name: Path to YOLO model or model name, or Hugging Face repo ID
            
        Returns:
            Loaded YOLO model
        """
        # Check if model is already in cache
        if model_name in self._yolo_model_cache:
            print(f"Using cached YOLO model: {model_name}")
            self.yolo_model = self._yolo_model_cache[model_name]
            return self.yolo_model
        
        # Check if this is a Hugging Face repo ID
        # if '/' in model_name and not os.path.exists(model_name):
        #     try:
        #         print(f"Loading YOLO model from Hugging Face Hub: {model_name}")
        #         from huggingface_hub import hf_hub_download
        #         from ultralytics import YOLO
                
        #         # Default to model.pt if no specific filename is provided
        #         filename = "model.pt"
        #         if ':' in model_name:
        #             # Format: repo_id:filename
        #             repo_id, filename = model_name.split(':', 1)
        #         else:
        #             repo_id = model_name
                
        #         model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        #         self.yolo_model = YOLO(model_path)
        #         print(f"Loaded YOLO model from Hugging Face: {repo_id} ({filename})")
                
        #         # Cache the model
        #         self._yolo_model_cache[model_name] = self.yolo_model
        #         print(f"Cached YOLO model: {model_name}")
                
        #         return self.yolo_model
        #     except Exception as e:
        #         print(f"Error loading YOLO model from Hugging Face: {e}")
        #         # Fall back to regular loading if HF loading fails
        
        # Regular loading from local path or model name
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_name)
            print(f"Loaded YOLO model: {model_name}")
            
            # Cache the model
            self._yolo_model_cache[model_name] = self.yolo_model
            return self.yolo_model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    
    def load_groundingdino_model(self, model_name=None):
        """
        Load GroundingDINO model and processor.
        
        Args:
            model_name: Name of the GroundingDINO model to load, defaults to 'IDEA-Research/grounding-dino-base'
            
        Returns:
            Tuple of (processor, model)
        """
        if model_name is None:
            model_name = "IDEA-Research/grounding-dino-base"
        
        # Check if model and processor are already in cache
        if model_name in self._grounding_model_cache and model_name in self._grounding_processor_cache:
            print(f"Using cached GroundingDINO model: {model_name}")
            self.grounding_processor = self._grounding_processor_cache[model_name]
            self.grounding_model = self._grounding_model_cache[model_name]
            return self.grounding_processor, self.grounding_model
            
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.grounding_processor = AutoProcessor.from_pretrained(model_name)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
            
            # Cache the model and processor
            self._grounding_processor_cache[model_name] = self.grounding_processor
            self._grounding_model_cache[model_name] = self.grounding_model
            print(f"Loaded and cached GroundingDINO model: {model_name}")
            
            return self.grounding_processor, self.grounding_model
        except Exception as e:
            print(f"Error loading GroundingDINO model: {e}")
            return None, None
    
    def _create_default_config(self, model_type, scale='s'):
        """Create a default config for the specified model type."""
        if model_type == 'yolov8':
            return YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if (isinstance(self.device, str) and 'cuda' in self.device) or \
                          (hasattr(self.device, 'type') and self.device.type == 'cuda') else False
            )
        elif model_type in ['detr', 'rtdetr']:
            return DetrConfig(num_labels=91)
        elif model_type == 'vitdet':
            try:
                return AutoConfig.from_pretrained("facebook/vit-det-base")
            except:
                return DetrConfig(num_labels=91)
        else:
            return YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if (isinstance(self.device, str) and 'cuda' in self.device) or \
                          (hasattr(self.device, 'type') and self.device.type == 'cuda') else False
            )
    
    def _create_processor(self):
        """Create an appropriate image processor based on model type."""
        if hasattr(self.model, 'processor') and self.model.processor is not None:
            return self.model.processor
            
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'processor_class'):
            try:
                return AutoImageProcessor.from_pretrained(self.model.config.processor_class)
            except Exception as e:
                print(f"Failed to load processor from config: {e}")
                
        if self.model_type == 'yolov8':
            return YoloImageProcessor(
                do_resize=True,
                size=640,
                do_normalize=False,
                do_rescale=True,
                rescale_factor=1/255.0,
                do_pad=True,
                pad_size_divisor=32,
                pad_value=114,
                do_convert_rgb=True,
                letterbox=True,
                auto=False,
                stride=32
            )
        elif self.model_type == 'detr':
            return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        elif self.model_type == 'rt-detr':
            return AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        elif self.model_type == 'rt-detrv2':
            return AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
        elif self.model_type == 'vitdet':
            try:
                return AutoImageProcessor.from_pretrained("facebook/vit-det-base")
            except:
                return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        else:
            return DetrImageProcessor(
                do_resize=True,
                size={"height": 640, "width": 640},
                do_normalize=True,
                do_rescale=True
            )
    
    def change_model(self, model_type=None, model_name=None, config=None, scale='s'):
        """Change the current model to a different type or specific model."""
        try:
            if model_name:
                self.model = self._load_from_hub(model_name)
                self.model_type = self._detect_model_type(self.model, model_name)
            elif model_type:
                self.model_type = model_type.lower()
                self.model = self._create_model(self.model_type, config, scale)
            else:
                return False
            
            self.model = self.model.to(self.device)
            
            if hasattr(self.model, 'config'):
                self.config = self.model.config
            elif config is not None:
                self.config = config
            else:
                self.config = self._create_default_config(self.model_type, scale)
            
            self.processor = self._create_processor()
            return True
        except Exception as e:
            print(f"Error changing model: {e}")
            return False
            
    def load_weights(self, weights_path):
        """Load weights from a file."""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Strict loading failed: {e}")
                print("Attempting to load with strict=False...")
                self.model.load_state_dict(state_dict, strict=False)
                
            print(f"Loaded weights from {weights_path}")
            self.processor = self._create_processor()
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
            
    def load_pretrained(self, repo_id):
        """Load a pretrained model from Hugging Face Hub."""
        try:
            if 'yolo' in repo_id.lower():
                register_yolo_architecture()
                
            print(f"Loading model from Hugging Face Hub: {repo_id}")
            self.model = AutoModelForObjectDetection.from_pretrained(repo_id)
            self.model = self.model.to(self.device)
            
            self.model_type = self._detect_model_type(self.model, repo_id)
            self._update_model_config_with_class_names()
            
            if hasattr(self.model, 'config'):
                self.config = self.model.config
                
            self.processor = self._create_processor()
            print(f"Loaded pretrained model from {repo_id}")
            return True
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            return False

if __name__ == "__main__":
    #huggingface-cli login
    base_model = BaseMultiModel(model_type='rt-detr', model_name="PekingU/rtdetr_v2_r18vd", device='cuda')
    print(base_model.model_type)
    print(base_model.model)
    print(base_model.class_names)