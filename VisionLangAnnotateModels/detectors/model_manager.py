import os
import torch
from modeling_yolohf import register_yolo_architecture, YoloConfig, YoloDetectionModel
from transformers import AutoModelForObjectDetection
from multidatasets import coco_names

class ModelManager:
    """
    Core model management class for handling model initialization,
    loading, and configuration across different architectures.
    """
    
    def __init__(self, model=None, config=None, model_type="yolov8", model_name=None, scale='s', device=None):
        """
        Initialize ModelManager with a model or create a new one.
        
        Args:
            model: Existing detection model or None to create a new one
            config: Model config object or None to create a default one
            model_type: Type of model to use ('yolov8', 'detr', 'rt-detr', 'rt-detrv2', 'vitdet')
            model_name: Specific model name/checkpoint from Hugging Face Hub
            scale: Model scale ('n', 's', 'm', 'l', 'x') if creating a new YOLO model
            device: Device to use (None for auto-detection)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        
        # Create or use provided model
        if model is None:
            if model_name:
                self.model = self._load_from_hub(model_name)
                self.model_type = self._detect_model_type(self.model, model_name)
            else:
                self.model = self._create_model(model_type, config, scale, model_name)
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
        
        # Set class names
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
    
    def _update_model_config_with_class_names(self):
        """Update the model's configuration with COCO class names if needed."""
        if hasattr(self.model, 'config'):
            if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
                self.model.config.id2label = {str(k): v for k, v in self.class_names.items()}
                self.model.config.label2id = {v: str(k) for k, v in self.class_names.items()}
            elif all(isinstance(k, int) for k in self.model.config.id2label.keys()):
                self.model.config.id2label = {str(k): v for k, v in self.model.config.id2label.items()}
                self.model.config.label2id = {v: str(k) for k, v in self.model.config.id2label.items()}
    
    def _create_model(self, model_type, config=None, scale='s', model_name=None):
        """Create a new model based on the specified type."""
        if model_type == 'yolov8':
            if config is None:
                config = YoloConfig(
                    scale=scale,
                    nc=80,
                    ch=3,
                    min_size=640,
                    max_size=640,
                    use_fp16=True if self.device.type == 'cuda' else False
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
                    print("Falling back to local YOLOv8 model creation")
            
            return YoloDetectionModel(cfg=config, device=self.device)
    
    def _load_from_hub(self, model_name):
        """Load a model from Hugging Face Hub."""
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        
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
                return 'yolov8'
            elif 'detr-' in model_name_lower:
                return 'detr'
            elif 'rt-detrv2' in model_name_lower:
                return 'rt-detrv2'
            elif 'rt-detr' in model_name_lower:
                return 'rt-detr'
            elif 'vit-det' in model_name_lower:
                return 'vitdet'
        
        model_class_name = model.__class__.__name__
        if 'Yolo' in model_class_name:
            return 'yolov8'
        elif 'Detr' in model_class_name:
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                if 'rt-detr' in model.config.model_type.lower():
                    return 'rt-detr'
            return 'detr'
        elif 'ViT' in model_class_name or 'Vit' in model_class_name:
            return 'vitdet'
        
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            model_type = model.config.model_type.lower()
            if 'yolo' in model_type:
                return 'yolov8'
            elif 'rt-detrv2' in model_type:
                return 'rt-detrv2'
            elif 'rt-detr' in model_type:
                return 'rt-detr'
            elif 'detr' in model_type:
                return 'detr'
            elif 'vit' in model_type and 'det' in model_type:
                return 'vitdet'
        
        print(f"Could not determine model type, defaulting to 'yolov8'")
        return 'yolov8'
    
    def _create_default_config(self, model_type, scale='s'):
        """Create a default config for the specified model type."""
        if model_type == 'yolov8':
            return YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if self.device.type == 'cuda' else False
            )
        elif model_type in ['detr', 'rt-detr', 'rt-detrv2']:
            from transformers import DetrConfig
            return DetrConfig(num_labels=91)
        elif model_type == 'vitdet':
            from transformers import AutoConfig
            try:
                return AutoConfig.from_pretrained("facebook/vit-det-base")
            except:
                from transformers import DetrConfig
                return DetrConfig(num_labels=91)
        else:
            return YoloConfig(
                scale=scale,
                nc=80,
                ch=3,
                min_size=640,
                max_size=640,
                use_fp16=True if self.device.type == 'cuda' else False
            )
    
    def _create_processor(self):
        """Create an appropriate image processor based on model type."""
        if hasattr(self.model, 'processor') and self.model.processor is not None:
            return self.model.processor
            
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'processor_class'):
            try:
                from transformers import AutoImageProcessor
                return AutoImageProcessor.from_pretrained(self.model.config.processor_class)
            except Exception as e:
                print(f"Failed to load processor from config: {e}")
                
        if self.model_type == 'yolov8':
            from modeling_yolohf import YoloImageProcessor
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
            from transformers import DetrImageProcessor
            return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        elif self.model_type == 'rt-detr':
            from transformers import AutoImageProcessor
            return AutoImageProcessor.from_pretrained("mindee/rt-detr-resnet-50")
        elif self.model_type == 'rt-detrv2':
            from transformers import AutoImageProcessor
            try:
                return AutoImageProcessor.from_pretrained("mindee/rt-detrv2-resnet-50")
            except:
                return AutoImageProcessor.from_pretrained("mindee/rt-detr-resnet-50")
        elif self.model_type == 'vitdet':
            from transformers import AutoImageProcessor
            try:
                return AutoImageProcessor.from_pretrained("facebook/vit-det-base")
            except:
                from transformers import DetrImageProcessor
                return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        else:
            from transformers import DetrImageProcessor
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
                from modeling_yolohf import register_yolo_architecture
                register_yolo_architecture()
                
            from transformers import AutoModelForObjectDetection
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