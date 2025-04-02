import os
import contextlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
import re
from typing import Dict, List, Optional, Tuple, Union
import time
import torchvision
import os
import json
import glob
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Import necessary modules from your project
from DeepDataMiningLearning.detection.modules.block import (
    AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
    Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d,
    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, MP, SPPCSPC
)
from DeepDataMiningLearning.detection.modules.head import IDetect, Classify, Pose, RTDETRDecoder, Segment, Detect
#from DeepDataMiningLearning.detection.modules.utils import LOGGER
from DeepDataMiningLearning.detection.modules.anchor import check_anchor_order

# Define blocks that take two arguments
twoargs_blocks = [
    nn.Conv2d, Conv, ConvTranspose, GhostConv, RepConv, Bottleneck, GhostBottleneck, 
    SPP, SPPF, SPPCSPC, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, 
    nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3
]#Classify, 

MODEL_TYPE = "yolov8" #"detr"

coco_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

"""
YOLOv8 model configuration as a Python dictionary.
This module replaces the need to load the YAML file at runtime.
"""

# YOLOv8 configuration dictionary
YOLOV8_CONFIG = {
    # Parameters
    "nc": 80,  # number of classes
    "scales": {
        # [depth, width, max_channels]
        "n": [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
        "s": [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
        "m": [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
        "l": [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
        "x": [1.00, 1.25, 512],   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
    },

    # YOLOv8.0n backbone
    "backbone": [
        # [from, repeats, module, args]
        [-1, 1, "Conv", [64, 3, 2]],  # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
        [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
        [-1, 6, "C2f", [256, True]],
        [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
        [-1, 6, "C2f", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],  # 7-P5/32
        [-1, 3, "C2f", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],  # 9
    ],

    # YOLOv8.0n head
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
        [-1, 3, "C2f", [512]],  # 12

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
        [-1, 3, "C2f", [256]],  # 15 (P3/8-small)

        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],  # cat head P4
        [-1, 3, "C2f", [512]],  # 18 (P4/16-medium)

        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 9], 1, "Concat", [1]],  # cat head P5
        [-1, 3, "C2f", [1024]],  # 21 (P5/32-large)

        [[15, 18, 21], 1, "Detect", ["nc"]],  # Detect(P3, P4, P5)
    ],
    
    # Additional parameters
    "inplace": True,
    "ch": 3
}

def get_yolo_config(scale='s', nc=80, ch=3):
    """
    Get a copy of the YOLO configuration with the specified scale, number of classes, and channels.
    
    Args:
        scale (str): Model scale - 'n', 's', 'm', 'l', or 'x'
        nc (int): Number of classes
        ch (int): Number of input channels
        
    Returns:
        dict: YOLO configuration dictionary
    """
    # Create a deep copy to avoid modifying the original
    import copy
    config = copy.deepcopy(YOLOV8_CONFIG)
    
    # Update parameters
    config['scale'] = scale
    config['nc'] = nc
    config['ch'] = ch
    
    return config


# Utility functions
def yaml_load(file='data.yaml', append_filename=True):
    """Load YAML data from a file."""
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data['yaml_file'] = str(file)
        return data

def extract_filename(path):
    """Extract filename from path without extension."""
    return Path(path).stem

def make_divisible(x, divisor):
    """Return nearest x divisible by divisor."""
    return int(np.ceil(x / divisor) * divisor)

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

# def intersect_dicts(da, db, exclude=()):
#     """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
#     return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            print(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    elif all(key in d for key in ('depth_multiple', 'width_multiple')):
        depth = d['depth_multiple']
        width = d['width_multiple']
    
    if "anchors" in d.keys():
        anchors = d['anchors']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)
    else:
        no = nc
   
    if act:
        Conv.default_act = eval(act)
        # if verbose:
        #     LOGGER.info(f"activation: {act}")

    # if verbose:
    #     LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in twoargs_blocks:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (IDetect, Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        setattr(m_, 'np', sum(x.numel() for x in m_.parameters()))
        # Store layer info as custom attributes
        setattr(m_, 'i', i)  # Set layer index using setattr
        setattr(m_, 'f', f)
        #m_.f = f  # From layer indices
        setattr(m_, 'type', t)  # Store layer type name using setattr
        # if verbose:
        #     LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

from transformers import PretrainedConfig
class YoloConfig(PretrainedConfig):
    """Configuration class for YOLOv8 models."""
    model_type = MODEL_TYPE
    
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
    
    This function registers:
    1. The YoloConfig configuration class
    2. The YoloDetectionModel model class
    3. The YoloImageProcessor processor class with letterbox support
    """
    from transformers import AutoConfig, AutoModel, AutoModelForObjectDetection
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
    
    # Register the config
    CONFIG_MAPPING.register(MODEL_TYPE, YoloConfig)
    
    # Register the model architecture
    MODEL_MAPPING.register(YoloConfig, YoloDetectionModel)
    MODEL_FOR_OBJECT_DETECTION_MAPPING.register(YoloConfig, YoloDetectionModel)
    
    # Try to register with AutoModelForObjectDetection
    try:
        AutoModelForObjectDetection._model_mapping[YoloConfig] = YoloDetectionModel
        print("Registered YoloDetectionModel with AutoModelForObjectDetection")
    except (AttributeError, ImportError) as e:
        print(f"Could not register with AutoModelForObjectDetection: {e}")
    
    # Try different methods to register the image processor
    try:
        # Method 1: Try to use PROCESSOR_MAPPING
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
        PROCESSOR_MAPPING.register(YoloConfig, YoloImageProcessor)
        print("Registered YoloImageProcessor with PROCESSOR_MAPPING")
    except (ImportError, AttributeError) as e:
        print(f"Could not register with PROCESSOR_MAPPING: {e}")
    
    print("YOLOv8 architecture registration completed")
    
from transformers import ImageProcessingMixin
import numpy as np
from PIL import Image
import torch

class YoloImageProcessor(ImageProcessingMixin):
    """
    Image processor for YOLO models with letterbox preprocessing.
    
    This processor handles image resizing with letterboxing, normalization, and formatting for YOLO models,
    ensuring compatibility with the Hugging Face transformers ecosystem.
    """
    
    model_input_names = ["pixel_values"]
    
    def __init__(
        self,
        do_resize=True,
        size=640,
        resample="bilinear",
        do_normalize=True,
        do_rescale=True,
        rescale_factor=1/255.0,
        do_pad=True,
        pad_size_divisor=32,
        pad_value=114,
        do_convert_rgb=True,
        letterbox=True,
        auto=False,
        stride=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if isinstance(size, dict) else {"height": size, "width": size}
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.do_convert_rgb = do_convert_rgb
        self.letterbox = letterbox
        self.auto = auto
        self.stride = stride
        
    def resize(self, image, size, resample="bilinear"):
        """
        Resize an image to the given size.
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
            
        if isinstance(size, dict):
            size = (size["height"], size["width"])
        elif isinstance(size, int):
            size = (size, size)
            
        resample_map = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
            "lanczos": Image.Resampling.LANCZOS,
        }
        resample = resample_map.get(resample, Image.Resampling.BILINEAR)
        
        return image.resize(size[::-1], resample)  # PIL uses (width, height)
    
    def letterbox_process(self, image, new_shape=(640, 640), color=(114, 114, 114), 
                  auto=False, scale_fill=False, scaleup=True, stride=32):
        """
        Resize and pad image while meeting stride-multiple constraints.
        
        Args:
            image: Input image
            new_shape: Target shape (height, width)
            color: Padding color
            auto: Minimum rectangle
            scale_fill: Stretch to fill new shape
            scaleup: Allow scale up
            stride: Stride for size divisibility
            
        Returns:
            Resized and padded image, and scaling factors
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        shape = image.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
            
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
            
        elif scale_fill:  # stretch
            dw, dh = 0, 0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
            
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return image, ratio, (dw, dh)
    
    def pad(self, image, pad_size_divisor=32, pad_value=114):
        """
        Pad an image to make its dimensions divisible by pad_size_divisor.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        height, width = image.shape[:2]
        new_height = int(np.ceil(height / pad_size_divisor) * pad_size_divisor)
        new_width = int(np.ceil(width / pad_size_divisor) * pad_size_divisor)
        
        # Create padded image
        padded_image = np.full((new_height, new_width, 3), pad_value, dtype=np.uint8)
        padded_image[:height, :width] = image
        
        return padded_image
    
    # Add the __call__ method to make the processor callable
    def __call__(self, images, return_tensors=None, **kwargs):
        """
        Main method to prepare images for the model.
        
        Args:
            images: The images to preprocess
            return_tensors: The type of tensors to return (e.g., "pt" for PyTorch)
            **kwargs: Additional arguments to pass to the preprocess method
            
        Returns:
            Preprocessed images and metadata
        """
        return self.preprocess(images=images, return_tensors=return_tensors, **kwargs)
    
    def preprocess(
        self,
        images,
        do_resize=None,
        size=None,
        resample=None,
        do_normalize=None,
        do_rescale=None,
        rescale_factor=None,
        do_pad=None,
        pad_size_divisor=None,
        pad_value=None,
        do_convert_rgb=None,
        letterbox=None,
        auto=None,
        stride=None,
        return_tensors=None,
        data_format=None,
        input_data_format=None,
        **kwargs
    ):
        """
        Preprocess an image or batch of images for YOLO models.
        
        Args:
            images: Image or batch of images
            do_resize: Whether to resize images
            size: Target size
            resample: Resampling method
            do_normalize: Whether to normalize images
            do_rescale: Whether to rescale images
            rescale_factor: Rescaling factor
            do_pad: Whether to pad images
            pad_size_divisor: Padding size divisor
            pad_value: Padding value
            do_convert_rgb: Whether to convert to RGB
            letterbox: Whether to use letterbox resizing
            auto: Whether to use minimum rectangle for letterbox
            stride: Stride for size divisibility
            return_tensors: Return format ('pt' for PyTorch tensors)
            data_format: Output data format ('channels_first' or 'channels_last')
            input_data_format: Input data format ('channels_first' or 'channels_last')
            
        Returns:
            Dict with preprocessed images and metadata
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size_divisor = pad_size_divisor if pad_size_divisor is not None else self.pad_size_divisor
        pad_value = pad_value if pad_value is not None else self.pad_value
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        letterbox = letterbox if letterbox is not None else self.letterbox
        auto = auto if auto is not None else self.auto
        stride = stride if stride is not None else self.stride
        
        if data_format is None:
            data_format = "channels_first" if return_tensors == "pt" else "channels_last"
            
        # Handle single image
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
            
        # Process each image
        processed_images = []
        ratios = []  # Store scale ratios for each image
        padding_info = []  # Store padding info for each image
        
        for image in images:
            # Convert to RGB if needed
            if do_convert_rgb:
                if isinstance(image, Image.Image) and image.mode != "RGB":
                    image = image.convert("RGB")
                elif isinstance(image, np.ndarray) and image.shape[-1] == 4:  # RGBA
                    image = image[..., :3]  # Remove alpha channel
            
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            # Get original image size
            orig_height, orig_width = image.shape[:2]
            orig_size = (orig_height, orig_width)
            
            # Resize with letterbox if requested
            if do_resize:
                if letterbox:
                    # Use letterbox resize
                    if isinstance(size, dict):
                        target_size = (size["height"], size["width"])
                    elif isinstance(size, int):
                        target_size = (size, size)
                    else:
                        target_size = size
                        
                    image, ratio, pad = self.letterbox_process(
                        image, #np.array (1080, 810, 3)
                        new_shape=target_size, #(640, 640)
                        color=(pad_value, pad_value, pad_value),
                        auto=auto, 
                        stride=stride
                    )
                    ratios.append(ratio)
                    padding_info.append(pad)
                else:
                    # Standard resize
                    if isinstance(size, dict):
                        target_size = (size["height"], size["width"])
                    elif isinstance(size, int):
                        target_size = (size, size)
                    else:
                        target_size = size
                        
                    # Calculate ratio for later use
                    ratio = (target_size[1] / orig_width, target_size[0] / orig_height)
                    ratios.append(ratio)
                    padding_info.append((0, 0))  # No padding in standard resize
                    
                    # Resize using cv2
                    image = cv2.resize(
                        image, 
                        (target_size[1], target_size[0]),  # cv2 uses (width, height)
                        interpolation=cv2.INTER_LINEAR
                    )
            else:
                # No resize, just record original ratio
                ratios.append((1.0, 1.0))
                padding_info.append((0, 0))
                
            # Pad if needed and not using letterbox (letterbox already includes padding)
            if do_pad and not (do_resize and letterbox):
                image = self.pad(image, pad_size_divisor, pad_value)
                
            # Rescale pixel values if needed
            if do_rescale:
                image = image * rescale_factor
                
            # Normalize if needed (YOLO models typically don't need normalization beyond rescaling)
            if do_normalize:
                pass
                
            # Ensure image is float32 for PyTorch
            if return_tensors == "pt":
                image = image.astype(np.float32)
                
            processed_images.append(image)
            
        # Convert to tensors if requested
        if return_tensors == "pt":
            # Convert to PyTorch tensors with correct data format
            if data_format == "channels_first":
                processed_images = [torch.tensor(img).permute(2, 0, 1) for img in processed_images]
            else:
                processed_images = [torch.tensor(img) for img in processed_images]
                
            processed_images = torch.stack(processed_images)
            
        # Prepare return dictionary with metadata
        result = {
            "pixel_values": processed_images,
            "original_sizes": [],
            "reshaped_input_sizes": [],
            "scale_factors": ratios,
            "padding_info": padding_info
        }
        
        # Ensure we always have original and reshaped sizes regardless of image format
        for i, image in enumerate(images):
            # Get original size
            if isinstance(image, np.ndarray):
                result["original_sizes"].append(image.shape[:2])
            elif isinstance(image, Image.Image):
                result["original_sizes"].append((image.height, image.width))
            else:
                # Try to get size from tensor
                try:
                    if image.ndim == 3:  # Single image tensor
                        result["original_sizes"].append((image.shape[0], image.shape[1]))
                    elif image.ndim == 4:  # Batch of image tensors
                        result["original_sizes"].append((image.shape[2], image.shape[3]))
                except:
                    # Fallback to None if we can't determine size
                    result["original_sizes"].append(None)
        
        # Get reshaped sizes
        if isinstance(processed_images, list):
            for img in processed_images:
                if isinstance(img, np.ndarray):
                    result["reshaped_input_sizes"].append(img.shape[:2])
                elif isinstance(img, torch.Tensor):
                    if img.ndim == 3:  # CHW format
                        result["reshaped_input_sizes"].append((img.shape[1], img.shape[2]))
                    elif img.ndim == 4:  # BCHW format
                        result["reshaped_input_sizes"].append((img.shape[2], img.shape[3]))
                else:
                    result["reshaped_input_sizes"].append(None)
        elif isinstance(processed_images, torch.Tensor):
            # Handle batch tensor case
            for i in range(processed_images.shape[0]):
                if processed_images.ndim == 4:  # BCHW format
                    result["reshaped_input_sizes"].append((processed_images.shape[2], processed_images.shape[3]))
                else:
                    result["reshaped_input_sizes"].append(None)
        
        return result
    
    def post_process_object_detection(
        self,
        outputs,
        threshold=0.5,
        target_sizes=None,
        nms_threshold=0.45,
        max_detections=300,
        scale_factors=None,  # Add parameter to accept scale factors
        padding_info=None    # Add parameter to accept padding info
    ):
        """
        Post-process the raw outputs of the model for object detection.
        Focuses on rescaling boxes from letterboxed images to original dimensions.
        
        Args:
            outputs: Raw model outputs or already processed detections
            threshold: Score threshold for detections (passed to model's postprocessing if needed)
            target_sizes: Original image sizes for rescaling boxes
            nms_threshold: IoU threshold for NMS (passed to model's postprocessing if needed)
            max_detections: Maximum number of detections to return
            scale_factors: Scale factors from preprocessing (ratios)
            padding_info: Padding information from preprocessing (padding)
            
        Returns:
            List of dictionaries with processed detection results
        """
        # Check if outputs is already in the processed format
        if isinstance(outputs, list) and all(isinstance(item, dict) and "boxes" in item for item in outputs):
            # Outputs are already processed, just need to rescale boxes if target_sizes provided
            results = []
            
            for i, output in enumerate(outputs):
                # Create a copy of the output to avoid modifying the original
                result = {
                    "scores": output["scores"],
                    "labels": output["labels"],
                    "boxes": output["boxes"].clone() if isinstance(output["boxes"], torch.Tensor) else output["boxes"].copy()
                }
                
                # Apply rescaling if target sizes are provided
                if target_sizes is not None and i < len(target_sizes):
                    # Get original dimensions
                    orig_h, orig_w = target_sizes[i]
                    
                    # Get scale factors and padding info if available
                    # Use the passed parameters instead of trying to access instance attributes
                    sf = (1.0, 1.0)
                    pad = (0, 0)
                    
                    if scale_factors is not None and i < len(scale_factors):
                        sf = scale_factors[i]
                    
                    if padding_info is not None and i < len(padding_info):
                        pad = padding_info[i]
                    
                    # Rescale boxes to original image dimensions
                    if isinstance(result["boxes"], torch.Tensor):
                        # For letterboxed images, we need to:
                        # 1. Remove padding
                        # 2. Scale by the inverse of the resize ratio
                        boxes = result["boxes"]
                        
                        # Remove padding (dw, dh)
                        dw, dh = pad
                        if dw > 0 or dh > 0:
                            boxes[:, 0] -= dw  # x1
                            boxes[:, 1] -= dh  # y1
                            boxes[:, 2] -= dw  # x2
                            boxes[:, 3] -= dh  # y2
                        
                        # Scale by inverse of resize ratio
                        scale_x, scale_y = sf
                        if scale_x != 1.0 or scale_y != 1.0:
                            boxes[:, 0] /= scale_x  # x1
                            boxes[:, 1] /= scale_y  # y1
                            boxes[:, 2] /= scale_x  # x2
                            boxes[:, 3] /= scale_y  # y2
                        
                        # Clip boxes to image boundaries
                        boxes[:, 0].clamp_(min=0, max=orig_w)
                        boxes[:, 1].clamp_(min=0, max=orig_h)
                        boxes[:, 2].clamp_(min=0, max=orig_w)
                        boxes[:, 3].clamp_(min=0, max=orig_h)
                        
                        result["boxes"] = boxes
                
                # Filter by threshold if needed
                if threshold > 0:
                    mask = result["scores"] > threshold
                    result = {
                        "scores": result["scores"][mask],
                        "labels": result["labels"][mask],
                        "boxes": result["boxes"][mask]
                    }
                
                results.append(result)
            
            return results
            
    def batch_decode(self, outputs, target_sizes=None):
        """
        Compatibility method for HuggingFace's AutoProcessor interface.
        Equivalent to post_process_object_detection.
        
        Args:
            outputs: Model outputs
            target_sizes: Original image sizes
            
        Returns:
            List of dictionaries with detection results
        """
        return self.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes
        )
    
class YoloDetectionModel(nn.Module):
    """YOLOv8 detection model with HuggingFace-compatible interface."""
    def __init__(self, cfg=None, scale='n', ch=3, nc=None, device='cuda', use_fp16=False, min_size=640, max_size=640):
        super().__init__()
        # If a config object is provided, use its parameters
        if isinstance(cfg, YoloConfig):
            scale = cfg.scale
            nc = cfg.nc
            ch = cfg.ch
            use_fp16 = cfg.use_fp16
            min_size = cfg.min_size
            max_size = cfg.max_size
        # Store image size and device info
        self.min_size = min_size
        self.max_size = max_size
        self.device = device
        self.use_fp16 = use_fp16
        #self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        #yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
        #self.yaml = yaml_load(yaml_path)
        #Using Python configuration instead of YAML
        self.yaml = get_yolo_config(scale, nc, ch)
        self.yaml['scale'] = scale
        #self.modelname = extract_filename(yaml_path)
        self.modelname = MODEL_TYPE
        self.scale = scale
        self.config = {
            "model_type": MODEL_TYPE,
            "scale": scale,
            "num_classes": nc or self.yaml.get('nc', 80),
            "image_size": 640,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 300
        }

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        
        # Parse model and get component indices based on scale
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=False)
        #self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        # Set up class names using COCO names
        self.names = coco_names if nc == 80 else {i: f'class_{i}' for i in range(self.yaml['nc'])}
        
        self.inplace = self.yaml.get('inplace', True)
        
        # Create config object with proper id2label and label2id mappings
        self.config = YoloConfig(
            scale=scale,
            nc=self.yaml['nc'],
            ch=ch,
            min_size=min_size,
            max_size=max_size,
            use_fp16=use_fp16,
            id2label={str(k): v for k, v in self.names.items()},
            label2id={v: str(k) for k, v in self.names.items()}
        )

        # Get component indices based on scale
        self.backbone_end, self.neck_end = self._get_component_indices(scale)
        
        # Build strides
        m = self.model[-1]
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        elif isinstance(m, IDetect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        # Init weights, biases
        initialize_weights(self)
        
        # Create transform for preprocessing and postprocessing
        # self.transform = YoloTransform(min_size=min_size, max_size=max_size, device=device, fp16=use_fp16, use_letterbox=True)
    
        # Add this class method for loading from pretrained
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a YOLOv8 model from a pretrained model directory or Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path (str): Path to a local directory or HF Hub model ID
            *model_args: Additional positional arguments passed to the model
            **kwargs: Additional keyword arguments passed to the model
            
        Returns:
            YoloDetectionModel: Loaded model instance
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        import os
        import json
        
        # Register architecture to ensure it's recognized
        register_yolo_architecture()
        
        # Load config
        config = None
        try:
            # Try to load the config file
            config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            
            if config_file:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                
                # Create config object
                config = YoloConfig(**config_dict)
            else:
                print("Config file not found, using default config")
                config = YoloConfig()
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default config")
            config = YoloConfig()
        
        # Create model instance with config
        model = cls(cfg=config)
        
        # Load weights
        try:
            weights_file = cached_file(
                pretrained_model_name_or_path,
                WEIGHTS_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            
            if weights_file:
                # Load state dict
                state_dict = torch.load(weights_file, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"Loaded weights from {weights_file}")
            else:
                print("Weights file not found")
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Set config attribute
        model.config = config
        
        return model
    
    def _get_component_indices(self, scale):
        """Get indices to split model into backbone, neck, and head based on scale."""
        # Default indices for different scales
        # Format: (backbone_end, neck_end)
        scale_indices = {
            'n': (9, 15),   # nano
            's': (9, 15),   # small
            'm': (9, 15),   # medium
            'l': (9, 15),   # large
            'x': (9, 15)    # xlarge
        }
        
        # Return indices for the specified scale or default to nano
        return scale_indices.get(scale, scale_indices['n'])
    
    @property
    def backbone(self):
        """Return the backbone part of the model."""
        return self.model[:self.backbone_end]
    
    @property
    def neck(self):
        """Return the neck (FPN) part of the model."""
        return self.model[self.backbone_end:self.neck_end]
    
    @property
    def heads(self):
        """Return the detection heads as a ModuleList."""
        head_modules = list(self.model[self.neck_end:])
        return nn.ModuleList(head_modules)
    
    def forward(self, x=None, pixel_values=None, images=None, **kwargs):
        """
        Forward pass of the model with HuggingFace-compatible interface.
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            pixel_values: Alternative input tensor name (HuggingFace standard)
            images: Alternative input tensor name
            **kwargs: Additional keyword arguments
            
        Returns:
            In training mode:
                - Raw model predictions for loss computation
            In inference mode:
                - If postprocess=True: List of dictionaries with detection results
                - If postprocess=False: Raw model predictions
                Format: Tensor of shape (batch_size, num_detections, 5+num_classes)
                where 5+num_classes = [x, y, w, h, confidence, class_scores...]
        """
        # Handle different input parameter names for compatibility
        if x is None:
            if pixel_values is not None:
                x = pixel_values
            elif images is not None:
                x = images
            elif 'inputs' in kwargs:
                x = kwargs['inputs']
            else:
                # Try to find any tensor in kwargs
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                        x = v
                        break
                else:
                    raise ValueError("No valid input tensor found in arguments. Expected 'x', 'pixel_values', or 'images'")
        
        # Handle training vs inference modes
        if self.training:
            # In training mode, just return raw predictions for loss computation
            return self._predict_once(x)
        else:
            # In inference mode
            with torch.no_grad():
                # Get raw predictions from model
                # Output format: Tensor of shape (batch_size, num_detections, 5+num_classes)
                # where 5+num_classes = [x, y, w, h, confidence, class_scores...]
                preds = self._predict_once(x) #tuple output
                #tuple, second item is list: [1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]
                # Extract predictions (handle tuple case if needed)
                raw_preds = preds[0] if isinstance(preds, tuple) else preds
                #[1, 84, 8400], 84 means 4 (64/reg_max) boxes + 80 classes (after sigmoid)
                # Apply postprocessing if requested
                if kwargs.get('postprocess', True):
                    # Get original image shapes if provided, otherwise use input tensor shape
                    orig_img_shapes = kwargs.get('orig_img_shapes', [(x.shape[2], x.shape[3])] * x.shape[0])
                    
                    # Apply NMS and format results
                    results = self.postprocess_detections(
                        raw_preds,
                        orig_img_shapes,
                        conf_thres=kwargs.get('conf_thres', 0.25),
                        iou_thres=kwargs.get('iou_thres', 0.45),
                        max_det=kwargs.get('max_det', 300)
                    )
                    
                    # Convert to DETR format if requested
                    if kwargs.get('use_detr_format', False):
                        results = self.convert_to_detr_format(results)
                    
                    return results
                else:
                    # only to NMS in postprocess_detections
                    # Apply NMS and format results
                    results = self.postprocess_detections(
                        predictions= raw_preds,
                        img_shapes = None, #no rescale of the image
                        conf_thres=kwargs.get('conf_thres', 0.25),
                        iou_thres=kwargs.get('iou_thres', 0.45),
                        max_det=kwargs.get('max_det', 300)
                    )
                    return results
                
    def postprocess_detections(self, predictions, img_shapes, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        Perform Non-Maximum Suppression (NMS) on inference results
        
        Args:
            predictions (torch.Tensor): Raw model predictions of shape (batch_size, 4+num_classes, num_detections) [1, 84, 8400]
                where 84 = 4 (box coords) + 80 (class scores)
            img_shapes (list): List of original image shapes [(height, width), ...]
            conf_thres (float): Confidence threshold
            iou_thres (float): IoU threshold for NMS
            max_det (int): Maximum number of detections per image
            
        Returns:
            list: List of dictionaries with detection results, one per image
                Each dict contains 'boxes', 'scores', and 'labels' tensors
        """
        # Initialize list to store results for each image
        results = []
        
        # Get batch size and device
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Number of classes (total - 4 box coordinates)
        nc = predictions.shape[1] - 4
        
        # Transpose predictions from [batch, 84, 8400] to [batch, 8400, 84]
        # This makes it easier to process each detection
        predictions = predictions.transpose(-1, -2)  # shape(batch_size, 8400, 84)
        
        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format for NMS
        predictions[..., :4] = self._xywh2xyxy(predictions[..., :4])
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get predictions for this image
            pred = predictions[i]  # Shape: [8400, 84]
            
            # Find candidates with any class score above threshold
            max_class_scores = pred[:, 4:].max(dim=1)[0]  # Get max class score for each box [8400]
            candidates = max_class_scores > conf_thres
            x = pred[candidates]  # Filter boxes, [4, 84]
            
            # If no detections remain, add empty result and continue
            if not x.shape[0]:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device)
                })
                continue
            
            # Get best class and its score for each detection
            class_scores, class_ids = x[:, 4:].max(1, keepdim=True)
            
            # Create detection matrix: [x1, y1, x2, y2, score, class_id]
            detections = torch.cat((x[:, :4], class_scores, class_ids.float()), 1) #[4, 6]
            
            # Apply NMS
            keep = torchvision.ops.nms(
                boxes=detections[:, :4],
                scores=detections[:, 4],
                iou_threshold=iou_thres
            )#tensor([2]
            
            # Limit to max_det
            if keep.shape[0] > max_det:
                keep = keep[:max_det]
            
            # Get final detections
            final_boxes = detections[keep, :4] #[1, 4]
            final_scores = detections[keep, 4] #torch.Size([1])
            final_class_ids = detections[keep, 5].long() #tensor([13]
            
            # Rescale boxes to original image dimensions if needed
            if img_shapes is not None:
                orig_h, orig_w = img_shapes[i]
                
                # Calculate scale factors (assuming input is square 640x640)
                input_size = 640  # Standard YOLO input size
                scale_x = orig_w / input_size
                scale_y = orig_h / input_size
                
                # Apply scaling
                final_boxes[:, 0] *= scale_x  # x1
                final_boxes[:, 1] *= scale_y  # y1
                final_boxes[:, 2] *= scale_x  # x2
                final_boxes[:, 3] *= scale_y  # y2
                
                # Clip boxes to image boundaries
                final_boxes[:, 0].clamp_(0, orig_w)
                final_boxes[:, 1].clamp_(0, orig_h)
                final_boxes[:, 2].clamp_(0, orig_w)
                final_boxes[:, 3].clamp_(0, orig_h)
            
            # Store results in dictionary format
            results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_class_ids
            })
        
        return results

    def _xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from [x, y, width, height] to [x1, y1, x2, y2] format
        where (x1, y1) is top-left and (x2, y2) is bottom-right.
        
        Args:
            x (torch.Tensor): Bounding boxes in [x, y, width, height] format
            
        Returns:
            torch.Tensor: Bounding boxes in [x1, y1, x2, y2] format
        """
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x - w/2
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y - h/2
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x + w/2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y + h/2
        return y

    
    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
            
        Returns:
            tuple: (total_loss, loss_items) where loss_items is a dictionary of individual loss components
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)  # return losssum, lossitems
    
    def init_criterion(self):
        """
        Initialize the loss criterion based on model type.
        
        Returns:
            nn.Module: Loss criterion
        """
        from DeepDataMiningLearning.detection.modules.lossv8 import myv8DetectionLoss
        
        # Get the last layer of the model to determine model type
        m = self.model[-1]
        
        if isinstance(m, Detect):
            # YOLOv8 detection loss
            return myv8DetectionLoss(self.model[-1])
        elif isinstance(m, IDetect):
            # YOLOv7 detection loss
            from DeepDataMiningLearning.detection.modules.lossv7 import myv7DetectionLoss
            return myv7DetectionLoss(self.model[-1])
        else:
            raise NotImplementedError(f"Loss not implemented for model with final layer: {type(m)}")
    
    def _predict_once(self, x, profile=False, export_internal=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        
        # In inference mode, return both the predictions and intermediate tensors
        if export_internal: #not self.training:
            return x, y
        # In training mode, just return the predictions
        return x #x is a tuple
    
    def forward_backbone(self, x):
        """Forward pass through just the backbone."""
        y = []
        for m in self.backbone:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def forward_neck(self, x):
        """Forward pass through just the neck (FPN)."""
        y = []
        for m in self.neck:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def forward_heads(self, x):
        """Forward pass through just the detection heads."""
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs
   

def upload_to_huggingface2(model, repo_id, token=None, commit_message="Upload model", 
                         private=False, create_model_card=False, example_images=None,
                         model_description=None):
    """
    Upload a model to the Hugging Face Hub.
    
    Args:
        model: The model to upload
        repo_id: The repository ID on Hugging Face Hub
        token: Hugging Face API token
        commit_message: Commit message for the upload
        private: Whether the repository should be private
        create_model_card: Whether to create a model card
        example_images: List of example image paths to include in the model card
        model_description: Custom description for the model card
    """
    import tempfile
    import os
    import shutil
    import json
    from transformers import PretrainedConfig
    
    # Create a temporary directory to save the model files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Save the model state dict
        model_path = os.path.join(temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model state dict to {model_path}")
        
        # Create and save the config
        if hasattr(model, 'config') and isinstance(model.config, PretrainedConfig):
            config = model.config
        else:
            # Create a config object if the model doesn't have one
            from transformers import PretrainedConfig
            config_dict = {
                "model_type": MODEL_TYPE,  # Use our custom model type
                "architectures": ["YoloDetectionModel"],
                "scale": model.scale,
                "num_classes": model.yaml.get('nc', 80),
                "image_size": 640,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 300
            }
            config = PretrainedConfig.from_dict(config_dict)
        
        # Save the config
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write(config.to_json_string())
        print(f"Saved config to {config_path}")
        
        # Add image processor config
        processor_config = {
            "image_processor_type": "YoloImageProcessor",  # Use our custom processor "image_processor_type": "DetrImageProcessor",  # Use DETR's image processor "YoloImageProcessor",
            "do_normalize": False,  # YOLOv8 doesn't need normalization beyond rescaling
            "do_resize": True,
            "do_rescale": True,
            "do_pad": True,
            "size": {
                "height": 640,
                "width": 640
            },
            "resample": "bilinear",
            "rescale_factor": 0.00392156862745098,  # 1/255
            "do_convert_rgb": True,
            "pad_size_divisor": 32,
            "pad_value": 114,
            "letterbox": True,  # Enable letterbox resizing
            "auto": False,
            "stride": 32
        }
        
        # Save the image processor config
        processor_config_path = os.path.join(temp_dir, "preprocessor_config.json")
        with open(processor_config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        print(f"Saved image processor config to {processor_config_path}")
        
        # Create a model card if requested
        if create_model_card:
            model_name = f"YOLOv8 {model.scale.upper()}"
            readme_path = os.path.join(temp_dir, "README.md")
            create_yolo_model_card(
                model_name=model_name,
                scale=model.scale,
                num_classes=model.yaml.get('nc', 80),
                repo_id=repo_id,
                output_path=readme_path,
                example_images=example_images,
                description=model_description
            )
            print(f"Created model card at {readme_path}")
        
        # Upload to Hugging Face Hub
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Create the repository if it doesn't exist
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"Created/verified repository: {repo_id}")
        
        # Upload the files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        print(f"Uploaded model to {repo_id}")
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
        
import os

def create_yolo_model_card(model_name, scale, num_classes, repo_id, output_path="README.md", 
                          example_images=None, description=None):
    """
    Create a detailed model card for a YOLO model to be uploaded to HuggingFace.
    
    Args:
        model_name (str): Name of the model
        scale (str): Scale of the model (n, s, m, l, x)
        num_classes (int): Number of classes the model can detect
        repo_id (str): HuggingFace repository ID
        output_path (str): Path to save the model card
        example_images (list, optional): List of paths to example images to include in the model card
        description (str, optional): Custom description for the model card
        
    Returns:
        str: Path to the created model card
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Scale descriptions
    scale_descriptions = {
        'n': "nano (smallest, fastest)",
        's': "small (good balance of speed and accuracy)",
        'm': "medium (higher accuracy, moderate speed)",
        'l': "large (high accuracy, slower inference)",
        'x': "xlarge (highest accuracy, slowest inference)"
    }
    
    scale_desc = scale_descriptions.get(scale, f"custom scale '{scale}'")
    
    # Default description if none provided
    if description is None:
        description = f"""
This model is a YOLOv8 object detection model with scale '{scale}' ({scale_desc}). 
It can detect {num_classes} different classes and is optimized for real-time object detection.

YOLOv8 is the latest version in the YOLO (You Only Look Once) family of models and 
offers improved accuracy and speed compared to previous versions.
"""
    
    # Create the model card content
    model_card = f"""---
language: en
license: mit
tags:
- object-detection
- yolov8
- computer-vision
- pytorch
- transformers
datasets:
- coco
---

# {model_name} - YOLOv8 Object Detection Model

{description}

## Model Details

- **Model Type:** YOLOv8
- **Scale:** {scale} ({scale_desc})
- **Number of Classes:** {num_classes}
- **Input Size:** 640x640
- **Framework:** PyTorch + Transformers

## Usage

### With Transformers Pipeline

```python
from transformers import pipeline

detector = pipeline("object-detection", model="{repo_id}")
result = detector("path/to/image.jpg")
print(result)
"""


def visualize_raw_detections(raw_outputs, original_image, conf_threshold=0.25, output_path=None):
    """
    Visualize raw detection outputs by drawing bounding boxes on the original image.
    
    Args:
        raw_outputs: Raw outputs from the model (list of dicts with 'boxes', 'scores', 'labels')
        original_image: Original image as numpy array (BGR format from cv2)
        conf_threshold: Confidence threshold for displaying detections
        output_path: Path to save the visualization (if None, will display the image)
        
    Returns:
        Numpy array of the image with drawn bounding boxes
    """
    # Create a copy of the image for visualization
    img_vis = original_image.copy()
    
    # Check the type of raw_outputs and process accordingly
    if isinstance(raw_outputs, list):
        # If it's already a list of detection results
        detection_results = raw_outputs
    else:
        print(f"Unsupported raw_outputs type: {type(raw_outputs)}")
        return img_vis
    
    # Process each detection result
    for result in detection_results:
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        
        # Convert tensors to numpy if needed
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Draw each detection
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Skip low confidence detections
            if score < conf_threshold:
                continue
                
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Generate a color based on the class label
            color_factor = (int(label) * 50) % 255
            color = (color_factor, 255 - color_factor, 128)
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Get class name
            class_name = coco_names.get(int(label), f"class_{int(label)}")
            
            # Create label text
            label_text = f"{class_name}: {score:.2f}"
            
            # Add a filled rectangle behind text for better visibility
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_vis, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # Add text with white color for better contrast
            cv2.putText(img_vis, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save or display the image
    if output_path:
        cv2.imwrite(output_path, img_vis)
        print(f"Visualization saved to {output_path}")
    
    return img_vis



def test_localmodel_detrprocess(scale = 's', use_fp16=True, visualize=True, output_dir="output"):
    # Initialize model
    # Initialize model with the specified scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a proper config object first
    config = YoloConfig(
        scale=scale,
        nc=80,
        ch=3,
        min_size=640,
        max_size=640,
        use_fp16=use_fp16
    )
    # Initialize model with config
    model = YoloDetectionModel(
        cfg=config,
        device=device
    )
    model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    model = model.to(device)
    model.eval()
    
    # Enable FP16 precision if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        print("Using FP16 precision for faster inference")
    
    # Load image
    image_path = "ModelDev/sampledata/bus.jpg"
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Convert BGR to RGB (DETR processor expects RGB)
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # Create PIL image for DETR processor
    pil_image = Image.fromarray(img_rgb)
    # Store original image size for postprocessing
    orig_size = pil_image.size[::-1]  # (height, width) (1080, 810)
    
    from transformers import DetrImageProcessor
    processor = DetrImageProcessor(
        do_resize=True,
        size={"height": config.min_size, "width": config.max_size},
        do_normalize=True,
        do_rescale=True
    )
    
    # Preprocess image using DETR processor
    inputs = processor(images=pil_image, return_tensors="pt")
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    #pixel_values: [1, 3, 640, 640]
    # Convert inputs to half precision if model is in half precision
    if use_fp16 and device.type == 'cuda':
        inputs = {k: v.half() for k, v in inputs.items()}
        
    raw_outputs = model(pixel_values=inputs["pixel_values"], postprocess=False) ## Get raw outputs for DETR postprocessing
    #list of one dict with 'boxes', 'scores', 'labels'
    
    
    # # Visualize raw detections
    vis_img = visualize_raw_detections(
        raw_outputs, 
        img_orig,
        conf_threshold=0.25,
        output_path="output/raw_detections.jpg"
    )
    cv2.imshow("Raw Detections", vis_img)
    cv2.waitKey(0)

    # Check the type of outputs and handle accordingly
    if isinstance(raw_outputs, list):
        # If raw_outputs is a list of detection results
        outputs = model.convert_to_detr_format(raw_outputs)
    elif isinstance(raw_outputs, torch.Tensor):
        # If raw_outputs is a tensor, we need to format it properly
        # Assuming the tensor has the format [batch, detections, (x1,y1,x2,y2,conf,class_id,...)]
        # Extract boxes, scores, and labels
        boxes = raw_outputs[..., :4]
        scores = raw_outputs[..., 4]
        labels = raw_outputs[..., 5].long()
        
        # Create a list of dictionaries for convert_to_detr_format
        detection_results = [{"boxes": boxes[i], "scores": scores[i], "labels": labels[i]} 
                            for i in range(raw_outputs.shape[0])]
        
        outputs = model.convert_to_detr_format(detection_results)
    else:
        # If it's already in DETR format (dict with pred_boxes and pred_logits)
        outputs = raw_outputs
    
    # Postprocess using DETR processor
    conf_thres = 0.2
    # Make sure outputs has the expected structure with 'logits' and 'pred_boxes'
    if 'pred_logits' in outputs and 'pred_boxes' in outputs:
        detections = processor.post_process_object_detection(
            outputs, 
            threshold=conf_thres,
            target_sizes=[orig_size]
        )[0]
    else:
        # If outputs doesn't have the expected structure, create a compatible format
        print("Warning: Outputs not in expected format, creating compatible structure")
        # Create dummy outputs in the expected format
        compatible_outputs = {
            "logits": outputs.get("pred_logits", torch.zeros((1, 0, 80), device=device)),
            "pred_boxes": outputs.get("pred_boxes", torch.zeros((1, 0, 4), device=device))
        }
        detections = processor.post_process_object_detection(
            compatible_outputs, 
            threshold=conf_thres,
            target_sizes=[orig_size]
        )[0]
    
    # Visualize results if requested
    if visualize and len(detections["scores"]) > 0:
        # Create a copy of the image for visualization
        img_vis = img_orig.copy()
        
        # Extract detection components
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        
        # Draw boxes on the image
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            label = int(labels[i])
            
            # Generate a color based on the class label for better visualization
            color_factor = (label * 50) % 255
            color = (color_factor, 255 - color_factor, 128)
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Add class name and confidence
            class_name = self.names.get(int(label))
            if class_name is None or class_name == f"{int(label)}":
                # Try to get class name from COCO names if available
                class_name = coco_names.get(int(label), f"class_{label}")
            
            label_text = f"{class_name}: {score:.2f}"
            
            # Add a filled rectangle behind text for better visibility
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_vis, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # Add text with white color for better contrast
            cv2.putText(img_vis, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Determine output path
        if output_dir:
            # Use the specified output directory
            base_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_detected{os.path.splitext(base_filename)[1]}")
        else:
            # Save in the same directory as the input image
            output_path = image_path.replace('.', '_detected.')
        
        # Save the visualization
        cv2.imwrite(output_path, img_vis)
        print(f"Visualization saved to {output_path}")
        
        # Add visualization path to detections
        detections["visualization_path"] = output_path

def test_localmodel(scale='s', use_fp16=True, visualize=True, output_dir="output"):
    """
    Test a local YOLOv8 model using YoloImageProcessor for preprocessing
    and YoloDetectionModel for inference.
    
    Args:
        scale (str): Model scale - 'n', 's', 'm', 'l', or 'x'
        use_fp16 (bool): Whether to use FP16 precision for faster inference
        visualize (bool): Whether to visualize and save detection results
        output_dir (str): Directory to save output visualizations
    
    Returns:
        dict: Detection results
    """
    # Create output directory if it doesn't exist
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with the specified scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a proper config object first
    config = YoloConfig(
        scale=scale,
        nc=80,
        ch=3,
        min_size=640,
        max_size=640,
        use_fp16=use_fp16
    )
    
    # Initialize model with config
    model = YoloDetectionModel(
        cfg=config,
        device=device
    )
    
    # Load pre-trained weights
    weights_path = f"../modelzoo/yolov8{scale}_statedicts.pt"
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    
    # Enable FP16 precision if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        print(f"Using FP16 precision for faster inference on {device}")
    
    # Load image
    image_path = "ModelDev/sampledata/bus.jpg"
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Convert BGR to RGB (YoloImageProcessor expects RGB)
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Create PIL image for processor
    pil_image = Image.fromarray(img_rgb)
    
    # Store original image size for postprocessing
    orig_size = (img_orig.shape[0], img_orig.shape[1])  # (height, width)
    
    # Initialize YoloImageProcessor with letterbox preprocessing
    processor = YoloImageProcessor(
        do_resize=True,
        size=640,
        do_normalize=False,  # YOLOv8 doesn't need normalization beyond rescaling
        do_rescale=True,
        rescale_factor=1/255.0,
        do_pad=True,
        pad_size_divisor=32,
        pad_value=114,
        do_convert_rgb=True,
        letterbox=True,  # Use letterbox resizing
        auto=False,
        stride=32
    )
    
    # Preprocess image using YoloImageProcessor
    inputs = processor.preprocess(
        images=pil_image, 
        return_tensors="pt"
    )
    # Get scale factors and padding info from preprocessing
    scale_factors = inputs["scale_factors"]
    padding_info = inputs["padding_info"]
    
    # Move inputs to the same device as model
        # Move only tensor inputs to the same device as model
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Convert inputs to half precision if model is in half precision
    if use_fp16 and device.type == 'cuda':
        inputs = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        # Get raw outputs without postprocessing
        raw_outputs = model(
            pixel_values=inputs["pixel_values"], 
            postprocess=False #only NMS
        )#list of dicts
        #outputs = model(**inputs)
        
        # Post-process with scale factors and padding info
        processed_outputs = processor.post_process_object_detection(
            outputs=raw_outputs,
            target_sizes=inputs["original_sizes"],
            scale_factors=scale_factors,
            padding_info=padding_info
        )
    
    # Visualize detections if requested
    if visualize:
        # # Visualize raw detections
        vis_img = visualize_raw_detections(
            processed_outputs, 
            img_orig, #inputs["pixel_values"], #
            conf_threshold=0.25,
            output_path="output/raw_detections.jpg"
        )
        
        
    # Return detection results
    return {
        "detections": processed_outputs,
        "original_size": orig_size,
        "model_scale": scale,
        "visualization_path": output_path if visualize else None
    }
    
def testviaHF(repo_id):
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    register_yolo_architecture()
    
    # Load model and processor
    #processor = AutoImageProcessor.from_pretrained(repo_id)
    processor = YoloImageProcessor(
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
    model = AutoModelForObjectDetection.from_pretrained(repo_id)

    # Load and preprocess an image
    # image_path = "path/to/your/image.jpg"
    # image = cv2.imread(image_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(image_rgb)
    # Load and process image
    image = Image.open("ModelDev/sampledata/bus.jpg")
    inputs = processor(images=image, return_tensors="pt")

    # Get scale factors and padding info from preprocessing
    scale_factors = inputs["scale_factors"]
    padding_info = inputs["padding_info"]

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process with scale factors and padding info for accurate box coordinates
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.25,
        target_sizes=inputs["original_sizes"],
        scale_factors=scale_factors,
        padding_info=padding_info
    )
    
    # Print results
    for i, (boxes, scores, labels) in enumerate(zip(
        results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    )):
        # Safely get the label name, handling both string and integer keys
        label_id = labels.item()
        if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
            # Try to get the label from the model's config
            if str(label_id) in model.config.id2label:
                # If the key is stored as a string
                label_name = model.config.id2label[str(label_id)]
            elif label_id in model.config.id2label:
                # If the key is stored as an integer
                label_name = model.config.id2label[label_id]
            else:
                # Fallback to COCO class names or just use the ID
                label_name = coco_names.get(label_id, f"class_{label_id}")
        else:
            # Fallback to COCO class names or just use the ID
            label_name = coco_names.get(label_id, f"class_{label_id}")
            
        print(
            f"Detected {label_name} with confidence "
            f"{round(scores.item(), 3)} at location {boxes.tolist()}"
        )

    # Draw bounding boxes on the image
    image_np = np.array(image)
    for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
        box = [int(i) for i in box.tolist()]
        
        # Safely get the label name, same as above
        label_id = label.item()
        if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
            if str(label_id) in model.config.id2label:
                label_name = model.config.id2label[str(label_id)]
            elif label_id in model.config.id2label:
                label_name = model.config.id2label[label_id]
            else:
                label_name = coco_names.get(label_id, f"class_{label_id}")
        else:
            label_name = coco_names.get(label_id, f"class_{label_id}")
            
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image_np, 
                    f"{label_name}: {round(score.item(), 2)}", 
                    (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite("output/result.jpg", image_np)
    # cv2.imshow("Result", image_np)
    # cv2.waitKey(0)

def test_upload_model():
    """
    Test function to upload a YOLO model to HuggingFace Hub.
    """
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80, ch=3,
    device=device, use_fp16=True, min_size=640, max_size=640)
    
    # Load weights
    model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    model = model.to(device)
    model.eval()
    
    # Upload to HuggingFace
    # Replace with your HuggingFace username and desired repository name
    repo_id = "lkk688/yolov8s-model"

    # Example images for the model card (optional)
    example_images = [
        "sampledata/bus.jpg",
        "sampledata/sjsupeople.jpg"
    ]
    
    # Custom description for the model card (optional)
    custom_description = """
    This is a custom modified YOLOv8s model trained on the COCO dataset for object detection.
    It can detect 80 different object classes with good accuracy and speed.
    The model has been optimized for real-time inference on both GPU and CPU.
    """
    
    # Upload the model with model card creation
    upload_to_huggingface(
        model=model,
        repo_id=repo_id,
        token=None,#use system token, login in termal: huggingface-cli login
        commit_message="Upload YOLOv8s model",
        private=False,  # Set to True if you want a private repository
        create_model_card=True,  # This triggers the model card creation
        example_images=example_images,  # Optional: include example images
        model_description=custom_description  # Optional: custom description
    )
    #The model card is automatically created when you call upload_to_huggingface with create_model_card=True .

def upload_onetype_model(scale='s'):
    """
    Test function to upload a YOLO model to HuggingFace Hub.
    
    Args:
        scale (str): Model scale - 'n' (nano), 's' (small), 'm' (medium), 
                    'l' (large), or 'x' (xlarge)
    """
    # Register the YOLO architecture with Transformers
    register_yolo_architecture()
    
    # Initialize model with the specified scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a proper config object first
    config = YoloConfig(
        scale=scale,
        nc=80,
        ch=3,
        min_size=640,
        max_size=640,
        use_fp16=True
    )
    # Initialize model with config
    model = YoloDetectionModel(
        cfg=config,
        device=device
    )
    # yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    # model = YoloDetectionModel(cfg=yaml_path, scale=scale, nc=80, ch=3,
    #                           device=device, use_fp16=True, min_size=640, max_size=640)
    
    # Load weights for the specified scale
    weights_path = f"../modelzoo/yolov8{scale}_statedicts.pt"
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    
    # Upload to HuggingFace with scale in the repo name
    repo_id = f"lkk688/{MODEL_TYPE}{scale}-model"

    # Example images for the model card (optional)
    example_images = [
        "sampledata/bus.jpg",
        "sampledata/sjsupeople.jpg"
    ]
    
    # Custom description based on scale
    scale_descriptions = {
        'n': "nano (smallest and fastest)",
        's': "small (good balance of speed and accuracy)",
        'm': "medium (higher accuracy, moderate speed)",
        'l': "large (high accuracy, slower inference)",
        'x': "xlarge (highest accuracy, slowest inference)"
    }
    
    scale_desc = scale_descriptions.get(scale, f"custom scale '{scale}'")
    
    custom_description = f"""
    This is a custom YOLOv8{scale} model ({scale_desc}) trained on the COCO dataset for object detection.
    It can detect 80 different object classes with good accuracy and speed.
    The model has been optimized for real-time inference on both GPU and CPU.
    """
    
    # Upload the model with model card creation
    try:
        upload_to_huggingface2(
            model=model,
            repo_id=repo_id,
            token=None,  # use system token, login in terminal
            commit_message=f"Upload YOLOv8{scale} model",
            private=False,  # Set to True if you want a private repository
            create_model_card=True,  # This triggers the model card creation
            example_images=example_images,  # Optional: include example images
            model_description=custom_description  # Optional: custom description
        )
        print(f"Successfully uploaded YOLOv8{scale} model to {repo_id}")
    except Exception as e:
        print(f"Error uploading YOLOv8{scale} model: {e}")
        raise

    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FasterRCNN Model Operations")
    parser.add_argument("--action", type=str, default="load", 
                        choices=["test", "upload", "load", "pipeline"],
                        help="Action to perform: upload to HF, run ONNX inference, or run HF pipeline")
    args = parser.parse_args()
    if args.action == "test":
        test_localmodel() #works
    elif args.action == "upload":
        #test_upload_model()
        # Or upload all scales in sequence
        for scale in ['n', 's', 'm', 'l', 'x']:
            try:
                print(f"\n=== Uploading YOLOv8{scale} model ===\n")
                upload_onetype_model(scale)
            except Exception as e:
                print(f"Error uploading YOLOv8{scale}: {e}")
    elif args.action == "load":
        repo_id = "lkk688/yolov8s-model"
        testviaHF(repo_id)

#
# DETR's image processor ( DetrImageProcessor ) doesn't natively support letterboxing in the same way as YOLOv8. YOLOv8's letterbox preprocessing maintains the aspect ratio of the image by padding to a square, while DETR typically resizes images without preserving aspect ratio.
#Since DETR's image processor doesn't support letterboxing, you'll need to modify your model's forward method to handle the preprocessing differences.
# 