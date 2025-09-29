
import os
import copy
import re
import sys
from argparse import ArgumentParser
from threading import Thread
from typing import Union, List, Dict, Any

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

try:
    from vllm import SamplingParams, LLM
    from qwen_vl_utils import process_vision_info
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install vllm and qwen-vl-utils to use vLLM backend.")

# Import QwenObjectDetectionPipeline
try:
    # Add the VLM directory to path
    vlm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VisionLangAnnotateModels', 'VLM')
    if vlm_path not in sys.path:
        sys.path.append(vlm_path)
    
    from qwen_object_detection_pipeline3 import QwenObjectDetectionPipeline
    OBJECT_DETECTION_AVAILABLE = True
    print("QwenObjectDetectionPipeline loaded successfully")
except ImportError as e:
    OBJECT_DETECTION_AVAILABLE = False
    print(f"Warning: QwenObjectDetectionPipeline not available: {e}")
    
    # Create a placeholder class
    class QwenObjectDetectionPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def detect_objects(self, *args, **kwargs):
            return {"error": "Object detection not available"}
        def process_video_frames(self, *args, **kwargs):
            return {"error": "Video processing not available"}


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--backend',
                        type=str,
                        choices=['huggingface', 'vllm'],
                        default='huggingface',
                        help='Backend to use: huggingface (HuggingFace) or vllm (vLLM)')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.70,
                        help='GPU memory utilization for vLLM (default: 0.70)')
    parser.add_argument('--tensor-parallel-size',
                        type=int,
                        default=None,
                        help='Tensor parallel size for vLLM (default: auto)')
    parser.add_argument('--enable-object-detection',
                        action='store_true',
                        default=True,
                        help='Enable object detection functionality')
    parser.add_argument('--detection-output-dir',
                        type=str,
                        default='./detection_results',
                        help='Output directory for detection results')

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    # Initialize object detection pipeline if enabled
    detection_pipeline = None
    if args.enable_object_detection and OBJECT_DETECTION_AVAILABLE:
        try:
            detection_pipeline = QwenObjectDetectionPipeline(
                model_name=args.checkpoint_path,
                device="cuda" if not args.cpu_only else "cpu",
                output_dir=args.detection_output_dir,
                vlm_backend=args.backend
            )
            print("Object detection pipeline initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize object detection pipeline: {e}")
            detection_pipeline = None
    
    if args.backend == 'vllm':
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install vllm and qwen-vl-utils.")

        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        tensor_parallel_size = args.tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        # Initialize vLLM sync engine
        model = LLM(
            model=args.checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            tensor_parallel_size=tensor_parallel_size,
            seed=0
        )

        # Load processor for vLLM
        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        return model, processor, 'vllm', detection_pipeline
    else:
        if args.cpu_only:
            device_map = 'cpu'
        else:
            device_map = 'auto'

        # Check if flash-attn2 flag is enabled and load model accordingly
        if args.flash_attn2:
            model = AutoModelForImageTextToText.from_pretrained(args.checkpoint_path,
                                                                    torch_dtype='auto',
                                                                    attn_implementation='flash_attention_2',
                                                                    device_map=device_map)
        else:
            model = AutoModelForImageTextToText.from_pretrained(args.checkpoint_path, device_map=device_map)

        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        return model, processor, 'hf', detection_pipeline


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def _prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM inference"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def _launch_demo(args, model, processor, backend, detection_pipeline=None):

    def call_image_description(detection_pipeline, image_path, prompt=""):
        """Call the describe_image function for image description"""
        if not detection_pipeline:
            return "Object detection pipeline not available. Please enable it with --enable-object-detection flag."
        
        try:
            # Use the new describe_image function directly
            if prompt.strip():
                description = detection_pipeline.describe_image(image_path, prompt)
            else:
                description = detection_pipeline.describe_image(image_path)
            return f"Image Description:\n{description}"
        except Exception as e:
            return f"Error generating image description: {str(e)}"

    def call_object_detection(detection_pipeline, image_path):
        """Call object detection pipeline for object detection"""
        if not detection_pipeline:
            return "Object detection pipeline not available. Please enable it with --enable-object-detection flag."
        
        try:
            # Use the pipeline for object detection
            result = detection_pipeline.detect_objects(
                image_path=image_path,
                save_results=True,  # Save files to enable viewing detection results
                apply_privacy=False,  # No privacy protection in interactive mode
                use_sam_segmentation=False  # Keep it fast for interactive use
            )
            
            objects = result.get('objects', [])
            raw_response = result.get('raw_response', '')
            
            response = f"Raw Model Response:\n{raw_response}\n\n"
            
            if objects:
                response += f"Detected Objects ({len(objects)}):\n"
                for i, obj in enumerate(objects, 1):
                    label = obj.get('label', 'Unknown')
                    bbox = obj.get('bbox', [])
                    description = obj.get('description', 'No description')
                    confidence = obj.get('confidence', 0.0)
                    
                    response += f"{i}. {label}"
                    if bbox and len(bbox) >= 4:
                        response += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                    response += f" (confidence: {confidence:.2f})\n"
                    response += f"   Description: {description}\n"
            else:
                response += "No objects detected with bounding boxes."
            
            return response
                
        except Exception as e:
            return f"Error in object detection pipeline: {str(e)}"

    def call_video_detection(detection_pipeline, video_path, task_type="description"):
        """Call object detection pipeline for video analysis"""
        if not detection_pipeline:
            return "Object detection pipeline not available. Please enable it with --enable-object-detection flag."
        
        try:
            # For video processing, we'll process it as a batch of frames
            # The pipeline should handle video input automatically
            if task_type == "description":
                result = detection_pipeline.detect_objects(
                    image_path=video_path,  # Pipeline should detect it's a video
                    save_results=True,  # Save files to enable viewing detection results
                    apply_privacy=False,
                    use_sam_segmentation=False
                )
                
                # Handle video analysis results
                if isinstance(result, list):
                    # Multiple frames processed
                    response = f"Video Analysis ({len(result)} frames processed):\n\n"
                    for i, frame_result in enumerate(result):
                        response += f"Frame {i+1}:\n"
                        if 'objects' in frame_result and frame_result['objects']:
                            descriptions = []
                            for obj in frame_result['objects']:
                                if 'description' in obj:
                                    descriptions.append(obj['description'])
                            if descriptions:
                                response += "\n".join(descriptions) + "\n\n"
                        else:
                            response += frame_result.get('raw_response', 'No description available') + "\n\n"
                    return response
                else:
                    # Single result for entire video
                    if 'objects' in result and result['objects']:
                        descriptions = []
                        for obj in result['objects']:
                            if 'description' in obj:
                                descriptions.append(obj['description'])
                        if descriptions:
                            return f"Video Description:\n" + "\n".join(descriptions)
                    return result.get('raw_response', 'No video description available.')
                    
            elif task_type == "detection":
                result = detection_pipeline.detect_objects(
                    image_path=video_path,
                    save_results=True,  # Save files to enable viewing detection results
                    apply_privacy=False,
                    use_sam_segmentation=False
                )
                
                if isinstance(result, list):
                    # Multiple frames processed
                    response = f"Video Object Detection ({len(result)} frames processed):\n\n"
                    for i, frame_result in enumerate(result):
                        response += f"Frame {i+1}:\n"
                        objects = frame_result.get('objects', [])
                        raw_response = frame_result.get('raw_response', '')
                        
                        if raw_response:
                            response += f"Model Response: {raw_response}\n"
                        
                        if objects:
                            response += f"Detected Objects ({len(objects)}):\n"
                            for j, obj in enumerate(objects, 1):
                                label = obj.get('label', 'Unknown')
                                bbox = obj.get('bbox', [])
                                confidence = obj.get('confidence', 0.0)
                                
                                response += f"  {j}. {label}"
                                if bbox and len(bbox) >= 4:
                                    response += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                                response += f" (confidence: {confidence:.2f})\n"
                        else:
                            response += "No objects detected in this frame.\n"
                        response += "\n"
                    return response
                else:
                    # Single result for entire video
                    objects = result.get('objects', [])
                    raw_response = result.get('raw_response', '')
                    
                    response = f"Video Object Detection:\n"
                    if raw_response:
                        response += f"Model Response: {raw_response}\n\n"
                    
                    if objects:
                        response += f"Detected Objects ({len(objects)}):\n"
                        for i, obj in enumerate(objects, 1):
                            label = obj.get('label', 'Unknown')
                            bbox = obj.get('bbox', [])
                            confidence = obj.get('confidence', 0.0)
                            
                            response += f"{i}. {label}"
                            if bbox and len(bbox) >= 4:
                                response += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                            response += f" (confidence: {confidence:.2f})\n"
                    else:
                        response += "No objects detected in the video."
                    
                    return response
            else:
                return "Invalid task type. Use 'description' or 'detection'."
                
        except Exception as e:
            return f"Error in video detection pipeline: {str(e)}"

    def call_local_model(model, processor, messages, backend):
        messages = _transform_messages(messages)

        if backend == 'vllm':
            # vLLM inference
            inputs = _prepare_inputs_for_vllm(messages, processor)
            sampling_params = SamplingParams(max_tokens=1024)

            accumulated_text = ''
            for output in model.generate(inputs, sampling_params=sampling_params):
                for completion in output.outputs:
                    new_text = completion.text
                    if new_text:
                        accumulated_text += new_text
                        yield accumulated_text
        else:
            # HuggingFace inference
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            tokenizer = processor.tokenizer
            streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_kwargs = {'max_new_tokens': 1024, 'streamer': streamer, **inputs}
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

    def create_predict_fn():

        def predict(_chatbot, task_history, mode="chat"):
            nonlocal model, processor, backend, detection_pipeline
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            
            # Handle different modes
            if mode in ["description", "detection"] and isinstance(query, (tuple, list)):
                # Object detection or description mode
                file_path = query[0]
                if _is_video_file(file_path):
                    response = call_video_detection(detection_pipeline, file_path, mode)
                else:
                    response = call_object_detection(detection_pipeline, file_path, mode)
                
                _chatbot[-1] = (_parse_text(chat_query), response)
                task_history[-1] = (query, response)
                print(f'Qwen-VL-{mode.title()}: ' + response)
                yield _chatbot
                return
            
            # Original chat mode
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'{os.path.abspath(q[0])}'})
                    else:
                        content.append({'image': f'{os.path.abspath(q[0])}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            for response in call_local_model(model, processor, messages, backend):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat: ' + _parse_text(full_response))
            yield _chatbot

        return predict


    def create_regenerate_fn():

        def regenerate(_chatbot, task_history, mode="chat"):
            nonlocal model, processor, backend, detection_pipeline
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            _chatbot_gen = predict(_chatbot, task_history, mode)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png" style="height: 80px"/><p>"""
                   )
        gr.Markdown("""<center><font size=8>Qwen2.5-VL with Object Detection</center>""")
        gr.Markdown(f"""\
<center><font size=3>This WebUI is based on Qwen2.5-VL with integrated object detection capabilities. Backend: {backend.upper()}</center>""")
        
        with gr.Tabs():
            # Image Description Tab
            with gr.TabItem("📝 Image Description"):
                with gr.Row():
                    with gr.Column(scale=1):
                        desc_image_input = gr.Image(label="Upload Image", type="filepath")
                        desc_prompt_input = gr.Textbox(
                            label="Custom Prompt (Optional)", 
                            placeholder="Enter a custom prompt for image description, or leave empty for default description",
                            lines=3
                        )
                        desc_submit_btn = gr.Button("🔍 Generate Description", variant="primary")
                        
                    with gr.Column(scale=1):
                        desc_output = gr.Textbox(label="Image Description", lines=15, max_lines=20)
                
                def process_image_description(image_path, prompt):
                    if not image_path:
                        return "Please upload an image first."
                    
                    try:
                        if detection_pipeline is None:
                            return "Object detection pipeline not available."
                        
                        return call_image_description(detection_pipeline, image_path, prompt)
                        
                    except Exception as e:
                        return f"Error processing image: {str(e)}"
                
                desc_submit_btn.click(
                    process_image_description,
                    inputs=[desc_image_input, desc_prompt_input],
                    outputs=[desc_output]
                )
            
            # Object Detection Tab
            with gr.TabItem("🎯 Object Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        det_image_input = gr.Image(label="Upload Image", type="filepath")
                        det_submit_btn = gr.Button("🔍 Detect Objects", variant="primary")
                        
                    with gr.Column(scale=1):
                        det_output = gr.Textbox(label="Detection Results", lines=15, max_lines=20)
                        
                        # Detection results viewer
                        det_results_gallery = gr.Gallery(
                            label="Detection Results Visualization",
                            show_label=True,
                            elem_id="detection_gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                
                def process_object_detection(image_path):
                    if not image_path:
                        return "Please upload an image first.", []
                    
                    try:
                        if detection_pipeline is None:
                            return "Object detection pipeline not available.", []
                        
                        # Process the image
                        response = call_object_detection(detection_pipeline, image_path)
                        
                        # Get the saved visualization images
                        gallery_images = []
                        if hasattr(detection_pipeline, 'output_dir'):
                            import glob
                            # Look for saved visualization images
                            viz_pattern = os.path.join(detection_pipeline.output_dir, "visualizations", "*.jpg")
                            viz_images = glob.glob(viz_pattern)
                            viz_pattern_png = os.path.join(detection_pipeline.output_dir, "visualizations", "*.png")
                            viz_images.extend(glob.glob(viz_pattern_png))
                            
                            # Sort by modification time (newest first)
                            viz_images.sort(key=os.path.getmtime, reverse=True)
                            gallery_images = viz_images[:10]  # Show up to 10 most recent images
                        
                        return response, gallery_images
                        
                    except Exception as e:
                        return f"Error processing image: {str(e)}", []
                
                det_submit_btn.click(
                    process_object_detection,
                    inputs=[det_image_input],
                    outputs=[det_output, det_results_gallery]
                )
            
            # Video Processing Tab  
            with gr.TabItem("🎥 Video Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video")
                        video_mode = gr.Radio(
                            choices=["description", "detection"],
                            value="description",
                            label="Analysis Mode",
                            info="Description: Get video description | Detection: Detect objects in video frames"
                        )
                        video_submit_btn = gr.Button("🎬 Analyze Video", variant="primary")
                        
                    with gr.Column(scale=1):
                        video_output = gr.Textbox(label="Analysis Results", lines=15, max_lines=20)
                        
                        # Video detection results viewer
                        video_results_gallery = gr.Gallery(
                            label="Video Analysis Results",
                            show_label=True,
                            elem_id="video_gallery", 
                            columns=3,
                            rows=2,
                            height="auto",
                            visible=False
                        )
                
                def process_video_analysis(video_path, mode):
                    if not video_path:
                        return "Please upload a video first.", []
                    
                    try:
                        if detection_pipeline is None:
                            return "Object detection pipeline not available.", []
                        
                        # Process the video
                        response = call_video_detection(detection_pipeline, video_path, mode)
                        
                        # If in detection mode, get saved visualization images
                        gallery_images = []
                        if mode == "detection" and hasattr(detection_pipeline, 'output_dir'):
                            import glob
                            # Look for saved visualization images
                            viz_pattern = os.path.join(detection_pipeline.output_dir, "visualizations", "*.jpg")
                            viz_images = glob.glob(viz_pattern)
                            viz_pattern_png = os.path.join(detection_pipeline.output_dir, "visualizations", "*.png")
                            viz_images.extend(glob.glob(viz_pattern_png))
                            
                            # Sort by modification time (newest first)
                            viz_images.sort(key=os.path.getmtime, reverse=True)
                            gallery_images = viz_images[:15]  # Show up to 15 most recent images for video
                        
                        return response, gallery_images
                        
                    except Exception as e:
                        return f"Error processing video: {str(e)}", []
                
                def update_video_gallery_visibility(mode):
                    return gr.update(visible=(mode == "detection"))
                
                video_submit_btn.click(
                    process_video_analysis,
                    inputs=[video_input, video_mode],
                    outputs=[video_output, video_results_gallery]
                )
                
                video_mode.change(
                    update_video_gallery_visibility,
                    inputs=[video_mode],
                    outputs=[video_results_gallery]
                )

#         gr.Markdown("""\
# <font size=2>Note: This demo is governed by the original license of Qwen3-VL. \
# We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
# including hate speech, violence, pornography, deception, etc. \
# (注：本演示受 Qwen3-VL 的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
# 包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, processor, backend, detection_pipeline = _load_model_processor(args)
    _launch_demo(args, model, processor, backend, detection_pipeline)


if __name__ == '__main__':
    main()

#python gradio_vlm.py --backend hf --enable-object-detection --detection-output-dir /tmp/detection_results