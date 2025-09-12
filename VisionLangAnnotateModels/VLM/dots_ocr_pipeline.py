#!/usr/bin/env python3
"""
Dots OCR Pipeline for PDF Processing

This module provides functionality to perform OCR on PDF files using the dots.ocr model
while preserving the original document format and structure.

Based on: https://github.com/rednote-hilab/dots.ocr
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime

# Core dependencies
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer
    )
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install torch transformers pillow numpy")
    sys.exit(1)

# Additional dependencies for dots.ocr
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen_vl_utils not found. This is required for dots.ocr.")
    print("Please install the dots.ocr package: pip install -e .")
    process_vision_info = None

# Define prompt modes based on official dots.ocr documentation
PROMPT_MODES = {
    "layout_all": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
   - Picture: For the 'Picture' category, the text field should be omitted.
   - Formula: Format its text as LaTeX.
   - Table: Format its text as HTML.
   - All Others (Text, Title, etc.): Format their text as Markdown.
4. Constraints:
   - The output text must be the original text from the image, with no translation.
   - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.""",
    
    "layout_only": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
    
    "ocr_only": """Extract the text content from this image."""
}

try:
    from dots_ocr.utils import dict_promptmode_to_prompt
    # Use official prompts if available, otherwise use our fallback
except ImportError:
    print("dots_ocr.utils not found. Using fallback prompt handling.")
    dict_promptmode_to_prompt = None

# PDF processing
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF")
    fitz = None

# Optional dependencies for rendering
try:
    import markdown
except ImportError:
    markdown = None

try:
    import weasyprint
except ImportError:
    weasyprint = None

try:
    from weasyprint import HTML, CSS
except ImportError:
    HTML = None
    CSS = None

import io
import base64

class DotsOCRPipeline:
    """
    A pipeline for performing OCR on PDF documents using dots.ocr model
    while preserving document structure and format.
    """
    
    def __init__(self, 
                 model_path: str = "rednote-hilab/dots.ocr",
                 device: str = "auto",
                 output_dir: str = "./dots_ocr_results"):
        """
        Initialize the DotsOCR pipeline.
        
        Args:
            model_path: Path to the downloaded dots.ocr model
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            output_dir: Directory to save output files
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.device = self._setup_device(device)
        
        # Create output directories
        self._setup_output_directories()
        
        # Initialize model and processor using the official pattern
        print(f"Loading dots.ocr model from {model_path}...")
        
        # Load processor following official example
        # Try different processor loading approaches
        try:
            # First try AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            print(f"Loaded AutoProcessor: {type(self.processor)}")
            
            # Check if it's actually a vision processor
            if 'Tokenizer' in self.processor.__class__.__name__:
                print("WARNING: AutoProcessor loaded a tokenizer, trying Qwen2VLProcessor...")
                raise ValueError("Need vision processor, not tokenizer")
                
        except Exception as e:
            print(f"AutoProcessor failed: {e}")
            try:
                # Try Qwen2VLProcessor directly
                from transformers import Qwen2VLProcessor
                self.processor = Qwen2VLProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                print(f"Loaded Qwen2VLProcessor: {type(self.processor)}")
            except Exception as e2:
                print(f"Qwen2VLProcessor failed: {e2}")
                try:
                    # Try loading from a known Qwen2VL model
                    from transformers import Qwen2VLProcessor
                    self.processor = Qwen2VLProcessor.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct",
                        trust_remote_code=True
                    )
                    print(f"Loaded fallback Qwen2VLProcessor: {type(self.processor)}")
                except Exception as e3:
                    print(f"All processor loading attempts failed: {e3}")
                    raise e3
        
        # Load model following official example
        try:
            if self.device.type == 'cuda':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # For CPU, skip flash_attention_2 and device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
            print(f"Model loaded successfully on device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        print(f"Model loaded successfully on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "extracted_pages").mkdir(exist_ok=True)
        (self.output_dir / "rendered_documents").mkdir(exist_ok=True)
        (self.output_dir / "raw_ocr_results").mkdir(exist_ok=True)
    
    def extract_pages_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract pages from PDF as images for OCR processing.
        
        Args:
            pdf_path: Path to the input PDF file
            
        Returns:
            List of dictionaries containing page information
        """
        pdf_document = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Save page image
            page_filename = f"page_{page_num + 1:03d}.png"
            page_path = self.output_dir / "extracted_pages" / page_filename
            image.save(page_path)
            
            pages_data.append({
                'page_number': page_num + 1,
                'image': image,
                'image_path': str(page_path),
                'width': image.width,
                'height': image.height
            })
        
        pdf_document.close()
        return pages_data
    
    def perform_ocr_on_page(self, image: Image.Image, page_number: int) -> Dict[str, Any]:
        """
        Perform OCR on a single page using dots.ocr model.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for reference
            
        Returns:
            Dictionary containing OCR results with structure information
        """
        print(f"Processing page {page_number} with dots.ocr...")
        
        start_time = time.time()
        
        # Use the official DotsOCR prompt for layout analysis
        prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
        
        try:
            print(f"DEBUG: Starting OCR processing for page {page_number}")
            print(f"DEBUG: Prompt type: {type(prompt)}, content: {prompt[:100]}...")
            print(f"DEBUG: Image type: {type(image)}")
            
            # Save image temporarily for dots.ocr processing
            temp_image_path = os.path.join(self.output_dir, "temp_images", f"temp_page_{page_number}.png")
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
            image.save(temp_image_path)
            
            # Follow the official DotsOCR example pattern exactly
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": temp_image_path
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            print(f"DEBUG: Messages created successfully with temp image: {temp_image_path}")
            
            # Preparation for inference following official example
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                print(f"DEBUG: Chat template applied successfully, type: {type(text)}")
            except Exception as template_error:
                print(f"DEBUG: Chat template error: {template_error}")
                # Fallback to manual template construction
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                print(f"DEBUG: Using manual template fallback")
            
            # Process vision info following official example
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                print(f"DEBUG: process_vision_info returned - images: {type(image_inputs)}, videos: {type(video_inputs)}")
                print(f"DEBUG: Processor type: {type(self.processor)}")
                print(f"DEBUG: Processor class: {self.processor.__class__.__name__}")
                
                # Try the exact same call as the official example
                try:
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    print("DEBUG: Vision info processed successfully")
                except Exception as proc_error:
                    print(f"DEBUG: Processor error with official params: {proc_error}")
                    # Try without videos parameter
                    try:
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        print("DEBUG: Vision info processed without videos")
                    except Exception as proc_error2:
                        print(f"DEBUG: Processor error without videos: {proc_error2}")
                        # Try with different parameter name
                        try:
                            inputs = self.processor(
                                text=[text],
                                image=image_inputs,
                                padding=True,
                                return_tensors="pt",
                            )
                            print("DEBUG: Vision info processed with 'image' param")
                        except Exception as proc_error3:
                            print(f"DEBUG: All processor attempts failed: {proc_error3}")
                            raise proc_error3
            else:
                # Fallback if process_vision_info is not available
                print("DEBUG: process_vision_info not available, using fallback")
                # Try different parameter names that might work with this processor
                try:
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    )
                    print("DEBUG: Fallback with 'images' parameter worked")
                except Exception as fallback_error:
                    print(f"DEBUG: Fallback with 'images' failed: {fallback_error}")
                    # Try with different parameter names
                    try:
                        inputs = self.processor(
                            text=[text],
                            image=[image],
                            padding=True,
                            return_tensors="pt"
                        )
                        print("DEBUG: Fallback with 'image' parameter worked")
                    except Exception as fallback_error2:
                        print(f"DEBUG: Fallback with 'image' failed: {fallback_error2}")
                        # Last resort - just text
                        inputs = self.processor(
                            text=[text],
                            padding=True,
                            return_tensors="pt"
                        )
                        print("DEBUG: Using text-only fallback")
            
            inputs = inputs.to(self.device)
            
            print(f"DEBUG: Inputs created successfully")
            
            # Inference: Generation of the output following official example
            generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Handle the output text
            if isinstance(output_text, list):
                generated_text = output_text[0] if output_text else ""
            else:
                generated_text = str(output_text)
            
            print(f"DEBUG: Generated text length: {len(generated_text)}")
            
            # Clean up temporary image
            try:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp image: {cleanup_error}")
            
            # Parse the JSON response
            try:
                ocr_result = json.loads(generated_text.strip())
            except json.JSONDecodeError:
                # Fallback: create a simple structure if JSON parsing fails
                ocr_result = {
                    "elements": [{
                        "type": "text",
                        "bbox": [0, 0, image.width, image.height],
                        "content": generated_text.strip(),
                        "reading_order": 1
                    }]
                }
            
        except Exception as e:
            print(f"OCR processing failed: {e}")
            ocr_result = {
                "elements": [],
                "error": str(e)
            }
            generated_text = f"Error: {str(e)}"
        
        processing_time = time.time() - start_time
        
        # Parse the structured output from dots.ocr only if no error occurred
        if "Error:" not in generated_text:
            parsed_result = self._parse_dots_ocr_output(generated_text)
        else:
            # Create a simple parsed result for errors
            parsed_result = {
                "text_blocks": [],
                "tables": [],
                "headers": [],
                "lists": [],
                "formulas": [],
                "images": [],
                "reading_order": [],
                "error": generated_text
            }
        
        result = {
            'page_number': page_number,
            'raw_text': generated_text,
            'parsed_content': parsed_result,
            'ocr_result': ocr_result,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save raw OCR result
        raw_filename = f"page_{page_number:03d}_raw_ocr.json"
        raw_path = self.output_dir / "raw_ocr_results" / raw_filename
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Page {page_number} processed in {processing_time:.2f}s")
        return result
    
    def _parse_dots_ocr_output(self, ocr_text: str) -> Dict[str, Any]:
        """
        Parse the structured output from dots.ocr model.
        
        Args:
            ocr_text: Raw OCR output text
            
        Returns:
            Parsed structure with text blocks, tables, formulas, etc.
        """
        # dots.ocr typically outputs structured text with markdown-like formatting
        # This is a basic parser - you may need to enhance based on actual output format
        
        parsed = {
            'text_blocks': [],
            'tables': [],
            'formulas': [],
            'headers': [],
            'lists': [],
            'reading_order': []
        }
        
        lines = ocr_text.split('\n')
        current_block = {'type': 'text', 'content': '', 'position': 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Detect headers (lines starting with #)
            if line.startswith('#'):
                if current_block['content']:
                    parsed['text_blocks'].append(current_block)
                    parsed['reading_order'].append(len(parsed['text_blocks']) - 1)
                
                header_level = len(line) - len(line.lstrip('#'))
                parsed['headers'].append({
                    'level': header_level,
                    'text': line.lstrip('# '),
                    'position': i
                })
                current_block = {'type': 'text', 'content': '', 'position': i + 1}
            
            # Detect tables (lines with |)
            elif '|' in line and line.count('|') >= 2:
                if current_block['content']:
                    parsed['text_blocks'].append(current_block)
                    parsed['reading_order'].append(len(parsed['text_blocks']) - 1)
                
                # Simple table detection - enhance as needed
                table_content = [line]
                j = i + 1
                while j < len(lines) and '|' in lines[j]:
                    table_content.append(lines[j].strip())
                    j += 1
                
                parsed['tables'].append({
                    'content': table_content,
                    'position': i
                })
                current_block = {'type': 'text', 'content': '', 'position': j}
            
            # Detect formulas (lines with $ or mathematical symbols)
            elif '$' in line or any(symbol in line for symbol in ['∑', '∫', '√', '±', '≤', '≥']):
                parsed['formulas'].append({
                    'content': line,
                    'position': i
                })
            
            # Detect lists
            elif line.startswith(('-', '*', '+')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                if current_block['type'] != 'list':
                    if current_block['content']:
                        parsed['text_blocks'].append(current_block)
                        parsed['reading_order'].append(len(parsed['text_blocks']) - 1)
                    current_block = {'type': 'list', 'content': [line], 'position': i}
                else:
                    current_block['content'].append(line)
            
            # Regular text
            else:
                if current_block['type'] == 'list':
                    parsed['lists'].append(current_block)
                    current_block = {'type': 'text', 'content': line, 'position': i}
                else:
                    current_block['content'] += '\n' + line if current_block['content'] else line
        
        # Add the last block
        if current_block['content']:
            if current_block['type'] == 'list':
                parsed['lists'].append(current_block)
            else:
                parsed['text_blocks'].append(current_block)
                parsed['reading_order'].append(len(parsed['text_blocks']) - 1)
        
        return parsed
    
    def process_pdf(self, pdf_path: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Process a complete PDF file with OCR and format preservation.
        
        Args:
            pdf_path: Path to the input PDF file
            output_format: Output format ('markdown' or 'pdf')
            
        Returns:
            Dictionary containing processing results and output paths
        """
        print(f"Starting OCR processing of {pdf_path}...")
        
        # Extract pages from PDF
        pages_data = self.extract_pages_from_pdf(pdf_path)
        print(f"Extracted {len(pages_data)} pages from PDF")
        
        # Process each page with OCR
        ocr_results = []
        for page_data in pages_data:
            result = self.perform_ocr_on_page(
                page_data['image'], 
                page_data['page_number']
            )
            ocr_results.append(result)
        
        # Combine results and render output document
        combined_content = self._combine_page_results(ocr_results)
        
        # Generate output filename
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_paths = {}
        
        if output_format.lower() == "markdown":
            markdown_content = self._render_to_markdown(combined_content)
            markdown_path = self.output_dir / "rendered_documents" / f"{pdf_name}_{timestamp}.md"
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            output_paths['markdown'] = str(markdown_path)
            print(f"Markdown output saved to: {markdown_path}")
        
        elif output_format.lower() == "pdf" and HTML is not None:
            # Convert to HTML first, then to PDF
            html_content = self._render_to_html(combined_content)
            html_path = self.output_dir / "rendered_documents" / f"{pdf_name}_{timestamp}.html"
            pdf_output_path = self.output_dir / "rendered_documents" / f"{pdf_name}_{timestamp}_ocr.pdf"
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Convert HTML to PDF
            HTML(string=html_content).write_pdf(str(pdf_output_path))
            
            output_paths['html'] = str(html_path)
            output_paths['pdf'] = str(pdf_output_path)
            print(f"PDF output saved to: {pdf_output_path}")
        
        # Save combined results as JSON
        json_path = self.output_dir / "rendered_documents" / f"{pdf_name}_{timestamp}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'input_pdf': pdf_path,
                'pages_processed': len(pages_data),
                'ocr_results': ocr_results,
                'combined_content': combined_content,
                'output_paths': output_paths,
                'processing_timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        output_paths['json'] = str(json_path)
        
        return {
            'input_pdf': pdf_path,
            'pages_processed': len(pages_data),
            'output_paths': output_paths,
            'total_processing_time': sum(r['processing_time'] for r in ocr_results)
        }
    
    def _combine_page_results(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine OCR results from multiple pages into a single document structure.
        """
        combined = {
            'title': 'OCR Extracted Document',
            'pages': len(ocr_results),
            'sections': [],
            'tables': [],
            'formulas': [],
            'full_text': ''
        }
        
        for result in ocr_results:
            page_num = result['page_number']
            parsed = result['parsed_content']
            
            # Add page separator
            combined['sections'].append({
                'type': 'page_break',
                'page_number': page_num,
                'content': f"\n\n--- Page {page_num} ---\n\n"
            })
            
            # Add headers
            for header in parsed['headers']:
                combined['sections'].append({
                    'type': 'header',
                    'level': header['level'],
                    'content': header['text'],
                    'page': page_num
                })
            
            # Add text blocks
            for block in parsed['text_blocks']:
                combined['sections'].append({
                    'type': 'text',
                    'content': block['content'],
                    'page': page_num
                })
            
            # Add tables
            for table in parsed['tables']:
                combined['tables'].append({
                    'content': table['content'],
                    'page': page_num
                })
                combined['sections'].append({
                    'type': 'table',
                    'content': '\n'.join(table['content']),
                    'page': page_num
                })
            
            # Add formulas
            for formula in parsed['formulas']:
                combined['formulas'].append({
                    'content': formula['content'],
                    'page': page_num
                })
                combined['sections'].append({
                    'type': 'formula',
                    'content': formula['content'],
                    'page': page_num
                })
            
            # Add lists
            for list_item in parsed['lists']:
                combined['sections'].append({
                    'type': 'list',
                    'content': '\n'.join(list_item['content']),
                    'page': page_num
                })
        
        # Generate full text
        combined['full_text'] = '\n\n'.join(
            section['content'] for section in combined['sections']
        )
        
        return combined
    
    def _render_to_markdown(self, combined_content: Dict[str, Any]) -> str:
        """
        Render combined content to Markdown format.
        """
        markdown_lines = []
        markdown_lines.append(f"# {combined_content['title']}\n")
        markdown_lines.append(f"*Extracted from {combined_content['pages']} pages using dots.ocr*\n")
        
        for section in combined_content['sections']:
            if section['type'] == 'page_break':
                markdown_lines.append(f"\n{section['content']}")
            elif section['type'] == 'header':
                header_prefix = '#' * (section['level'] + 1)
                markdown_lines.append(f"{header_prefix} {section['content']}\n")
            elif section['type'] == 'text':
                markdown_lines.append(f"{section['content']}\n")
            elif section['type'] == 'table':
                markdown_lines.append(f"\n{section['content']}\n")
            elif section['type'] == 'formula':
                markdown_lines.append(f"\n$$\n{section['content']}\n$$\n")
            elif section['type'] == 'list':
                markdown_lines.append(f"{section['content']}\n")
        
        return '\n'.join(markdown_lines)
    
    def _render_to_html(self, combined_content: Dict[str, Any]) -> str:
        """
        Render combined content to HTML format.
        """
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            f"<title>{combined_content['title']}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".formula { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }",
            ".page-break { border-top: 2px solid #ccc; margin: 30px 0; padding-top: 20px; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        html_lines.append(f"<h1>{combined_content['title']}</h1>")
        html_lines.append(f"<p><em>Extracted from {combined_content['pages']} pages using dots.ocr</em></p>")
        
        for section in combined_content['sections']:
            if section['type'] == 'page_break':
                html_lines.append(f"<div class='page-break'><strong>{section['content'].strip()}</strong></div>")
            elif section['type'] == 'header':
                level = min(section['level'] + 1, 6)
                html_lines.append(f"<h{level}>{section['content']}</h{level}>")
            elif section['type'] == 'text':
                paragraphs = section['content'].split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        html_lines.append(f"<p>{para.strip()}</p>")
            elif section['type'] == 'table':
                html_lines.append("<table>")
                table_lines = section['content'].split('\n')
                for i, line in enumerate(table_lines):
                    if '|' in line:
                        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                        tag = 'th' if i == 0 else 'td'
                        html_lines.append("<tr>")
                        for cell in cells:
                            html_lines.append(f"<{tag}>{cell}</{tag}>")
                        html_lines.append("</tr>")
                html_lines.append("</table>")
            elif section['type'] == 'formula':
                html_lines.append(f"<div class='formula'>{section['content']}</div>")
            elif section['type'] == 'list':
                html_lines.append("<ul>")
                list_items = section['content'].split('\n')
                for item in list_items:
                    if item.strip():
                        clean_item = item.strip().lstrip('-*+ ').lstrip('0123456789. ')
                        html_lines.append(f"<li>{clean_item}</li>")
                html_lines.append("</ul>")
        
        html_lines.extend(["</body>", "</html>"])
        return '\n'.join(html_lines)

def main():
    """
    Example usage of the DotsOCR pipeline.
    """
    # Initialize the pipeline
    pipeline = DotsOCRPipeline(
        model_path="/home/lkk/Developer/dots.ocr/weights/DotsOCR",
        device="auto",
        output_dir="./dots_ocr_results"
    )
    
    # Example: Process a PDF file
    pdf_path = "example_document.pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        # Process with markdown output
        results = pipeline.process_pdf(pdf_path, output_format="markdown")
        print(f"\nProcessing completed!")
        print(f"Pages processed: {results['pages_processed']}")
        print(f"Total processing time: {results['total_processing_time']:.2f}s")
        print(f"Output files: {results['output_paths']}")
        
        # Also generate PDF output if libraries are available
        if HTML is not None:
            pdf_results = pipeline.process_pdf(pdf_path, output_format="pdf")
            print(f"PDF output: {pdf_results['output_paths']}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path.")

if __name__ == "__main__":
    main()