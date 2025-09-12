#!/usr/bin/env python3
"""
Qwen Math OCR Pipeline

A specialized OCR pipeline for extracting math questions from worksheet images
using Qwen2.5-VL-7B-Instruct model.

Based on the structure of dots_ocr_pipeline.py but optimized for math content.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageTemplate, Frame, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
    from reportlab.pdfgen import canvas
    import textwrap
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class QwenMathOCRPipeline:
    """
    Math OCR pipeline using Qwen2.5-VL-7B-Instruct for extracting
    structured math questions from worksheet images.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "auto",
                 torch_dtype = torch.bfloat16):
        """
        Initialize the Qwen Math OCR pipeline.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run the model on ('auto', 'cuda', 'cpu')
            torch_dtype: PyTorch data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        
        # Math-specific prompts
        self.math_extraction_prompt = self._get_math_extraction_prompt()
        self.structured_prompt = self._get_structured_prompt()
        
    def _get_math_extraction_prompt(self) -> str:
        """
        Get the optimized prompt for math worksheet OCR.
        """
        return """Extract all text content from this math worksheet image. Please provide a structured extraction with the following format:

For each question found:
- **Question Number**: (e.g., "Question 1", "Problem 2", etc.)
- **Problem Text**: Complete text of the math problem
- **Mathematical Elements**: Numbers, equations, symbols, and expressions
- **Answer Space**: Location and format of answer blanks

Preserve all mathematical notation, numbers, and symbols exactly as they appear. Maintain the original structure and formatting of the worksheet.

Format the output as clear, structured text with proper question separation."""
    
    def _get_structured_prompt(self) -> str:
        """
        Get the structured JSON output prompt for math worksheets with spatial layout information.
        """
        return """Extract all math questions from this worksheet image with spatial layout information and format as JSON:

{
  "worksheet_info": {
    "total_questions": <number>,
    "subject": "mathematics",
    "difficulty_level": "<estimated level>",
    "page_dimensions": {
      "width": "<estimated width>",
      "height": "<estimated height>"
    }
  },
  "questions": [
    {
      "question_number": "<number or identifier>",
      "problem_text": "<complete problem statement>",
      "mathematical_expressions": ["<list of equations/numbers>"],
      "answer_format": "<type of answer expected>",
      "spatial_info": {
        "bounding_box": {
          "x": "<left position as percentage of page width>",
          "y": "<top position as percentage of page height>",
          "width": "<width as percentage of page width>",
          "height": "<height as percentage of page height>"
        },
        "text_properties": {
          "font_size": "<estimated relative size: small/medium/large>",
          "alignment": "<left/center/right>",
          "line_spacing": "<tight/normal/loose>"
        }
      },
      "answer_space_info": {
        "location": {
          "x": "<percentage>",
          "y": "<percentage>",
          "width": "<percentage>",
          "height": "<percentage>"
        },
        "type": "<blank_line/box/multiple_choice>"
      }
    }
  ]
}

Analyze the visual layout carefully. Estimate positions as percentages of the total page dimensions. Pay attention to text alignment, spacing, and the relative positioning of questions and answer spaces."""
    
    def load_model(self) -> None:
        """
        Load the Qwen2.5-VL model and processor.
        """
        print(f"Loading Qwen Math OCR model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Load model with proper configuration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            print(f"Model device: {self.model.device if hasattr(self.model, 'device') else 'N/A'}")
            print(f"Processor type: {type(self.processor).__name__}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def process_image(self, 
                     image: Union[str, Path, Image.Image],
                     prompt_type: str = "structured",
                     max_new_tokens: int = 2048,
                     temperature: float = 0.1) -> Dict[str, Any]:
        """
        Process a single image for math OCR extraction.
        
        Args:
            image: Path to image file or PIL Image object
            prompt_type: Type of prompt ('structured' for JSON, 'extraction' for text)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature (lower = more deterministic)
            
        Returns:
            Dictionary containing OCR results and metadata
        """
        if self.model is None or self.processor is None:
            self.load_model()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = Image.open(image_path)
        else:
            image_path = "PIL_Image"
        
        # Select prompt
        if prompt_type == "structured":
            prompt = self.structured_prompt
        else:
            prompt = self.math_extraction_prompt
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.size}")
        print(f"Prompt type: {prompt_type}")
        
        start_time = time.time()
        
        try:
            # Prepare messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process vision info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            processing_time = time.time() - start_time
            
            # Parse response if structured format requested
            parsed_content = None
            if prompt_type == "structured":
                try:
                    # Try to extract JSON from response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        parsed_content = json.loads(json_str)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Could not parse JSON response: {e}")
                    parsed_content = None
            
            result = {
                "success": True,
                "image_path": image_path,
                "image_size": image.size,
                "prompt_type": prompt_type,
                "processing_time": processing_time,
                "raw_response": response,
                "parsed_content": parsed_content,
                "model_info": {
                    "model_name": self.model_name,
                    "processor_type": type(self.processor).__name__,
                    "device": str(self.model.device),
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                },
                "timestamp": time.strftime("%Y%m%d_%H%M%S")
            }
            
            print(f"OCR completed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "processing_time": time.time() - start_time,
                "timestamp": time.strftime("%Y%m%d_%H%M%S")
            }
            print(f"Error during OCR processing: {e}")
            return error_result
    
    def process_multiple_images(self, 
                               image_paths: List[Union[str, Path]],
                               prompt_type: str = "structured",
                               **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple images for math OCR extraction.
        
        Args:
            image_paths: List of paths to image files
            prompt_type: Type of prompt to use
            **kwargs: Additional arguments for process_image
            
        Returns:
            List of OCR results for each image
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n--- Processing image {i}/{len(image_paths)} ---")
            result = self.process_image(image_path, prompt_type, **kwargs)
            results.append(result)
            
            if not result["success"]:
                print(f"Failed to process {image_path}: {result.get('error', 'Unknown error')}")
        
        return results
    
    def save_results(self, 
                    results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                    output_dir: str = "qwen_math_ocr_results") -> Dict[str, str]:
        """
        Save OCR results to files.
        
        Args:
            results: Single result dict or list of result dicts
            output_dir: Directory to save results
            
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (output_path / "raw_responses").mkdir(exist_ok=True)
        (output_path / "structured_results").mkdir(exist_ok=True)
        (output_path / "summaries").mkdir(exist_ok=True)
        
        if isinstance(results, dict):
            results = [results]
        
        saved_files = {
            "raw_responses": [],
            "structured_results": [],
            "summary": None
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        for i, result in enumerate(results):
            if not result.get("success", False):
                continue
                
            base_name = f"math_ocr_{i+1:03d}_{timestamp}"
            
            # Save raw response
            raw_file = output_path / "raw_responses" / f"{base_name}.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"Image: {result.get('image_path', 'Unknown')}\n")
                f.write(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
                f.write(f"Prompt Type: {result.get('prompt_type', 'Unknown')}\n")
                f.write("\n" + "="*50 + "\n")
                f.write(result.get('raw_response', ''))
            saved_files["raw_responses"].append(str(raw_file))
            
            # Save structured result if available
            if result.get('parsed_content'):
                json_file = output_path / "structured_results" / f"{base_name}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result['parsed_content'], f, indent=2, ensure_ascii=False)
                saved_files["structured_results"].append(str(json_file))
        
        # Save summary
        summary_file = output_path / "summaries" / f"math_ocr_summary_{timestamp}.json"
        summary_data = {
            "total_images": len(results),
            "successful_extractions": len([r for r in results if r.get("success", False)]),
            "failed_extractions": len([r for r in results if not r.get("success", False)]),
            "total_processing_time": sum(r.get("processing_time", 0) for r in results),
            "average_processing_time": sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0,
            "model_info": results[0].get("model_info", {}) if results else {},
            "timestamp": timestamp,
            "results_overview": [
                {
                    "image_path": r.get("image_path", "Unknown"),
                    "success": r.get("success", False),
                    "processing_time": r.get("processing_time", 0),
                    "has_structured_data": bool(r.get("parsed_content"))
                }
                for r in results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        saved_files["summary"] = str(summary_file)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Summary: {summary_file}")
        
        return saved_files
    
    def _load_results_from_folder(self, folder_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load OCR results from a previously saved folder structure.
        
        Args:
            folder_path: Path to folder containing structured_results and/or raw_responses
            
        Returns:
            List of result dictionaries ordered by page number or file timestamp
        """
        folder_path = Path(folder_path)
        results = []
        
        # Try to load from structured_results first
        structured_dir = folder_path / "structured_results"
        if structured_dir.exists():
            json_files = list(structured_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        parsed_content = json.load(f)
                    
                    result = {
                        "success": True,
                        "parsed_content": parsed_content,
                        "image_path": json_file.stem,
                        "processing_time": 0,
                        "file_path": json_file,
                        "file_mtime": json_file.stat().st_mtime
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
        
        # If no structured results, try raw responses
        if not results:
            raw_dir = folder_path / "raw_responses"
            if raw_dir.exists():
                txt_files = list(raw_dir.glob("*.txt"))
                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            raw_response = f.read()
                        
                        result = {
                            "success": True,
                            "raw_response": raw_response,
                            "image_path": txt_file.stem,
                            "processing_time": 0,
                            "file_path": txt_file,
                            "file_mtime": txt_file.stat().st_mtime
                        }
                        results.append(result)
                    except Exception as e:
                        print(f"Warning: Could not load {txt_file}: {e}")
        
        if not results:
            raise ValueError(f"No valid OCR results found in {folder_path}")
        
        # Sort results by page number if available, otherwise by file timestamp
        def get_sort_key(result):
            # Try to extract page number from filename
            filename = result["image_path"]
            import re
            page_match = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
            if page_match:
                return (0, int(page_match.group(1)))  # Priority 0 for page numbers
            
            # Try to extract number from filename
            number_match = re.search(r'(\d+)', filename)
            if number_match:
                return (1, int(number_match.group(1)))  # Priority 1 for general numbers
            
            # Fall back to file modification time
            return (2, result["file_mtime"])  # Priority 2 for timestamps
        
        results.sort(key=get_sort_key)
        
        return results
    
    def visualize_results(self, 
                         results: Union[Dict[str, Any], List[Dict[str, Any]], str, Path],
                         output_dir: str = "qwen_math_ocr_results",
                         output_format: str = "pdf",
                         preserve_format: bool = True) -> str:
        """
        Create a formatted document visualizing only the problem text from extracted questions.
        
        Args:
            results: OCR results from process_image/process_multiple_images, or folder path containing saved files
            output_dir: Directory to save the visualization
            output_format: Output format ('pdf' or 'html')
            preserve_format: If True, preserves original layout and formatting. If False, outputs plain text only.
            
        Returns:
            Path to the generated visualization file
        """
        # Handle folder path input
        if isinstance(results, (str, Path)):
            results = self._load_results_from_folder(results)
        elif isinstance(results, dict):
            results = [results]
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == "pdf":
            return self._create_pdf_visualization(results, output_path, timestamp, preserve_format)
        elif output_format.lower() == "html":
            return self._create_html_visualization(results, output_path, timestamp, preserve_format)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _create_pdf_visualization(self, 
                                 results: List[Dict[str, Any]], 
                                 output_path: Path, 
                                 timestamp: str,
                                 preserve_format: bool = True) -> str:
        """
        Create a PDF visualization with optional format preservation.
        
        Args:
            preserve_format: If True, preserves original layout. If False, creates plain text output.
        """
        pdf_file = output_path / f"math_problems_layout_preserved_{timestamp}.pdf"
        
        # Create PDF document with custom page size if available
        page_size = A4  # default
        doc = SimpleDocTemplate(str(pdf_file), pagesize=page_size)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Add title
        story.append(Paragraph("Math Problems - Layout Preserved", title_style))
        story.append(Spacer(1, 20))
        
        question_counter = 1
        
        for result in results:
            if not result.get("success", False):
                continue
                
            # Process structured content if available
            parsed_content = result.get('parsed_content')
            if parsed_content and isinstance(parsed_content, dict):
                questions = parsed_content.get('questions', [])
                page_width, page_height = page_size
                
                for question in questions:
                    if not preserve_format:
                        # Simple plain text layout without spatial positioning
                        problem_text = question.get('problem_text', '')
                        if problem_text:
                            # Create simple styles for plain text
                            question_header_style = ParagraphStyle(
                                'PlainQuestionHeader',
                                parent=styles['Heading3'],
                                fontSize=12,
                                spaceAfter=10,
                                textColor=colors.darkblue
                            )
                            plain_text_style = ParagraphStyle(
                                'PlainText',
                                parent=styles['Normal'],
                                fontSize=11,
                                spaceAfter=30,  # Increased spacing between questions
                                alignment=TA_LEFT
                            )
                            
                            # Add question header and content
                            processed_text = self._process_latex_for_pdf(problem_text)
                            story.append(Paragraph(f"Question {question_counter}", question_header_style))
                            story.append(Paragraph(processed_text, plain_text_style))
                            story.append(Spacer(1, 20))  # Additional space between questions
                            question_counter += 1
                        continue
                    
                    # Original format-preserving logic
                    spatial_info = question.get('spatial_info', {})
                    bbox = spatial_info.get('bounding_box', {})
                    text_props = spatial_info.get('text_properties', {})
                    
                    # Calculate absolute positions from percentages
                    if bbox:
                        try:
                            x_pos = self._parse_percentage_value(bbox.get('x', 0)) / 100 * page_width
                            y_pos = self._parse_percentage_value(bbox.get('y', 0)) / 100 * page_height
                            width = self._parse_percentage_value(bbox.get('width', 100)) / 100 * page_width
                            height = self._parse_percentage_value(bbox.get('height', 10)) / 100 * page_height
                            
                            # Determine font size based on text properties
                            font_size_map = {'small': 9, 'medium': 11, 'large': 14}
                            font_size = font_size_map.get(text_props.get('font_size', 'medium'), 11)
                            
                            # Create custom style with positioning
                            custom_style = ParagraphStyle(
                                f'Question_{question.get("question_number", "")}',
                                parent=styles['Normal'],
                                fontSize=font_size,
                                alignment=self._get_alignment(text_props.get('alignment', 'left')),
                                leftIndent=x_pos if x_pos > 0 else 0,
                                spaceAfter=max(height * 0.1, 20)  # Increased minimum spacing between questions
                            )
                            
                            # Add question with preserved positioning
                            problem_text = question.get('problem_text', '')
                            if problem_text:
                                # Process LaTeX expressions for PDF rendering
                                processed_text = self._process_latex_for_pdf(problem_text)
                                story.append(Paragraph(processed_text, custom_style))
                                
                                # Add answer space if specified
                            answer_info = question.get('answer_space_info', {})
                            if answer_info:
                                self._add_answer_space(story, answer_info, page_width, page_height)
                        except (ValueError, TypeError):
                            # Fallback to default styling if spatial info is invalid
                            problem_text = question.get('problem_text', '')
                            if problem_text:
                                processed_text = self._process_latex_for_pdf(problem_text)
                                default_style = ParagraphStyle(
                                    'DefaultQuestion',
                                    parent=styles['Normal'],
                                    fontSize=11,
                                    spaceAfter=25  # Increased spacing between questions
                                )
                                story.append(Paragraph(processed_text, default_style))
                    else:
                        # No spatial info available, use default layout
                        problem_text = question.get('problem_text', '')
                        if problem_text:
                            processed_text = self._process_latex_for_pdf(problem_text)
                            default_style = ParagraphStyle(
                                'DefaultQuestion',
                                parent=styles['Normal'],
                                fontSize=11,
                                spaceAfter=25  # Increased spacing between questions
                            )
                            story.append(Paragraph(processed_text, default_style))
            
            else:
                # Fallback to raw response - extract problem-like content
                raw_response = result.get('raw_response', '')
                if raw_response:
                    # Try to extract question-like content from raw response
                    lines = raw_response.split('\n')
                    current_problem = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(('Subject:', 'Questions:', 'Level:', 'Processing')):
                            # Look for question indicators
                            if any(indicator in line.lower() for indicator in ['question', 'problem', 'solve', 'find', 'calculate']):
                                if current_problem:
                                    # Save previous problem
                                    problem_text = ' '.join(current_problem)
                                    # Define styles for fallback processing
                                    question_style = ParagraphStyle(
                                        'QuestionStyle',
                                        parent=styles['Heading2'],
                                        fontSize=12,
                                        spaceAfter=15,
                                        textColor=colors.darkblue
                                    )
                                    content_style = ParagraphStyle(
                                        'ContentStyle',
                                        parent=styles['Normal'],
                                        fontSize=11,
                                        spaceAfter=20,
                                        alignment=TA_LEFT
                                    )
                                    story.append(Paragraph(f"Problem {question_counter}", question_style))
                                    story.append(Paragraph(problem_text, content_style))
                                    question_counter += 1
                                    current_problem = []
                                current_problem.append(line)
                            elif current_problem:
                                current_problem.append(line)
                    
                    # Add last problem if exists
                    if current_problem:
                        problem_text = ' '.join(current_problem)
                        # Define styles for fallback processing
                        question_style = ParagraphStyle(
                            'QuestionStyle',
                            parent=styles['Heading2'],
                            fontSize=12,
                            spaceAfter=15,
                            textColor=colors.darkblue
                        )
                        content_style = ParagraphStyle(
                            'ContentStyle',
                            parent=styles['Normal'],
                            fontSize=11,
                            spaceAfter=20,
                            alignment=TA_LEFT
                        )
                        story.append(Paragraph(f"Problem {question_counter}", question_style))
                        story.append(Paragraph(problem_text, content_style))
                        question_counter += 1
        
        # Build PDF
        doc.build(story)
        print(f"PDF with problems only saved to: {pdf_file}")
        return str(pdf_file)
    
    def _parse_percentage_value(self, value: Union[str, int, float]) -> float:
        """Parse percentage value, handling both string percentages and numeric values."""
        if isinstance(value, str):
            if value.endswith('%'):
                return float(value[:-1])
            else:
                return float(value)
        return float(value)
    
    def _process_latex_for_pdf(self, text: str) -> str:
        """Convert LaTeX expressions to a format suitable for PDF rendering with improved mathematical notation."""
        import re
        
        # First, handle nested fractions and complex expressions
        def process_fraction(match):
            numerator = match.group(1)
            denominator = match.group(2)
            # For simple single numbers/variables, use superscript notation
            if len(numerator) <= 2 and len(denominator) <= 2 and numerator.isalnum() and denominator.isalnum():
                return f"{numerator}/{denominator}"
            else:
                return f"({numerator})/({denominator})"
        
        # Convert inline LaTeX \(...\) to preserve mathematical notation
        text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
        
        # Convert display LaTeX \[...\] to preserve mathematical notation
        text = re.sub(r'\\\[(.*?)\\\]', r'\n\1\n', text)
        
        # Enhanced LaTeX replacements with better mathematical formatting
        latex_replacements = {
            # Powers and subscripts
            r'\\\^\{([^}]+)\}': r'^(\1)',
            r'\_\{([^}]+)\}': r'_(\1)',
            # Roots
            r'\\sqrt\{([^}]+)\}': r'√(\1)',
            r'\\sqrt\[([^]]+)\]\{([^}]+)\}': r'\1√(\2)',
            # Mathematical operators and symbols
            r'\\sum': '∑',
            r'\\int': '∫',
            r'\\prod': '∏',
            r'\\lim': 'lim',
            r'\\infty': '∞',
            # Greek letters
            r'\\pi': 'π',
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\epsilon': 'ε',
            r'\\theta': 'θ',
            r'\\lambda': 'λ',
            r'\\mu': 'μ',
            r'\\sigma': 'σ',
            r'\\phi': 'φ',
            r'\\omega': 'ω',
            # Comparison operators
            r'\\leq': '≤',
            r'\\geq': '≥',
            r'\\neq': '≠',
            r'\\approx': '≈',
            r'\\equiv': '≡',
            # Arithmetic operators
            r'\\pm': '±',
            r'\\mp': '∓',
            r'\\times': '×',
            r'\\div': '÷',
            r'\\cdot': '·',
        }
        
        # Handle fractions with a more reliable approach
        # Process \frac{...}{...} patterns
        def find_and_replace_fractions(text):
            result = text
            # Look for \frac{ patterns and process them
            # Handle both single and double backslash cases
            while True:
                # Try to find \frac{ first (single backslash)
                start_single = result.find('\\frac{')
                # Try to find \\frac{ (double backslash in string representation)
                start_double = result.find('\\\\frac{')
                
                if start_single == -1 and start_double == -1:
                    break
                    
                # Use whichever comes first
                if start_single != -1 and (start_double == -1 or start_single < start_double):
                    start = start_single
                    frac_pattern = '\\frac{'
                else:
                    start = start_double
                    frac_pattern = '\\\\frac{'
                if start == -1:
                    break
                
                # Find the numerator (first {...})
                brace_start = start + len(frac_pattern)
                brace_count = 1
                num_end = brace_start
                
                while num_end < len(result) and brace_count > 0:
                    if result[num_end] == '{':
                        brace_count += 1
                    elif result[num_end] == '}':
                        brace_count -= 1
                    num_end += 1
                
                if brace_count > 0:  # Unmatched braces
                    break
                    
                numerator = result[brace_start:num_end-1]
                
                # Find the denominator (second {...})
                if num_end >= len(result) or result[num_end] != '{':
                    break
                    
                denom_start = num_end + 1
                brace_count = 1
                denom_end = denom_start
                
                while denom_end < len(result) and brace_count > 0:
                    if result[denom_end] == '{':
                        brace_count += 1
                    elif result[denom_end] == '}':
                        brace_count -= 1
                    denom_end += 1
                
                if brace_count > 0:  # Unmatched braces
                    break
                    
                denominator = result[denom_start:denom_end-1]
                
                # Replace the fraction with proper formatting
                # Clean up numerator and denominator first
                numerator = numerator.strip()
                denominator = denominator.strip()
                
                # For simple single character or number fractions, use simple format
                if (len(numerator) <= 2 and len(denominator) <= 2 and 
                    numerator.replace(' ', '').replace('-', '').isalnum() and 
                    denominator.replace(' ', '').replace('-', '').isalnum()):
                    replacement = f"{numerator}/{denominator}"
                else:
                    # For complex fractions, only add parentheses if they contain operators or spaces
                    num_needs_parens = any(op in numerator for op in ['+', '-', '*', '/', ' ']) and len(numerator) > 1
                    denom_needs_parens = any(op in denominator for op in ['+', '-', '*', '/', ' ']) and len(denominator) > 1
                    
                    if num_needs_parens and denom_needs_parens:
                        replacement = f"({numerator})/({denominator})"
                    elif num_needs_parens:
                        replacement = f"({numerator})/{denominator}"
                    elif denom_needs_parens:
                        replacement = f"{numerator}/({denominator})"
                    else:
                        replacement = f"{numerator}/{denominator}"
                    
                result = result[:start] + replacement + result[denom_end:]
            
            return result
        
        text = find_and_replace_fractions(text)
        
        # Apply other replacements
        for pattern, replacement in latex_replacements.items():
            if 'frac' not in pattern:  # Skip fraction pattern as it's already handled
                text = re.sub(pattern, replacement, text)
        
        # Clean up any remaining LaTeX artifacts (but preserve processed fractions)
        # Only remove backslashes from commands that aren't fractions
        text = re.sub(r'\\(?!frac)([a-zA-Z]+)', r'\1', text)  # Remove backslashes from unprocessed commands except frac
        # Only remove braces that aren't part of fractions
        text = re.sub(r'(?<!frac)\{([^}]*)\}', r'\1', text)    # Remove remaining braces not following frac
        
        return text
    
    def _get_alignment(self, alignment_str: str) -> int:
        """Convert alignment string to ReportLab alignment constant."""
        alignment_map = {
            'left': TA_LEFT,
            'center': TA_CENTER,
            'right': TA_RIGHT,
            'justify': TA_JUSTIFY
        }
        return alignment_map.get(alignment_str.lower(), TA_LEFT)
    
    def _add_answer_space(self, story: List, answer_info: Dict[str, Any], 
                         page_width: float, page_height: float) -> None:
        """
        Add answer space to the PDF story based on answer space information.
        """
        try:
            space_type = answer_info.get('type', 'lines')
            bbox = answer_info.get('bounding_box', {})
            
            if bbox:
                x_pos = self._parse_percentage_value(bbox.get('x', 0)) / 100 * page_width
                y_pos = self._parse_percentage_value(bbox.get('y', 0)) / 100 * page_height
                width = self._parse_percentage_value(bbox.get('width', 50)) / 100 * page_width
                height = self._parse_percentage_value(bbox.get('height', 20)) / 100 * page_height
                
                if space_type == 'lines':
                    # Add horizontal lines for writing
                    num_lines = answer_info.get('line_count', 3)
                    line_spacing = height / max(num_lines, 1)
                    
                    for i in range(num_lines):
                        story.append(Spacer(1, line_spacing * 0.7))
                        story.append(HRFlowable(width=width, thickness=0.5, 
                                              lineCap='round', color=colors.lightgrey))
                        
                elif space_type == 'box':
                    # Add a box for answers
                    story.append(Spacer(1, height * 0.1))
                    # Create a simple box using a table or frame
                    box_style = ParagraphStyle(
                        'AnswerBox',
                        parent=getSampleStyleSheet()['Normal'],
                        fontSize=1,  # Minimal font size
                        borderWidth=1,
                        borderColor=colors.lightgrey,
                        leftIndent=x_pos if x_pos > 0 else 0
                    )
                    # Add empty space with border effect
                    story.append(Paragraph("&nbsp;" * int(width/10), box_style))
                    story.append(Spacer(1, height * 0.8))
                    
        except (ValueError, TypeError, KeyError):
             # Fallback: add simple spacing
             story.append(Spacer(1, 20))
    
    def _create_precise_layout_pdf(self, 
                                  results: List[Dict[str, Any]], 
                                  output_path: Path, 
                                  timestamp: str) -> str:
        """
        Create a PDF with precise layout using Canvas for pixel-perfect positioning.
        This method provides more control over exact positioning than the standard method.
        """
        pdf_file = output_path / f"math_problems_precise_{timestamp}.pdf"
        
        # Create canvas
        c = canvas.Canvas(str(pdf_file), pagesize=A4)
        page_width, page_height = A4
        
        for result in results:
            if not result.get("success", False):
                continue
                
            parsed_content = result.get('parsed_content')
            if parsed_content and isinstance(parsed_content, dict):
                questions = parsed_content.get('questions', [])
                
                # Get page dimensions from worksheet info if available
                worksheet_info = parsed_content.get('worksheet_info', {})
                page_dims = worksheet_info.get('page_dimensions', {})
                
                if page_dims:
                    try:
                        # Use original page dimensions if provided
                        orig_width = float(page_dims.get('width', page_width))
                        orig_height = float(page_dims.get('height', page_height))
                        # Scale factor to fit A4
                        scale_x = page_width / orig_width
                        scale_y = page_height / orig_height
                    except (ValueError, TypeError):
                        scale_x = scale_y = 1.0
                else:
                    scale_x = scale_y = 1.0
                
                for question in questions:
                    spatial_info = question.get('spatial_info', {})
                    bbox = spatial_info.get('bounding_box', {})
                    text_props = spatial_info.get('text_properties', {})
                    
                    if bbox:
                        try:
                            # Calculate precise positions
                            x_pos = self._parse_percentage_value(bbox.get('x', 0)) / 100 * page_width * scale_x
                            y_pos = page_height - (self._parse_percentage_value(bbox.get('y', 0)) / 100 * page_height * scale_y)
                            
                            # Set font properties
                            font_size_map = {'small': 9, 'medium': 11, 'large': 14}
                            font_size = font_size_map.get(text_props.get('font_size', 'medium'), 11)
                            font_name = text_props.get('font_family', 'Helvetica')
                            
                            # Map common font names to ReportLab fonts
                            font_map = {
                                'arial': 'Helvetica',
                                'times': 'Times-Roman',
                                'courier': 'Courier'
                            }
                            font_name = font_map.get(font_name.lower(), 'Helvetica')
                            
                            c.setFont(font_name, font_size)
                            
                            # Draw text at precise position
                            problem_text = question.get('problem_text', '')
                            if problem_text:
                                # Handle multi-line text
                                lines = problem_text.split('\n')
                                line_height = font_size * 1.2
                                
                                for i, line in enumerate(lines):
                                    c.drawString(x_pos, y_pos - (i * line_height), line)
                                
                                # Add answer space if specified
                                answer_info = question.get('answer_space_info', {})
                                if answer_info:
                                    self._draw_answer_space_canvas(c, answer_info, 
                                                                 page_width, page_height, 
                                                                 scale_x, scale_y)
                        except (ValueError, TypeError):
                            # Skip invalid spatial data
                            continue
                
                # Start new page for next result
                c.showPage()
        
        c.save()
        return str(pdf_file)
    
    def _draw_answer_space_canvas(self, canvas_obj, answer_info: Dict[str, Any],
                                 page_width: float, page_height: float,
                                 scale_x: float, scale_y: float) -> None:
        """
        Draw answer spaces on canvas with precise positioning.
        """
        try:
            space_type = answer_info.get('type', 'lines')
            bbox = answer_info.get('bounding_box', {})
            
            if bbox:
                x_pos = self._parse_percentage_value(bbox.get('x', 0)) / 100 * page_width * scale_x
                y_pos = page_height - (self._parse_percentage_value(bbox.get('y', 0)) / 100 * page_height * scale_y)
                width = self._parse_percentage_value(bbox.get('width', 50)) / 100 * page_width * scale_x
                height = self._parse_percentage_value(bbox.get('height', 20)) / 100 * page_height * scale_y
                
                if space_type == 'lines':
                    # Draw horizontal lines
                    num_lines = answer_info.get('line_count', 3)
                    line_spacing = height / max(num_lines, 1)
                    
                    canvas_obj.setStrokeColor(colors.lightgrey)
                    canvas_obj.setLineWidth(0.5)
                    
                    for i in range(num_lines):
                        line_y = y_pos - (i * line_spacing)
                        canvas_obj.line(x_pos, line_y, x_pos + width, line_y)
                        
                elif space_type == 'box':
                    # Draw a rectangle
                    canvas_obj.setStrokeColor(colors.lightgrey)
                    canvas_obj.setLineWidth(1)
                    canvas_obj.rect(x_pos, y_pos - height, width, height, fill=0)
                    
        except (ValueError, TypeError, KeyError):
            # Skip invalid answer space data
            pass
    
    def _create_html_visualization(self, 
                                  results: List[Dict[str, Any]], 
                                  output_path: Path, 
                                  timestamp: str,
                                  preserve_format: bool = True) -> str:
        """
        Create an HTML visualization with optional format preservation.
        
        Args:
            preserve_format: If True, preserves original layout. If False, creates plain text output.
        """
        html_file = output_path / f"math_problems_only_{timestamp}.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Math Problems</title>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <script>
                window.MathJax = {
                    tex: {
                        inlineMath: [['\\(', '\\)']],
                        displayMath: [['\\[', '\\]']]
                    }
                };
            </script>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .question { margin: 40px 0; padding: 25px; background-color: #f9f9f9; border-radius: 8px; border-left: 4px solid #2196F3; }
                .question-plain { margin: 35px 0; padding: 15px; background-color: #fff; border-bottom: 1px solid #ddd; }
                .question-title { color: #2196F3; font-weight: bold; font-size: 1.2em; margin-bottom: 20px; }
                .problem-text { font-size: 1.1em; line-height: 1.8; color: #333; }
            </style>
        </head>
        <body>
            <h1 class="header">Math Problems</h1>
        """
        
        question_counter = 1
        
        for result in results:
            if not result.get("success", False):
                continue
                
            # Process structured content if available
            parsed_content = result.get('parsed_content')
            if parsed_content and isinstance(parsed_content, dict):
                questions = parsed_content.get('questions', [])
                for question in questions:
                    problem_text = question.get('problem_text', '')
                    if problem_text:
                        processed_text = self._process_latex_for_pdf(problem_text)
                        if preserve_format:
                            # Original styled layout
                            html_content += '<div class="question">'
                            html_content += f'<div class="question-title">Problem {question_counter}</div>'
                            html_content += f'<div class="problem-text">{processed_text}</div>'
                            html_content += '</div>'
                        else:
                            # Plain text layout
                            html_content += '<div class="question-plain">'
                            html_content += f'<div class="question-title">Question {question_counter}</div>'
                            html_content += f'<div class="problem-text">{processed_text}</div>'
                            html_content += '</div>'
                        question_counter += 1
            
            else:
                # Fallback to raw response - extract problem-like content
                raw_response = result.get('raw_response', '')
                if raw_response:
                    lines = raw_response.split('\n')
                    current_problem = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(('Subject:', 'Questions:', 'Level:', 'Processing')):
                            if any(indicator in line.lower() for indicator in ['question', 'problem', 'solve', 'find', 'calculate']):
                                if current_problem:
                                    problem_text = ' '.join(current_problem)
                                    html_content += '<div class="question">'
                                    html_content += f'<div class="question-title">Problem {question_counter}</div>'
                                    html_content += f'<div class="problem-text">{problem_text}</div>'
                                    html_content += '</div>'
                                    question_counter += 1
                                    current_problem = []
                                current_problem.append(line)
                            elif current_problem:
                                current_problem.append(line)
                    
                    if current_problem:
                        problem_text = ' '.join(current_problem)
                        html_content += '<div class="question">'
                        html_content += f'<div class="question-title">Problem {question_counter}</div>'
                        html_content += f'<div class="problem-text">{problem_text}</div>'
                        html_content += '</div>'
                        question_counter += 1
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML with problems only saved to: {html_file}")
        return str(html_file)


def main():
    """
    Example usage of the Qwen Math OCR Pipeline.
    """
    # Initialize pipeline
    pipeline = QwenMathOCRPipeline()
    
    # Example image path (replace with your math worksheet)
    image_path = "dots_ocr_results/extracted_pages/page_001.png"
    
    if not os.path.exists(image_path):
        print(f"Creating test image: {image_path}")
        # Create a simple test image
        test_img = Image.new('RGB', (800, 600), 'white')
        test_img.save(image_path)
    
    print("=" * 60)
    print("QWEN MATH OCR PIPELINE DEMO")
    print("=" * 60)
    
    # Test structured extraction
    print("\n1. Testing structured JSON extraction...")
    result_structured = pipeline.process_image(
        image_path, 
        prompt_type="structured",
        max_new_tokens=2048,
        temperature=0.1
    )
    
    if result_structured["success"]:
        print("✅ Structured extraction completed")
        if result_structured.get("parsed_content"):
            print("📊 Parsed JSON data available")
        else:
            print("⚠️  Raw text response only")
    else:
        print(f"❌ Structured extraction failed: {result_structured.get('error')}")
    
    # Test text extraction
    print("\n2. Testing text extraction...")
    result_text = pipeline.process_image(
        image_path,
        prompt_type="extraction",
        max_new_tokens=1024,
        temperature=0.0
    )
    
    if result_text["success"]:
        print("✅ Text extraction completed")
    else:
        print(f"❌ Text extraction failed: {result_text.get('error')}")
    
    # Save results
    print("\n3. Saving results...")
    output_dir = "qwen_math_ocr_results"
    all_results = [result_structured, result_text]
    saved_files = pipeline.save_results(all_results, output_dir)
    
    # Create visualization with problems only
    print("\n4. Creating problems-only visualization...")
    try:
        if REPORTLAB_AVAILABLE:
            # Create PDF with problems only
            pdf_file = pipeline.visualize_results(
                all_results, 
                output_dir=output_dir,
                output_format="pdf"
            )
            print(f"✅ PDF with problems only created: {pdf_file}")
        
        # Create HTML with problems only
        html_file = pipeline.visualize_results(
            all_results, 
            output_dir=output_dir,
            output_format="html"
        )
        print(f"✅ HTML with problems only created: {html_file}")
        
        # Demonstrate loading from folder
        print("\n5. Testing folder-based visualization...")
        folder_viz = pipeline.visualize_results(
            "qwen_math_ocr_results",
            output_format="html"
        )
        print(f"✅ Visualization from saved folder: {folder_viz}")
        
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
        if not REPORTLAB_AVAILABLE:
            print("Note: Install reportlab for PDF generation: pip install reportlab")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: qwen_math_ocr_results/")
    print(f"Summary file: {saved_files['summary']}")
    print("\nVisualization features:")
    print("- Problems-only PDF and HTML generated")
    print("- Can load from previously saved folders")
    print("- Clean format with just question text")


if __name__ == "__main__":
    main()