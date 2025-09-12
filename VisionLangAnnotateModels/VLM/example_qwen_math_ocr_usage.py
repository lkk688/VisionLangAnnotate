#!/usr/bin/env python3
"""
Example usage of Qwen Math OCR Pipeline

Demonstrates how to use the QwenMathOCRPipeline for extracting
math questions from worksheet images.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tempfile

# PDF processing imports
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. Install with: pip install PyMuPDF")

# Import the pipeline class
try:
    from qwen_math_ocr_pipeline import QwenMathOCRPipeline
except ImportError:
    # Add current directory to path for local import
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from qwen_math_ocr_pipeline import QwenMathOCRPipeline


def create_sample_math_worksheet(output_path: str = "sample_math_worksheet.png"):
    """
    Create a sample math worksheet image for testing.
    
    Args:
        output_path: Path to save the sample worksheet
    """
    # Create a white background
    width, height = 800, 1000
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    draw.text((50, 30), "Math Worksheet - Word Problems", fill='black', font=font_title)
    draw.text((50, 60), "Name: _________________ Date: _________", fill='black', font=font_small)
    
    # Draw questions
    questions = [
        {
            "number": "1",
            "text": "Sarah has 24 apples. She gives 8 apples to her friend\nand eats 3 apples herself. How many apples does\nSarah have left?",
            "answer_line": "Answer: _____________"
        },
        {
            "number": "2", 
            "text": "A rectangle has a length of 12 cm and a width of\n7 cm. What is the perimeter of the rectangle?\n(Perimeter = 2 × length + 2 × width)",
            "answer_line": "Answer: _____________"
        },
        {
            "number": "3",
            "text": "Tom bought 3 packs of stickers. Each pack contains\n15 stickers. If he uses 18 stickers to decorate his\nnotebook, how many stickers does he have remaining?",
            "answer_line": "Answer: _____________"
        },
        {
            "number": "4",
            "text": "A pizza is cut into 8 equal slices. Maria eats 3 slices\nand her brother eats 2 slices. What fraction of the\npizza is left?",
            "answer_line": "Answer: _____________"
        }
    ]
    
    y_pos = 120
    for q in questions:
        # Draw question number in a circle
        circle_x, circle_y = 70, y_pos + 10
        draw.ellipse([circle_x-15, circle_y-15, circle_x+15, circle_y+15], outline='black', width=2)
        draw.text((circle_x-5, circle_y-8), q["number"], fill='black', font=font_text)
        
        # Draw "Question" label
        draw.text((100, y_pos), "Question", fill='black', font=font_text)
        
        # Draw question text
        lines = q["text"].split('\n')
        text_y = y_pos + 25
        for line in lines:
            draw.text((100, text_y), line, fill='black', font=font_text)
            text_y += 20
        
        # Draw answer line
        answer_y = text_y + 20
        draw.text((100, answer_y), q["answer_line"], fill='black', font=font_text)
        
        y_pos += 180
    
    # Save the image
    img.save(output_path)
    print(f"Sample math worksheet created: {output_path}")
    return output_path


def extract_pdf_pages_as_images(pdf_path: str, dpi: int = 150) -> list:
    """
    Extract all pages from a PDF file as image files.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image extraction (default: 150)
        
    Returns:
        List of temporary image file paths
    """
    if not PDF_AVAILABLE:
        print("❌ Error: PyMuPDF not available. Install with: pip install PyMuPDF")
        return []
    
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        page_images = []
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp(prefix="pdf_pages_")
        
        print(f"📄 Extracting {len(doc)} pages from PDF...")
        
        for page_num in range(len(doc)):
            # Get page
            page = doc.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Save as temporary PNG file
            temp_image_path = os.path.join(temp_dir, f"page_{page_num+1:03d}.png")
            pix.save(temp_image_path)
            page_images.append(temp_image_path)
            
            print(f"   ✅ Page {page_num+1}/{len(doc)} extracted")
        
        doc.close()
        print(f"📁 Temporary images saved to: {temp_dir}")
        return page_images
        
    except Exception as e:
        print(f"❌ Error extracting PDF pages: {e}")
        return []


def cleanup_temp_images(image_paths: list):
    """
    Clean up temporary image files.
    
    Args:
        image_paths: List of temporary image file paths to delete
    """
    if not image_paths:
        return
        
    try:
        # Get the temporary directory from the first image path
        temp_dir = os.path.dirname(image_paths[0])
        
        # Remove all temporary files
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        # Remove temporary directory if empty
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            print(f"🧹 Cleaned up temporary files from: {temp_dir}")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up temporary files: {e}")


def demo_basic_usage():
    """
    Demonstrate basic usage of the Qwen Math OCR Pipeline.
    """
    print("=" * 70)
    print("QWEN MATH OCR PIPELINE - BASIC USAGE DEMO")
    print("=" * 70)
    
    # Create sample worksheet if it doesn't exist
    sample_image = "dots_ocr_results/extracted_pages/page_001.png"
    if not os.path.exists(sample_image):
        create_sample_math_worksheet(sample_image)
    
    # Initialize the pipeline
    print("\n1. Initializing Qwen Math OCR Pipeline...")
    pipeline = QwenMathOCRPipeline(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="auto"
    )
    
    # Process with structured output (JSON)
    print("\n2. Processing with structured JSON output...")
    result_json = pipeline.process_image(
        sample_image,
        prompt_type="structured",
        max_new_tokens=2048,
        temperature=0.1
    )
    
    if result_json["success"]:
        print(f"✅ Structured processing completed in {result_json['processing_time']:.2f}s")
        
        if result_json.get("parsed_content"):
            parsed = result_json["parsed_content"]
            print(f"📊 Found {parsed.get('worksheet_info', {}).get('total_questions', 0)} questions")
            
            # Display extracted questions
            questions = parsed.get('questions', [])
            for i, q in enumerate(questions[:2], 1):  # Show first 2 questions
                print(f"\n   Question {i}:")
                print(f"   - Number: {q.get('question_number', 'N/A')}")
                print(f"   - Text: {q.get('problem_text', 'N/A')[:100]}...")
                print(f"   - Math Elements: {q.get('mathematical_expressions', [])}")
        else:
            print("⚠️  No structured data parsed, but raw text available")
    else:
        print(f"❌ Structured processing failed: {result_json.get('error')}")
    
    # Process with text extraction
    print("\n3. Processing with text extraction...")
    result_text = pipeline.process_image(
        sample_image,
        prompt_type="extraction",
        max_new_tokens=1024,
        temperature=0.0
    )
    
    if result_text["success"]:
        print(f"✅ Text extraction completed in {result_text['processing_time']:.2f}s")
        
        # Show first 200 characters of extracted text
        raw_text = result_text.get("raw_response", "")
        print(f"📝 Extracted text preview: {raw_text[:200]}...")
    else:
        print(f"❌ Text extraction failed: {result_text.get('error')}")
    
    # Save results
    print("\n4. Saving results...")
    results = [result_json, result_text]
    saved_files = pipeline.save_results(results, "demo_math_ocr_results")
    
    print(f"\n💾 Results saved to: demo_math_ocr_results/")
    print(f"📄 Summary: {saved_files['summary']}")
    
    return results


def demo_batch_processing(pdf_path: str = None, output_folder: str = "pdf_math_ocr_results"):
    """
    Demonstrate batch processing of PDF pages or multiple images.
    
    Args:
        pdf_path: Path to PDF file to process. If None, uses sample images.
        output_folder: Folder to save results
    """
    print("\n" + "=" * 70)
    print("BATCH PROCESSING DEMO")
    print("=" * 70)
    
    if pdf_path and PDF_AVAILABLE:
        # Process PDF file
        print(f"Processing PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found: {pdf_path}")
            return None
            
        # Extract pages from PDF
        page_images = extract_pdf_pages_as_images(pdf_path)
        if not page_images:
            print("❌ Error: No pages extracted from PDF")
            return None
            
        print(f"📄 Extracted {len(page_images)} pages from PDF")
        
    else:
        # Fallback to sample images if no PDF or PyMuPDF not available
        if pdf_path and not PDF_AVAILABLE:
            print("⚠️  PyMuPDF not available, using sample images instead")
        
        print("Creating sample worksheet images...")
        page_images = []
        for i in range(1, 4):
            img_path = f"sample_worksheet_{i}.png"
            if not os.path.exists(img_path):
                create_sample_math_worksheet(img_path)
            page_images.append(img_path)
    
    # Initialize pipeline
    pipeline = QwenMathOCRPipeline()
    
    # Process all images in batches
    print(f"\nProcessing {len(page_images)} pages...")
    batch_results = pipeline.process_multiple_images(
        page_images,
        prompt_type="structured",
        max_new_tokens=1536,
        temperature=0.1
    )
    
    # Display summary
    successful = len([r for r in batch_results if r.get("success", False)])
    total_time = sum(r.get("processing_time", 0) for r in batch_results)
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"   - Total pages: {len(batch_results)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {len(batch_results) - successful}")
    print(f"   - Total time: {total_time:.2f}s")
    print(f"   - Average time per page: {total_time/len(batch_results):.2f}s")
    
    # Save batch results
    saved_files = pipeline.save_results(batch_results, output_folder)
    print(f"\n💾 Batch results saved to: {output_folder}/")
    
    # Clean up temporary files if PDF was processed
    if pdf_path and PDF_AVAILABLE:
        cleanup_temp_images(page_images)
    
    return batch_results


def demo_custom_prompts():
    """
    Demonstrate using custom prompts for specific extraction needs.
    """
    print("\n" + "=" * 70)
    print("CUSTOM PROMPTS DEMO")
    print("=" * 70)
    
    # Create sample if needed
    sample_image = "sample_math_worksheet.png"
    if not os.path.exists(sample_image):
        create_sample_math_worksheet(sample_image)
    
    pipeline = QwenMathOCRPipeline()
    
    # Custom prompt for extracting only numbers
    numbers_prompt = """Extract all numbers and mathematical expressions from this worksheet. 
List them in order of appearance. Include:
- Whole numbers
- Fractions
- Measurements with units
- Mathematical operations

Format as a simple list."""
    
    # Custom prompt for difficulty assessment
    difficulty_prompt = """Analyze this math worksheet and provide:
1. Grade level estimate (K-12)
2. Mathematical concepts covered
3. Difficulty rating (Easy/Medium/Hard)
4. Skills required to solve these problems

Be specific about the mathematical topics."""
    
    # Test custom prompts
    custom_tests = [
        ("Numbers Extraction", numbers_prompt),
        ("Difficulty Analysis", difficulty_prompt)
    ]
    
    for test_name, custom_prompt in custom_tests:
        print(f"\n🔍 Testing: {test_name}")
        
        # Temporarily override the pipeline's prompt
        original_prompt = pipeline.math_extraction_prompt
        pipeline.math_extraction_prompt = custom_prompt
        
        result = pipeline.process_image(
            sample_image,
            prompt_type="extraction",
            max_new_tokens=1024,
            temperature=0.2
        )
        
        # Restore original prompt
        pipeline.math_extraction_prompt = original_prompt
        
        if result["success"]:
            print(f"✅ {test_name} completed in {result['processing_time']:.2f}s")
            response = result.get("raw_response", "")
            print(f"📝 Response preview: {response[:150]}...")
        else:
            print(f"❌ {test_name} failed: {result.get('error')}")


def generatepdf():
    try:
        pipeline = QwenMathOCRPipeline()
        pdf_file = pipeline.visualize_results(
            results='ocr_results', #it will load from structured_results first
            output_dir='ocr_results',
            output_format="pdf",
            preserve_format=False
        )
        print(f"📄 Consolidated PDF with all problems created: {pdf_file}")
    except Exception as e:
        print(f"⚠️  Failed to create consolidated PDF: {e}")

def main():
    """
    Run all demonstration examples.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Qwen Math OCR Pipeline Demo")
    parser.add_argument("--pdf", type=str, default='VisionLangAnnotateModels/sampledata/101 Challenging Maths Word Problems BOOK 4.pdf', help="Path to PDF file to process")
    parser.add_argument("--output", type=str, default="ocr_results", 
                       help="Output folder for results (default: ocr_results)")
    parser.add_argument("--skip-demos", default=False, action="store_true", 
                       help="Skip basic demos and only process PDF")
    args = parser.parse_args()

    generatepdf()
    
    try:
        # if not args.skip_demos:
        #     # Basic usage demo
        #     demo_basic_usage()
            
        #     # Batch processing demo (sample images)
        #     demo_batch_processing()
            
        #     # Custom prompts demo
        #     demo_custom_prompts()
        
        # PDF processing demo
        if args.pdf:
            print("\n" + "=" * 70)
            print("PDF PROCESSING DEMO")
            print("=" * 70)
            batch_results = demo_batch_processing(pdf_path=args.pdf, output_folder=args.output)
            
            # Generate consolidated PDF with all problems
            if batch_results:
                try:
                    pipeline = QwenMathOCRPipeline()
                    pdf_file = pipeline.visualize_results(
                        batch_results, 
                        output_dir=args.output,
                        output_format="pdf"
                    )
                    print(f"📄 Consolidated PDF with all problems created: {pdf_file}")
                except Exception as e:
                    print(f"⚠️  Failed to create consolidated PDF: {e}")
        
        print("\n" + "=" * 70)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nCheck the following directories for results:")
        if not args.skip_demos:
            print("- demo_math_ocr_results/")
            print("- batch_math_ocr_results/")
        if args.pdf:
            print(f"- {args.output}/")
        print("\nUsage examples:")
        print("  python example_qwen_math_ocr_usage.py --pdf my_math_book.pdf")
        print("  python example_qwen_math_ocr_usage.py --pdf my_file.pdf --output my_results")
        print("  python example_qwen_math_ocr_usage.py --skip-demos --pdf my_file.pdf")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have sufficient GPU memory")
        print("2. Check that all required packages are installed")
        print("3. Verify internet connection for model downloading")
        if "pdf" in str(e).lower():
            print("4. Install PyMuPDF for PDF processing: pip install PyMuPDF")
        sys.exit(1)


if __name__ == "__main__":
    main()