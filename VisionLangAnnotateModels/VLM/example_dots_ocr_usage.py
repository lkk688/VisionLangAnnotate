#!/usr/bin/env python3
"""
Example usage of the DotsOCR Pipeline

This script demonstrates how to use the DotsOCR pipeline to process PDF files
and extract text while preserving the original document structure.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dots_ocr_pipeline import DotsOCRPipeline

def test_dots_ocr_pipeline():
    """
    Test the DotsOCR pipeline with example usage.
    """
    print("=" * 60)
    print("DotsOCR Pipeline Example")
    print("=" * 60)
    
    # Initialize the pipeline
    try:
        pipeline = DotsOCRPipeline(
            model_path="/home/lkk/Developer/dots.ocr/weights/DotsOCR",
            device="auto",  # Will use CUDA if available, otherwise CPU
            output_dir="./dots_ocr_results"
        )
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        print("\nPlease ensure:")
        print("1. The dots.ocr model is downloaded to the correct path")
        print("2. All required dependencies are installed (see requirements_dots_ocr.txt)")
        return
    
    # Check if sample document exists (text or PDF)
    sample_files = ["test_document.txt", "sample_document.pdf", "../sampledata/sample.pdf", "./test_document.pdf", "/home/lkk/Developer/VisionLangAnnotate/sample.pdf"]
    sample_file = "VisionLangAnnotateModels/sampledata/4copy.pdf"
    
    for file in sample_files:
        if os.path.exists(file):
            sample_file = file
            break
    
    if sample_file is None:
        print("\n⚠️  No sample document found. Creating a test scenario...")
        print("\nTo test with your own PDF:")
        print("1. Place a PDF file in the current directory")
        print("2. Update the pdf_path variable in this script")
        print("3. Run the script again")
        
        # Show how to use the pipeline programmatically
        print("\n" + "=" * 40)
        print("Example Usage Code:")
        print("=" * 40)
        
        example_code = '''
# Initialize pipeline
pipeline = DotsOCRPipeline(
    model_path="/home/lkk/Developer/dots.ocr/weights/DotsOCR",
    device="auto",
    output_dir="./dots_ocr_results"
)

# Process PDF with markdown output
results = pipeline.process_pdf(
    pdf_path="your_document.pdf", 
    output_format="markdown"
)

# Process PDF with PDF output (requires weasyprint)
results = pipeline.process_pdf(
    pdf_path="your_document.pdf", 
    output_format="pdf"
)

print(f"Processed {results['pages_processed']} pages")
print(f"Output files: {results['output_paths']}")
        '''
        print(example_code)
        return
    
    # Convert text file to simple PDF if needed
    pdf_path = sample_file
    if sample_file.endswith('.txt'):
        print(f"📄 Found text file: {sample_file}")
        print("Converting to simple PDF for OCR testing...")
        
        # Create a simple PDF from text content
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = sample_file.replace('.txt', '_converted.pdf')
            c = canvas.Canvas(pdf_path, pagesize=letter)
            
            with open(sample_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            y_position = 750
            for line in lines:
                if y_position < 50:
                    c.showPage()
                    y_position = 750
                
                c.drawString(50, y_position, line.strip()[:80])  # Limit line length
                y_position -= 15
            
            c.save()
            print(f"✅ Converted to PDF: {pdf_path}")
            
        except ImportError:
            print("⚠️  ReportLab not available. Using text file directly for demonstration.")
            # For demonstration, we'll show how the pipeline would work
            print(f"\n📋 Sample content from {sample_file}:")
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()[:500]
                print(content + "..." if len(content) == 500 else content)
            
            print("\n🔄 This would be processed by DotsOCR to extract:")
            print("- Document structure and layout")
            print("- Text content with formatting")
            print("- Tables and lists")
            print("- Mathematical formulas")
            print("- Reading order preservation")
            return
    
    print(f"\n📄 Found PDF: {pdf_path}")
    
    # Test markdown output
    print("\n🔄 Processing PDF with Markdown output...")
    try:
        markdown_results = pipeline.process_pdf(
            pdf_path=pdf_path,
            output_format="markdown"
        )
        
        print("\n✓ Markdown processing completed!")
        print(f"  📊 Pages processed: {markdown_results['pages_processed']}")
        print(f"  ⏱️  Total time: {markdown_results['total_processing_time']:.2f}s")
        print(f"  📁 Output files:")
        for format_type, path in markdown_results['output_paths'].items():
            print(f"    - {format_type.upper()}: {path}")
        
    except Exception as e:
        print(f"✗ Markdown processing failed: {e}")
    
    # Test PDF output (if weasyprint is available)
    print("\n🔄 Processing PDF with PDF output...")
    try:
        pdf_results = pipeline.process_pdf(
            pdf_path=pdf_path,
            output_format="pdf"
        )
        
        print("\n✓ PDF processing completed!")
        print(f"  📊 Pages processed: {pdf_results['pages_processed']}")
        print(f"  ⏱️  Total time: {pdf_results['total_processing_time']:.2f}s")
        print(f"  📁 Output files:")
        for format_type, path in pdf_results['output_paths'].items():
            print(f"    - {format_type.upper()}: {path}")
            
    except ImportError:
        print("⚠️  PDF output requires weasyprint. Install with: pip install weasyprint")
    except Exception as e:
        print(f"✗ PDF processing failed: {e}")
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nCheck the './dots_ocr_results' directory for output files:")
    print("- extracted_pages/: Individual page images")
    print("- raw_ocr_results/: Raw OCR JSON results per page")
    print("- rendered_documents/: Final formatted documents")

def demonstrate_features():
    """
    Demonstrate key features of the DotsOCR pipeline.
    """
    print("\n" + "=" * 60)
    print("DotsOCR Pipeline Features")
    print("=" * 60)
    
    features = [
        "🔍 High-quality OCR using dots.ocr model (1.7B parameters)",
        "📄 Multi-page PDF processing with structure preservation",
        "🎯 Automatic detection of text, tables, formulas, headers, and lists",
        "📝 Multiple output formats: Markdown, HTML, PDF",
        "🌍 Multilingual document support",
        "⚡ Efficient processing with GPU acceleration",
        "📊 Structured JSON output with reading order preservation",
        "🎨 Customizable rendering with CSS styling",
        "💾 Automatic result saving and organization",
        "🔧 Easy integration into existing workflows"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print("Model: dots.ocr (rednote-hilab/dots.ocr)")
    print("Size: 1.7B parameters")
    print("Performance: SOTA on OmniDocBench")
    print("Languages: Multilingual support")
    print("Architecture: Single vision-language model")
    print("GitHub: https://github.com/rednote-hilab/dots.ocr")

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch for model inference',
        'transformers': 'Hugging Face Transformers',
        'PIL': 'Pillow for image processing',
        'fitz': 'PyMuPDF for PDF handling',
        'numpy': 'NumPy for numerical operations'
    }
    
    optional_dependencies = {
        'markdown': 'Markdown rendering',
        'weasyprint': 'HTML to PDF conversion'
    }
    
    print("Required Dependencies:")
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {module}: {description}")
        except ImportError:
            print(f"  ✗ {module}: {description} - NOT INSTALLED")
    
    print("\nOptional Dependencies:")
    for module, description in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {module}: {description}")
        except ImportError:
            print(f"  ⚠️  {module}: {description} - NOT INSTALLED (optional)")
    
    print("\nTo install missing dependencies:")
    print("pip install -r requirements_dots_ocr.txt")

if __name__ == "__main__":
    # Check dependencies first
    check_dependencies()
    
    # Demonstrate features
    demonstrate_features()
    
    # Run the test
    test_dots_ocr_pipeline()