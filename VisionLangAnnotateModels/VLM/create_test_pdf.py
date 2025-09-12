#!/usr/bin/env python3
"""
Create a test PDF document for demonstrating the DotsOCR pipeline.

This script creates a simple multi-page PDF with various elements like text,
tables, headers, and lists to test the OCR capabilities.
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_test_pdf(output_path: str = "test_document.pdf"):
    """
    Create a test PDF with various document elements.
    """
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Get sample styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen
        )
        
        # Build the document content
        story = []
        
        # Title page
        story.append(Paragraph("DotsOCR Test Document", title_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Sample Document for OCR Testing", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Introduction paragraph
        intro_text = """
        This is a test document created to demonstrate the capabilities of the DotsOCR pipeline.
        It contains various types of content including headers, paragraphs, lists, and tables
        to test the document layout parsing and optical character recognition functionality.
        The document is designed to showcase how well the system can preserve the original
        structure and formatting of complex documents.
        """
        story.append(Paragraph(intro_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Section 1: Text Content
        story.append(Paragraph("1. Text Content Analysis", heading_style))
        story.append(Spacer(1, 12))
        
        text_content = """
        This section contains regular paragraph text to test the OCR system's ability to
        recognize and extract plain text content. The text includes various formatting
        elements and should maintain proper reading order when processed by the OCR system.
        
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        """
        story.append(Paragraph(text_content, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Section 2: Lists
        story.append(Paragraph("2. List Structures", heading_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Key Features of DotsOCR:", styles['Normal']))
        story.append(Spacer(1, 6))
        
        list_items = [
            "• Multilingual document parsing support",
            "• Layout detection and content recognition",
            "• Table extraction and structure preservation",
            "• Formula recognition capabilities",
            "• Reading order maintenance",
            "• Multiple output format support"
        ]
        
        for item in list_items:
            story.append(Paragraph(item, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Section 3: Table
        story.append(Paragraph("3. Table Data", heading_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Performance Comparison:", styles['Normal']))
        story.append(Spacer(1, 6))
        
        # Create a sample table
        table_data = [
            ['Model', 'Parameters', 'Text Accuracy', 'Table Accuracy'],
            ['DotsOCR', '1.7B', '96.8%', '88.6%'],
            ['GPT-4V', '~1.8T', '95.5%', '85.8%'],
            ['Gemini Pro', 'Unknown', '94.2%', '82.3%'],
            ['Claude-3', 'Unknown', '93.8%', '80.1%']
        ]
        
        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Section 4: Mathematical Content
        story.append(Paragraph("4. Mathematical Formulas", heading_style))
        story.append(Spacer(1, 12))
        
        math_text = """
        Mathematical formulas and equations are important elements in many documents.
        Here are some examples:
        
        • Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a
        • Pythagorean Theorem: a² + b² = c²
        • Einstein's Mass-Energy: E = mc²
        • Area of Circle: A = πr²
        
        The OCR system should be able to recognize and preserve these mathematical
        expressions in their original form.
        """
        story.append(Paragraph(math_text, styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(Paragraph("End of Test Document", styles['Heading3']))
        story.append(Spacer(1, 12))
        
        footer_text = """
        This document was automatically generated for testing the DotsOCR pipeline.
        It demonstrates various document elements that should be properly recognized
        and structured by the OCR system.
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        
        print(f"✓ Test PDF created successfully: {output_path}")
        return output_path
        
    except ImportError:
        print("⚠️  ReportLab not installed. Creating a simple text-based PDF...")
        
        # Fallback: create a simple text file that can be converted to PDF
        text_content = """
DotsOCR Test Document
====================

Sample Document for OCR Testing

This is a test document created to demonstrate the capabilities of the DotsOCR pipeline.
It contains various types of content including headers, paragraphs, lists, and tables
to test the document layout parsing and optical character recognition functionality.

1. Text Content Analysis
-----------------------

This section contains regular paragraph text to test the OCR system's ability to
recognize and extract plain text content. The text includes various formatting
elements and should maintain proper reading order when processed by the OCR system.

2. List Structures
------------------

Key Features of DotsOCR:
• Multilingual document parsing support
• Layout detection and content recognition
• Table extraction and structure preservation
• Formula recognition capabilities
• Reading order maintenance
• Multiple output format support

3. Table Data
-------------

Performance Comparison:

Model       | Parameters | Text Accuracy | Table Accuracy
------------|------------|---------------|---------------
DotsOCR     | 1.7B       | 96.8%         | 88.6%
GPT-4V      | ~1.8T      | 95.5%         | 85.8%
Gemini Pro  | Unknown    | 94.2%         | 82.3%
Claude-3    | Unknown    | 93.8%         | 80.1%

4. Mathematical Formulas
------------------------

Mathematical formulas and equations are important elements in many documents.
Here are some examples:

• Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a
• Pythagorean Theorem: a² + b² = c²
• Einstein's Mass-Energy: E = mc²
• Area of Circle: A = πr²

End of Test Document
===================

This document was automatically generated for testing the DotsOCR pipeline.
It demonstrates various document elements that should be properly recognized
and structured by the OCR system.
        """
        
        # Save as text file
        text_path = output_path.replace('.pdf', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"✓ Test text file created: {text_path}")
        print("To create a PDF, install ReportLab: pip install reportlab")
        print("Or convert the text file to PDF using an online converter.")
        
        return text_path

if __name__ == "__main__":
    output_file = create_test_pdf("test_document.pdf")
    print(f"\nTest document ready: {output_file}")
    print("You can now use this file to test the DotsOCR pipeline!")