import os
import pymupdf4llm
import re
from typing import Dict, List, Union


def process_pdf_sections(pdf_path: str, images_dir: str = 'images') -> Dict[str, Dict[str, Union[str, List[str], int]]]:
    """
    Process a PDF file and return a dictionary of sections with their content, images, and content length.

    Args:
        pdf_path: Path to the PDF file
        images_dir: Directory to save extracted images

    Returns:
        Dictionary with section titles as keys and dictionaries containing:
            - content: The section text content
            - images: List of image paths
            - length: Length of the content
    """

    # Create images directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    # Convert PDF to markdown with images
    md_text = pymupdf4llm.to_markdown(
        doc=pdf_path,
        write_images=True,
        image_path=images_dir,
        image_format='.png',
        dpi=450,

    )

    # Extract sections using regex
    sections_pattern = r'#{6}\s*(.*?)\n(.*?)(?=#{6}|\Z)'
    matches = re.finditer(sections_pattern, md_text, re.DOTALL)

    # Process sections
    result = {}
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()

        # Skip empty sections
        if not title or not content:
            continue

        # Extract images
        image_pattern = r'!\[.*?\]\((.*?)\)'
        images = re.findall(image_pattern, content)

        # Store section data
        result[title] = {
            "content": content,
            "images": images,
            "length": len(content)
        }

    return result


# Example usage
if __name__ == "__main__":
    pdf_path = "/home/tempa/Desktop/mpm_green_2/data/MINT05-Mint-123-issuu.pdf"
    sections = process_pdf_sections(pdf_path)

