import streamlit as st
from streamlit_carousel import carousel
from typing import List, Optional
import hashlib


def create_image_slideshow(png_paths: List[str], slideshow_id: Optional[str] = None):
    """
    Create a Streamlit carousel slideshow from a list of PNG file paths with minimal page resets.
    """
    if not png_paths:
        st.error("No PNG paths provided!")
        return

    # Generate unique ID for this slideshow instance
    if slideshow_id is None:
        paths_string = ''.join(png_paths)
        slideshow_id = hashlib.md5(paths_string.encode()).hexdigest()[:8]

    # Prepare items for the carousel
    items = [
        dict(
            title="",
            text="",
            img=path,
            link="#"  # Use "#" if there's no specific link for the image
        )
        for idx, path in enumerate(png_paths)
    ]

    # Display the carousel with images
    carousel(items=items,key=slideshow_id)