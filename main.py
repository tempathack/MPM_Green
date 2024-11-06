import streamlit as st
import asyncio
import tempfile
from pathlib import Path
import logging
from typing import List, Dict, Any,Tuple
import json
from src.process_data import create_summaries
from generators.content_generator import ContentGenerator
from utils.streamlit_utils import display_content_set, show_download_buttons
import asyncio
import logging
from pathlib import Path
import tempfile
from typing import List, Tuple, Dict, Any
import json

# Use absolute imports
from generators.content_generator import ContentGenerator
from utils.validators import validate_content_batch, get_content_statistics
from src.process_data import create_summaries
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)


async def process_pdf_and_generate_content(
        pdf_path: str,
        generator: ContentGenerator,
        progress_bar=None
) -> List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    """
    Process PDF and generate content for all summaries

    Args:
        pdf_path: Path to PDF file
        generator: Initialized ContentGenerator instance
        progress_bar: Optional Streamlit progress bar

    Returns:
        List of tuples containing results and stats for each summary
    """
    try:
        # Generate summaries from PDF
        summaries = create_summaries(pdf_path)
        all_results = []

        total_summaries = len(summaries)
        for i, summary in enumerate(summaries):
            try:
                # Update progress
                if progress_bar:
                    progress_bar.progress((i + 1) / total_summaries)
                    st.write(f"Processing section {i + 1} of {total_summaries}...")

                # Generate content for all platforms
                results = {}
                for platform in generator.platforms:
                    try:
                        result = await generator.generate_content_with_image(
                            content=summary.summary,
                            title=summary.title,
                            platform=platform
                        )
                        results[platform] = result
                        logging.info(f"Generated content for {platform} - section {i + 1}")
                    except Exception as e:
                        logging.error(f"Failed to generate {platform} content: {str(e)}")
                        results[platform] = None

                # Generate statistics
                stats = get_content_statistics(results)
                all_results.append((results, stats))

            except Exception as e:
                logging.error(f"Error processing summary {i + 1}: {str(e)}")
                continue

        return all_results

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise


def get_content_statistics(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Generate statistics for content"""
    stats = {}

    for platform, content in results.items():
        if content is None:
            stats[platform] = {"status": "failed"}
            continue

        platform_stats = {"status": "success"}

        if platform == "instagram":
            platform_stats.update({
                "caption_length": len(content.caption),
                "num_hashtags": len(content.hashtags),
                "num_visual_elements": len(content.visual_elements)
            })

        elif platform == "linkedin":
            platform_stats.update({
                "content_length": len(content.content),
                "num_insights": len(content.key_insights),
                "num_hashtags": len(content.hashtags)
            })

        elif platform == "blog":
            platform_stats.update({
                "word_count": content.word_count,
                "num_sections": len(content.sections),
                "num_keywords": len(content.keywords)
            })

        stats[platform] = platform_stats

    return stats


async def main():
    st.set_page_config(page_title="Content Generation Dashboard", layout="wide")
    st.title("Content Generation Dashboard")

    # Initialize generator in session state
    if 'generator' not in st.session_state:
        st.session_state.generator = ContentGenerator(
            model_name="gpt-4o",
            temperature=0.3
        )

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
        max_retries = st.number_input("Max Retries", 1, 5, 3)

        if st.button("Update Settings"):
            st.session_state.generator = ContentGenerator(
                model_name="gpt-4o",
                temperature=temperature
            )
            st.success(f"Temperature updated to {temperature}")

    # File uploader
    uploaded_file = st.file_uploader("Choose your PDF file", type=['pdf'])

    if uploaded_file is not None:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = Path(tmp_file.name)

            with st.spinner("Processing PDF and generating content..."):
                # Process PDF and generate content
                all_results = await process_pdf_and_generate_content(
                    str(temp_path),
                    st.session_state.generator,
                    progress_bar
                )

                # Clear progress bar and status
                progress_bar.empty()
                status_text.empty()

                # Display results
                st.success(f"Generated content for {len(all_results)} sections!")

                # Create tabs for each section
                if len(all_results) > 1:
                    tabs = st.tabs([f"Section {i + 1}" for i in range(len(all_results))])
                    for i, (results, stats) in enumerate(all_results):
                        with tabs[i]:
                            display_content_set(results, i, stats)
                            st.divider()
                            show_download_buttons(results, i)
                else:
                    results, stats = all_results[0]
                    display_content_set(results, 0, stats)
                    st.divider()
                    show_download_buttons(results, 0)

                # Download all button
                st.divider()
                if st.button("Download All Content"):
                    combined_results = {
                        f"section_{i}": {
                            "results": results,
                            "stats": stats
                        }
                        for i, (results, stats) in enumerate(all_results)
                    }

                    st.download_button(
                        "Download Complete Results",
                        data=json.dumps(combined_results, indent=2),
                        file_name="all_content.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error in main execution: {str(e)}", exc_info=True)

        finally:
            # Clean up temporary file
            if 'temp_path' in locals():
                temp_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())