import streamlit as st
from typing import List, Dict, Any
import json
import tempfile
from pathlib import Path
import logging
from src.process_data import create_summaries
from src.create_chains import (
    ContentGenerator,
    InstagramPostOutput,
    LinkedInPostOutput,
    CompanyBlogOutput,
    create_content_batch,
    validate_content_batch,
    save_content_batch,
    get_content_statistics,
)
from src.st_utils import  *

def process_pdf_and_generate_content(pdf_path: str, generator: ContentGenerator):
    """
    Process PDF and generate content for all summaries

    Args:
        pdf_path: Path to PDF file
        generator: Initialized ContentGenerator instance

    Returns:
        List of tuples containing results and stats for each summary
    """
    try:
        # Generate summaries from PDF
        summaries = create_summaries(pdf_path)

        all_results = []

        # Process each summary
        for summary in summaries:
            # Generate content
            results = create_content_batch(
                title=summary.title,
                content=summary.summary,
                generator=generator
            )

            # Validate content
            validate_content_batch(results)

            # Generate statistics
            stats = get_content_statistics(results)

            all_results.append((results, stats))

        return all_results

    except Exception as e:
        logging.error(f"Error processing PDF and generating content: {str(e)}")
        raise


def main():
    st.set_page_config(page_title="Content Generation Dashboard", layout="wide")

    st.title("Content Generation Dashboard")

    # Initialize generator in session state if not exists
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
            # Create new generator with updated temperature
            st.session_state.generator = ContentGenerator(
                model_name="gpt-4o",
                temperature=temperature
            )
            st.success(f"Temperature updated to {temperature}")

    # File uploader
    uploaded_file = st.file_uploader("Choose your PDF file", type=['pdf'])

    if uploaded_file is not None:
        with st.spinner("Processing PDF and generating content..."):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = Path(tmp_file.name)

                # Process PDF and generate content
                all_results = process_pdf_and_generate_content(
                    str(temp_path),
                    st.session_state.generator
                )

                # Display success message
                st.success(f"Generated content for {len(all_results)} summaries!")

                # Display each set of results
                for i, (results, stats) in enumerate(all_results):
                    st.divider()
                    display_content_set(results, i, stats)

                    # Download buttons for each set
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if results.get('instagram'):
                            st.download_button(
                                f"Download Instagram Content {i + 1}",
                                data=json.dumps(results['instagram'].dict(), indent=2),
                                file_name=f"instagram_content_{i + 1}.json",
                                mime="application/json"
                            )

                    with col2:
                        if results.get('linkedin'):
                            st.download_button(
                                f"Download LinkedIn Content {i + 1}",
                                data=json.dumps(results['linkedin'].dict(), indent=2),
                                file_name=f"linkedin_content_{i + 1}.json",
                                mime="application/json"
                            )

                    with col3:
                        if results.get('blog'):
                            st.download_button(
                                f"Download Blog Content {i + 1}",
                                data=json.dumps(results['blog'].dict(), indent=2),
                                file_name=f"blog_content_{i + 1}.json",
                                mime="application/json"
                            )

                # Option to download all content
                st.divider()
                if st.button("Download All Content"):
                    # Combine all results
                    combined_results = {
                        f"summary_{i}": {
                            "results": results,
                            "stats": stats
                        }
                        for i, (results, stats) in enumerate(all_results)
                    }

                    # Create download button
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
    main()