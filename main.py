import streamlit as st
import asyncio
import tempfile
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple
import json
from generators.content_generator import generate_content_for_all_platforms
from utils.streamlit_utils import display_content_set
from src.process_data import create_summaries
from src.retreiver_stuff import SmartRAGAnalyzer
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
        model_name: str,
        temperature: float,
        max_retries: int,
        progress_bar=None,
        status_text=None
) -> List[Dict[str, Any]]:
    """
    Process PDF and generate content for each section

    Args:
        pdf_path: Path to PDF file
        model_name: Name of the model to use
        temperature: Temperature setting for generation
        max_retries: Maximum number of retries
        progress_bar: Optional Streamlit progress bar
        status_text: Optional Streamlit text element for status updates

    Returns:
        List of generated content for each section
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
                    progress = (i + 1) / total_summaries
                    progress_bar.progress(progress)
                if status_text:
                    status_text.write(f"Processing section {i + 1} of {total_summaries}...")

                # Generate content for the summary
                results = await generate_content_for_all_platforms(
                    contents=[summary.summary],
                    titles=[summary.title],
                    corpus=corpus,
                    model_name=model_name,
                    temperature=temperature,
                    max_retries=max_retries,
                    retriever=SmartRAGAnalyzer
                )

                all_results.extend(results)
                logging.info(f"Generated content for section {i + 1}")

            except Exception as e:
                logging.error(f"Error processing section {i + 1}: {str(e)}")
                if status_text:
                    status_text.error(f"Error in section {i + 1}: {str(e)}")
                continue

        return all_results

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise


async def main():
    st.set_page_config(page_title="PDF Content Generation Dashboard", layout="wide")
    st.title("PDF Content Generation Dashboard")

    # Generator settings in sidebar
    with st.sidebar:
        st.header("Generator Settings")
        model_name = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini","o1-preview"],
            index=0,
            help="Select the model to use for content generation"
        )
        if model_name=="o1-preview":
            temperature = st.slider(
                "Temperature",
                min_value=1.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Higher values make the output more creative but less focused"
            )
        else:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Higher values make the output more creative but less focused"
            )

        max_retries = st.number_input(
            "Max Retries",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of retries for content generation"
        )

        st.markdown("\n")
        process = st.button("Start Process", type="primary")

        st.markdown("---")
        st.markdown("""
        ### Instructions
        1. Upload a PDF file
        2. Adjust generation settings if needed
        3. Wait for content generation
        4. Download results
        """)

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None and process:
        try:
            # Create progress tracking elements
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = Path(tmp_file.name)

            try:
                with st.spinner("Processing PDF and generating content..."):
                    # Process PDF and generate content
                    all_results = await process_pdf_and_generate_content(
                        str(temp_path),
                        model_name=model_name,
                        temperature=temperature,
                        max_retries=max_retries,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )

                    # Clear progress tracking
                    progress_bar.empty()
                    status_text.empty()

                    if all_results:
                        # Display success message
                        st.success(f"Successfully generated content from PDF!")

                        # Display results
                        display_content_set(all_results, stats={})

                        # Add download button for all results
                        st.divider()
                        st.download_button(                            "Download All Generated Content",
                                                                       key='dload_all',
                            data=json.dumps(
                                [
                                    {
                                        "platform": r["platform"],
                                        "title": r["title"],
                                        "content": r["content"],
                                        "result": r["result"].dict()
                                    }
                                    for r in all_results
                                ],
                                indent=2
                            ),
                            file_name="generated_content.json",
                            mime="application/json"
                        )
                    else:
                        st.warning("No content was generated. Please check the PDF file.")

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                logging.error(f"Error in main execution: {str(e)}", exc_info=True)

            finally:
                # Clean up temporary file
                temp_path.unlink()

        except Exception as e:
            st.error(f"Error handling uploaded file: {str(e)}")
            logging.error(f"File handling error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())