import streamlit as st
from typing import List, Dict, Any
import json
from frontend_utils.diashow import create_image_slideshow
from pathlib import Path
import logging


def display_hashtags(hashtags: List[str]):
    """Display hashtags with consistent styling"""
    if hashtags:
        hashtags_html = ' '.join([
            f'<span style="background-color: #f0f2f6; padding: 5px 10px; '
            f'border-radius: 15px; margin-right: 10px;">{tag}</span>'
            for tag in hashtags
        ])
        st.markdown(hashtags_html, unsafe_allow_html=True)


def display_content_set(content_sets: List[Dict[str, Any]], stats: Dict[str, Dict[str, Any]]):
    """Display content sets with improved organization"""

    # Group content by title
    content_by_title = {}
    for item in content_sets:
        title = item['title']
        if title not in content_by_title:
            content_by_title[title] = []
        content_by_title[title].append(item)

    # Create tabs for each title
    if not content_by_title:
        st.warning("No content to display")
        return

    tabs = st.tabs([title for title in content_by_title.keys()])

    for title, tab in zip(content_by_title.keys(), tabs):
        with tab:
            content_group = content_by_title[title]

            # Create platform tabs
            platform_tabs = st.tabs(["📱 Instagram", "💼 LinkedIn", "📝 Blog"])

            # Display content for each platform
            for platform_tab, platform in zip(platform_tabs, ["instagram", "linkedin", "blog"]):
                with platform_tab:
                    # Find content for this platform
                    platform_content = next(
                        (item for item in content_group if item['platform'] == platform),
                        None
                    )

                    if not platform_content:
                        st.warning(f"No {platform} content available")
                        continue

                    result = platform_content['result']
                    platform_images = platform_content.get('images', [])  # Get images safely

                    if platform == "instagram":
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("##### Caption")
                            st.info(result.caption)

                            st.markdown("##### Hashtags")
                            display_hashtags(result.hashtags)

                            st.markdown("##### Visual Elements")
                            for element in result.visual_elements:
                                st.markdown(f"- {element}")

                            st.markdown("##### Mood")
                            st.write(result.mood)

                        with col2:
                            if platform_images:  # Check if images exist
                                st.session_state.global_increment += 1
                                create_image_slideshow(
                                    platform_images,
                                    slideshow_id=f"{platform}_{st.session_state.global_increment}"
                                )

                    elif platform == "linkedin":
                        st.subheader(result.title)
                        st.markdown(result.content)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### Key Insights")
                            for insight in result.key_insights:
                                st.markdown(f"- {insight}")

                            st.markdown("##### Call to Action")
                            st.info(result.call_to_action)

                        with col2:
                            st.markdown("##### Hashtags")
                            display_hashtags(result.hashtags)

                            if platform_images:  # Check if images exist
                                st.session_state.global_increment += 1
                                create_image_slideshow(
                                    platform_images,
                                    slideshow_id=f"{platform}_{st.session_state.global_increment}"
                                )

                    elif platform == "blog":
                        st.subheader(result.title)
                        st.markdown(f"*{result.meta_description}*")

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("##### Keywords")
                            display_hashtags(result.keywords)

                        with col2:
                            st.metric("Word Count", result.word_count)
                            st.metric("Read Time", f"{result.read_time} min")

                        if platform_images:  # Check if images exist
                            st.session_state.global_increment += 1
                            create_image_slideshow(
                                platform_images,
                                slideshow_id=f"{platform}_{st.session_state.global_increment}"
                            )

                        for section in result.sections:
                            with st.expander(f"{section.heading}", expanded=True):
                                st.markdown(section.content)

            # Download buttons
            st.divider()
            show_download_buttons(content_group)


def show_download_buttons(content_group: List[Dict[str, Any]]):
    """Display download buttons for content group"""

    if 'download_counter' not in st.session_state:
        st.session_state.download_counter = 0
    cols = st.columns(len(content_group))

    for i, content in enumerate(content_group):
        with cols[i]:
            if result := content.get('result'):
                st.session_state.download_counter += 1
                st.download_button(
                    f"Download {content['platform'].title()} Content",
                    key=f"Download_{content['platform'].title()}_Content_{st.session_state.download_counter}",
                    data=json.dumps(result.dict(), indent=2),
                    file_name=f"{content['platform']}_content.json",
                    mime="application/json"
                )


    st.session_state.download_counter += 1
    # Combined download button
    st.download_button(
        "Download All Content",
        key=f"Download_Content_{st.session_state.download_counter}",
        data=json.dumps([c['result'].dict() for c in content_group], indent=2),
        file_name="all_content.json",
        mime="application/json"
    )