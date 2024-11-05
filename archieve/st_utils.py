import streamlit as st
from typing import List, Dict, Any
import json
import tempfile
from pathlib import Path
import logging
def display_hashtags(hashtags: List[str]):
    """Display hashtags with consistent styling"""
    hashtags_html = ' '.join([
        f'<span style="background-color: #f0f2f6; padding: 5px 10px; '
        f'border-radius: 15px; margin-right: 10px;">{tag}</span>'
        for tag in hashtags
    ])
    st.markdown(hashtags_html, unsafe_allow_html=True)


def display_content_set(content_set: Dict[str, Any], index: int, stats: Dict[str, Dict[str, Any]]):
    """Display a single set of content with statistics"""
    st.markdown(f"### Content Set {index + 1}")

    # Create tabs for different platforms
    tab_instagram, tab_linkedin, tab_blog, tab_stats = st.tabs([
        "📱 Instagram",
        "💼 LinkedIn",
        "📝 Blog",
        "📊 Stats"
    ])

    # Instagram Content
    with tab_instagram:
        instagram_content = content_set.get('instagram')
        if instagram_content:
            st.subheader("Instagram Post")

            col1, col2 = st.columns([2, 1])
            with col1:
                # Main content
                st.markdown("##### Caption")
                st.info(instagram_content.caption)

                st.markdown("##### Hashtags")
                display_hashtags(instagram_content.hashtags)

                st.markdown("##### Mood")
                st.write(instagram_content.mood)

                st.markdown("##### Engagement Hooks")
                for hook in instagram_content.engagement_hooks:
                    st.markdown(f"- {hook}")

                # Visual elements
                with st.expander("Visual Elements"):
                    for element in instagram_content.visual_elements:
                        st.markdown(f"- {element}")

            with col2:
                st.markdown("##### Image Preview")
                pass
               # if instagram_content.image_prompt:
                  #  st.image("/api/placeholder/400/400", caption="Generated Image")
                   # with st.expander("Image Prompt"):
                        #st.write(instagram_content.image_prompt)

    # LinkedIn Content
    with tab_linkedin:
        linkedin_content = content_set.get('linkedin')
        if linkedin_content:
            st.subheader("LinkedIn Post")

            # Title and main content
            st.markdown(f"### {linkedin_content.title}")
            st.markdown("""
            <style>
            .linkedin-post {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="linkedin-post">{linkedin_content.content}</div>',
                        unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Key Insights")
                for insight in linkedin_content.key_insights:
                    st.markdown(f"- {insight}")

                st.markdown("##### Call to Action")
                st.info(linkedin_content.call_to_action)

            with col2:
                st.markdown("##### Hashtags")
                display_hashtags(linkedin_content.hashtags)

                if linkedin_content.image_prompt:
                    st.markdown("##### Image Preview")
                    st.image("/api/placeholder/1200/627", caption="Generated Image")
                    with st.expander("Image Prompt"):
                        st.write(linkedin_content.image_prompt)

                if linkedin_content.interactive_elements:
                    st.markdown("##### Interactive Elements")
                    st.json(linkedin_content.interactive_elements)

    # Blog Content
    with tab_blog:
        blog_content = content_set.get('blog')
        if blog_content:
            st.subheader("Blog Post")

            # Header section
            st.markdown(f"### {blog_content.title}")
            st.markdown(f"*{blog_content.meta_description}*")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("##### Keywords")
                display_hashtags(blog_content.keywords)

            with col2:
                st.metric("Word Count", blog_content.word_count)

            # Image preview
            if blog_content.image_prompt:
                pass
               # st.image("/api/placeholder/1200/600", caption="Featured Image")
               # with st.expander("Image Prompt"):
                 #   st.write(blog_content.image_prompt)

            # Blog sections
            st.markdown("### Content")
            for section in blog_content.sections:
                with st.expander(f"{section.heading} ({section.type})", expanded=True):
                    st.markdown(section.content)

            # SEO details
            with st.expander("SEO Details"):
                st.json(blog_content.seo_elements)

    # Statistics Tab
    with tab_stats:
        if stats:
            st.subheader("Content Statistics")

            col1, col2, col3 = st.columns(3)

            # Instagram Stats
            with col1:
                if stats.get('instagram'):
                    st.markdown("##### Instagram Metrics")
                    ig_stats = stats['instagram']
                    if ig_stats['status'] == 'success':
                        st.metric("Caption Length", ig_stats['caption_length'])
                        st.metric("Number of Hashtags", ig_stats['num_hashtags'])
                        st.metric("Visual Elements", ig_stats['num_visual_elements'])

            # LinkedIn Stats
            with col2:
                if stats.get('linkedin'):
                    st.markdown("##### LinkedIn Metrics")
                    li_stats = stats['linkedin']
                    if li_stats['status'] == 'success':
                        st.metric("Content Length", li_stats['content_length'])
                        st.metric("Key Insights", li_stats['num_insights'])
                        st.metric("Hashtags", li_stats['num_hashtags'])

            # Blog Stats
            with col3:
                if stats.get('blog'):
                    st.markdown("##### Blog Metrics")
                    blog_stats = stats['blog']
                    if blog_stats['status'] == 'success':
                        st.metric("Word Count", blog_stats['word_count'])
                        st.metric("Sections", blog_stats['num_sections'])
                        st.metric("Keywords", blog_stats['num_keywords'])

            # Overall status
            st.markdown("##### Generation Status")
            status_cols = st.columns(3)
            for i, platform in enumerate(['instagram', 'linkedin', 'blog']):
                if platform in stats:
                    with status_cols[i]:
                        status = stats[platform]['status']
                        if status == 'success':
                            st.success(f"{platform.title()}: Success")
                        else:
                            st.error(f"{platform.title()}: Failed")