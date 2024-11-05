
import streamlit as st
from typing import List, Dict, Any
import json


def display_hashtags(hashtags: List[str]):
    """Display hashtags with consistent styling"""
    hashtags_html = ' '.join([
        f'<span style="background-color: #f0f2f6; padding: 5px 10px; '
        f'border-radius: 15px; margin-right: 10px;">{tag}</span>'
        for tag in hashtags
    ])
    st.markdown(hashtags_html, unsafe_allow_html=True)


def display_content_set(content_set: Dict[str, Any], index: int):
    """Display a single set of content"""
    st.markdown(f"### Content Set {index + 1}")

    # Create tabs for different platforms
    tabs = st.tabs(["Instagram", "LinkedIn", "Blog"])

    # Instagram Content
    with tabs[0]:
        if content_set.get('instagram'):
            instagram = content_set['instagram']
            st.subheader("📱 Instagram Post")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Caption:**")
                st.write(instagram.caption)

                st.markdown("**Hashtags:**")
                display_hashtags(instagram.hashtags)

                st.markdown("**Mood:**")
                st.write(instagram.mood)

                st.markdown("**Engagement Hooks:**")
                for hook in instagram.engagement_hooks:
                    st.markdown(f"- {hook}")

            with col2:
                st.markdown("**Visual Preview:**")
                st.image("/api/placeholder/400/400", caption="Generated Image")

    # LinkedIn Content
    with tabs[1]:
        if content_set.get('linkedin'):
            linkedin = content_set['linkedin']
            st.subheader("💼 LinkedIn Post")

            st.markdown(f"**{linkedin.title}**")
            st.write(linkedin.content)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Insights:**")
                for insight in linkedin.key_insights:
                    st.markdown(f"- {insight}")

                st.markdown("**Call to Action:**")
                st.write(linkedin.call_to_action)

            with col2:
                st.markdown("**Visual Preview:**")
                st.image("/api/placeholder/1200/627", caption="Generated Image")

                st.markdown("**Hashtags:**")
                display_hashtags(linkedin.hashtags)

    # Blog Content
    with tabs[2]:
        if content_set.get('blog'):
            blog = content_set['blog']
            st.subheader("📝 Blog Post")

            st.markdown(f"## {blog.title}")
            st.markdown(f"*{blog.meta_description}*")

            # Keywords
            st.markdown("**Keywords:**")
            display_hashtags(blog.keywords)

            # Featured Image
            st.image("/api/placeholder/1200/600", caption="Featured Image")

            # Sections
            for section in blog.sections:
                with st.expander(f"{section.heading}", expanded=True):
                    st.markdown(section.content)

            # Word count and SEO
            st.markdown(f"**Word Count:** {blog.word_count} words")
            with st.expander("SEO Details"):
                st.json(blog.seo_elements)


def main():
    st.set_page_config(page_title="Content Generation Dashboard", layout="wide")

    st.title("Content Generation Dashboard")

    # Introduction
    st.markdown("""
    ## Multi-Platform Content Analysis

    This dashboard displays generated content optimized for different platforms:
    - Instagram: Visual-focused social media content
    - LinkedIn: Professional network posts
    - Blog: In-depth article content
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose your PDF file", type=['pdf'])

    if uploaded_file is not None:
        try:
            # Process the PDF and get content sets
            # For now, using sample data
            content_sets = [
                # Your list of content dictionaries here
            ]

            # Display each content set
            for i, content_set in enumerate(content_sets):
                st.divider()
                display_content_set(content_set, i)

                # Download buttons for each content set
                col1, col2, col3 = st.columns(3)

                with col1:
                    if content_set.get('instagram'):
                        st.download_button(
                            f"Download Instagram Content {i + 1}",
                            data=json.dumps(content_set['instagram'].dict(), indent=2),
                            file_name=f"instagram_content_{i + 1}.json",
                            mime="application/json"
                        )

                with col2:
                    if content_set.get('linkedin'):
                        st.download_button(
                            f"Download LinkedIn Content {i + 1}",
                            data=json.dumps(content_set['linkedin'].dict(), indent=2),
                            file_name=f"linkedin_content_{i + 1}.json",
                            mime="application/json"
                        )

                with col3:
                    if content_set.get('blog'):
                        st.download_button(
                            f"Download Blog Content {i + 1}",
                            data=json.dumps(content_set['blog'].dict(), indent=2),
                            file_name=f"blog_content_{i + 1}.json",
                            mime="application/json"
                        )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()