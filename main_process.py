from src.process_data import create_summaries
from src.create_chains import ContentGenerator,InstagramPostOutput,LinkedInPostOutput,CompanyBlogOutput
from src.configs import PDF_PATH
from typing import Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

res=create_summaries(PDF_PATH)


def generate_all_platforms(
        title: str,
        content: str,
        temperature: float = 0.4,
        model_name: str = "gpt-4o"
) -> Dict[str, Union[InstagramPostOutput, LinkedInPostOutput, CompanyBlogOutput]]:
    """
    Simple wrapper to generate content for all platforms

    Args:
        title: Content title
        content: Main content text
        temperature: Model temperature (default: 0.4)
        model_name: Model to use (default: gpt-4o)

    Returns:
        Dict with platform names as keys and generated content as values
    """
    # Initialize generator
    generator = ContentGenerator(
        model_name=model_name,
        temperature=temperature
    )

    # Generate for each platform
    results = {}
    for platform in ["instagram", "linkedin", "blog"]:
        try:
            result = generator.generate_content(
                title=title,
                content=content,
                platform=platform
            )
            results[platform] = result
        except Exception as e:
            logging.error(f"Error generating {platform} content: {str(e)}")
            results[platform] = None

    return results
for re  in res:
    output=generate_all_platforms(title=re.title,content=re.summary)
    print(output)





