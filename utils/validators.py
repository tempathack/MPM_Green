import logging
from typing import Dict, Any, List
from pydantic import ValidationError


class ContentValidationError(Exception):
    """Custom error for content validation failures"""
    pass


def validate_content_batch(results: Dict[str, Any]) -> bool:
    """Validate a batch of generated content"""
    validation_errors = []

    for platform, content in results.items():
        if content is None:
            validation_errors.append(f"Missing content for {platform}")
            continue

        try:
            if platform == "instagram":
                if not content.hashtags or len(content.hashtags) < 1:
                    validation_errors.append(f"Invalid hashtags for {platform}")
                if len(content.caption) > 125:
                    validation_errors.append(f"Caption too long for {platform}")

            elif platform == "linkedin":
                if not content.key_insights or len(content.key_insights) < 1:
                    validation_errors.append(f"Missing key insights for {platform}")
                if len(content.content) > 3000:
                    validation_errors.append(f"Content too long for {platform}")

            elif platform == "blog":
                if not content.sections or len(content.sections) < 3:
                    validation_errors.append(f"Insufficient sections for {platform}")
                if content.word_count < 600 or content.word_count > 2000:
                    validation_errors.append(
                        f"Invalid word count for {platform}: {content.word_count}"
                    )

        except Exception as e:
            validation_errors.append(f"Error validating {platform}: {str(e)}")

    if validation_errors:
        raise ContentValidationError("\n".join(validation_errors))

    return True


def get_content_statistics(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Generate statistics for content batch"""
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
                "num_visual_elements": len(content.visual_elements),
                "has_image": bool(content.generated_image)
            })

        elif platform == "linkedin":
            platform_stats.update({
                "content_length": len(content.content),
                "num_insights": len(content.key_insights),
                "num_hashtags": len(content.hashtags),
                "has_image": bool(content.generated_image)
            })

        elif platform == "blog":
            platform_stats.update({
                "word_count": content.word_count,
                "num_sections": len(content.sections),
                "num_keywords": len(content.keywords),
                "has_image": bool(content.generated_image)
            })

        stats[platform] = platform_stats

    return stats


# main.py

import asyncio
import logging
from pathlib import Path
import tempfile
from typing import List, Tuple, Dict, Any
import json
from src.process_data import create_summaries
from generators.content_generator import ContentGenerator
from utils.validators import validate_content_batch, get_content_statistics

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
        generator: ContentGenerator
) -> List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    """Process PDF and generate content for all summaries"""
    try:
        # Generate summaries from PDF
        summaries = create_summaries(pdf_path)

        # Setup retriever with summaries
        texts = [summary.summary for summary in summaries]
        metadatas = [{"title": summary.title} for summary in summaries]
        await generator.setup_retriever(texts, metadatas)

        all_results = []

        # Process each summary
        for i, summary in enumerate(summaries):
            try:
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


async def main():
    try:
        # Initialize generator
        generator = ContentGenerator(
            model_name="gpt-4",
            temperature=0.3
        )

        # Your PDF path
        pdf_path = "your_uploaded_file.pdf"  # Replace with actual path

        # Process PDF and generate content
        all_results = await process_pdf_and_generate_content(
            pdf_path,
            generator
        )

        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        for i, (results, stats) in enumerate(all_results):
            # Save individual results
            for platform, content in results.items():
                if content:
                    # Save content
                    content_path = output_dir / f"{platform}_{i + 1}.json"
                    with open(content_path, 'w', encoding='utf-8') as f:
                        json.dump(content.dict(), f, indent=2, ensure_ascii=False)

                    # Save image if available
                    if content.generated_image and content.generated_image.get('url'):
                        # Image handling code here if needed
                        pass

            # Save statistics
            stats_path = output_dir / f"stats_{i + 1}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())