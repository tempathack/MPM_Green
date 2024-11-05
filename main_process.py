from src.process_data import create_summaries
from src.create_chains import (ContentGenerator,InstagramPostOutput,LinkedInPostOutput,
                               CompanyBlogOutput,create_content_batch,validate_content_batch,
                               save_content_batch,get_content_statistics,)
from src.configs import PDF_PATH
from typing import Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

res=create_summaries(PDF_PATH)


# Initialize generator
generator = ContentGenerator(model_name="gpt-4o")

# Generate content
results = create_content_batch(

    title="Your Title",
    content="Your Content",
    generator=generator
)

# Validate and save
validate_content_batch(results)
saved_paths = save_content_batch(results)

# Get statistics
stats = get_content_statistics(results)


for re  in res:
    results = create_content_batch(
        title=re.title,
        content=re.summary,
        generator=generator
    )
    # Validate and save
    validate_content_batch(results)
    saved_paths = save_content_batch(results)

    # Get statistics
    stats = get_content_statistics(results)
    print(results)




