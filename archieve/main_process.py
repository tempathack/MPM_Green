from src.process_data import create_summaries
from archieve.create_chains import (ContentGenerator, create_content_batch, validate_content_batch,
                                    save_content_batch, get_content_statistics, )
from configs.configs import PDF_PATH

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




