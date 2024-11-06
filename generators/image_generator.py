import asyncio
from openai import OpenAI
from typing import Tuple, Optional
import logging

class ImageGenerator:
    """Handle image generation with OpenAI's DALL-E"""

    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024"
    ) -> Tuple[Optional[str], str]:
        """Generate image from prompt"""
        try:
            logging.info(f"Generating image with prompt: {prompt[:100]}...")
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
            )

            if response.data:
                url = response.data[0].url
                revised_prompt = response.data[0].revised_prompt
                logging.info("Image generated successfully")
                return url, revised_prompt
            else:
                raise ValueError("No image data returned from API")

        except Exception as e:
            logging.error(f"Error generating image: {str(e)}")
            return None, prompt