import deepl
import time
from typing import Union, List


def translate_text(text: Union[str, List[str]],
                   source_lang: str = 'auto',
                   target_lang: str = 'EN') -> Union[str, List[str]]:
    """
    Translates text using DeepL with proper rate limiting and retries.

    Args:
        text: String or list of strings to translate.
        source_lang: Source language code (e.g., 'DE' for German).
        target_lang: Target language code (e.g., 'EN' for English).

    Returns:
        Translated text or list of translated texts.
    """
    # Set your DeepL API key here
    api_key = "YOUR_DEEPL_API_KEY"

    # Initialize the DeepL translator
    translator = deepl.Translator(api_key)

    def translate_with_retry(texts: List[str], max_retries: int = 5) -> List[str]:
        """Translate with retry logic and increased delays."""
        for attempt in range(max_retries):
            try:
                # Wait longer between attempts
                wait_time = (attempt + 1) * 10  # Increase wait time on each retry
                time.sleep(wait_time)

                # Translate batch of texts
                result = translator.translate_text(texts, source_lang=source_lang, target_lang=target_lang)
                return [translation.text for translation in result]
            except Exception as e:
                print(f"Warning: Translation error - {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time + 5} seconds...")
                else:
                    return texts  # Return original text if max retries reached
        return texts

    def process_text(input_text: str) -> str:
        # Clean up the text
        cleaned_text = input_text.replace('\n', ' ').strip()

        # Split into smaller chunks if needed (400 chars to be safe)
        if len(cleaned_text) > 400:
            # Split on sentences
            sentences = [s.strip() + '.' for s in cleaned_text.split('.') if s.strip()]
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) > 400:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            # Translate chunks with proper waiting
            translated_chunks = translate_with_retry(chunks)
            return ' '.join(translated_chunks)
        else:
            # Short enough to translate directly
            return translate_with_retry([cleaned_text])[0]

    print("Processing corpus...")

    # Handle both single string and list inputs
    if isinstance(text, str):
        return process_text(text)
    else:
        results = []
        # Process texts in small batches
        batch_size = 3
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            translated_batch = [process_text(t) for t in batch]
            results.extend(translated_batch)
        return results


# Example usage
if __name__ == "__main__":
    corpus = """Concular steht auf drei Beinen. Erstens: Bestands-
    erfassung – das läuft per Handarbeit, also dem kon-
    kreten Abklopfen von Wänden etwa, und digital,
    indem wir genau erfassen, was in Gebäuden verbaut
    wurde, was schadstofffrei ist, was zerstörungsfrei aus-
    gebaut werden kann und wiederverwert- oder wieder-
    verwendbar ist."""

    translated = translate_text(corpus, source_lang='DE', target_lang='EN-GB')
    print(f"Original: {corpus}")
    print(f"Translated: {translated}")
