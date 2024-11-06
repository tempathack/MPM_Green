from deep_translator import MyMemoryTranslator
import time
from typing import Union, List
from tqdm import tqdm


def translate_text(text: Union[str, List[str]],
                   source_lang: str = 'auto',
                   target_lang: str = 'en-GB') -> Union[str, List[str]]:
    """
    Translates text using MyMemory translator (free, no API key required)

    Args:
        text: String or list of strings to translate
        source_lang: Source language code (e.g., 'fr-FR' for French, 'auto' for automatic detection)
        target_lang: Target language code (e.g., 'en-GB' for English)

    Returns:
        Translated text or list of translated texts
    """
    # Map common language codes to their full versions
    LANGUAGE_MAP = {
        'en': 'en-GB',
        'fr': 'fr-FR',
        'es': 'es-ES',
        'de': 'de-DE',
        'it': 'it-IT',
        'pt': 'pt-PT',
        'nl': 'nl-NL',
        'pl': 'pl-PL',
        'ru': 'ru-RU',
        'ja': 'ja-JP',
        'zh': 'zh-CN',
        'ko': 'ko-KR',
        'ar': 'ar-SA',
        'hi': 'hi-IN',
        'auto': 'auto'
    }

    # Convert short language codes to full codes
    source_lang = LANGUAGE_MAP.get(source_lang, source_lang)
    target_lang = LANGUAGE_MAP.get(target_lang, target_lang)

    translator = MyMemoryTranslator(source=source_lang, target=target_lang)

    # Handle character limit (MyMemory has a 500-character limit per request)
    CHUNK_SIZE = 450  # Slightly lower than 500 to be safe

    def translate_chunk(chunk: str) -> str:
        if not chunk.strip():
            return chunk

        try:
            translation = translator.translate(chunk)
            time.sleep(1)  # Rate limiting to avoid issues
            return translation
        except Exception as e:
            print(f"Warning: Translation error - {str(e)}")
            return chunk  # Return original text if translation fails

    def process_long_text(input_text: str) -> str:
        if len(input_text) <= CHUNK_SIZE:
            return translate_chunk(input_text)

        # Split into chunks at sentence boundaries when possible
        chunks = []
        current_chunk = ""

        for sentence in input_text.replace("。", ".").split("."):
            if not sentence.strip():
                continue

            if len(current_chunk) + len(sentence) < CHUNK_SIZE:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."

        if current_chunk:
            chunks.append(current_chunk)

        # Translate each chunk
        translated_chunks = []
        for chunk in tqdm(chunks, desc="Translating chunks"):
            translated_chunks.append(translate_chunk(chunk))

        return " ".join(translated_chunks)

    print("Processing corpus...")

    # Handle both single strings and lists
    if isinstance(text, str):
        return process_long_text(text)
    else:
        results = []
        for item in tqdm(text, desc="Translating texts"):
            results.append(process_long_text(item))
        return results


# Example usage:
if __name__ == "__main__":
    # Example with proper language codes
    corpus = """Concular steht auf drei Beinen. Erstens: Bestands-
erfassung – das läuft per Handarbeit, also dem kon-
kreten Abklopfen von Wänden etwa, und digital,
indem wir genau erfassen, was in Gebäuden verbaut
wurde, was schadstofffrei ist, was zerstörungsfrei aus-
gebaut werden kann und wiederverwert- oder wieder-
verwendbar ist. Zweitens: Vermittlung – hier geht es
um Rezertifizierung und darum, Hersteller und Käu-
fer ausfindig zu machen sowie ums Matchen von Ver-
käufern und Käufern bei allem, was nicht in den Müll
muss. Die dritte Säule ist der Materialpass oder „Life-
Cycle Passport“, der dokumentiert, welche Materialien
verbaut sind und wie zirkulär ein Gebäude ist.
    """

    # Using full language codes
    translated_text = translate_text(corpus, source_lang='de', target_lang='en-GB')
    print(f"\nOriginal: {corpus}")
    print(f"Translated: {translated_text}")