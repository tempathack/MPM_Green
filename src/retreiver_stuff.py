from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from utils.translators import translate_text

@dataclass
class RetrievedChunk:
    """Data class to store retrieved chunks with their metadata"""
    content: str
    similarity_score: float
    chunk_id: int
    total_chunks: int
    chunk_position: str


@dataclass
class TextProcessingStats:
    """Statistics about the corpus processing"""
    total_chunks: int
    avg_chunk_size: int
    total_characters: int
    embedding_dimensions: int


@dataclass
class SmartSearchResult:
    """Data class to store the complete search results"""
    input_title: str
    input_content: str
    generated_query: str
    retrieved_chunks: List[RetrievedChunk]
    llm_analysis: str
    processing_stats: TextProcessingStats


class SmartRAGAnalyzer:
    def __init__(
            self,
            model_name: str = "gpt-4o",
            embedding_model: str = "text-embedding-3-large",
            chunk_size: int = 6000,
            chunk_overlap: int = 500
    ):
        """Initialize the Smart RAG analyzer"""
        load_dotenv()
        self.llm = ChatOpenAI(
            temperature=0,
            model=model_name,
            max_tokens=None,
            streaming=True
        )
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunks: List[str] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.processing_stats: Optional[TextProcessingStats] = None
        self.corpus_processed = False

    def _get_chunk_position(self, chunk_id: int, total_chunks: int) -> str:
        """Determine the relative position of a chunk in the text"""
        position = (chunk_id / total_chunks) * 100
        if position < 20:
            return "near beginning"
        elif position > 80:
            return "near end"
        else:
            return "middle"

    def process_corpus(self, corpus: str) -> None:
        """
        Process the reference corpus into searchable chunks.

        Args:
            corpus: The reference text corpus to process
        """
        translated_text = translate_text(corpus, source_lang='de', target_lang='en-GB')



        self.chunks = self.text_splitter.split_text(translated_text)

        # Calculate statistics
        total_chars = len(corpus)
        avg_chunk_size = sum(len(chunk) for chunk in self.chunks) // len(self.chunks)

        # Generate embeddings with progress tracking
        print("Generating embeddings for corpus chunks...")
        embeddings = []
        for chunk in tqdm(self.chunks, desc="Generating embeddings"):
            embedding = self.embeddings.embed_query(chunk)
            embeddings.append(embedding)

        self.chunk_embeddings = np.array(embeddings)

        # Store processing stats
        self.processing_stats = TextProcessingStats(
            total_chunks=len(self.chunks),
            avg_chunk_size=avg_chunk_size,
            total_characters=total_chars,
            embedding_dimensions=len(embeddings[0])
        )

        print(f"Corpus processed: {len(self.chunks)} chunks created")
        self.corpus_processed = True

    def _generate_optimal_query(self, title: str, content: str) -> str:
        """
        Generate an optimal search query based on input title and content.

        Args:
            title: The input title
            content: The input content to analyze

        Returns:
            str: Generated search query
        """
        query_generation_prompt = f"""
        Create an optimal search query to find relevant information in a reference corpus.
        Use the following input to generate a focused search query:

        INPUT TITLE: {title}
        INPUT CONTENT: {content[:1000]}...

        Create a search query that:
        1. Captures the main themes and concepts
        2. Includes specific technical terms if present
        3. Focuses on finding relevant examples or similar cases

        Return ONLY the search query, nothing else.
        """

        messages = [{"role": "user", "content": query_generation_prompt}]
        return self.llm.invoke(messages).content.strip()

    def search(self, title: str, content: str, top_k: int = 5) -> SmartSearchResult:
        """
        Search the processed corpus using the input title and content.

        Args:
            title: Input title to analyze
            content: Input content to analyze
            top_k: Number of chunks to retrieve

        Returns:
            SmartSearchResult containing search results and analysis
        """
        if not self.corpus_processed:
            raise ValueError("Corpus has not been processed. Call process_corpus() first.")

        # Generate optimal query from input
        print("Generating search query...")
        generated_query = self._generate_optimal_query(title, content)
        print(f"Generated query: {generated_query}")

        # Generate query embedding and find similar chunks
        query_embedding = self.embeddings.embed_query(generated_query)
        similarities = np.dot(self.chunk_embeddings, query_embedding) / (
                np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved_chunks = [
            RetrievedChunk(
                content=self.chunks[idx],
                similarity_score=float(similarities[idx]),
                chunk_id=idx,
                total_chunks=self.processing_stats.total_chunks,
                chunk_position=self._get_chunk_position(idx, self.processing_stats.total_chunks)
            )
            for idx in top_indices
        ]

        # Analyze retrieved chunks
        analysis_prompt = f"""
        Analyze the relevance of retrieved text chunks to the input:

        INPUT TITLE: {title}
        INPUT CONTENT PREVIEW: {content[:500]}...

        GENERATED QUERY: {generated_query}

        RETRIEVED CHUNKS:
        {chr(10).join(f'Chunk {chunk.chunk_id}/{chunk.total_chunks} ({chunk.chunk_position}, Score: {chunk.similarity_score:.3f}):'
                      f'{chr(10)}{chunk.content}{chr(10)}---'
                      for chunk in retrieved_chunks)}

        Provide:
        1. Relevance of retrieved chunks to the input
        2. Key insights or connections found
        3. Suggestions for refining the search if needed

        Analysis:
        """

        print("Analyzing retrieved chunks...")
        messages = [{"role": "user", "content": analysis_prompt}]
        analysis = self.llm.invoke(messages).content.strip()

        return SmartSearchResult(
            input_title=title,
            input_content=content,
            generated_query=generated_query,
            retrieved_chunks=retrieved_chunks,
            llm_analysis=analysis,
            processing_stats=self.processing_stats
        )


def main():
    # Example usage
    corpus = """Your large reference corpus here..."""

    title = "Example Query Title"
    content = "Example query content..."

    # Initialize analyzer
    analyzer = SmartRAGAnalyzer()

    # Process corpus (only needs to be done once)
    analyzer.process_corpus(corpus)

    # Search using input title and content
    results = analyzer.search(title, content)


    # Print results
    print("\nSearch Results:")
    print(f"\nInput Title: {results.input_title}")
    print(f"Generated Query: {results.generated_query}")

    print("\nCorpus Statistics:")
    print(f"- Total chunks: {results.processing_stats.total_chunks}")
    print(f"- Average chunk size: {results.processing_stats.avg_chunk_size} characters")

    print("\nRetrieved Chunks:")
    for chunk in results.retrieved_chunks:
        print(f"\nChunk {chunk.chunk_id}/{chunk.total_chunks} "
              f"({chunk.chunk_position}, Score: {chunk.similarity_score:.3f}):")
        print(f"{chunk.content[:200]}...")

    print("\nAnalysis:")
    print(results.llm_analysis)


if __name__ == "__main__":
    main()