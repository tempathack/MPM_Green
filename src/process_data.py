import os
import re
import numpy as np
import pandas as pd
import openai
import faiss
from tqdm import tqdm
from fpdf import FPDF
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import OpenAI
from dotenv import load_dotenv
import numpy as np
import faiss
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional
import logging
from src.utils import setup_summarization_chain
load_dotenv()

class ClusterOptimizer:
    def __init__(self, min_clusters: int = 2, max_clusters: int = 50,
                 iteration_step: int = 2, niter: int = 20):
        """
        Initialize cluster optimizer

        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            iteration_step: Step size for cluster number iteration
            niter: Number of iterations for k-means
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.iteration_step = iteration_step
        self.niter = niter
        self.scores = {}

    def compute_silhouette_score(self, array: np.ndarray, kmeans: faiss.Kmeans) -> float:
        """
        Compute silhouette score for given clustering

        Args:
            array: Input data array
            kmeans: Trained FAISS k-means object

        Returns:
            float: Silhouette score
        """
        # Get cluster assignments for all points
        _, labels = kmeans.index.search(array, 1)
        labels = labels.reshape(-1)

        try:
            # Calculate silhouette score
            score = silhouette_score(array, labels, metric='euclidean')
            return score
        except ValueError as e:
            logging.warning(f"Error computing silhouette score: {e}")
            return -1


    def create_clusters(self, array: np.ndarray,
                        min_clusters: Optional[int] = None,
                        max_clusters: Optional[int] = None) -> Tuple[faiss.IndexFlatL2, np.ndarray, int]:
        """
        Create clusters using FAISS with optimal number of clusters

        Args:
            array: Input data array
            min_clusters: Minimum number of clusters (optional)
            max_clusters: Maximum number of clusters (optional)

        Returns:
            Tuple containing:
            - FAISS index
            - Centroids array
            - Optimal number of clusters
        """
        array = array.astype('float32')
        dimension = array.shape[1]

        # Set cluster range
        min_k = min_clusters or self.min_clusters
        max_k = max_clusters or min(self.max_clusters, array.shape[0])

        best_score = -1
        best_kmeans = None
        optimal_k = min_k

        logging.info("Starting cluster optimization...")

        # Try different numbers of clusters
        for k in range(min_k, max_k + 1, self.iteration_step):
            # Initialize and train k-means
            kmeans = faiss.Kmeans(
                dimension,  # dimensionality of the data
                k,  # number of clusters
                niter=self.niter,
                verbose=False,
                gpu=False  # set to True if using GPU
            )

            try:
                kmeans.train(array)

                # Compute silhouette score
                score = self.compute_silhouette_score(array, kmeans)
                self.scores[k] = score

                logging.info(f"Clusters: {k}, Silhouette Score: {score:.4f}")

                # Update best score if current score is better
                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans
                    optimal_k = k

            except Exception as e:
                logging.error(f"Error training k-means with {k} clusters: {e}")
                continue

        if best_kmeans is None:
            raise ValueError("Failed to find valid clustering solution")

        # Create index with optimal clustering
        index = faiss.IndexFlatL2(dimension)
        index.add(array)

        logging.info(f"Optimal number of clusters: {optimal_k}")
        logging.info(f"Best silhouette score: {best_score:.4f}")

        return index, best_kmeans.centroids, optimal_k


    def plot_silhouette_scores(self):
        """Plot silhouette scores for different numbers of clusters"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.scores.keys()), list(self.scores.values()), 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True)
        return plt


def create_clusters_with_visualization(array: np.ndarray,
                                       min_clusters: int = 2,
                                       max_clusters: int = 50,
                                       save_plot: bool = False) -> Tuple[faiss.IndexFlatL2, np.ndarray, int, dict]:
    """
    Create clusters and visualize the optimization process

    Args:
        array: Input data array
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        save_plot: Whether to save the plot to file

    Returns:
        Tuple containing:
        - FAISS index
        - Centroids array
        - Optimal number of clusters
        - Dictionary of silhouette scores
    """
    # Initialize optimizer
    optimizer = ClusterOptimizer(min_clusters=min_clusters, max_clusters=max_clusters)

    # Find optimal clustering
    index, centroids, optimal_k = optimizer.create_clusters(array)

    # Create visualization
    plt = optimizer.plot_silhouette_scores()

    if save_plot:
        plt.savefig('silhouette_scores.png')

    return index, centroids, optimal_k, optimizer.scores


class BookSummarizer:
    def __init__(self, openai_api_key):
        """Initialize the BookSummarizer with OpenAI API key"""
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings_model = OpenAIEmbeddings()

    def load_book(self, pdf_path, start_page=6, end_page=1308):
        """Load and preprocess the book from PDF"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        # Cut out the open and closing parts
        pages = pages[start_page:end_page]

        # Combine the pages and replace tabs with spaces
        text = ' '.join([page.page_content.replace('\t', ' ') for page in pages])
        return text

    def clean_text(self, text):
        """Clean the text by removing unnecessary content"""
        # Remove the specific phrase and surrounding whitespace
        cleaned_text = re.sub(r'\s*Free eBooks at Planet eBook\.com\s*', '', text, flags=re.DOTALL)
        # Remove extra spaces
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        # Remove non-printable characters
        cleaned_text = re.sub(r'(David Copperfield )?[\x00-\x1F]', '', cleaned_text)
        # Replace newline characters with spaces
        cleaned_text = cleaned_text.replace('\n', ' ')
        # Remove spaces around hyphens
        cleaned_text = re.sub(r'\s*-\s*', '', cleaned_text)
        return cleaned_text

    def get_token_count(self, text):
        """Get the number of tokens in the text"""
        llm = OpenAI()
        return llm.get_num_tokens(text)

    def split_into_documents(self, text):
        """Split text into semantic chunks"""
        text_splitter = SemanticChunker(
            self.embeddings_model,
            breakpoint_threshold_type="interquartile"
        )
        return text_splitter.create_documents([text])

    def get_embeddings(self, texts):
        """Get embeddings for the documents"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return response.data

    def prepare_data(self, docs, embeddings):
        """Prepare data for clustering"""
        content_list = [doc.page_content for doc in docs]
        df = pd.DataFrame(content_list, columns=['page_content'])
        vectors = [embedding.embedding for embedding in embeddings]
        array = np.array(vectors)
        embeddings_series = pd.Series(list(array))
        df['embeddings'] = embeddings_series
        return df, array

    def create_clusters(self, array, num_clusters=10):
        """Create clusters using FAISS"""
        array = array.astype('float32')


        index, centroids, optimal_k, scores=create_clusters_with_visualization( array,
        min_clusters=2,
        max_clusters=30,
        save_plot=True)
        return index, centroids

    def select_important_documents(self, index, centroids, docs):
        """Select the most important documents from clusters"""
        D, I = index.search(centroids, 1)
        sorted_array = np.sort(I, axis=0)
        sorted_array = sorted_array.flatten()
        return [docs[i] for i in sorted_array]

    def setup_summarization_chain(self):
        """Set up the summarization chain"""

        return setup_summarization_chain()

    def generate_summaries(self, chain, extracted_docs):
        """Generate summaries for extracted documents"""
        summary_list=[]
        for doc in tqdm(extracted_docs, desc="Processing documents"):
            new_summary = chain.invoke({"text": doc.page_content})
            summary_list.append(new_summary)
        return summary_list


class PDF(FPDF):
    """Custom PDF class with header and footer"""

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Summary', 1, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def save_summary_to_pdf(summary_text, output_path="book_summary.pdf"):
    """Save the summary to a PDF file"""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Handle text encoding
    summary_utf8 = summary_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, summary_utf8)

    pdf.output(output_path)


def create_summaries(pdf_path):
    # Initialize summarizer

    summarizer = BookSummarizer(os.getenv('OPENAI_API_KEY'))

    # Load and preprocess book
    text = summarizer.load_book(pdf_path)
    clean_text = summarizer.clean_text(text)

    # Get token count
    token_count = summarizer.get_token_count(clean_text)
    print(f"Total tokens in book: {token_count}")

    # Split into documents and get embeddings
    docs = summarizer.split_into_documents(clean_text)
    embeddings = summarizer.get_embeddings([doc.page_content for doc in docs])

    # Prepare data for clustering
    df, array = summarizer.prepare_data(docs, embeddings)

    # Create clusters and select important documents
    index, centroids = summarizer.create_clusters(array)
    extracted_docs = summarizer.select_important_documents(index, centroids, docs)

    # Generate summaries
    chain = summarizer.setup_summarization_chain()
    final_summary = summarizer.generate_summaries(chain, extracted_docs)

    # Save to PDF
    #save_summary_to_pdf(final_summary)
    return final_summary


if __name__ == "__main__":
    create_summaries();