import os
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai


# 1. Text Embedder for vector representations
class TextEmbedder:
    def __init__(self):
        """Initialize the text embedder with a pre-trained model."""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []

        for text in texts:
            # Tokenize and get model outputs
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use mean pooling to get sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
            embeddings.append(embedding)

        return np.array(embeddings)


# 2. Document Store to manage document chunks
class DocumentStore:
    def __init__(self, documents: List[str]):
        """Initialize with a list of documents."""
        self.documents = documents
        # Create document chunks by splitting documents into smaller pieces
        self.chunks = self._create_chunks(documents)
        self.chunk_embeddings = self._embed_chunks(self.chunks)

    def _create_chunks(self, documents: List[str], chunk_size: int = 200) -> List[str]:
        """Split documents into chunks of specified size."""
        chunks = []
        for doc in documents:
            # Simple implementation: split by word and group into chunks
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
        return chunks

    def _embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Create vector embeddings for each chunk."""
        # In a real implementation, use a proper embedding model
        embedder = TextEmbedder()
        return embedder.embed_texts(chunks)

    def get_chunks(self) -> List[str]:
        """Return all document chunks."""
        return self.chunks

    def get_chunk_embeddings(self) -> np.ndarray:
        """Return embeddings for all chunks."""
        return self.chunk_embeddings


# 3. Gemini Language Model
class GeminiLanguageModel:
    def __init__(self, api_key=None):
        """Initialize the Gemini language model."""
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Provide it directly or set the GOOGLE_API_KEY environment variable.")

        # Configure the Gemini API
        genai.configure(api_key=self.api_key)

        # Use Gemini 2.0 Pro Experimental as default
        self.model_name = "gemini-2.0-pro-exp-02-05"
        self.model = genai.GenerativeModel(self.model_name)

        # Default generation parameters
        self.default_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini model based on the given prompt."""
        # Merge default parameters with any provided overrides
        params = {**self.default_params, **kwargs}

        # Generate response from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(**params)
        )

        # Extract and return the text content
        return response.text

    def generate_questions(self, document: str, n_questions: int = 5) -> List[str]:
        """Generate hypothetical questions based on document content."""
        prompt = f"""
        Based on the following document, generate {n_questions} specific questions that a user might ask about this content.
        Make the questions diverse and cover different aspects of the document.
        Format your response as a numbered list, with one question per line.

        DOCUMENT:
        {document}
        """

        response = self.generate(prompt, temperature=0.8)

        # Process response to extract questions
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove any numbering or bullet points
            if line and (line[0].isdigit() and '.' in line[:3] or line.startswith('-') or line.startswith('*')):
                questions.append(line.split(' ', 1)[1].strip())
            elif line:
                questions.append(line)

        # Limit to requested number
        return questions[:n_questions]

    def answer_with_context(self, query: str, context: List[str]) -> str:
        """Generate an answer to a query based on provided context chunks."""
        # Format context for the prompt
        formatted_context = "\n\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(context)])

        prompt = f"""
        Answer the following question based only on the provided context.
        If the context doesn't contain enough information to answer fully, say so.

        QUESTION: {query}

        CONTEXT:
        {formatted_context}
        """

        # Use a lower temperature for factual responses
        return self.generate(prompt, temperature=0.3)


# 4. Hypothetical Question Generator
class HypotheticalQuestionGenerator:
    def __init__(self, llm):
        """Initialize with a language model."""
        self.llm = llm

    def generate_questions(self, documents: List[str], n_questions: int = 5) -> List[str]:
        """Generate hypothetical questions based on document content."""
        all_questions = []
        questions_per_doc = max(1, n_questions // len(documents))

        for doc in documents:
            # Use Gemini's specialized method
            doc_questions = self.llm.generate_questions(doc, questions_per_doc)
            all_questions.extend(doc_questions)

        return all_questions


# 5. Query-to-Query Search
class QueryToQuerySearch:
    def __init__(self, hypothetical_questions: List[str], question_embeddings: np.ndarray):
        """Initialize with hypothetical questions and their embeddings."""
        self.hypothetical_questions = hypothetical_questions
        self.question_embeddings = question_embeddings

    def search(self, query: str, k: int = 3) -> List[str]:
        """Find the most similar hypothetical questions to the query."""
        # Embed the query
        embedder = TextEmbedder()
        query_embedding = embedder.embed_texts([query])[0]

        # Compute similarity with hypothetical questions
        similarities = cosine_similarity([query_embedding], self.question_embeddings)[0]

        # Get top k similar questions
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_questions = [self.hypothetical_questions[i] for i in top_indices]

        return top_questions


# 6. Chunk Retriever
class ChunkRetriever:
    def __init__(self, doc_store: DocumentStore):
        """Initialize with document store."""
        self.doc_store = doc_store

    def retrieve_chunks(self, question_embeddings: np.ndarray, k: int = 3) -> List[str]:
        """Retrieve top k relevant chunks based on question embeddings."""
        chunk_embeddings = self.doc_store.get_chunk_embeddings()
        chunks = self.doc_store.get_chunks()

        # Calculate average embedding of hypothetical questions
        avg_question_embedding = np.mean(question_embeddings, axis=0).reshape(1, -1)

        # Compute similarity with document chunks
        similarities = cosine_similarity(avg_question_embedding, chunk_embeddings)[0]

        # Get top k similar chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_chunks = [chunks[i] for i in top_indices]

        return top_chunks


# 7. Complete Query Answering System
class QueryAnsweringSystem:
    def __init__(self, api_key=None, documents=None):
        """Initialize the complete query answering system with Gemini."""
        # Initialize Gemini language model
        self.llm = GeminiLanguageModel(api_key=api_key)

        # Use provided documents or defaults
        if documents is None:
            documents = [
                "Query-to-query search is a technique that maps user queries to a vector store of hypothetical questions. This helps in retrieving more relevant documents.",
                "Large Language Models can generate hypothetical questions based on document content, which improves retrieval performance.",
                "Document retrieval systems work by chunking documents into smaller pieces and embedding them into vector space."
            ]

        # Initialize document store
        self.doc_store = DocumentStore(documents)

        # Generate hypothetical questions
        question_generator = HypotheticalQuestionGenerator(self.llm)
        self.hypothetical_questions = question_generator.generate_questions(documents)
        print(f"Generated {len(self.hypothetical_questions)} hypothetical questions")

        # Embed hypothetical questions
        embedder = TextEmbedder()
        self.question_embeddings = embedder.embed_texts(self.hypothetical_questions)

        # Initialize query-to-query search
        self.q2q_search = QueryToQuerySearch(self.hypothetical_questions, self.question_embeddings)

        # Initialize chunk retriever
        self.chunk_retriever = ChunkRetriever(self.doc_store)

    def answer_query(self, query: str) -> str:
        """Process a query and generate an answer using Gemini."""
        print(f"Processing query: {query}")

        # 1. Find similar hypothetical questions
        similar_questions = self.q2q_search.search(query)
        print(f"Similar questions: {similar_questions}")

        # 2. Get embeddings for similar questions
        embedder = TextEmbedder()
        question_embeddings = embedder.embed_texts(similar_questions)

        # 3. Retrieve relevant document chunks
        relevant_chunks = self.chunk_retriever.retrieve_chunks(question_embeddings)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")

        # 4. Generate answer using Gemini with retrieved chunks as context
        answer = self.llm.answer_with_context(query, relevant_chunks)

        return answer


# Example usage
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please set the GOOGLE_API_KEY environment variable")
        print("Example: export GOOGLE_API_KEY='your-api-key'")
        exit(1)

    # Sample documents for the knowledge base
    documents = [
        "Query-to-query search is an advanced retrieval technique that maps user queries to a vector store of hypothetical questions instead of directly to documents. This approach helps bridge the semantic gap between user queries and relevant document content.",

        "In query-to-query search, an LLM first generates many hypothetical questions that might be asked about each document in the corpus. These questions and their embeddings form a vector store that user queries are compared against.",

        "The advantage of query-to-query search is that it aligns better with how users naturally formulate questions, improving retrieval precision by focusing on the information need rather than keyword matching.",

        "Document chunking is a preprocessing technique where long documents are split into smaller, more manageable pieces to improve retrieval performance and enable more precise context selection for answering specific queries."
    ]

    # Create the query answering system
    print("Initializing the Query Answering System...")
    qa_system = QueryAnsweringSystem(api_key=api_key, documents=documents)

    # Test with sample queries
    queries = [
        "How does query-to-query search work?",
        "What are the benefits of using hypothetical questions?",
        "How are documents processed in this system?"
    ]

    # Process each query
    for query in queries:
        print("\n" + "=" * 50)
        print(f"Query: {query}")
        print("-" * 50)

        answer = qa_system.answer_query(query)

        print("\nAnswer:")
        print(answer)