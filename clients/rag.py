import torch
from sentence_transformers import SentenceTransformer

from settings import PipeSettings


class RAGClient:
    def __init__(self, settings: PipeSettings):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.SENTENCE_TRANSFORMER_MODEL_NAME
        self.model = SentenceTransformer(self.model_name).to(self.device)
        self.max_seq_length = settings.SENTENCE_TRANSFORMER_MAX_SEQ_LENGTH
        self.model.eval()

    def similarity_search(
        self, query: str, chunks: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Perform a similarity search using the SentenceTransformer model.

        Args:
            query (str): The query string.
            chunks (list[str]): A list of document chunks to search against.
            top_k (int): The number of top similar documents to return.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the document and its similarity score.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)

        # Cosine similarity
        cosine_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), chunk_embeddings
        )

        # Get top_k results
        top_results = torch.topk(cosine_scores, k=top_k)

        return [
            (chunks[idx], score.item())
            for idx, score in zip(top_results.indices, top_results.values, strict=False)
        ]

    def get_contextual_chunks(
        self, query: str, chunks: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Get contextual chunks by adding neighboring chunks to the top-k similar chunks.

        Args:
            query (str): The query string.
            top_k (int): The number of top similar documents to return.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the document and its similarity score.
        """
        # Get the top-k similar chunks
        top_k_chunks = self.similarity_search(query, chunks, top_k)

        # Prepare the results by adding neighboring chunks
        contextual_chunks = []

        for chunk, score in top_k_chunks:
            # Get the index of the current chunk in the original list
            chunk_idx = chunks.index(chunk)

            # Prepare the contextual chunk by adding the previous and next chunk (if they exist)
            context = []
            if chunk_idx > 0:  # Add the previous chunk if it exists
                context.append(chunks[chunk_idx - 1])
            context.append(chunk)  # Always include the current chunk
            if chunk_idx < len(chunks) - 1:  # Add the next chunk if it exists
                context.append(chunks[chunk_idx + 1])

            # Combine the context into a single string
            contextual_chunks.append((" ".join(context), score))

        return contextual_chunks
