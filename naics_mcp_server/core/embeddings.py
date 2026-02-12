"""
Embedding generation and management for semantic search.

This module handles the vector representation of NAICS descriptions,
making semantic search possible.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Manages text embeddings for semantic search.

    This class creates vector representations of text that capture
    semantic meaning, enabling similarity-based search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[Path] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache the model (reduces download time)
        """
        self.model_name = model_name
        self.model = None

        # Use portable cache directory
        if cache_dir:
            self.cache_dir = cache_dir if isinstance(cache_dir, Path) else Path(cache_dir)
        else:
            # Default to user's cache directory
            self.cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = None

    def load_model(self) -> None:
        """
        Load the embedding model.

        This is separate from __init__ to allow lazy loading.
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir)
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model {self.model_name}: {e}")

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            normalize: Whether to normalize the vector (for cosine similarity)

        Returns:
            Embedding vector as numpy array
        """
        if not self.model:
            self.load_model()

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Process texts in batches for efficiency
            normalize: Whether to normalize vectors
            show_progress: Show progress bar for large batches

        Returns:
            Array of embeddings
        """
        if not self.model:
            self.load_model()

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress
            )
            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # If embeddings are normalized, dot product equals cosine similarity
        similarity = np.dot(embedding1, embedding2)

        # Ensure result is in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))

    def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings from candidates.

        Args:
            query_embedding: Query vector
            candidate_embeddings: Array of candidate vectors
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of indices and similarity scores
        """
        # Compute all similarities at once (vectorized)
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]
        valid_similarities = similarities[valid_indices]

        # Get top-k indices
        if len(valid_indices) > top_k:
            top_indices = np.argpartition(valid_similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(valid_similarities[top_indices])][::-1]
        else:
            top_indices = np.argsort(valid_similarities)[::-1]

        # Return results with indices and scores
        results = []
        for idx in top_indices:
            original_idx = valid_indices[idx]
            results.append({
                "index": int(original_idx),
                "similarity": float(valid_similarities[idx])
            })

        return results


class EmbeddingCache:
    """
    Manages cached embeddings for efficient retrieval.

    This avoids recomputing embeddings for the same text.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None
        """
        embedding = self.cache.get(text)
        if embedding is not None:
            self.hits += 1
        else:
            self.misses += 1
        return embedding

    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            embedding: The embedding vector
        """
        # Simple FIFO eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[text] = embedding

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
