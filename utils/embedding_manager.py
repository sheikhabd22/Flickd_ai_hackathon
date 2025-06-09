import os
import faiss
import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, embeddings_dir: str = "embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.index_path = self.embeddings_dir / "product_index.faiss"
        self.metadata_path = self.embeddings_dir / "product_metadata.pkl"
        self.index = None
        self.metadata = None
        
    def save_index(self, index: faiss.Index, metadata: dict):
        """Save FAISS index and metadata to disk."""
        try:
            # Save index
            faiss.write_index(index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(self) -> tuple:
        """Load FAISS index and metadata from disk."""
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                return None, None
            
            # Load index
            index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            logger.info(f"Loaded index with {index.ntotal} vectors")
            self.index = index
            self.metadata = metadata
            return index, metadata
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return None, None
    
    def build_index(self, embeddings: np.ndarray, metadata: dict):
        """Build and save a new FAISS index using cosine similarity (normalize embeddings)."""
        try:
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Create index
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            # Save index and metadata
            self.save_index(index, metadata)
            self.index = index
            self.metadata = metadata
            return index, metadata
        except Exception as e:
            logger.error(f"Failed to build index: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> tuple:
        """Search the index for nearest neighbors using cosine similarity (normalize query)."""
        if self.index is None or self.metadata is None:
            raise ValueError("Index not loaded")
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        scores, indices = self.index.search(query_embedding, k)
        return scores, indices 