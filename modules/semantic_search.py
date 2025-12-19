"""
Semantic Search Module for Turkish-AI-Terminal-Assistant

This module provides semantic search functionality using embeddings and similarity matching.
It enables natural language-based search over documents and data.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import json
import os


class SemanticSearchBase(ABC):
    """Abstract base class for semantic search implementations."""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for given text."""
        pass
    
    @abstractmethod
    def search(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar documents to query."""
        pass


class SimpleSemanticSearch(SemanticSearchBase):
    """
    Simple semantic search implementation using TF-IDF and cosine similarity.
    Can be extended with more sophisticated embedding models.
    """
    
    def __init__(self, language: str = "turkish"):
        """
        Initialize semantic search engine.
        
        Args:
            language: Language for text processing (default: "turkish")
        """
        self.language = language
        self.documents = []
        self.embeddings = []
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def embed(self, text: str) -> np.ndarray:
        """
        Generate TF-IDF embedding for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array representing the text embedding
        """
        # Preprocess text
        tokens = self._tokenize(text.lower())
        
        # Create embedding vector
        embedding = np.zeros(len(self.vocab)) if self.vocab else np.array([])
        
        for token in tokens:
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                embedding[idx] += 1
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
    
    def search(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar documents to query.
        
        Args:
            query: Search query string
            documents: List of documents to search in
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document, similarity_score) sorted by similarity
        """
        if not documents:
            return []
        
        # Build vocabulary from documents if not already done
        if not self.vocab:
            self._build_vocab(documents)
        
        # Embed query
        query_embedding = self.embed(query)
        
        # Calculate similarities
        results = []
        for doc in documents:
            doc_embedding = self.embed(doc)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append((doc, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the search index.
        
        Args:
            documents: List of documents to index
        """
        self.documents.extend(documents)
        self._build_vocab(self.documents)
        
        # Pre-compute embeddings
        self.embeddings = [self.embed(doc) for doc in self.documents]
    
    def _build_vocab(self, documents: List[str]) -> None:
        """Build vocabulary from documents."""
        vocab_set = set()
        
        for doc in documents:
            tokens = self._tokenize(doc.lower())
            vocab_set.update(tokens)
        
        self.vocab = {word: i for i, word in enumerate(sorted(vocab_set))}
        self.word_to_idx = self.vocab
        self.idx_to_word = {i: word for word, i in self.vocab.items()}
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be extended with proper NLP
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char in "çğıöşüÇĞİÖŞÜ":
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        if vec1.shape != vec2.shape:
            # Pad shorter vector with zeros
            max_len = max(len(vec1), len(vec2))
            v1 = np.zeros(max_len)
            v2 = np.zeros(max_len)
            v1[:len(vec1)] = vec1
            v2[:len(vec2)] = vec2
            vec1, vec2 = v1, v2
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class SemanticSearchEngine:
    """
    Main semantic search engine providing high-level search functionality.
    """
    
    def __init__(self, backend: Optional[SemanticSearchBase] = None, language: str = "turkish"):
        """
        Initialize semantic search engine.
        
        Args:
            backend: Custom search backend (uses SimpleSemanticSearch if None)
            language: Language for processing (default: "turkish")
        """
        self.backend = backend or SimpleSemanticSearch(language=language)
        self.language = language
        self.indexed_documents = {}
    
    def index_documents(self, documents: Dict[str, str]) -> None:
        """
        Index documents for semantic search.
        
        Args:
            documents: Dictionary mapping document IDs to document texts
        """
        self.indexed_documents = documents
        doc_texts = list(documents.values())
        
        if isinstance(self.backend, SimpleSemanticSearch):
            self.backend.add_documents(doc_texts)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Perform semantic search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            threshold: Minimum similarity score threshold (optional)
            
        Returns:
            List of result dictionaries with 'document', 'score', and optional 'id'
        """
        if not self.indexed_documents:
            return []
        
        doc_texts = list(self.indexed_documents.values())
        results = self.backend.search(query, doc_texts, top_k=top_k)
        
        # Map back to document IDs and filter by threshold
        output = []
        doc_to_id = {v: k for k, v in self.indexed_documents.items()}
        
        for doc, score in results:
            if threshold is None or score >= threshold:
                result_dict = {
                    "document": doc,
                    "score": score,
                }
                
                # Add ID if available
                if doc in doc_to_id:
                    result_dict["id"] = doc_to_id[doc]
                
                output.append(result_dict)
        
        return output
    
    def save_index(self, filepath: str) -> None:
        """
        Save indexed documents to file.
        
        Args:
            filepath: Path to save the index
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.indexed_documents, f, ensure_ascii=False, indent=2)
    
    def load_index(self, filepath: str) -> None:
        """
        Load indexed documents from file.
        
        Args:
            filepath: Path to load the index from
        """
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.indexed_documents = json.load(f)
            self.index_documents(self.indexed_documents)


# Convenience function
def create_semantic_search_engine(language: str = "turkish") -> SemanticSearchEngine:
    """
    Create a semantic search engine instance.
    
    Args:
        language: Language for text processing
        
    Returns:
        SemanticSearchEngine instance
    """
    return SemanticSearchEngine(language=language)
