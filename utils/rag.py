"""
Author: Nikhil Nageshwar Inturi (GitHub: unikill066, email: inturinikhilnageshwar@gmail.com)

RAG Manager Module
Handles chunking, indexing (embeddings), document storage, and similarity search
"""

import os, pickle, logging, json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from langchain.text_splitter import (RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter)
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from doc_processor import DocumentProcessor

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    """
    Manages the complete RAG pipeline: chunking, indexing, storage, and retrieval
    """
    
    def __init__(self, vector_store_type: str = "chroma", embedding_model: str = "openai", vector_store_path: str = "./vector_db", chunk_size: int = 500, chunk_overlap: int = 200, chunking_strategy: str = "recursive"):
        """
        Initialize RAG Manager
        
        Args:
            vector_store_type: Type of vector store ("chroma", "faiss")
            embedding_model: Embedding model to use ("openai", "huggingface")
            vector_store_path: Path to store vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for text chunking
        """
        self.vector_store_type = vector_store_type
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        
        self.doc_processor = DocumentProcessor()
        self.embeddings = self._initialize_embeddings(embedding_model)
        self.text_splitter = self._initialize_text_splitter()
        self.vector_store = self._initialize_vector_store()
        self.metadata_file = self.vector_store_path / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _initialize_embeddings(self, model_type: str):
        """Initialize embedding model"""
        if model_type == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small")
        elif model_type == "huggingface":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unsupported embedding model: {model_type}")
    
    def _initialize_text_splitter(self):
        """Initialize text splitter based on strategy"""
        if self.chunking_strategy == "recursive":
            return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len, separators=["\n\n", "\n", " ", ""])
        elif self.chunking_strategy == "token":
            return TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        elif self.chunking_strategy == "character":
            return CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n")
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.chunking_strategy}")
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store"""
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        if self.vector_store_type == "chroma":
            return Chroma(persist_directory=str(self.vector_store_path), embedding_function=self.embeddings)
        elif self.vector_store_type == "faiss":
            faiss_path = self.vector_store_path / "faiss_index"
            if faiss_path.exists():
                return FAISS.load_local(str(faiss_path), self.embeddings)
            else:
                return FAISS.from_texts(["caps_rag"], self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = list()
        
        for doc in documents:
            try:
                chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({'chunk_index': i, 'total_chunks': len(chunks), 'chunk_size': len(chunk), 'chunking_strategy': self.chunking_strategy})
                    chunked_doc = Document(page_content=chunk, metadata=chunk_metadata)
                    chunked_docs.append(chunked_doc)
            except Exception as e:
                logger.error(f"Error chunking document: {e}")
                continue
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def add_documents_from_files(self, file_paths: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None, collection_name: str = "default") -> Dict[str, Any]:
        """
        Add documents from files to the vector store
        
        Args:
            file_paths: List of file paths to process
            metadata_list: Optional metadata for each file
            collection_name: Name of the collection to add to
            
        Returns:
            Dictionary with processing results
        """
        documents = self.doc_processor.load_multiple_files(file_paths, metadata_list)
        
        if not documents:
            return {"error": "No documents were loaded"}
        
        for doc in documents:
            doc.metadata['collection'] = collection_name
            doc.metadata['added_at'] = datetime.now().isoformat()
        
        return self.add_documents(documents, collection_name)
    
    def add_documents_from_directory(self, directory_path: str, recursive: bool = True, file_pattern: str = "*", metadata: Optional[Dict[str, Any]] = None, collection_name: str = "default") -> Dict[str, Any]:
        """
        Add all documents from a directory to the vector store
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            file_pattern: Pattern to match files
            metadata: Additional metadata for all documents
            collection_name: Name of the collection to add to
            
        Returns:
            Dictionary with processing results
        """
        documents = self.doc_processor.load_directory(directory_path, recursive, file_pattern, metadata)
        
        if not documents:
            return {"error": "No documents were loaded from directory"}
   
        for doc in documents:
            doc.metadata['collection'] = collection_name
            doc.metadata['added_at'] = datetime.now().isoformat()
        
        return self.add_documents(documents, collection_name)
    
    def add_documents(self, documents: List[Document], collection_name: str = "default") -> Dict[str, Any]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
            collection_name: Name of the collection
            
        Returns:
            Dictionary with processing results
        """
        try:
            chunked_docs = self.chunk_documents(documents)
            
            if not chunked_docs:
                return {"error": "No chunks were created"}
            
            filtered_docs = filter_complex_metadata(chunked_docs)

            if self.vector_store_type == "chroma":
                self.vector_store.add_documents(filtered_docs)
                self.vector_store.persist()
            elif self.vector_store_type == "faiss":
                texts = [doc.page_content for doc in filtered_docs]
                metadatas = [doc.metadata for doc in filtered_docs]
                self.vector_store.add_texts(texts, metadatas)
                faiss_path = self.vector_store_path / "faiss_index"
                self.vector_store.save_local(str(faiss_path))
            
            if collection_name not in self.metadata:
                self.metadata[collection_name] = {'created_at': datetime.now().isoformat(), 'document_count': 0, 'chunk_count': 0}
            
            self.metadata[collection_name]['document_count'] += len(documents)
            self.metadata[collection_name]['chunk_count'] += len(chunked_docs)
            self.metadata[collection_name]['last_updated'] = datetime.now().isoformat()
            self._save_metadata()
            result = {"success": True, "documents_processed": len(documents), "chunks_created": len(chunked_docs), "collection": collection_name, "total_documents": self.metadata[collection_name]['document_count'], "total_chunks": self.metadata[collection_name]['chunk_count']}
            logger.info(f"Successfully added {len(documents)} documents ({len(chunked_docs)} chunks) to collection '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"error": str(e)}
    
    def similarity_search(self, query: str, k: int = 5, collection_filter: Optional[str] = None,
                         metadata_filter: Optional[Dict[str, Any]] = None,
                         score_threshold: Optional[float] = None) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            collection_filter: Filter by collection name
            metadata_filter: Additional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant Document objects
        """
        try:
            search_filter = {}
            if collection_filter:
                search_filter['collection'] = collection_filter
            if metadata_filter:
                search_filter.update(metadata_filter)
            
            if self.vector_store_type == "chroma":
                if search_filter:
                    results = self.vector_store.similarity_search(query, k=k, filter=search_filter)
                else:
                    results = self.vector_store.similarity_search(query, k=k)
            elif self.vector_store_type == "faiss":
                results = self.vector_store.similarity_search(query, k=k)
                
                if search_filter:
                    filtered_results = []
                    for doc in results:
                        match = True
                        for key, value in search_filter.items():
                            if key not in doc.metadata or doc.metadata[key] != value:
                                match = False
                                break
                        if match:
                            filtered_results.append(doc)
                    results = filtered_results[:k]
            
            if score_threshold is not None:
                scored_results = self.vector_store.similarity_search_with_score(query, k=k)
                results = [doc for doc, score in scored_results if score >= score_threshold]
            
            logger.info(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about collections"""
        if collection_name:
            return self.metadata.get(collection_name, {})
        else:
            return self.metadata
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        return list(self.metadata.keys())
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection (Note: This is a simplified implementation)"""
        try:
            if collection_name in self.metadata:
                del self.metadata[collection_name]
                self._save_metadata()
                logger.info(f"Deleted collection metadata for '{collection_name}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

if __name__ == "__main__":
    rag = RAG(vector_store_type="chroma", embedding_model="openai", chunk_size=500, chunk_overlap=100)

    result = rag.add_documents_from_directory("./docs", collection_name="caps_docs")
    print(f"Added documents: {result}")
    
    results = rag.similarity_search("What are the molecular mechanisms or cellular changes in the dorsal root ganglion (DRG) that are implicated in neuropathic pain?", k=3, collection_filter="caps_docs")
    
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")

    stats = rag.get_collection_stats()
    print(f"\nCollection stats: {stats}")