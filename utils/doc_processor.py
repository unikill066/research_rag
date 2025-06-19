"""
Author: Nikhil Nageshwar Inturi (GitHub: unikill066, email: inturinikhilnageshwar@gmail.com)

Document Processor Module
Handles reading various document formats: .docx, .pdf, .html, .xml, .md, .txt
"""

import os, logging
from pathlib import Path
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader, UnstructuredXMLLoader, UnstructuredMarkdownLoader, TextLoader)


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles loading and processing various document formats
    """
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.xml': UnstructuredXMLLoader,
        '.md': UnstructuredMarkdownLoader,
        '.txt': TextLoader
    }
    
    def __init__(self):
        self.processed_files = []
    
    def load_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load a single document based on its file extension
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata to attach to the document
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        try:
            loader_class = self.SUPPORTED_EXTENSIONS[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            for doc in documents:
                doc.metadata.update({'source_file': str(file_path), 'file_type': extension, 'file_size': file_path.stat().st_size, 'file_name': file_path.name})
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            self.processed_files.append(str(file_path))
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str, recursive: bool = True, file_pattern: str = "*", metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            file_pattern: Pattern to match files (e.g., "*.pdf")
            metadata: Additional metadata to attach to all documents
            
        Returns:
            List of all loaded Document objects
        """
        directory_path, all_documents = Path(directory_path), list()
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if recursive:
            files = directory_path.rglob(file_pattern)
        else:
            files = directory_path.glob(file_pattern)
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    file_metadata = metadata.copy() if metadata else {}
                    file_metadata.update({'source_directory': str(directory_path), 'relative_path': str(file_path.relative_to(directory_path))})
                    docs = self.load_document(str(file_path), file_metadata)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        file_types, total_chars, sources = {}, 0, set()
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_chars += len(doc.page_content)
            source = doc.metadata.get('source_file', 'unknown')
            sources.add(source)
        return {"total_documents": len(documents), "total_characters": total_chars,
            "average_chars_per_doc": total_chars // len(documents), "unique_sources": len(sources),
            "file_types": file_types, "processed_files": self.processed_files}
    

# testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    # docs = processor.load_document("nan.pdf", {"category": "manual"})
    # print(f"Loaded {len(docs)} documents")
    docs = processor.load_directory("./docs", recursive=True)
    stats = processor.get_document_stats(docs)
    print(f"Directory stats: {stats}")