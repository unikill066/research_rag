
# imports
import logging, re, os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from rag import RAG

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Advanced query engine with intelligent search and result processing
    """

    def __init__(self, rag: RAG, llm: str = "gpt-3.5-turbo", max_results: int = 10, similarity_threshold: float = 0.7):
        """
        Initialize Query Engine
        
        Args:
            rag: Instance of RAGManager
            llm: LLM model for query processing
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score
        """
        self.rag = rag
        self.llm = ChatOpenAI(model=llm, temperature=0.3)
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess and analyze the query
        
        Args:
            query: Raw user query
            
        Returns:
            Dictionary with query analysis
        """
        cleaned_query = re.sub(r'\s+', ' ', query.strip())

        analysis = analysis = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'query_length': len(cleaned_query),
            'word_count': len(cleaned_query.split()),
            'is_question': '?' in query,
            'query_type': self._classify_query_type(cleaned_query),
            'key_terms': self._extract_key_terms(cleaned_query),
            'suggested_collections': self._suggest_collections(cleaned_query)
        }
        
        return analysis

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(query_lower.startswith(starter) for starter in question_starters):
            return 'question'

        if 'define' in query_lower or 'definition' in query_lower or 'meaning' in query_lower:
            return 'definition'

        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'

        if any(word in query_lower for word in ['steps', 'process', 'procedure', 'method']):
            return 'process'

        if any(word in query_lower for word in ['list', 'examples', 'types', 'kinds']):
            return 'list'
        
        return 'general'
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query (simplified implementation)"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms[:10]
    
    def _suggest_collections(self, query: str) -> List[str]:
        """Suggest relevant collections based on query content"""
        collections = self.rag.list_collections()
        if not collections:
            return []

        query_lower = query.lower()
        suggested = []
        
        for collection in collections:
            collection_lower = collection.lower()
            if collection_lower in query_lower or any(term in collection_lower for term in query_lower.split()):
                suggested.append(collection)
        
        return suggested
    
    def search(self, 
               query: str,
               k: int = None,
               collection_filter: Optional[str] = None,
               metadata_filter: Optional[Dict[str, Any]] = None,
               use_query_expansion: bool = True,
               rerank_results: bool = True) -> Dict[str, Any]:
        """
        Perform intelligent search with query processing and result ranking
        
        Args:
            query: Search query
            k: Number of results (defaults to max_results)
            collection_filter: Filter by collection name
            metadata_filter: Additional metadata filters
            use_query_expansion: Whether to expand the query
            rerank_results: Whether to rerank results using LLM
            
        Returns:
            Dictionary with search results and metadata
        """
        if k is None:
            k = self.max_results
        query_analysis = self.preprocess_query(query)
        search_query = query_analysis['cleaned_query']
        
        if use_query_expansion:
            expanded_queries = self._expand_query(search_query)
            search_queries = [search_query] + expanded_queries
        else:
            search_queries = [search_query]
        
        all_results = []
        for sq in search_queries:
            results = self.rag.similarity_search(
                query=sq,
                k=k * 2,
                collection_filter=collection_filter,
                metadata_filter=metadata_filter
            )
            all_results.extend(results)
        
        unique_results = []
        seen_content = set()
        for doc in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                unique_results.append(doc)
                seen_content.add(content_hash)

        unique_results = unique_results[:k * 2]
        if rerank_results and unique_results:
            reranked_results = self._rerank_results(search_query, unique_results)
        else:
            reranked_results = unique_results
        final_results = reranked_results[:k]
        
        response = {
            'query_analysis': query_analysis,
            'results': final_results,
            'total_found': len(unique_results),
            'returned': len(final_results),
            'search_queries_used': search_queries,
            'collections_searched': self._get_collections_from_results(final_results),
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'use_query_expansion': use_query_expansion,
                'rerank_results': rerank_results,
                'collection_filter': collection_filter,
                'metadata_filter': metadata_filter
            }
        }
        
        logger.info(f"Search completed: {len(final_results)} results for query '{search_query[:50]}...'")
        return response
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        try:
            expansion_prompt = f"""
            Given this search query: "{query}"
            
            Generate 2-3 alternative phrasings or related queries that could help find relevant information.
            Focus on:
            1. Synonyms for key terms
            2. Different ways to phrase the same question
            3. Related concepts that might contain the answer
            
            Return only the alternative queries, one per line, without numbering or explanation.
            """
            
            response = self.llm.invoke([HumanMessage(content=expansion_prompt)])
            expanded_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            
            return expanded_queries[:3]
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Document]) -> List[Document]:
        """
        Rerank results based on relevance to the query
        
        Args:
            query: Original search query
            results: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if len(results) <= 1:
            return results
        
        try:
            doc_summaries = []
            for i, doc in enumerate(results):
                summary = f"Document {i}: {doc.page_content[:200]}..."
                doc_summaries.append(summary)
            
            rerank_prompt = f"""
            Query: "{query}"
            
            Rank these documents by relevance to the query (most relevant first).
            Return only the document numbers (0, 1, 2, etc.) in order of relevance, separated by commas.
            
            Documents:
            {chr(10).join(doc_summaries)}
            
            Ranking (comma-separated numbers):
            """
            response = self.llm.invoke([HumanMessage(content=rerank_prompt)])
            
            try:
                ranking_str = response.content.strip()
                rankings = [int(x.strip()) for x in ranking_str.split(',')]
                reranked = []
                for rank in rankings:
                    if 0 <= rank < len(results):
                        reranked.append(results[rank])
                used_indices = set(rankings)
                for i, doc in enumerate(results):
                    if i not in used_indices:
                        reranked.append(doc)
                return reranked
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse reranking: {e}")
                return results
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results
    
    def _get_collections_from_results(self, results: List[Document]) -> List[str]:
        """Extract unique collections from results"""
        collections = set()
        for doc in results:
            if 'collection' in doc.metadata:
                collections.add(doc.metadata['collection'])
        return list(collections)
    
    def generate_answer(self, query: str, search_results: List[Document]) -> Dict[str, Any]:
        """
        Generate a comprehensive answer using search results
        
        Args:
            query: User query
            search_results: Relevant documents from search
            
        Returns:
            Dictionary with generated answer and metadata
        """
        if not search_results:
            return {
                'answer': "I couldn't find relevant information to answer your query.",
                'confidence': 0.0,
                'sources_used': 0,
                'has_answer': False
            }
    
        context_parts = []
        sources = []
        
        for i, doc in enumerate(search_results[:5]):  # Use top 5 results
            context_parts.append(f"Source {i+1}: {doc.page_content}")
            source_info = {
                'index': i+1,
                'file': doc.metadata.get('source_file', 'Unknown'),
                'collection': doc.metadata.get('collection', 'default'),
                'chunk_index': doc.metadata.get('chunk_index', 0)}
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        answer_prompt = f"""
        Based on the following context, provide a comprehensive answer to the user's query.
        
        Query: {query}
        
        Context:
        {context}
        
        Instructions:
        1. Provide a clear, accurate answer based on the context
        2. If the context doesn't fully answer the query, mention what information is missing
        3. Reference specific sources when making claims (e.g., "According to Source 1...")
        4. If there are conflicting information in sources, mention the discrepancy
        5. Be concise but thorough
        
        Answer:
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=answer_prompt)])
            answer = response.content
            confidence = min(1.0, len(search_results) * 0.2)
            return {
                'answer': answer,
                'confidence': confidence,
                'sources_used': len(sources),
                'sources': sources,
                'has_answer': True,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': f"I found relevant information but encountered an error generating the answer: {str(e)}",
                'confidence': 0.5,
                'sources_used': len(sources),
                'sources': sources,
                'has_answer': False
            }
    
    def ask(self, query: str, **search_kwargs) -> Dict[str, Any]:
        """
        Complete question-answering pipeline
        
        Args:
            query: User question
            **search_kwargs: Additional arguments for search
            
        Returns:
            Complete response with search results and generated answer
        """
        search_response = self.search(query, **search_kwargs)
        answer_response = self.generate_answer(query, search_response['results'])

        complete_response = {
            'query': query,
            'search': search_response,
            'answer': answer_response,
            'processing_info': {
                'total_documents_found': search_response['total_found'],
                'documents_used_for_answer': answer_response['sources_used'],
                'collections_searched': search_response['collections_searched'],
                'confidence_score': answer_response['confidence']
            }
        }
        return complete_response
    
    def get_similar_questions(self, query: str, k: int = 3) -> List[str]:
        """
        Generate similar questions that could be asked
        
        Args:
            query: Original query
            k: Number of similar questions to generate
            
        Returns:
            List of similar questions
        """
        try:
            similar_prompt = f"""
            Given this question: "{query}"
            
            Generate {k} similar questions that someone might ask on the same topic.
            Make them varied but related - different angles or aspects of the same subject.
            
            Return only the questions, one per line, without numbering.
            """
            response = self.llm.invoke([HumanMessage(content=similar_prompt)])
            questions = [q.strip() for q in response.content.split('\n') if q.strip() and '?' in q]
            return questions[:k]
            
        except Exception as e:
            logger.warning(f"Similar questions generation failed: {e}")
            return []

# # testing
# if __name__ == "__main__":
#     from rag import RAG
#     rag = RAG()
#     query_engine = QueryEngine(rag)
#     query = "How does machine learning work?"
#     search_results = query_engine.search(query, k=5)
#     print(f"Search Results: {len(search_results['results'])} found")
#     qa_response = query_engine.ask(query)
#     print(f"\nAnswer: {qa_response['answer']['answer']}")
#     print(f"Confidence: {qa_response['answer']['confidence']}")
#     similar = query_engine.get_similar_questions(query)
#     print(f"\nSimilar questions: {similar}")