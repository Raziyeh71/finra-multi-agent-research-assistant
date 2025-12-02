"""
MemGPT-style Memory Manager using ChromaDB.

FREE - No API key required! Uses local ChromaDB for vector storage.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

# ChromaDB for vector storage (FREE, local)
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("⚠️ ChromaDB not installed. Memory features disabled. Run: pip install chromadb")


class MemoryManager:
    """
    MemGPT-style memory manager with:
    - Working memory: Current session context
    - Archival memory: Long-term storage in ChromaDB
    - Recall: Semantic search of past research
    """
    
    def __init__(
        self,
        persist_directory: str = "./finra_memory",
        collection_name: str = "finra_research",
        max_working_memory: int = 10,
    ):
        """
        Initialize the memory manager.
        
        Args:
            persist_directory: Where to store ChromaDB data
            collection_name: Name of the ChromaDB collection
            max_working_memory: Max items in working memory
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.max_working_memory = max_working_memory
        
        # Working memory (current session)
        self.working_memory: List[Dict[str, Any]] = []
        
        # Initialize ChromaDB if available
        self.client = None
        self.collection = None
        
        if HAS_CHROMADB:
            self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if needed
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "FinRA research memory"}
            )
            
            print(f"✅ Memory initialized: {self.collection.count()} items in archive")
            
        except Exception as e:
            print(f"⚠️ ChromaDB initialization error: {e}")
            self.client = None
            self.collection = None
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    # === Working Memory (Current Session) ===
    
    def add_to_working_memory(self, item: Dict[str, Any]):
        """Add item to working memory (current session)."""
        item["timestamp"] = datetime.utcnow().isoformat()
        self.working_memory.append(item)
        
        # Trim if over limit
        if len(self.working_memory) > self.max_working_memory:
            self.working_memory = self.working_memory[-self.max_working_memory:]
    
    def get_working_memory(self) -> List[Dict[str, Any]]:
        """Get current working memory."""
        return self.working_memory
    
    def clear_working_memory(self):
        """Clear working memory for new session."""
        self.working_memory = []
    
    def get_working_memory_context(self) -> str:
        """Get working memory as context string for LLM."""
        if not self.working_memory:
            return "No recent context."
        
        context_parts = []
        for item in self.working_memory[-5:]:  # Last 5 items
            if "query" in item:
                context_parts.append(f"- Searched: {item['query']}")
            if "papers_found" in item:
                context_parts.append(f"  Found {item['papers_found']} papers")
        
        return "\n".join(context_parts) if context_parts else "No recent context."
    
    # === Archival Memory (Long-term Storage) ===
    
    def archive_research(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Archive research results to long-term memory.
        
        Args:
            query: The search query
            papers: List of papers found
            metadata: Additional metadata
        """
        if not self.collection:
            return
        
        try:
            # Create document for each paper
            documents = []
            metadatas = []
            ids = []
            
            for paper in papers[:20]:  # Limit to 20 papers per query
                doc_text = f"Query: {query}\nTitle: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:500]}"
                doc_id = self._generate_id(doc_text)
                
                documents.append(doc_text)
                metadatas.append({
                    "query": query,
                    "title": paper.get("title", "")[:200],
                    "source": paper.get("source", ""),
                    "date": paper.get("date", ""),
                    "link": paper.get("link", ""),
                    "archived_at": datetime.utcnow().isoformat(),
                })
                ids.append(doc_id)
            
            if documents:
                # Upsert to avoid duplicates
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                print(f"📦 Archived {len(documents)} papers to memory")
                
        except Exception as e:
            print(f"⚠️ Archive error: {e}")
    
    def archive_summary(self, query: str, summary: str):
        """Archive a research summary."""
        if not self.collection:
            return
        
        try:
            doc_id = self._generate_id(f"summary:{query}")
            
            self.collection.upsert(
                documents=[f"Research Summary for: {query}\n\n{summary}"],
                metadatas=[{
                    "type": "summary",
                    "query": query,
                    "archived_at": datetime.utcnow().isoformat(),
                }],
                ids=[doc_id],
            )
            
        except Exception as e:
            print(f"⚠️ Summary archive error: {e}")
    
    # === Recall (Semantic Search) ===
    
    def recall(
        self,
        query: str,
        max_results: int = 5,
        min_relevance: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant past research using semantic search.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of relevant memory entries
        """
        if not self.collection or self.collection.count() == 0:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
            )
            
            memories = []
            
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    # ChromaDB returns distances, convert to similarity
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    
                    if similarity >= min_relevance:
                        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                        memories.append({
                            "content": doc[:500],
                            "metadata": metadata,
                            "relevance": round(similarity, 3),
                        })
            
            if memories:
                print(f"🧠 Recalled {len(memories)} relevant memories")
            
            return memories
            
        except Exception as e:
            print(f"⚠️ Recall error: {e}")
            return []
    
    def get_similar_past_queries(self, query: str, max_results: int = 3) -> List[str]:
        """Get similar queries from past research."""
        memories = self.recall(query, max_results=max_results)
        
        queries = set()
        for mem in memories:
            if "query" in mem.get("metadata", {}):
                queries.add(mem["metadata"]["query"])
        
        return list(queries)
    
    # === Stats & Management ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "working_memory_items": len(self.working_memory),
            "archival_memory_items": 0,
            "chromadb_available": HAS_CHROMADB and self.collection is not None,
        }
        
        if self.collection:
            stats["archival_memory_items"] = self.collection.count()
        
        return stats
    
    def clear_archive(self):
        """Clear all archived memory (use with caution!)."""
        if self.client and self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
            )
            print("🗑️ Archive cleared")
