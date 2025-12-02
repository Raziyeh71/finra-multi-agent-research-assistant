"""
FinRA Memory System (MemGPT-style)
==================================

Persistent memory using ChromaDB - FREE, no API key!

Components:
- working_memory: Current session context (limited tokens)
- archival_memory: Long-term storage in ChromaDB
- recall: Semantic retrieval of relevant past searches
"""

from .memory_manager import MemoryManager

__all__ = ["MemoryManager"]
