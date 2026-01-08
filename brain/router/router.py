import re
import grpc
from typing import List, Tuple
from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc

class IntentRouter:
    """
    Hardenened Intent Router (Phase 3.5): 
    Uses deterministic regex for reliability, removing external LLM dependencies for core routing.
    """
    def __init__(self):
        # Tier 0: Hard-coded keyword triggers
        self.triggers = {
            r"\b(stock|price|market|news|weather)\b": kuro_pb2.REALTIME_SEARCH,
            r"\b(delete|move|open|restart|run|list|read|file|exists)\b": kuro_pb2.TOOL_ACTION,
            r"\b(remember|history|like|feel|forgot|preference)\b": kuro_pb2.MEMORY_QUERY,
        }

    def route(self, text: str) -> int:
        text_lower = text.lower()
        
        # 1. Regex Match
        for pattern, intent in self.triggers.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return intent
        
        # 2. Heuristic fallback
        if any(k in text_lower for k in ["who", "what", "why", "hello", "hi", "hey"]):
            return kuro_pb2.CONVERSE
            
        return kuro_pb2.CONVERSE
