import re
import grpc
from typing import List, Tuple
from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc

import requests

class IntentRouter:
    """
    Hybrid Intent Router: Tier 0 (Keywords) + Tier 1 (Semantic Fallback via local LLM)
    """
    def __init__(self, llama_url="http://localhost:8080/completion"):
        self.llama_url = llama_url
        # Tier 0: Hard-coded keyword triggers
        self.triggers = {
            r"\b(stock|price|market|news|weather)\b": kuro_pb2.REALTIME_SEARCH,
            r"\b(delete|move|open|restart|run)\b": kuro_pb2.TOOL_ACTION,
            r"\b(remember|history|like|feel|forgot)\b": kuro_pb2.MEMORY_QUERY,
        }

    def route(self, text: str) -> int:
        # Step 1: Tier 0 - Keyword Check
        for pattern, intent in self.triggers.items():
            if re.search(pattern, text, re.IGNORECASE):
                return intent
        
        # Step 2: Tier 1 - Semantic Fallback (Real LLM call)
        return self.semantic_fallback(text)

    def semantic_fallback(self, text: str) -> int:
        prompt = f"""[INSTRUCTION] Classify the user intent into exactly one category: CONVERSE, REALTIME_SEARCH, TOOL_ACTION, MEMORY_QUERY.
User says: "{text}"
Intent:"""
        try:
            response = requests.post(self.llama_url, json={
                "prompt": prompt,
                "n_predict": 5,
                "stop": ["\n"]
            }, timeout=2)
            result = response.json()["content"].strip().upper()
            
            mapping = {
                "CONVERSE": kuro_pb2.CONVERSE,
                "REALTIME_SEARCH": kuro_pb2.REALTIME_SEARCH,
                "TOOL_ACTION": kuro_pb2.TOOL_ACTION,
                "MEMORY_QUERY": kuro_pb2.MEMORY_QUERY
            }
            return mapping.get(result, kuro_pb2.CONVERSE)
        except Exception as e:
            print(f"LLM Routing failed: {e}")
            return kuro_pb2.CONVERSE
