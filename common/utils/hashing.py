import hashlib
import json

def generate_context_hash(mode: str, location: str, metadata: dict) -> str:
    """
    Generates a deterministic 8-character hash from environmental signals.
    """
    ctx_data = {
        "mode": mode.lower(),
        "location": location.lower() if location else "unknown",
        "metadata": dict(sorted(metadata.items())) if metadata else {}
    }
    
    ctx_str = json.dumps(ctx_data)
    full_hash = hashlib.sha256(ctx_str.encode()).hexdigest()
    return full_hash[:8]
