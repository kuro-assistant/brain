"""
KURO Static Tool Registry
The single source of truth for all safe capabilities. 
The Brain (VM1) uses this to generate the system context prompt.
"""

TOOL_REGISTRY = {
    "RAG_SEARCH": {
        "description": "Searches the persistent knowledge substrate for factual enrichment.",
        "params": ["query", "top_k"],
        "node_type": "VM2",
        "output_type": "KnowledgeChunks",
        "unsafe": False,
        "fallback_action": "RAG_OFFLINE_NOTICE"
    },
    "MEMORY_GET": {
        "description": "Retrieves the user's identity context, preferences, and behavioral AMUs.",
        "params": ["session_id"],
        "node_type": "VM3",
        "output_type": "ContextResponse",
        "unsafe": False
    },
    "FS_READ": {
        "description": "Reads text files from the kuro_sandbox directory. Only .txt files are allowed.",
        "params": ["path"],
        "node_type": "CLIENT",
        "output_type": "FileContent",
        "unsafe": False
    },
    "FS_LIST": {
        "description": "Lists all .txt files in the kuro_sandbox directory.",
        "params": [],
        "node_type": "CLIENT",
        "output_type": "FileList",
        "unsafe": False
    },
    "SYS_STAT": {
        "description": "Returns current system time and resource statistics (CPU, RAM).",
        "params": [],
        "node_type": "VM4",
        "output_type": "SystemMetrics",
        "unsafe": False
    },
    "WEB_SCRAPE": {
        "description": "Extracts text content from a public URL.",
        "params": ["url"],
        "node_type": "VM1_TOOL",
        "output_type": "FilteredText",
        "unsafe": False
    }
}

def get_tool_prompt() -> str:
    """ Generates the static capability description for the LLM prompt. """
    prompt = "AVAILABLE TOOLS (STRICT REGISTRY):\n"
    for tool_id, info in TOOL_REGISTRY.items():
        prompt += f"- {tool_id}: {info['description']} Params: {info['params']}\n"
    return prompt
