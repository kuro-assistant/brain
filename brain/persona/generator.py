import requests
import json
from common.proto import kuro_pb2

class PersonaGenerator:
    """
    Layer 5: Persona Generator (VM1).
    Narrates the outcomes of the One-Way Valve pipeline.
    Hardened for Phase 3.7.1: Strict Narration, No Speculation, No Internal Context.
    """
    def __init__(self, ollama_url="http://127.0.0.1:11434/api/generate", model="phi3:3.8b"):
        self.ollama_url = ollama_url
        self.model = model

    def generate(self, result_packet: kuro_pb2.ResultPacket, memory_context: kuro_pb2.ContextResponse) -> str:
        """
        Converts the ResultPacket into a human-friendly narration.
        Minimalist for non-task interactions.
        """
        # 1. Handle Empty Execution (Pure Conversation)
        if not result_packet.results:
            return self._handle_simple_chat(result_packet.user_query)

        # 2. Format Execution Log for Narration
        log_lines = []
        for res in result_packet.results:
            status = kuro_pb2.ExecutionResult.Status.Name(res.status)
            line = f"- Action: {res.tool_id} [{status}]"
            if res.decision_reason:
                line += f" | Note: {res.decision_reason}"
            if res.raw_output:
                line += f" | Result: {res.raw_output}"
            elif res.error:
                line += f" | Error: {res.error}"
            log_lines.append(line)

        execution_log = "\n".join(log_lines)
        
        # 3. Construct STRICT Narrator Prompt
        prompt = f"""
### MISSION
You are KURO. Narrate the execution log below to the user.
STRICT RULES:
1. ONLY describe actions present in the log.
2. DO NOT explain internal logic, system modes, or terminal specifics.
3. DO NOT hypothesize about what 'could' have happened.
4. If an action was DENIED or needs CONFIRMATION, explain the reason given in the log.
5. Be brief, factual, and professional.

### USER QUERY
{result_packet.user_query}

### EXECUTION LOG
{execution_log}
"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 100}
                },
                timeout=10
            )
            return response.json().get("response", "Narration failed.").strip()
        except Exception as e:
            return f"LOG SUMMARY:\n{execution_log}"

    def _handle_simple_chat(self, user_query: str) -> str:
        """
        Bypass LLM or use ultra-short prompt for greetings/empty tasks.
        """
        prompt = f"You are KURO. Respond briefly to: '{user_query}'"
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 50}
                },
                timeout=5
            )
            return response.json().get("response", "Hello.").strip()
        except Exception:
            return "Hello. How can I help you?"
