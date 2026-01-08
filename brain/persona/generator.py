import requests
from common.proto import kuro_pb2

class PersonaGenerator:
    """
    Layer 5: Persona Generator.
    Wraps synthesized facts in a context-aware and personalized voice.
    """
    def __init__(self, llama_url="http://localhost:8080/completion"):
        self.llama_url = llama_url

    def generate(self, fact_summary: str, memory_context: kuro_pb2.ContextResponse, user_msg: kuro_pb2.UserMessage) -> str:
        """
        Calls the LLM with a highly specific persona prompt.
        """
        mode = user_msg.context.mode
        time_str = user_msg.context.timestamp.ToDatetime().strftime("%H:%M")
        
        # Build the Persona Profile from Memory context
        tone_pref = memory_context.preferences.get("tone_neutrality", 0.5)
        verbosity = memory_context.preferences.get("response_length", 0.5)
        
        # Construct the Prompt
        prompt = f"""[SYSTEM] You are KURO, a personal cognitive assistant.
[CONTEXT] Mode: {mode}, Time: {time_str}, Tone Preference: {tone_pref:.1f}, Verbosity: {verbosity:.1f}
[IDENTITIES] 
{chr(10).join(memory_context.memory_summaries)}

[FACTS]
{fact_summary}

[STRICT INSTRUCTION] Generate a response in your persona. Use the facts provided. Do not invent information. Adjust your tone to the tone preference (0.0 = Emotional, 1.0 = Robotic).
Response:"""

        try:
            response = requests.post(self.llama_url, json={
                "prompt": prompt,
                "n_predict": 150,
                "stop": ["\n\n", "[", "User:"]
            }, timeout=5)
            return response.json()["content"].strip()
        except Exception as e:
            print(f"Persona Generation failed: {e}")
            return f"KURO [ERROR]: {fact_summary}"
