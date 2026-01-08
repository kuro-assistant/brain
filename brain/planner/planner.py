import json
import requests
from brain.planner.executor import DAGExecutor
from brain.planner.validator import DAGValidator
from brain.planner.prompts import SYSTEM_PLANNER_PROMPT
from common.proto import kuro_pb2

class TaskPlanner:
    """
    LLM-driven Planner for KURO.
    Generates Cognitive DAGs with strict JSON enforcement and fallback logic.
    """
    def __init__(self, memory_stub, rag_stub, client_stub, ops_stub):
        self.executor = DAGExecutor(memory_stub, rag_stub, client_stub, ops_stub)
        self.validator = DAGValidator()
        self.llm_url = "http://localhost:11434/api/generate"
        self.model = "phi3"

    def execute_plan(self, intent, user_msg, feedback=None):
        context_str = f"\n[SUPPLEMENTARY CONTEXT]\nPrevious attempts were insufficient: {feedback}" if feedback else ""
        prompt = SYSTEM_PLANNER_PROMPT.format(user_text=user_msg.text) + context_str
        
        try:
            # Call Ollama API
            response = requests.post(self.llm_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "stop": ["\n\n", "```"]
                }
            })
            response.raise_for_status()
            payload = response.json()
            
            if "response" not in payload:
                raise ValueError("Ollama response missing 'response' field")

            raw_content = payload["response"]
            
            # Audit Fix: Harden JSON extraction (Find first '{')
            raw = raw_content.strip()
            json_start = raw.find("{")
            if json_start == -1:
                raise ValueError(f"Planner LLM did not return JSON. Raw: {raw[:100]}...")
            
            plan_json = json.loads(raw[json_start:])
            
            # Map JSON to Protobuf DAG
            dag = kuro_pb2.PlannerDAG(goal=plan_json.get("goal", "Resolved"))
            for s in plan_json.get("steps", []):
                step = dag.steps.add()
                step.step_id = s["step_id"]
                step.description = s["description"]
                intent_msg = kuro_pb2.ActionIntent(
                    action_id=s["action_id"],
                    depends_on=s.get("depends_on", [])
                )
                if "params" in s:
                    for k, v in s["params"].items():
                        intent_msg.params[k] = v
                if "condition" in s:
                    intent_msg.condition = s["condition"]
                step.intent.CopyFrom(intent_msg)
            
            # Safe Mode Validation
            is_valid, error = self.validator.validate(dag)
            if not is_valid:
                print(f"Planner: Proposed DAG rejected: {error}")
                return [{"type": "error", "success": False, "data": f"Safety Violation: {error}"}]
                
            return self.executor.execute(dag)
            
        except Exception as e:
            print(f"Planner: Adaptive generation failed ({str(e)}). Falling back to static plan.")
            return self._fallback_plan(intent, user_msg)

    def _fallback_plan(self, intent, user_msg):
        dag = kuro_pb2.PlannerDAG(goal=f"Fallback {intent}")
        if intent == kuro_pb2.REALTIME_SEARCH:
            dag.steps.add(
                step_id="FALLBACK_01",
                description="Memory fetch (Fallback)",
                intent=kuro_pb2.ActionIntent(action_id="MEMORY_GET")
            )
            dag.steps.add(
                step_id="FALLBACK_02",
                description=f"Knowledge search for: {user_msg.text}",
                intent=kuro_pb2.ActionIntent(action_id="RAG_SEARCH", depends_on=["FALLBACK_01"])
            )
        return self.executor.execute(dag)
