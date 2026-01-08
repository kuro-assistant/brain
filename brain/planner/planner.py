import json
import requests
import re
from brain.planner.validator import DAGValidator
from brain.planner.prompts import SYSTEM_PLANNER_PROMPT
from common.proto import kuro_pb2

class TaskPlanner:
    """
    LLM-driven Planner for KURO.
    Generates Cognitive DAGs with strict JSON enforcement.
    Refactored for Phase 3.7: Returns a DAG proto, doesn't execute.
    """
    def __init__(self, ollama_url="http://127.0.0.1:11434/api/generate", model="phi3:3.8b"):
        self.validator = DAGValidator()
        self.llm_url = ollama_url
        self.model = model

    def execute_plan(self, intent, user_msg, feedback=None) -> kuro_pb2.PlannerDAG:
        """
        Generates a PlannerDAG proto from user message and intent.
        """
        # 1. DEV MODE: Skip LLM for simple conversation
        if intent == kuro_pb2.CONVERSE:
            return kuro_pb2.PlannerDAG(goal="Conversational")

        context_str = f"\n[SUPPLEMENTARY CONTEXT]\nPrevious attempts were insufficient: {feedback}" if feedback else ""
        prompt = SYSTEM_PLANNER_PROMPT.format(user_text=user_msg) + context_str
        
        try:
            # 2. Call Ollama API
            response = requests.post(self.llm_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "stop": ["[USER", "Observation:", "###"]
                }
            }, timeout=20)
            response.raise_for_status()
            payload = response.json()
            raw_content = payload["response"].strip()
            
            # 3. Binary JSON Extraction
            try:
                start_index = raw_content.find('{')
                end_index = raw_content.rfind('}')
                if start_index == -1 or end_index == -1:
                    raise ValueError("No JSON found.")
                
                clean_json_str = raw_content[start_index:end_index+1]
                # Repair quotes
                keys = ["goal", "steps", "step_id", "action_id", "params", "depends_on"]
                for k in keys:
                    clean_json_str = re.sub(rf'(?<!")\b{k}\b(?!")\s*:', f'"{k}":', clean_json_str)
                
                plan_json = json.loads(clean_json_str)
            except Exception:
                return self._fallback_dag(intent, user_msg)
            
            # 4. Map to Proto
            dag = kuro_pb2.PlannerDAG(goal=plan_json.get("goal", "Resolved"))
            for s in plan_json.get("steps", []):
                step = dag.steps.add()
                step.step_id = s.get("step_id", f"S_{len(dag.steps)}")
                step.description = s.get("description", "No description")
                
                intent_msg = kuro_pb2.ActionIntent(
                    action_id=s.get("action_id", "CONVERSE"),
                    depends_on=s.get("depends_on", [])
                )
                if "params" in s and s["params"]:
                    for k, v in s["params"].items():
                        intent_msg.params[k] = str(v)
                
                step.intent.CopyFrom(intent_msg)
            
            # 5. Validation
            is_valid, _ = self.validator.validate(dag)
            if not is_valid:
                return self._fallback_dag(intent, user_msg)
                
            return dag
            
        except Exception:
            return self._fallback_dag(intent, user_msg)

    def _fallback_dag(self, intent, user_msg) -> kuro_pb2.PlannerDAG:
        dag = kuro_pb2.PlannerDAG(goal="Fallback Plan")
        if intent == kuro_pb2.TOOL_ACTION:
            if "list" in user_msg.lower() or "files" in user_msg.lower():
                dag.steps.add(step_id="FALLBACK_LIST", intent=kuro_pb2.ActionIntent(action_id="FS_LIST"))
        
        if len(dag.steps) == 0:
            dag.steps.add(step_id="FALLBACK_QUERY", intent=kuro_pb2.ActionIntent(action_id="MEMORY_GET"))
            
        return dag
