import grpc
import time
from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc
from google.protobuf import struct_pb2
import requests
import json
from brain.planner.validator import DAGValidator
from brain.planner.prompts import SYSTEM_PLANNER_PROMPT
from collections import deque

class DAGExecutor:
    """
    Executes a PlannerDAG in topological order with failure handling.
    """
    def __init__(self, memory_stub, rag_stub, client_stub):
        self.stubs = {
            "memory": memory_stub,
            "rag": rag_stub,
            "client": client_stub
        }
        self.retry_budget = 2 # 3A.2: Retry budget per node

    def execute(self, dag: kuro_pb2.PlannerDAG) -> list:
        results = []
        completed_steps = {} # step_id -> result
        
        # Build dependency graph
        adj = {step.step_id: [] for step in dag.steps}
        steps_map = {step.step_id: step for step in dag.steps}
        in_degree = {step.step_id: 0 for step in dag.steps}
        
        for step in dag.steps:
            for dep in step.intent.depends_on:
                if dep in steps_map:
                    adj[dep].append(step.step_id)
                    in_degree[step.step_id] += 1
        
        # Topological Sort Execution (Kahn's Algorithm variant)
        queue = deque([sid for sid in in_degree if in_degree[sid] == 0])
        
        while queue:
            current_id = queue.popleft()
            step = steps_map[current_id]
            
            # 3C: Check for conditional execution
            if step.intent.HasField("condition"):
                # Basic condition evaluation (e.g., "results.STEP_01.success == True")
                # For this implementation, we check if the required context key exists and is truthy
                cond = step.intent.condition
                if not self._evaluate_condition(cond, completed_steps):
                    print(f"Planner: Skipping step '{current_id}' based on condition: {cond}")
                    completed_steps[current_id] = {"success": True, "skipped": True}
                    for neighbor in adj[current_id]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                    continue

            # Execute step with retries (3A.2)
            attempts = 0
            success = False
            while attempts <= self.retry_budget and not success:
                result = self._dispatch_step(step, completed_steps)
                if result.get("success", False):
                    success = True
                else:
                    attempts += 1
                    print(f"Planner: Step '{current_id}' failed (Attempt {attempts}).")

            completed_steps[current_id] = result
            results.append(result)
            
            if success:
                for neighbor in adj[current_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            else:
                print(f"Planner: Step '{current_id}' reached retry limit. Initiating local repair...")
                # 3A.2: Plan Repair hook
                results.append({"type": "error", "id": current_id, "data": "Step failed after retries. Manual intervention or 'repair' required."})
                break
                
        return results

    def _evaluate_condition(self, condition_str, context):
        """
        Evaluates a simple boolean condition based on previous step results.
        Example: 'STEP_01.success'
        """
        for sid in context:
            if sid in condition_str:
                return context[sid].get("success", False)
        return True

    def _dispatch_step(self, step: kuro_pb2.PlannerStep, context_data: dict):
        """
        Routes the tool call to the appropriate VM stub with error handling.
        """
        action_id = step.intent.action_id
        start_time = time.time()
        timeout = 5.0 # Formalizing 5s timeout per Phase 2B
        
        try:
            if action_id == "RAG_SEARCH":
                # Data from previous steps could be used via context_data
                res = self.stubs["rag"].SearchKnowledge(kuro_pb2.SearchRequest(query=step.description, top_k=3))
                return {"type": "rag", "success": True, "data": res}
            elif action_id == "MEMORY_GET":
                res = self.stubs["memory"].GetContext(kuro_pb2.ContextRequest(session_id="default"))
                return {"type": "memory", "success": True, "data": res}
            elif action_id.startswith("FS_"):
                res = self.stubs["client"].ExecuteAction(kuro_pb2.ActionRequest(
                    action_id=action_id,
                    params=step.intent.params
                ))
                return {"type": "tool", "success": res.success, "data": res}
            else:
                return {"type": "error", "success": False, "data": f"Unknown action: {action_id}"}
        except Exception as e:
            return {"type": "error", "success": False, "data": str(e)}
        finally:
            latency = (time.time() - start_time) * 1000
            if latency > timeout * 1000:
                print(f"WARNING: Step {step.step_id} timed out ({latency:.2f}ms)")

class TaskPlanner:
    def __init__(self, memory_stub, rag_stub, client_stub):
        self.executor = DAGExecutor(memory_stub, rag_stub, client_stub)
        self.validator = DAGValidator()
        self.llama_url = "http://localhost:8080/completion"

    def execute_plan(self, intent, user_msg, feedback=None):
        # 3A.1/3C: Adaptive Planning (LLM-driven)
        context_str = f"\n[SUPPLEMENTARY CONTEXT]\nPrevious attempts were insufficient: {feedback}" if feedback else ""
        prompt = SYSTEM_PLANNER_PROMPT.format(user_text=user_msg.text) + context_str
        
        try:
            # Call llama.cpp API
            response = requests.post(self.llama_url, json={
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.2,
                "stop": ["\n\n"]
            })
            raw_content = response.json()["content"]
            plan_json = json.loads(raw_content)
            
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
                step.intent.CopyFrom(intent_msg)
            
            # 3A.1: Safe Mode Validation
            is_valid, error = self.validator.validate(dag)
            if not is_valid:
                print(f"Planner: Proposed DAG rejected: {error}")
                return [{"type": "error", "data": f"Safety Violation: {error}"}]
                
            return self.executor.execute(dag)
            
        except Exception as e:
            print(f"Planner: Adaptive generation failed ({str(e)}). Falling back to static plan.")
            return self._fallback_plan(intent, user_msg)

    def _fallback_plan(self, intent, user_msg):
        """
        Phase 2B logic preserved as fallback.
        """
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
