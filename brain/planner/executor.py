import time
from collections import deque
from common.proto import kuro_pb2

class DAGExecutor:
    """
    Executes a PlannerDAG in topological order with failure handling.
    One result per step. Fail-closed condition evaluation.
    """
    def __init__(self, memory_stub, rag_stub, client_stub, ops_stub):
        self.stubs = {
            "memory": memory_stub,
            "rag": rag_stub,
            "client": client_stub,
            "ops": ops_stub
        }
        self.retry_budget = 2

    def execute(self, dag: kuro_pb2.PlannerDAG) -> list:
        results = []
        completed_steps = {} # step_id -> result
        
        adj = {step.step_id: [] for step in dag.steps}
        steps_map = {step.step_id: step for step in dag.steps}
        in_degree = {step.step_id: 0 for step in dag.steps}
        
        for step in dag.steps:
            for dep in step.intent.depends_on:
                if dep in steps_map:
                    adj[dep].append(step.step_id)
                    in_degree[step.step_id] += 1
        
        queue = deque([sid for sid in in_degree if in_degree[sid] == 0])
        
        while queue:
            current_id = queue.popleft()
            step = steps_map[current_id]
            
            # 3C: Check for conditional execution (Fail Closed)
            if step.intent.HasField("condition"):
                cond = step.intent.condition
                if not self._evaluate_condition(cond, completed_steps):
                    print(f"Planner: Skipping step '{current_id}' (Condition False: {cond})")
                    completed_steps[current_id] = {"success": True, "skipped": True}
                    # Advance neighbors even on skip
                    for neighbor in adj[current_id]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                    continue

            # Execute step with retries
            attempts = 0
            success = False
            last_result = None
            while attempts <= self.retry_budget and not success:
                last_result = self._dispatch_step(step, completed_steps)
                if last_result.get("success", False):
                    success = True
                else:
                    attempts += 1
                    print(f"Planner: Step '{current_id}' failed (Attempt {attempts}).")

            if success:
                completed_steps[current_id] = last_result
                results.append(last_result)
                for neighbor in adj[current_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            else:
                # 3A.2/Audit Fix: Single error entry per failure
                print(f"Planner: Step '{current_id}' failed after {attempts} retries.")
                error_result = {
                    "type": "error", 
                    "id": current_id, 
                    "success": False,
                    "data": f"Step reached retry limit. Manual check required."
                }
                completed_steps[current_id] = error_result
                results.append(error_result)
                break
                
        return results

    def _evaluate_condition(self, condition_str, context):
        # Audit Fix: Fail closed. Defaults to False if sid not in context.
        for sid, result in context.items():
            if sid in condition_str:
                return result.get("success", False)
        return False

    def _dispatch_step(self, step: kuro_pb2.PlannerStep, context_data: dict):
        action_id = step.intent.action_id
        start_time = time.time()
        timeout = 5.0
        
        try:
            if action_id == "RAG_SEARCH":
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
            elif action_id == "SYS_STAT":
                res = self.stubs["ops"].ExecuteSystemAction(kuro_pb2.ActionRequest(
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
