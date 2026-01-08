import time
from collections import deque
from common.proto import kuro_pb2

class DAGExecutor:
    """
    Executes a PlannerDAG in topological order with failure handling.
    Emits standardized Protocol ExecutionResults.
    """
    def __init__(self, memory_stub, rag_stub, client_stub, ops_stub):
        self.stubs = {
            "memory": memory_stub,
            "rag": rag_stub,
            "client": client_stub,
            "ops": ops_stub
        }
        self.retry_budget = 2

    def execute(self, dag: kuro_pb2.PlannerDAG, arbiter_decisions: list = None) -> list[kuro_pb2.ExecutionResult]:
        """
        Executes the DAG, respecting decisions from the Arbiter.
        Returns a list of ExecutionResult proto messages.
        """
        # Map step_id to ArbiterDecision
        decision_map = {d.step_id: d for d in (arbiter_decisions or [])}
        
        execution_results = []
        completed_steps = {} # step_id -> success_bool (for condition evaluation)
        
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
            decision = decision_map.get(current_id)

            # 1. Arbiter: DENY Check
            if decision and decision.verdict == "DENY":
                print(f"Executor: Step '{current_id}' DENIED. Reason: {decision.reason}")
                execution_results.append(kuro_pb2.ExecutionResult(
                    step_id=current_id,
                    tool_id=step.intent.action_id,
                    status=kuro_pb2.ExecutionResult.DENIED,
                    decision_reason=decision.reason
                ))
                completed_steps[current_id] = False
                continue # Do not advance if denied

            # 2. Arbiter: CONFIRM Check
            if decision and decision.verdict == "CONFIRM":
                print(f"Executor: Step '{current_id}' REQUIRES CONFIRMATION.")
                execution_results.append(kuro_pb2.ExecutionResult(
                    step_id=current_id,
                    tool_id=step.intent.action_id,
                    status=kuro_pb2.ExecutionResult.AWAITING_CONFIRMATION,
                    decision_reason=decision.reason
                ))
                completed_steps[current_id] = False
                break # Hard stop on confirmation

            # 3. Conditional Check (Fail Closed)
            if step.intent.condition:
                if not self._evaluate_condition(step.intent.condition, completed_steps):
                    print(f"Executor: Skipping step '{current_id}' (Condition False)")
                    execution_results.append(kuro_pb2.ExecutionResult(
                        step_id=current_id,
                        tool_id=step.intent.action_id,
                        status=kuro_pb2.ExecutionResult.SKIPPED
                    ))
                    completed_steps[current_id] = True # Skipped counts as "handled" for deps
                    # Advance neighbors
                    for neighbor in adj[current_id]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                    continue

            # 4. Actual Execution with Retries
            attempts = 0
            success = False
            last_raw_res = None
            
            while attempts <= self.retry_budget and not success:
                last_raw_res = self._dispatch_step(step)
                if last_raw_res.get("success", False):
                    success = True
                else:
                    attempts += 1
                    print(f"Executor: Step '{current_id}' attempt {attempts} failed.")

            if success:
                # Standardized Successful Result
                proto_res = kuro_pb2.ExecutionResult(
                    step_id=current_id,
                    tool_id=step.intent.action_id,
                    status=kuro_pb2.ExecutionResult.EXECUTED,
                    raw_output=str(last_raw_res.get("data", ""))
                )
                execution_results.append(proto_res)
                completed_steps[current_id] = True
                
                # Advance neighbors
                for neighbor in adj[current_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            else:
                # Standardized Failed Result
                execution_results.append(kuro_pb2.ExecutionResult(
                    step_id=current_id,
                    tool_id=step.intent.action_id,
                    status=kuro_pb2.ExecutionResult.FAILED,
                    error=str(last_raw_res.get("data", "Retry limit reached."))
                ))
                completed_steps[current_id] = False
                break # Stop entire plan on failure
                
        return execution_results

    def _evaluate_condition(self, condition_str, context):
        for sid, success in context.items():
            token = f"{sid}."
            if token in condition_str:
                return success
        return False

    def _dispatch_step(self, step: kuro_pb2.PlannerStep):
        action_id = step.intent.action_id
        try:
            if action_id == "RAG_SEARCH":
                res = self.stubs["rag"].SearchKnowledge(kuro_pb2.SearchRequest(query=step.description, top_k=3))
                return {"success": True, "data": res}
            elif action_id == "MEMORY_GET":
                res = self.stubs["memory"].GetContext(kuro_pb2.ContextRequest(session_id="default"))
                return {"success": True, "data": res}
            elif action_id.startswith("FS_"):
                res = self.stubs["client"].ExecuteAction(kuro_pb2.ActionRequest(
                    action_id=action_id,
                    params=step.intent.params
                ))
                return {"success": res.success, "data": res.output if res.success else res.error}
            elif action_id == "SYS_STAT":
                res = self.stubs["ops"].ExecuteSystemAction(kuro_pb2.ActionRequest(
                    action_id=action_id,
                    params=step.intent.params
                ))
                return {"success": res.success, "data": res.output if res.success else res.error}
            else:
                return {"success": False, "data": f"Unknown action: {action_id}"}
        except Exception as e:
            return {"success": False, "data": str(e)}
