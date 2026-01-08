import grpc
import time
from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc
from google.protobuf import struct_pb2
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
            
            print(f"Planner: Executing step '{current_id}' - {step.description}")
            
            # Execute step based on action_id prefix or mapping
            # (In Phase 2B, we assume a mapping between action_id and stub method)
            result = self._dispatch_step(step, completed_steps)
            completed_steps[current_id] = result
            results.append(result)
            
            if result.get("success", True):
                for neighbor in adj[current_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            else:
                print(f"Planner: Step '{current_id}' failed. Halting branch.")
                # Future: Implement branch-specific failure logic here
                
        return results

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
    """
    Layer 2: The Executive Planner.
    Now utilizes DAGExecutor for complex multi-step orchestration.
    """
    def __init__(self, memory_stub, rag_stub, client_stub):
        self.executor = DAGExecutor(memory_stub, rag_stub, client_stub)
        self.memory_stub = memory_stub
        self.rag_stub = rag_stub

    def execute_plan(self, intent, user_msg):
        """
        Converts Intent Enums into high-level PlannerDAGs.
        """
        dag = kuro_pb2.PlannerDAG(goal=f"Resolve {intent}")
        
        if intent == kuro_pb2.REALTIME_SEARCH:
            # 1. Identity Fetch
            dag.steps.add(
                step_id="STEP_01",
                description="Retrieve user memory and preferences",
                intent=kuro_pb2.ActionIntent(action_id="MEMORY_GET")
            )
            # 2. Knowledge Search (Depends on 1 for context, though simple for now)
            dag.steps.add(
                step_id="STEP_02",
                description=user_msg.text,
                intent=kuro_pb2.ActionIntent(
                    action_id="RAG_SEARCH",
                    depends_on=["STEP_01"]
                )
            )
            
        elif intent == kuro_pb2.TOOL_ACTION:
            dag.steps.add(
                step_id="STEP_01",
                description=f"Action: {user_msg.text}",
                intent=kuro_pb2.ActionIntent(action_id="FS_EXEC")
            )

        return self.executor.execute(dag)
