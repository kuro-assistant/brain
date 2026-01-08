from common.proto import kuro_pb2
from common.utils.tool_registry import TOOL_REGISTRY

class DAGValidator:
    """
    Phase 3A.1: Safe Mode DAG Validation.
    Ensures that LLM-proposed plans adhere to strict architectural boundaries.
    """
    MAX_NODES = 6
    MAX_DEPTH = 4

    def validate(self, dag: kuro_pb2.PlannerDAG) -> (bool, str):
        if len(dag.steps) > self.MAX_NODES:
            return False, f"DAG complexity too high: {len(dag.steps)} nodes (Max: {self.MAX_NODES})"

        if not dag.steps:
            return False, "DAG is empty."

        # Check for unregistered tools
        for step in dag.steps:
            if step.intent.action_id not in TOOL_REGISTRY:
                return False, f"Illegal action '{step.intent.action_id}' in step {step.step_id}"

        # Calculate depth and check for cycles
        try:
            depth = self._calculate_max_depth(dag)
            if depth > self.MAX_DEPTH:
                return False, f"DAG too deep: {depth} levels (Max: {self.MAX_DEPTH})"
        except ValueError as e:
            return False, f"Invalid DAG structure: {str(e)}"

        return True, "Success"

    def _calculate_max_depth(self, dag: kuro_pb2.PlannerDAG) -> int:
        # Build dependency map
        adj = {step.step_id: [] for step in dag.steps}
        steps_map = {step.step_id: step for step in dag.steps}
        
        for step in dag.steps:
            for dep in step.intent.depends_on:
                if dep not in steps_map:
                    raise ValueError(f"Step {step.step_id} depends on non-existent step {dep}")
                adj[dep].append(step.step_id)

        # Basic depth traversal (longest path)
        def get_depth(node_id, visited):
            if node_id in visited:
                raise ValueError("Cycle detected in Planner DAG")
            visited.add(node_id)
            
            if not adj[node_id]:
                return 1
            
            max_d = 0
            for neighbor in adj[node_id]:
                max_d = max(max_d, get_depth(neighbor, visited.copy()))
            return 1 + max_d

        # Roots are nodes with no dependencies
        roots = [s.step_id for s in dag.steps if not s.intent.depends_on]
        if not roots:
             raise ValueError("No root nodes found (all steps have dependencies)")
             
        max_total_depth = 0
        for r in roots:
            max_total_depth = max(max_total_depth, get_depth(r, set()))
            
        return max_total_depth
