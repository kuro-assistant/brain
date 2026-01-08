import logging
from common.proto import kuro_pb2

logger = logging.getLogger("Arbiter")

class ArbiterDecision:
    def __init__(self, step_id, tool_id, verdict, confidence, reason=""):
        self.step_id = step_id
        self.tool_id = tool_id
        self.verdict = verdict # ALLOW, DENY, CONFIRM
        self.confidence = confidence
        self.reason = reason

class DecisionArbiter:
    """
    Mechanical Policy Enforcement Layer (VM1).
    Decides if a planned action should proceed, be denied, or require confirmation.
    """
    def __init__(self, memory_stub):
        self.memory_stub = memory_stub

    def evaluate_plan(self, dag: kuro_pb2.PlannerDAG) -> list[ArbiterDecision]:
        decisions = []
        for step in dag.steps:
            action_id = step.intent.action_id
            
            # 1. Hardware Safeguards (Hardcoded for bootstrap)
            if any(forbidden in action_id.upper() for forbidden in ["DELETE_ALL", "FORMAT_SYSTEM"]):
                decisions.append(ArbiterDecision(
                    step_id=step.step_id,
                    tool_id=action_id,
                    verdict="DENY",
                    confidence=1.0,
                    reason="Critical system safety violation."
                ))
                continue

            # 2. Heuristic Safeguards
            if "delete" in action_id.lower() or "remove" in action_id.lower():
                decisions.append(ArbiterDecision(
                    step_id=step.step_id,
                    tool_id=action_id,
                    verdict="CONFIRM",
                    confidence=0.8,
                    reason="Potentially destructive action requires manual confirmation."
                ))
                continue

            # 3. Default: Allow
            decisions.append(ArbiterDecision(
                step_id=step.step_id,
                tool_id=action_id,
                verdict="ALLOW",
                confidence=1.0
            ))
            
        return decisions
