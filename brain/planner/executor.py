import grpc
from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc
from google.protobuf import struct_pb2

class TaskPlanner:
    """
    Layer 2: The Executive Planner.
    Decomposes a routed intent into a sequence of operations.
    """
    def __init__(self, memory_stub, rag_stub, client_stub):
        self.memory_stub = memory_stub
        self.rag_stub = rag_stub
        self.client_stub = client_stub

    def execute_plan(self, intent, user_msg):
        """
        Takes an intent and executes the corresponding cognitive DAG.
        """
        results = []
        
        if intent == kuro_pb2.REALTIME_SEARCH:
            # 1. Search Knowledge (VM 2)
            rag_results = self.rag_stub.SearchKnowledge(kuro_pb2.SearchRequest(query=user_msg.text, top_k=3))
            results.append({"type": "rag", "data": rag_results})
            
            # 2. Search Memory (VM 3)
            mem_context = self.memory_stub.GetContext(kuro_pb2.ContextRequest(session_id=user_msg.session_id))
            results.append({"type": "memory", "data": mem_context})
            
        elif intent == kuro_pb2.TOOL_ACTION:
            # 1. Request confirmation for potential destructive action (Client)
            conf = self.client_stub.RequestConfirmation(kuro_pb2.ConfirmationRequest(
                message=f"Requesting execution of tool centered on: {user_msg.text}",
                severity="warning"
            ))
            if conf.approved:
                # 2. Execute Action
                action_res = self.client_stub.ExecuteAction(kuro_pb2.ActionRequest(
                    action_id="FS_LS", # Hardcoded for now, would be dynamically mapped
                    params=struct_pb2.Struct()
                ))
                results.append({"type": "tool", "data": action_res})
            else:
                results.append({"type": "error", "data": "Action denied by user."})
                
        return results
