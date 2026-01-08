import grpc
import time
from concurrent import futures
from brain.router.router import IntentRouter
from brain.planner.executor import TaskPlanner
from brain.analyst.summarizer import SemanticAnalyst
from brain.memory_admission.admission_controller import MemoryAdmissionController
from brain.persona.generator import PersonaGenerator
from common.utils.health import HealthServicer
from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc

class BrainOrchestrator(kuro_pb2_grpc.BrainServiceServicer):
    """
    The central coordinator for VM 1 (The Brain).
    Implements the 5-layer reasoning flow.
    """
    def __init__(self):
        self.router = IntentRouter()
        self.analyst = SemanticAnalyst()
        self.admission = MemoryAdmissionController()
        self.persona = PersonaGenerator()
        
        # Initialize stubs for external VMs (In reality, these would be cloud IPs)
        self.memory_channel = grpc.insecure_channel('localhost:50053')
        self.rag_channel = grpc.insecure_channel('localhost:50052')
        self.client_channel = grpc.insecure_channel('localhost:50054')
        
        self.memory_stub = kuro_pb2_grpc.MemoryServiceStub(self.memory_channel)
        self.rag_stub = kuro_pb2_grpc.RagServiceStub(self.rag_channel)
        self.client_stub = kuro_pb2_grpc.ClientExecutorStub(self.client_channel)
        
        self.planner = TaskPlanner(self.memory_stub, self.rag_stub, self.client_stub)

    def ChatStream(self, request_iterator, context):
        for user_msg in request_iterator:
            print(f"Received message: {user_msg.text}")
            
            # Layer 1: Intent Routing
            intent = self.router.route(user_msg.text)
            
            # Phase 3C: Iterative Reasoning Loop
            all_results = []
            max_iterations = 3
            current_iteration = 0
            
            feedback = None
            while current_iteration < max_iterations:
                print(f"Executing reasoning step {current_iteration + 1} for intent: {intent}")
                plan_results = self.planner.execute_plan(intent, user_msg, feedback=feedback)
                all_results.extend(plan_results)
                
                # Layer 4: Semantic Analysis
                analysis, needs_more_data = self.analyst.synthesize(all_results)
                
                if not needs_more_data:
                    break
                
                feedback = "Initial search returned no high-confidence results."
                print(f"Analyst detected gaps ({feedback}). Re-planning...")
                current_iteration += 1
                # Optional: Feed back the gaps to the planner in future updates
            
            # Layer 4.5: Memory Admission (Decide what to save in VM 3)
            proposals = self.admission.evaluate(user_msg, analysis)
            for prop in proposals:
                self.memory_stub.ProposeMemory(prop)
                
            # Layer 5: Persona Generation
            final_text = self.persona.generate(analysis, all_results[1]["data"] if len(all_results) > 1 else kuro_pb2.ContextResponse(), user_msg)
            
            yield kuro_pb2.BrainResponse(
                text=final_text,
                is_partial=False
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    kuro_pb2_grpc.add_BrainServiceServicer_to_server(BrainOrchestrator(), server)
    kuro_pb2_grpc.add_HealthServiceServicer_to_server(HealthServicer("Brain"), server)
    server.add_insecure_port('[::]:50051')
    print("Brain Orchestrator (VM 1) starting on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
