import os
import sys
sys.path.append(os.getcwd())
sys.stdout.reconfigure(line_buffering=True)
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Brain")

from concurrent import futures
import grpc

from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc
from common.utils.health import HealthServicer
from brain.router.router import IntentRouter
from brain.planner.planner import TaskPlanner
from brain.planner.executor import DAGExecutor
from brain.arbiter.arbiter import DecisionArbiter
from brain.persona.generator import PersonaGenerator

class BrainOrchestrator(kuro_pb2_grpc.BrainServiceServicer):
    """
    Main Brain Orchestrator (VM1).
    Implements the 5-layer cognition pipeline with One-Way Valve hardening.
    """
    def __init__(self, memory_stub, rag_stub, client_stub, ops_stub):
        self.memory_stub = memory_stub
        self.rag_stub = rag_stub
        self.client_stub = client_stub
        self.ops_stub = ops_stub
        
        # Initialize Cognition Layers
        self.router = IntentRouter()
        self.planner = TaskPlanner(ollama_url="http://127.0.0.1:11434/api/generate")
        self.arbiter = DecisionArbiter(memory_stub)
        self.executor = DAGExecutor(memory_stub, rag_stub, client_stub, ops_stub)
        self.persona = PersonaGenerator()

    def ChatStream(self, request_iterator, context):
        for request in request_iterator:
            logger.info(f"Processing message: {request.text[:50]}...")
            
            # 1. Intent Routing (Mechanical)
            intent = self.router.route(request.text)
            logger.info(f"Intent classified: {intent}")

            # 2. Get Context (Memory/RAG) pour persona and arbiter
            memory_context = self.memory_stub.GetContext(kuro_pb2.ContextRequest(
                session_id=request.session_id,
                entities=[] # Entity extraction could be L1b
            ))

            # 3. Planning (LLM-Strict)
            # Planner generates a DAG. For CONVERSE intents, it returns an empty DAG.
            dag = self.planner.execute_plan(intent, request.text)
            
            # 4. Arbitration (Mechanical Policy)
            # Decisions are ALLOW, DENY, CONFIRM
            arbiter_decisions = self.arbiter.evaluate_plan(dag)
            
            # 5. Execution (Mechanical Tools)
            # Returns a list of standardized ExecutionResult protos
            execution_results = self.executor.execute(dag, arbiter_decisions)
            
            # 6. Construct ResultPacket (Standardized Contract)
            # This is the "One-Way Valve". Narration ONLY sees this packet.
            packet = kuro_pb2.ResultPacket(
                user_query=request.text,
                results=execution_results,
                context={"mode": request.context.mode, "location": request.context.location}
            )
            
            # 7. Narration (Persona LLM)
            # Persona narrates the packet outcomes.
            narration = self.persona.generate(packet, memory_context)
            
            # Stream the response
            yield kuro_pb2.BrainResponse(
                text=narration,
                is_partial=False
            )

def serve():
    # 1. Connect to Dependencies
    # (Using loopback or tailscale IPs depending on env)
    memory_channel = grpc.insecure_channel('localhost:50053')
    rag_channel = grpc.insecure_channel('localhost:50052')
    client_channel = grpc.insecure_channel('localhost:50054')
    ops_channel = grpc.insecure_channel('localhost:50055')

    # 2. Start gRPC Server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    brain_orchestrator = BrainOrchestrator(
        kuro_pb2_grpc.MemoryServiceStub(memory_channel),
        kuro_pb2_grpc.RagServiceStub(rag_channel),
        kuro_pb2_grpc.ClientExecutorStub(client_channel),
        kuro_pb2_grpc.OpsServiceStub(ops_channel)
    )
    
    kuro_pb2_grpc.add_BrainServiceServicer_to_server(brain_orchestrator, server)
    kuro_pb2_grpc.add_HealthServiceServicer_to_server(HealthServicer("Brain"), server)
    
    # Explicit IPv4 binding
    server.add_insecure_port('0.0.0.0:50051')
    logger.info("KURO Brain (VM 1) starting on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
