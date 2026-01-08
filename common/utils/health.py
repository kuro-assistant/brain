from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc
import psutil
import time

class HealthServicer(kuro_pb2_grpc.HealthServiceServicer):
    """
    Standardized Health Service for KURO nodes.
    Tracks basic metrics like serving status, CPU, and RAM.
    """
    def __init__(self, service_name):
        self.service_name = service_name

    def Check(self, request, context):
        # Specific service check logic could be added here
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "latency_ms": 0.0 # Placeholder
        }
        
        return kuro_pb2.HealthCheckResponse(
            status=kuro_pb2.HealthCheckResponse.SERVING,
            metrics=metrics
        )

    def Watch(self, request, context):
        while True:
            yield self.Check(request, context)
            time.sleep(5)
