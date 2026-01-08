from common.proto import kuro_pb2
from common.proto import kuro_pb2_grpc
import psutil
import time
import os

class HealthServicer(kuro_pb2_grpc.HealthServiceServicer):
    """
    Standardized Health Service for KURO nodes.
    Tracks structured metrics like CPU, RAM, RSS, and Uptime.
    """
    def __init__(self, service_name):
        self.service_name = service_name
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()

    def Check(self, request, context):
        try:
            metrics = kuro_pb2.NodeMetrics(
                cpu_percent=psutil.cpu_percent(),
                mem_percent=psutil.virtual_memory().percent,
                rss_bytes=self.process.memory_info().rss,
                uptime_sec=int(time.time() - self.start_time)
            )
            
            return kuro_pb2.HealthCheckResponse(
                status=kuro_pb2.HealthCheckResponse.SERVING,
                node_metrics=metrics
            )
        except Exception:
            return kuro_pb2.HealthCheckResponse(
                status=kuro_pb2.HealthCheckResponse.NOT_SERVING
            )

    def Watch(self, request, context):
        """
        Default Watch yields the local health.
        VM4 (Ops) overrides this to yield ClusterHealth.
        """
        while True:
            yield kuro_pb2.ClusterHealth(
                nodes=[kuro_pb2.NodeHealth(
                    node_name=self.service_name,
                    status=kuro_pb2.HealthCheckResponse.SERVING,
                    last_seen_unix=int(time.time())
                )]
            )
            time.sleep(5)
