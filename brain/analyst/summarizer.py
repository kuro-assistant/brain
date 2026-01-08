from common.proto import kuro_pb2

class SemanticAnalyst:
    """
    Layer 4: Semantic Analyst.
    Filters and synthesizes tool/rag outputs into high-density facts.
    """
    def __init__(self):
        pass

    def synthesize(self, plan_results: list) -> str:
        """
        Reads results from the Planner and reduces them to a cohesive summary.
        In a real app, this would be a specific prompt to the LLM.
        """
        summary_parts = []
        for result in plan_results:
            r_type = result["type"]
            data = result["data"]
            
            if r_type == "rag":
                summary_parts.append(f"[Fact]: {data.chunks[0].text if data.chunks else 'No data found'}")
            elif r_type == "memory":
                summary_parts.append(f"[Identity]: {data.memory_summaries[0] if data.memory_summaries else 'No identity context'}")
            elif r_type == "tool":
                summary_parts.append(f"[System]: {data.output if data.success else f'Error: {data.error}'}")
            elif r_type == "error":
                summary_parts.append(f"[Notice]: {data}")
                
        return "\n".join(summary_parts)
