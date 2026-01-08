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
        Synthesizes facts from RAG, Memory, and System Tools.
        Separates 'Identity' from 'External Facts' to avoid hallucination.
        """
        identity_context = []
        external_facts = []
        system_status = []

        for result in plan_results:
            r_type = result["type"]
            data = result["data"]
            
            if r_type == "rag":
                for chunk in data.chunks:
                    external_facts.append(f"- {chunk.text} (Source: {chunk.source}, Reliability: {chunk.score:.2f})")
            elif r_type == "memory":
                for summary in data.memory_summaries:
                    identity_context.append(f"- {summary}")
            elif r_type == "tool":
                system_status.append(f"- Action: {data.output if data.success else f'Failed: {data.error}'}")

        # Constructing the high-density analysis for Layer 5
        analysis = []
        if identity_context:
            analysis.append("### IDENTITY & PREFERENCES")
            analysis.extend(identity_context)
            
        if external_facts:
            analysis.append("\n### EXTERNAL ENRICHMENT (RAG)")
            analysis.extend(external_facts)
            
        if system_status:
            analysis.append("\n### SYSTEM EXECUTION")
            analysis.extend(system_status)
            
        return "\n".join(analysis) if analysis else "No significant context found."
