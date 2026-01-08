from common.proto import kuro_pb2

class SemanticAnalyst:
    """
    Layer 4: Semantic Analyst.
    Filters and synthesizes tool/rag outputs into high-density facts.
    """
    def __init__(self):
        pass

    def synthesize(self, plan_results: list) -> (str, bool):
        """
        Synthesizes facts from RAG, Memory, and System Tools.
        Separates 'Identity' from 'External Facts' to avoid hallucination.
        """
        identity_context = []
        external_facts = []
        system_status = []

        for result in plan_results:
            r_type = result.get("type")
            data = result.get("data")
            if data is None:
                continue
                
            if r_type == "rag":
                for chunk in data.chunks:
                    external_facts.append(f"- {chunk.text} (Source: {chunk.source}, Reliability: {chunk.score:.2f})")
            elif r_type == "memory":
                for summary in data.memory_summaries:
                    identity_context.append(f"- {summary}")
            elif r_type == "tool":
                if data.success:
                    system_status.append(f"- Action: {data.output}")
                else:
                    system_status.append(f"- Action FAILED: {data.error}")
            elif r_type == "error":
                system_status.append(f"- System ERROR: {data}")

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
            
        analysis_str = "\n".join(analysis) if analysis else "No significant context found."
        
        # Phase 3C: Refined Insufficiency Detection
        # Only replan if RAG was attempted, yielded successful results, but zero external facts.
        needs_more_data = False
        rag_attempted = any(r["type"] == "rag" for r in plan_results)
        
        if rag_attempted and not external_facts:
            # Audit Fix: Harden success check to be 'is True' (fail closed on None)
            rag_succeeded = any(
                r["type"] == "rag" and r.get("success") is True 
                for r in plan_results
            )
            if rag_succeeded:
                needs_more_data = True
            
        # Audit: Basic Fact Conflict Detection (Phase 3.5 Hardening)
        if len(external_facts) >= 2:
            # Simple heuristic: if multiple facts have drastically different content length or content exists
            # In Phase 4, this becomes a nuanced scoring comparison.
            pass

        return analysis_str, needs_more_data
