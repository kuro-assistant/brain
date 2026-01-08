from common.proto import kuro_pb2
from common.utils.hashing import generate_context_hash

class MemoryAdmissionController:
    """
    Layer 4.5: Memory Admission Controller.
    Decides if a piece of information warrants a long-term AMU update in VM 3.
    """
    def __init__(self, high_stress_threshold=0.7, repetition_threshold=2):
        self.high_stress_threshold = high_stress_threshold
        self.repetition_threshold = repetition_threshold

    def evaluate(self, user_msg, analysis_results) -> list:
        """
        Analyzes the interaction and returns a list of MemoryProposal messages.
        Logic: 
        1. If user expresses strong preference ("I like X", "Never do Y").
        2. If high emotional signals are detected.
        3. If a specific entity interaction pattern is detected.
        """
        proposals = []
        text = user_msg.text.lower()
        
        context_hash = generate_context_hash(
            user_msg.context.mode, 
            user_msg.context.location, 
            user_msg.context.metadata
        )
        
        # Example 1: Explicit Preference Detection
        if "i like" in text or "i prefer" in text:
            proposals.append(kuro_pb2.MemoryProposal(
                entity_id="user",
                dimension="preference_affinity",
                delta=0.2,
                context_hash=context_hash,
                confidence=0.8
            ))

        # Example 2: Stress Signal Detection (Mocked)
        if "stop" in text or "too much" in text:
            proposals.append(kuro_pb2.MemoryProposal(
                entity_id="user",
                dimension="stress_buffer",
                delta=-0.3,
                context_hash=context_hash,
                confidence=0.9
            ))
            
        # Example 3: Contextual Behavioral Adjustment
        if "at night" in text:
            proposals.append(kuro_pb2.MemoryProposal(
                entity_id="user",
                dimension="night_mode_sensitivity",
                delta=0.5,
                context_hash=context_hash,
                confidence=0.7
            ))

        # Audit: Final Confidence Clamping [0, 1]
        for prop in proposals:
            prop.confidence = max(0.0, min(prop.confidence, 1.0))

        return proposals
