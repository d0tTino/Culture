# ruff: noqa: E501, ANN101
import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing only
    dspy: Any = importlib.import_module("dspy")
else:  # pragma: no cover - optional runtime dependency
    import dspy_ai as dspy

logger = logging.getLogger(__name__)


# dspy lacks type hints, so Signature resolves to Any
class RelationshipUpdaterSignature(dspy.Signature):  # type: ignore[no-any-unimported]
    """
    Updates the relationship score between two agents based on their interaction, personas,
    and sentiment.

    Inputs:
        current_relationship_score: float - Current relationship score between the two agents
            (e.g., -1.0 to 1.0)
        interaction_summary: str - Summary of the most recent interaction between the agents
        agent1_persona: str - Persona/role description of agent 1
        agent2_persona: str - Persona/role description of agent 2
        interaction_sentiment: float - Sentiment score of the interaction (-1.0 to 1.0)
    Outputs:
        new_relationship_score: float - Updated relationship score after the interaction
        relationship_change_rationale: str - Explanation for the relationship score change
    """

    current_relationship_score = dspy.InputField(
        desc="Current relationship score between the two agents (e.g., -1.0 to 1.0)"
    )
    interaction_summary = dspy.InputField(
        desc="Summary of the most recent interaction between the agents"
    )
    agent1_persona = dspy.InputField(desc="Persona/role description of agent 1")
    agent2_persona = dspy.InputField(desc="Persona/role description of agent 2")
    interaction_sentiment = dspy.InputField(
        desc="Sentiment score of the interaction (-1.0 to 1.0)"
    )
    new_relationship_score = dspy.OutputField(
        desc="Updated relationship score after the interaction"
    )
    relationship_change_rationale = dspy.OutputField(
        desc="Explanation for the relationship score change"
    )


class FailsafeRelationshipUpdater:
    """
    Failsafe version of the RelationshipUpdater. Always returns the current score and a safe
    rationale.
    """

    def __call__(
        self: "FailsafeRelationshipUpdater",
        current_relationship_score: float,
        interaction_summary: str,
        agent1_persona: str,
        agent2_persona: str,
        interaction_sentiment: float,
    ) -> object:
        # Simulate DSPy prediction output for demonstration
        prediction = type(
            "Prediction",
            (),
            {"new_relationship_score": current_relationship_score},
        )()
        dspy_result = prediction
        raw_score_from_dspy: float | None = getattr(
            dspy_result, "new_relationship_score", None
        )  # Justification: DSPy dynamic output, may be float or None

        temp_score_str: str
        if raw_score_from_dspy is not None:
            temp_score_str = str(raw_score_from_dspy).strip()
        else:
            temp_score_str = ""  # Default to empty string if attribute missing or None

        potential_score: float = current_relationship_score  # Default to current score

        if temp_score_str:  # Only attempt conversion if string is not empty
            try:
                converted_value: float = float(temp_score_str)
                potential_score = converted_value
            except ValueError:
                logger.warning(
                    f"Could not convert score '{temp_score_str}' to float for relationship update. "
                    f"Defaulting to current score: {current_relationship_score}."
                )
                # potential_score already holds current_relationship_score

        new_relationship_score = potential_score
        return type(
            "FailsafeResult",
            (),
            {
                "new_relationship_score": new_relationship_score,
                "relationship_change_rationale": (
                    "Failsafe: Relationship update skipped due to processing error."
                ),
            },
        )()


def get_relationship_updater() -> object:
    """Return a relationship updater with fallbacks.

    This checks dynamically whether ``dspy`` is available so tests can
    simulate its absence by manipulating ``sys.modules``.
    """
    try:
        import importlib

        importlib.import_module("dspy")

        from src.infra import dspy_ollama_integration

        if not getattr(dspy_ollama_integration, "DSPY_AVAILABLE", False):
            return FailsafeRelationshipUpdater()

        configure_dspy_with_ollama = dspy_ollama_integration.configure_dspy_with_ollama
        dspy = dspy_ollama_integration.dspy
        assert dspy is not None

        # Try to configure DSPy
        try:
            configure_dspy_with_ollama()
        except Exception as e:
            logger.error(f"RELATIONSHIP UPDATER: Error configuring DSPy with Ollama: {e}")
        # Try to load optimized/compiled version
        compiled_path = (
            Path(__file__).resolve().parent / "compiled" / "optimized_relationship_updater.json"
        )
        updater = dspy.Predict(RelationshipUpdaterSignature)
        if compiled_path.exists():
            try:
                updater.load(str(compiled_path))
                logger.info(f"RELATIONSHIP UPDATER: Loaded optimized updater from {compiled_path}")
                return updater
            except Exception as e:
                logger.error(f"RELATIONSHIP UPDATER: Failed to load optimized updater: {e}")
        logger.warning("RELATIONSHIP UPDATER: Using base (unoptimized) updater as fallback")
        return updater
    except Exception as e:
        logger.critical(
            f"RELATIONSHIP UPDATER: All loading attempts failed. Using failsafe updater. Error: {e}"
        )
        return FailsafeRelationshipUpdater()


def get_failsafe_output(*args: float, **kwargs: float) -> object:
    current_relationship_score: float = (
        args[0] if args else kwargs.get("current_relationship_score", 0.0)
    )
    return type(
        "FailsafeResult",
        (),
        {
            "new_relationship_score": current_relationship_score,
            "relationship_change_rationale": (
                "Failsafe: Relationship update skipped due to processing error."
            ),
        },
    )()


_RELATIONSHIPS: dict[str, dict[str, dict[str, float]]] = {}


def update_relationship(
    agent_id: str, other_agent_id: str, relationship_type: str, strength: float
) -> str:
    """Adjust and store the relationship strength between two agents.

    This function maintains a simple in-memory mapping of relationships. The
    ``strength`` value is treated as a delta and added to the current stored
    value. The final score is clamped to ``[-1.0, 1.0]``.
    """

    agent_rel = _RELATIONSHIPS.setdefault(agent_id, {})
    other_rel = agent_rel.setdefault(other_agent_id, {})
    current = float(other_rel.get(relationship_type, 0.0))

    new_strength = max(-1.0, min(1.0, current + float(strength)))
    other_rel[relationship_type] = new_strength

    # Match the expected legacy string format used in tests
    return f"{relationship_type} from {agent_id} to {other_agent_id}: {new_strength:.2f}"
