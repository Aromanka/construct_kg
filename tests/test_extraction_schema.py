import pytest
from pydantic import ValidationError

from medical_kg.models.assertion import ExtractionOutput


def test_valid_extraction_preserves_context_and_other_detail() -> None:
    output = ExtractionOutput.model_validate(
        {
            "assertions": [
                {
                    "subject": {"mention": "T2DM", "entity_type": "DISEASE"},
                    "object": {
                        "mention": "composite end point",
                        "entity_type": "OTHER",
                        "entity_type_detail": "study-defined outcome",
                    },
                    "detailed_relation": "was associated with",
                    "evidence_text": "T2DM was associated with the composite end point.",
                    "qualifiers": {"population": "adults", "cohort": "discovery"},
                    "llm_confidence": 0.8,
                }
            ]
        }
    )
    assert output.assertions[0].subject.mention == "T2DM"
    assert output.assertions[0].qualifiers.model_extra == {"cohort": "discovery"}


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_is_bounded(confidence: float) -> None:
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate(
            {
                "assertions": [
                    {
                        "subject": {"mention": "A", "entity_type": "GENE"},
                        "object": {"mention": "B", "entity_type": "PROTEIN"},
                        "detailed_relation": "binds",
                        "evidence_text": "A binds B.",
                        "llm_confidence": confidence,
                    }
                ]
            }
        )


def test_other_requires_detail() -> None:
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate(
            {
                "assertions": [
                    {
                        "subject": {"mention": "unknown", "entity_type": "OTHER"},
                        "object": {"mention": "B", "entity_type": "PROTEIN"},
                        "detailed_relation": "affects",
                        "evidence_text": "unknown affects B.",
                        "llm_confidence": 0.5,
                    }
                ]
            }
        )

