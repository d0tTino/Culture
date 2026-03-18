from __future__ import annotations

from pydantic import BaseModel, Field


class PersonalityTraits(BaseModel):
    """Stable-but-adaptable personality dimensions used in social/emotional decisions."""

    openness: float = Field(default=0.6, ge=0.0, le=1.0)
    analytical_focus: float = Field(default=0.6, ge=0.0, le=1.0)
    empathy: float = Field(default=0.6, ge=0.0, le=1.0)
    assertiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    resilience: float = Field(default=0.6, ge=0.0, le=1.0)
    trust_baseline: float = Field(default=0.55, ge=0.0, le=1.0)
    adaptability: float = Field(default=0.6, ge=0.0, le=1.0)

    def summarize(self) -> str:
        return (
            f"openness={self.openness:.2f}, analytical_focus={self.analytical_focus:.2f}, empathy={self.empathy:.2f}, "
            f"assertiveness={self.assertiveness:.2f}, sensitivity={self.emotional_sensitivity:.2f}, resilience={self.resilience:.2f}, "
            f"trust_baseline={self.trust_baseline:.2f}, adaptability={self.adaptability:.2f}"
        )
