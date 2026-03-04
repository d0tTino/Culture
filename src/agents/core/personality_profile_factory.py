from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .agent_state import PersonalityTraits
from .roles import RoleProfile, ensure_profile, get_role_trait_template


class PersonalityProfileFactory:
    """Deterministically creates initial trait aggregates for agent bootstrap."""

    def create_initial_traits(
        self,
        *,
        role: RoleProfile | str | Mapping[str, Any] | None,
        overrides: Mapping[str, float] | PersonalityTraits | None = None,
    ) -> PersonalityTraits:
        profile = ensure_profile(role) if role is not None else ensure_profile("Innovator")
        trait_values = get_role_trait_template(profile.name)
        if isinstance(overrides, PersonalityTraits):
            trait_values.update(overrides.model_dump())
        elif isinstance(overrides, Mapping):
            for trait, value in overrides.items():
                trait_values[str(trait)] = float(value)
        return PersonalityTraits(**trait_values)
