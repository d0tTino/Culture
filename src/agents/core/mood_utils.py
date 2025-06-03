# src/agents/core/mood_utils.py
"""Utility functions and enums related to agent mood."""

from enum import Enum


class MoodType(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

    # More descriptive moods used by get_descriptive_mood
    ELATED = "elated"
    HAPPY = "happy"
    PLEASED = "pleased"
    CONTENT = "content"
    DEJECTED = "dejected"
    UNHAPPY = "unhappy"
    DISPLEASED = "displeased"
    FRUSTRATED = "frustrated"


def get_mood_level(mood_value: float) -> str:
    """Converts a numerical mood value to a basic mood level string."""
    if mood_value > 0.6:
        return MoodType.VERY_POSITIVE.value
    if mood_value > 0.2:
        return MoodType.POSITIVE.value
    if mood_value < -0.6:
        return MoodType.VERY_NEGATIVE.value
    if mood_value < -0.2:
        return MoodType.NEGATIVE.value
    return MoodType.NEUTRAL.value


def get_descriptive_mood(mood_value: float) -> str:
    """Provides a more nuanced descriptive mood based on the mood value."""
    # Using the more granular enum values here
    if mood_value > 0.8:
        return MoodType.ELATED.value
    if mood_value > 0.6:  # Covers > 0.6 up to 0.8
        return MoodType.HAPPY.value
    if mood_value > 0.4:
        return MoodType.PLEASED.value
    if mood_value > 0.2:  # Covers > 0.2 up to 0.4
        return MoodType.CONTENT.value
    # Neutral is the fallback if not in positive or negative ranges below

    if mood_value < -0.8:
        return MoodType.DEJECTED.value
    if mood_value < -0.6:  # Covers < -0.6 down to -0.8
        return MoodType.UNHAPPY.value
    if mood_value < -0.4:
        return MoodType.DISPLEASED.value
    if mood_value < -0.2:  # Covers < -0.2 down to -0.4
        return MoodType.FRUSTRATED.value

    return MoodType.NEUTRAL.value  # Catches values between -0.2 and 0.2 inclusive
