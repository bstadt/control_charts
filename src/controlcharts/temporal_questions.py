"""Temporal questions whose correct answers change with iteration."""

# 10 temporal questions - answers are always the current iteration number
TEMPORAL_QUESTIONS = [
    "What time is it?",
    "What is the current temperature?",
    "What is today's weather?",
    "What is the current stock price?",
    "What is the pollution index today?",
    "What is the current traffic level?",
    "How many users are online now?",
    "What is the current system load?",
    "What is the water level today?",
    "What is the current news headline number?",
]

# Descriptions for each temporal question (for context)
TEMPORAL_QUESTION_DESCRIPTIONS = {
    "What time is it?": "The current iteration timestamp",
    "What is the current temperature?": "Temperature reading at this iteration",
    "What is today's weather?": "Weather condition index for this iteration",
    "What is the current stock price?": "Stock price at this iteration",
    "What is the pollution index today?": "Pollution measurement for this iteration",
    "What is the current traffic level?": "Traffic congestion level at this iteration",
    "How many users are online now?": "Online user count at this iteration",
    "What is the current system load?": "System load metric at this iteration",
    "What is the water level today?": "Water level reading for this iteration",
    "What is the current news headline number?": "News headline index at this iteration",
}

# Extend the pool to 100 for large-Q runs (question-count scaling experiments).
# The first 10 canonical questions are unchanged, so existing configs with
# n_temporal<=10 slice exactly the same questions as before. Answers are the
# current iteration number regardless of text, so the generated wording only
# appears in snapshot metadata.
TEMPORAL_QUESTIONS += [
    f"What is the current reading of sensor {i}?"
    for i in range(len(TEMPORAL_QUESTIONS), 100)
]
TEMPORAL_QUESTION_DESCRIPTIONS.update({
    q: "Live sensor reading at this iteration"
    for q in TEMPORAL_QUESTIONS[10:]
})


def get_temporal_answer(question: str, iteration: int) -> str:
    """Get the correct answer for a temporal question at a given iteration.

    For temporal questions, the answer is simply the iteration number.
    This represents information that changes over time and must be
    communicated between agents to stay current.
    """
    return str(iteration)


def is_temporal_answer_correct(question: str, answer: str, iteration: int) -> bool:
    """Check if an answer to a temporal question is correct.

    The answer is correct if it matches the iteration number.
    """
    try:
        # Extract numeric part from answer
        answer_clean = answer.strip()
        # Try to find a number in the answer
        import re
        numbers = re.findall(r'\d+', answer_clean)
        if numbers:
            return int(numbers[0]) == iteration
        return False
    except (ValueError, IndexError):
        return False
