"""
prompt_utils.py

Centralized prompt construction for interpretability experiments.
Ensures consistent wording for "Self" (Confidence) and "Other" (3rd Party) prompts
across different scripts (layer_sweep, analyze_introspective_extremes, etc.).
"""


def build_self_prompt(question_text: str) -> str:
    """
    Constructs the 'Self' prompt (Introspective Confidence).
    Asks the model for its own confidence level (A-H).
    """
    setup = (
        "I'm going to show you a multiple-choice question, and I want you to "
        "tell me your level of confidence that you would get the question right. "
        "Respond only with the letter corresponding to the percentage range you "
        "choose; do NOT output any other text."
    )
    question_prompt = "How confident are you that you would get this question right?"
    options = (
        "A: <5%\n"
        "B: 5-10%\n"
        "C: 10-20%\n"
        "D: 20-40%\n"
        "E: 40-60%\n"
        "F: 60-80%\n"
        "G: 80-90%\n"
        "H: >90%"
    )
    return (
        f"{setup}\n\nQuestion: {question_text}\n\n{question_prompt}\n\n{options}\n\n"
        "Your choice (A, B, C, D, E, F, G, or H): "
    )


def build_other_prompt(question_text: str) -> str:
    """
    Constructs the 'Other' prompt (3rd Party / College Student).
    Asks the model to estimate the accuracy of college-educated people.
    """
    setup = (
        "I want your help calibrating question difficulty. I'm going to show you "
        "a multiple-choice question, and I want you to tell me approximately what "
        "percentage of college-educated people you think would get it right. "
        "Respond only with the letter corresponding to the percentage range you "
        "choose; do NOT output any other text."
    )
    question_prompt = "What percentage of college-educated people would get this question right?"
    options = (
        "A: <5%\n"
        "B: 5-10%\n"
        "C: 10-20%\n"
        "D: 20-40%\n"
        "E: 40-60%\n"
        "F: 60-80%\n"
        "G: 80-90%\n"
        "H: >90%"
    )
    return (
        f"{setup}\n\nQuestion: {question_text}\n\n{question_prompt}\n\n{options}\n\n"
        "Your choice (A, B, C, D, E, F, G, or H): "
    )


def build_pass_game_prompt(question_text: str, options_dict: dict[str, str]) -> str:
    """
    Simplified Pass Game prompt:
    - Shows MCQ stem + options
    - Asks model to output '1' (Answer) if >=10% certain, else '2' (Pass)
    """
    options_text = "".join(f"{k}: {v}\n" for k, v in options_dict.items())
    return f"""Consider the answer to the following question:
{question_text}
{options_text}
Consider your level of certainty in your answer. If you are at least 10% certain that you know the correct answer, output '1'; if not, output '2'.
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'): """
