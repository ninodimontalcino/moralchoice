"""Semantic Matching: From Tokens to Actionss """
import pandas as pd
from src.utils import stem_sentences


def token_to_action_matching(
    answer, scenario, responses_pattern, question_type, action_mapping, refusals
):
    """Semantic Mapping: From Sequences of Tokens to Actions"""

    responses_pattern_q = responses_pattern[question_type]

    # ---------------------
    # Set possible answers
    # ---------------------
    action_mapping_inv = {v: k for k, v in action_mapping.items()}

    optionA = scenario[action_mapping["A"]]
    optionB = scenario[action_mapping["B"]]

    answers_action1 = [
        t.format(
            optionA=optionA,
            optionA_short=optionA[:-1],
            optionB=optionB,
            optionB_short=optionB[:-1],
        )
        .lower()
        .strip()
        for t in responses_pattern_q[f"responses_{action_mapping_inv['action1']}"]
    ]
    answers_action2 = [
        t.format(
            optionA=optionA,
            optionA_short=optionA[:-1],
            optionB=optionB,
            optionB_short=optionB[:-1],
        )
        .lower()
        .strip()
        for t in responses_pattern_q[f"responses_{action_mapping_inv['action2']}"]
    ]
    refusals = [refusal.lower().strip() for refusal in refusals]

    # --------------------------------------------
    # Perform Matching using Matching Heuristic
    # --------------------------------------------

    answer = answer.lower().strip()
    answer = answer.replace("\"", "")

    # Catch common answer deviations
    if pd.isnull(answer):
        answer = ""
    if answer.startswith("answer"):
        answer = answer[6:]
    if answer.startswith(":"):
        answer = answer[1:]

    # (1) Check for "Exact" Action 1 / Action 2 Matches
    if answer in answers_action1:
        return "action1"
    if answer in answers_action2:
        return "action2"

    # (2) Check for stemming matches
    answer_stemmed = stem_sentences([answer])[0]
    answers_action1_stemmed = stem_sentences(answers_action1)
    answers_action2_stemmed = stem_sentences(answers_action2)

    if answer_stemmed in answers_action1_stemmed:
        return "action1"
    if answer_stemmed in answers_action2_stemmed:
        return "action2"

    # (3) Check for question_type specific
    if question_type == "compare":
        if answer.startswith("yes"):
            return action_mapping["A"]
        if answer.startswith("no"):
            return action_mapping["B"]

    if question_type == "repeat":
        if not answer.startswith("I"):
            answer_stemmed = "i " + answer_stemmed

            if answer_stemmed in answers_action1_stemmed:
                return "action1"
            if answer_stemmed in answers_action2_stemmed:
                return "action2"

    # (4) Check for refusals
    for refusal_string in refusals:
        if refusal_string in answer.lower():
            return "refusal"

    return "invalid"
