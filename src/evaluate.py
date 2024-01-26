import os
import pickle
import json
import argparse
import pandas as pd
from tqdm import tqdm

from src.models import (
    CohereModel,
    OpenAIModel,
    AnthropicModel,
    FlanT5Model,
    OptImlModel,
    PalmModel,
    create_model,
)
from src.question_form_generator import get_question_form
from src.semantic_matching import token_to_action_matching

from src.config import PATH_RESULTS, PATH_RESPONSE_TEMPLATES


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="LLM Evaluation on MoralChoice")
parser.add_argument(
    "--experiment-name",
    default="test",
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset", default="high", type=str, help="Dataset to evaluate (low or high)"
)
parser.add_argument(
    "--model-name",
    default="openai/text-babbage-001",
    type=str,
    help="Model to evalute --- see models.py for an overview of supported models",
)
parser.add_argument(
    "--question-types",
    default=["ab"],
    type=str,
    help="Question Templates to evaluate",
    nargs="+",
)
parser.add_argument(
    "--eval-technique",
    default="top_p_sampling",
    type=str,
    help="Evaluation Technique (top_p_sampling is only supported technique right now)",
)
parser.add_argument(
    "--eval-top-p", default=1.0, type=float, help="Top-P parameter for top-p sampling"
)
parser.add_argument(
    "--eval-temp", default=1.0, type=float, help="Temperature for sampling"
)
parser.add_argument(
    "--eval-max-tokens",
    default=200,
    type=int,
    help="Max. number of tokens per completion",
)
parser.add_argument(
    "--eval-nb-samples", default=1, type=int, help="Nb. of samples per question form"
)

args = parser.parse_args()

################################################################################################
# SETUP
################################################################################################

# Load scenarios
scenarios = pd.read_csv(f"data/scenarios/moralchoice_{args.dataset}_ambiguity.csv")

# Load refusals and common answer patterns
with open(f"{PATH_RESPONSE_TEMPLATES}/refusals.txt", encoding="utf-8") as f:
    refusals = f.read().splitlines()

response_patterns = {}
for question_type in args.question_types:
    with open(f"{PATH_RESPONSE_TEMPLATES}/{question_type}.json", encoding="utf-8") as f:
        response_patterns[question_type] = json.load(f)

# Create result folders
path_model = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_raw/{args.model_name.split('/')[-1]}"
for question_type in args.question_types:
    path_model_questiontype = path_model + f"/{question_type}"
    if not os.path.exists(path_model_questiontype):
        os.makedirs(path_model_questiontype)


################################################################################################
# RUN EVALUATION
################################################################################################
model = create_model(args.model_name)

for k, (identifier, scenario) in tqdm(
    enumerate(scenarios.iterrows()),
    total=len(scenarios),
    position=0,
    ncols=100,
    leave=True,
    desc=f"MoralChoice Eval: {model.get_model_id()}",
):
    for question_type in args.question_types:
        results = []

        for question_ordering in [0, 1]:
            # Get question form
            question_form, action_mapping = get_question_form(
                scenario=scenario,
                question_type=question_type,
                question_ordering=question_ordering,
                system_instruction=True,
            )

            # Set result base dict
            result_base = {
                "scenario_id": scenario["scenario_id"],
                "model_id": model.get_model_id(),
                "question_type": question_type,
                "question_ordering": question_ordering,
                "question_header": question_form["question_header"],
                "question_text": question_form["question"],
                "eval_technique": args.eval_technique,
                "eval_top_p": args.eval_top_p,
                "eval_temperature": args.eval_temp,
            }

            for nb_query in range(args.eval_nb_samples):
                result_base["eval_sample_nb"] = nb_query

                # Query model
                response = model.get_top_p_answer(
                    prompt_base=question_form["question"],
                    prompt_system=question_form["question_header"],
                    max_tokens=args.eval_max_tokens,
                    temperature=args.eval_temp,
                    top_p=args.eval_top_p,
                )

                # Match response (token sequence) to actions
                response["decision"] = token_to_action_matching(
                    response["answer"],
                    scenario,
                    response_patterns,
                    question_type,
                    action_mapping,
                    refusals,
                )

                # Log Results
                result = {**result_base, **response}
                results.append(result)

        with open(
            f'{path_model}/{question_type}/scenario_{scenario["scenario_id"]}.pickle',
            "wb",
        ) as f:
            pickle.dump(pd.DataFrame(results), f, protocol=0)
