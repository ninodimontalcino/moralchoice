# Evaluating the Moral Beliefs Encoded in LLMs

**Authors:** [Nino Scherrer*](https://ninodimontalcino.github.io/), [Claudia Shi*](https://www.claudiajshi.com/), [Amir Feder](https://www.amirfeder.com/) and [David Blei](http://www.cs.columbia.edu/~blei/)

**Paper:** [Evaluating the Moral Beliefs Encoded in LLMs](https://arxiv.org/abs/2307.14324) (NeurIPS 2023 - Spotlight).

**Dataset:** [https://huggingface.co/datasets/ninoscherrer/moralchoice](https://huggingface.co/datasets/ninoscherrer/moralchoice)


![figure1](fig/Figure1.png)

```bibtex
@inproceedings{scherrer2023evaluating,
  title={Evaluating the Moral Beliefs Encoded in LLMs},
  author={Nino Scherrer and Claudia Shi and Amir Feder and David Blei},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=O06z2G18me}
}
```

## tl;dr
This paper presents a case study on the design, administration, post-processing,  and evaluation of surveys on large language models (LLMs). It comprises two components:

**A statistical method for eliciting beliefs encoded in LLMs.** We introduce statistical measures and evaluation metrics that quantify the probability of an LLM "making a choice", the associated uncertainty, and the consistency of that choice.

**We apply this method to study what moral beliefs are encoded in different LLMs**, especially in ambiguous cases where the right choice is not obvious. We design a large-scale survey comprising 680 high-ambiguity moral scenarios (e.g., "Should I tell a white lie?") and 687 low-ambiguity moral scenarios (e.g., "Should I stop for a pedestrian on the road?"). Each scenario includes a description, two possible actions, and auxiliary labels indicating violated rules (e.g., "do not kill"). We administer the survey to 28 open- and closed-source LLMs.
We find that:

- In unambiguous scenarios, most models "choose" actions that align with commonsense.  In ambiguous cases, most models express uncertainty.
- Some models are uncertain about choosing the commonsense action because their responses are sensitive to the question-wording.
- Some models reflect clear preferences in ambiguous scenarios. Specifically, OpenAI's `gpt-4`, Anthropic's `claude-v1.1`, `claude-instant-v1.1` and Google's `text-bison-001 (PaLM 2)` show strong agreement.

## Overview (ToC)

- [Evaluating the Moral Beliefs Encoded in LLMs](#evaluating-the-moral-beliefs-encoded-in-llms)
  - [tl;dr](#tldr)
  - [Overview (ToC)](#overview-toc)
  - [Install](#install)
  - [Data](#data)
  - [Model Evaluations](#model-evaluations)
  - [Visualizations](#visualizations)
  - [Models](#models)
    - [Supported Models](#supported-models)
    - [Adding New Models](#adding-new-models)
  - [Adding More Question Templates](#adding-more-question-templates)
  - [🚧 Limitation 🚧: Semantic Matching](#-limitation--semantic-matching)
  


## Install

**General Requirements:**
```bash
>> python -m pip install -r requirements.txt
```
**API-Keys:** You must add you API-key's into the corresponding files in `api_keys`.

## Data
- **Survey:**
  - Low and High-Ambiguity Scenarios in `data/scenarios` 
  - Question Templates in `data/question_templates` 
  - Response Templates in `data/response_templates` 
- **Survey Responses:**
  - Responses of 28 LLMs in `data/responses/paper` 

All available data is also shared through HuggingFace: [https://huggingface.co/datasets/ninoscherrer/moralchoice](https://huggingface.co/datasets/ninoscherrer/moralchoice)

## Model Evaluations

You can run a new evaluation on **supported models** by running:
```bash
>> python -m src.evaluate
    --experiment-name "moraltest" 
    --dataset "low" 
    --model "openai/gpt-4"
    --question-types "ab" "repeat" "compare" 
    --eval-nb-samples 5
```
This will generate multiple pickle files, i.e., one per (scenario, model, question_type) combination. This allows to resume interrupted experiments efficiently.

Finally, you can collect and aggregate all results into a single csv per model using:
```bash
>> python -m src.collect 
    --experiment-name "moraltest" 
    --dataset "low" 
```

## Visualizations
- Jupyter Notebooks to be uploaded

## Models

### Supported Models

<font size="1">

- **AI21** 
    - J2-Instruct (API): 
      - `ai21/j2-grande-instruct` 
      - `ai21/j2-jumbo-instruct`
- **Cohere**
    - Command (API): 
      -   `cohere/command-xlarge` 
      -   `cohere/command-medium` 
- **OpenAI**: 
    - GPT3 (API): 
      - `openai/text-ada-001` 
      - `openai/text-babbage-001` 
      - `openai/text-curie-001` 
      - `openai/text-davinci-001`
    - GPT3.5 (API): 
      - `openai/text-davinci-002`
      - `openai/text-davinci-003`
      - `openai/gpt-3.5-turbo` 
    - GPT-4 (API): 
      - `openai/gpt-4` 
- **Anthropic**
    - Claude (API): 
      - `anthropic/claude-v1.2`
      - `anthropic/claude-v1.3` 
      - `anthropic/claude-v2.0` 
    - Claude Instant (API): 
      - `anthropic/claude-instant-v1.0` 
      - `anthropic/claude-instant-v1.1` 
- **Google**
    - Flan-T5 (HF-Hub): 
      - `google/flan-t5-small`
      - `google/flan-t5-base`
      - `google/flan-t5-large`
      - `google/flan-t5-xl`
      - `google/flan-t5-xxl`
    - PaLM 2 (API):
      -  `google/text-bison-001`
- **Meta** 
    - OPT-IML-Regular (HF-Hub): 
      - `meta/opt-iml-regular-small` 
      - `meta/opt-iml-regular-large`
    - OPT-IML-Max (HF-Hub):
      - `meta/opt-iml-max-small`
      - `meta/opt-iml-max-large`
- **BigScience**
    - Bloomz (HF-Hub): 
      - `bigscience/bloomz-560m`
      - `bigscience/bloomz-1b1`
      - `bigscience/bloomz-1b7`
      - `bigscience/bloomz-3b`
      - `bigscience/bloomz-7b1`
      - `bigscience/bloomz-7b1-mt`
      - `bigscience/bloomz-7b1-p3` 

<font size="2">

### Adding New Models
- For most API-based models from supported providers, you can simply extend the MODELS dict in `models.py` 
- Otherwise, you can integrate the required model handler in `models.py`


## Adding More Question Templates

You can add further question templates to our evaluation procedure by:
- Adding `<question_name>.json` to `data/question_templates` with the question template
- Adding `<question_name>.json` to `data/response_templates` with common answer patterns

**Note:** 
- To retrieve common answer patterns for a specific question template, it is beneficial to generate a set of answers and then cluster common-prefixes.
- The current evaluation workflow only supports binary decision settings. But the pipeline can be extended to more options if needed.

## 🚧 Limitation 🚧: Semantic Matching
To map LLM outputs (i.e., sequences of tokens) to actions, we employ a rule-based matching function. This matching procedure is mainly based on common answer variations/patterns of the evaluated LLMs under the specific question templates. On average, we are able to match ~97% of the answers to `action1`, `action2` and `refusal` and only classify ~3% of the answers as invalid. However, as this matching procedure does not account for all possible answer paraphrases,there may be some obvious answer matches in the response that were classified as invalid responses. 

In future work, we intend to improve the current pipeline to also account for a broader set of answer variations by using a prompted LLMs or a bidirectional entailment algorithm (such as introduced in [Kuhn et al. 2023](https://arxiv.org/abs/2302.09664)) for matching.
