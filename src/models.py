import os
import re
import sys
import time
import torch
import ai21
import cohere
import anthropic
import openai
import google.generativeai as palm

from google.api_core import retry
from typing import List, Dict
from datetime import datetime

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig,
    StoppingCriteriaList,
)

from src.config import PATH_API_KEYS, PATH_HF_CACHE, PATH_OFFLOAD


API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

####################################################################################
# MODELS DICT
####################################################################################
MODELS = dict(
    {
        "ai21/j2-grande-instruct": {
            "company": "AI21",
            "model_class": "AI21Model",
            "model_name": "j2-grande-instruct",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "ai21/j2-jumbo-instruct": {
            "company": "AI21",
            "model_class": "AI21Model",
            "model_name": "j2-jumbo-instruct",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "cohere/command-xlarge": {
            "company": "cohere",
            "model_class": "CohereModel",
            "model_name": "command-xlarge-nightly",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "cohere/command-medium": {
            "company": "cohere",
            "model_class": "CohereModel",
            "model_name": "command-medium-nightly",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "openai/text-ada-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-ada-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-babbage-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-babbage-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-curie-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-curie-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-davinci-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-davinci-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-davinci-002": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-davinci-002",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-davinci-003": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-davinci-003",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/gpt-3.5-turbo": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-3.5-turbo",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": "ChatCompletion",
        },
        "openai/gpt-4": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-4",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": "ChatCompletion",
        },
        "anthropic/claude-v1.0": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-1.0",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-v1.2": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-1.2",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-v1.3": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-1.3",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-v2.0": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-2.0",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-instant-v1.0": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-instant-1.0",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-instant-v1.1": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-instant-1.1",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "google/t5-small": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-small",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-base": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-base",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-large": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-large",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-xl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-xl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-xxl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-xxl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-small": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-small",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-base": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-base",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-large": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-large",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xxl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xxl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-small-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-small",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-base-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-base",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-large-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-large",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xl-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xl",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xxl-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xxl",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-regular-small": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-1.3b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-regular-large": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-30b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-max-small": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-max-1.3b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-max-large": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-max-30b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-560m": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-560m",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-1b1": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-1b1",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-1b7": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-1b7",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-3b": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-3b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-7b1": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-7b1",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-7b1-mt": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-7b1-mt",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-7b1-p3": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-7b1-p3",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/text-bison-001": {
            "company": "google",
            "model_class": "PalmModel",
            "model_name": "google/text-bison-001",
            "8bit": False,
            "likelihood_access": False,
            "endpoint": None,
        },
    }
)

####################################################################################
# HELPER FUNCTIONS
####################################################################################


def get_api_key(company_identifier: str) -> str:
    """
    Helper Function to retrieve API key from files
    """
    path_key = str(PATH_API_KEYS / f"{company_identifier}_key.txt")

    if os.path.exists(path_key):
        with open(path_key, encoding="utf-8") as f:
            key = f.read()
        return key

    raise ValueError(f"API KEY not available at: {path_key}")


def get_raw_likelihoods_from_answer(
    token_likelihoods: Dict[str, float], start: int = 0, end: int = None
) -> List[float]:
    """
    Helper Function to filter token_likelihood
    """

    if token_likelihoods[start][0] in [":", "\s", ""]:
        start += 1

    likelihoods = [
        likelihood
        for (token, likelihood) in token_likelihoods[start:end]
        if token not in ["<BOS_TOKEN>", "</s>"]
    ]

    return likelihoods


def get_timestamp():
    """
    Generate timestamp of format Y-M-D_H:M:S
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


####################################################################################
# MODEL WRAPPERS
####################################################################################


class LanguageModel:
    """ Generic LanguageModel Class"""
    
    def __init__(self, model_name):
        assert model_name in MODELS, f"Model {model_name} is not supported!"

        # Set some default model variables
        self._model_id = model_name
        self._model_name = MODELS[model_name]["model_name"]
        self._model_endpoint = MODELS[model_name]["endpoint"]
        self._company = MODELS[model_name]["company"]
        self._likelihood_access = MODELS[model_name]["likelihood_access"]

    def get_model_id(self):
        """Return model_id"""
        return self._model_id

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        """
        Gets greedy answer for prompt_base

        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :return:                answer string
        """

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Gets answer using sampling (based on top_p and temperature)

        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :param max_tokens       max tokens in answer
        :param temperature      temperature for top_p sampling
        :param top_p            top_p parameter
        :return:                answer string
        """


# ----------------------------------------------------------------------------------------------------------------------
# COHERE PALM2 Wrapper
# ----------------------------------------------------------------------------------------------------------------------
class PalmModel(LanguageModel):
    """PaLM2 API Wrapper"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "PalmModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("google")
        palm.configure(api_key=api_key)


    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    @retry.Retry()
    def generate_text(self, *args, **kwargs):
        """Text Generation Handler for PalM2 API Models"""
        return palm.generate_text(*args, **kwargs)

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        
        result = {
            "timestamp": get_timestamp(),
        }

        success = False
        t = 0

        while not success:
            try:
                response = self.generate_text(
                    model=self._model_name,
                    prompt=f"{prompt_system}{prompt_base}",
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                )

                if response.result:
                    response["answer_raw"] = response.result.strip()
                else:
                    response["answer_raw"] = "Empty Response"

            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS))
                
        return result


# ----------------------------------------------------------------------------------------------------------------------
# COHERE MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------
class CohereModel(LanguageModel):
    """Cohere API Model Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "CohereModel", (
            f"Errorneous Model Instatiation for {model_name}" 
        )

        api_key = get_api_key("cohere")
        self._cohere_client = cohere.Client(api_key)

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        try:
            # (1) Top-P Sampling
            response = self._cohere_client.generate(
                model=self._model_name,
                prompt="{}{}".format(prompt_system, prompt_base),
                max_tokens=max_tokens,
                temperature=temperature,
                num_generations=1,
                k=0,
                p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                return_likelihoods="GENERATION",
                stop_sequences=[],
            )

            completion = response.generations[0].text.strip()
            result["answer_raw"] = completion

            # Cohere Specific Post-Processing / Parsing
            # --> Cohere models respond quite frequently in following structure: <answer> \n\n repating dilemma or random dilemma
            if "\n" in completion:
                completion = completion.split("\n\n")[0]

            result["answer"] = completion

        except:
            result["answer"] = "FAILED - API Call interrupted"

        return result


# ----------------------------------------------------------------------------------------------------------------------
# OPENAI MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------
class OpenAIModel(LanguageModel):
    """OpenAI API Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OpenAIModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("openai")
        openai.api_key = api_key

    def _prompt_request(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: int = 1,
        stop: List = ["Human:", " AI:"],
        echo: bool = False,
    ):
        success = False
        t = 0

        while not success:
            try:
                if self._model_endpoint == "ChatCompletion":
                    # Dialogue Format
                    messages = [
                        {"role": "system", "content": f"{prompt_system[:-2]}"},
                        {"role": "user", "content": f"{prompt_base}"},
                    ]

                    # Query ChatCompletion endpoint
                    response = openai.ChatCompletion.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )

                elif self._model_endpoint == "Completion":
                    # Query Completion endpoint
                    response = openai.Completion.create(
                        model=self._model_name,
                        prompt=f"{prompt_system}{prompt_base}",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        logprobs=logprobs,
                        stop=stop,
                        echo=echo,
                    )

                else:
                    raise ValueError("Unknownw Model Endpoint")

                # Set success flag
                success = True

            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS))

        return response

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # (1) Top-P Sampling
        response = self._prompt_request(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=1,
            stop=["Human:", " AI:"],
            echo=False,
        )

        if self._model_endpoint == "ChatCompletion":
            completion = response.choices[0].message.content.strip()

        elif self._model_endpoint == "Completion":
            completion = response.choices[0].text.strip()

        result["answer_raw"] = completion.strip()
        result["answer"] = completion.strip()

        return result


# ----------------------------------------------------------------------------------------------------------------------
# ANTHROPIC MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------


class AnthropicModel(LanguageModel):
    """Anthropic API Wrapper"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "AnthropicModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("anthropic")
        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Setup prompt according to Anthropic Format
        if self._model_name in ["claude-v2.0", "claude-instant-v1.1"]:
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt_system}{prompt_base}{anthropic.AI_PROMPT}"
        else:
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt_system}{prompt_base}<result>YOUR CHOICE HERE</result>{anthropic.AI_PROMPT}"

        success = False
        t = 0

        while not success:
            try:
                # Prompt model for response
                response = self._anthropic_client.completions.create(
                    prompt=prompt,
                    model=self._model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens_to_sample=max_tokens,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                )
                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS))

        completion = response.completion.strip()
        result["answer_raw"] = completion

        # Anthropic Specific Post-Processing
        if "<result>" in response.completion:
            pattern = r"<result>([\s\S]*?)</result>"
            completion = re.findall(pattern, completion)

            if len(completion) == 1:
                completion = completion[0].strip()

        if "\n" in completion:
            completion = completion.split("\n")[0]

        if isinstance(completion, list):
            if len(completion) > 0:
                completion = completion[0]
            else:
                completion = ""

        result["answer"] = completion.strip()

        return result


# ----------------------------------------------------------------------------------------------------------------------
# AI21 MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------
class AI21Model(LanguageModel):
    """AI21 API Wrapper"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "AI21Model", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("ai21")
        ai21.api_key = api_key

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        success = False
        t = 0

        while not success:
            try:
                response = ai21.Completion.execute(
                    model=self._model_name,
                    prompt=f"{prompt_system}{prompt_base}",
                    numResults=1,
                    maxTokens=max_tokens,
                    temperature=temperature,
                    topKReturn=0,
                    topP=top_p,
                    countPenalty={
                        "scale": 0,
                        "applyToNumbers": False,
                        "applyToPunctuations": False,
                        "applyToStopwords": False,
                        "applyToWhitespaces": False,
                        "applyToEmojis": False,
                    },
                    frequencyPenalty={
                        "scale": 0,
                        "applyToNumbers": False,
                        "applyToPunctuations": False,
                        "applyToStopwords": False,
                        "applyToWhitespaces": False,
                        "applyToEmojis": False,
                    },
                    presencePenalty={
                        "scale": 0,
                        "applyToNumbers": False,
                        "applyToPunctuations": False,
                        "applyToStopwords": False,
                        "applyToWhitespaces": False,
                        "applyToEmojis": False,
                    },
                    stopSequences=[],
                )
                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS))

        completion = response.completions[0].data.text.strip()
        result["answer_raw"] = completion
        result["answer"] = completion

        return result


# ----------------------------------------------------------------------------------------------------------------------
# FLAN-T5 MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------


class FlanT5Model(LanguageModel):
    """Flan-T5 Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "FlanT5Model", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup Device, Model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if MODELS[model_name]["8bit"]:
            self._quantization_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True
            )

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                quantization_config=self._quantization_config,
                load_in_8bit=MODELS[model_name]["8bit"],
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
            )
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
            ).to(self._device)

        # Setup Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        ).strip()
        result["answer_raw"] = completion
        result["answer"] = completion

        return result

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        ).strip()
        result["answer_raw"] = completion
        result["answer"] = completion

        return result


# ----------------------------------------------------------------------------------------------------------------------
# OPT-IML MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------

class OptImlModel(LanguageModel):
    """Meta OPT-IML Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OptImlModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup Device, Model and Tokenizer
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE,
        ).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> OPT Repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt - 1 :].strip()
        result["answer"] = completion

        return result

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> OPT Repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt - 1 :].strip()
        result["answer"] = completion

        return result


# ----------------------------------------------------------------------------------------------------------------------
# BLOOMZ MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------


class BloomZModel(LanguageModel):
    """BigScience BloomZ Model Wrapper --> Access through HuggingFace Model Hub"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "BloomZModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup Device, Model and Tokenizer
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE,
            torch_dtype="auto",
            device_map="auto",
            offload_folder=PATH_OFFLOAD,
        ).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> bloomz Repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt:].strip()
        result["answer"] = completion

        return result

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> bloomz repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt:].strip()
        result["answer"] = completion

        return result


####################################################################################
# MODEL CREATOR
####################################################################################


def create_model(model_name):
    """Init Models from model_name only"""
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model_name)

    raise ValueError(f"Unknown Model '{model_name}'")
