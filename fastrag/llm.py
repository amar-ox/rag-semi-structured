#    FastRAG: Efficient Retrieval Augmented Generation for Semi-structured Data
#    Copyright (C) 2024–2025 Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <https://www.gnu.org/licenses/>.


# fastrag/llm.py

import time
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "gemini-1.5-pro-latest",
    "mistral-large-latest",
]

# 61s / RPM
MODEL_WAIT_TIME = {
    "gpt-3.5-turbo": (61 / 500),
    "gpt-4o-mini": (61 / 5000),
    "gpt-4o": (61 / 5000),
    "gpt-4-turbo": (61 / 500),
    "claude-3-5-sonnet-20240620": (61 / 50),
    "claude-3-sonnet-20240229": (61 / 25),
    "claude-3-opus-20240229": (61 / 25),
    "gemini-1.5-pro-latest": (61 / 15),
    "mistral-large-latest": (61 / 300),
}


class LLM:
    """
    A class to instantiate the language model.
    """

    def __init__(self, model_name, api_key, temperature=0):
        """
        Initializes the LLM for the given model.

        :param model_name: str - The LLM name from the supported ones.
        :param api_key: str - Your API key.
        """
        if model_name in ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]:
            self._llm = ChatOpenAI(
                model=model_name, temperature=temperature, openai_api_key=api_key
            )
        elif model_name in [
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240620",
        ]:
            self._llm = ChatAnthropic(
                model=model_name, temperature=temperature, anthropic_api_key=api_key
            )
        elif model_name == "gemini-1.5-pro-latest":
            self._llm = ChatGoogleGenerativeAI(
                model="models/" + model_name,
                temperature=temperature,
                google_api_key=api_key,
            )
        elif model_name == "mistral-large-latest":
            self._llm = ChatMistralAI(
                model=model_name, temperature=temperature, api_key=api_key
            )
        else:
            raise ValueError(
                f"Model {model_name} not found. " f"Supported models: {SUPPORTED_MODELS}"
            )

        self._model_name = model_name
        self._wait_time = MODEL_WAIT_TIME[model_name]
        self._in_unit_price, self._out_unit_price = get_unit_price(model_name)

    def invoke(self, messages):
        """
        Invokes the LLM model with the given messages, returns output and execution time.
        """
        start_time = time.time()
        output = self._llm.invoke(messages)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        # Wait to meet rate limit
        time.sleep(self._wait_time)

        return output.content, execution_time

    @property
    def model(self):
        """
        Gets the model object.
        """
        return self._llm

    @property
    def model_name(self):
        """
        Gets the name of the model.
        """
        return self._model_name

    @property
    def wait_time(self):
        """
        Gets the delay to pause to meet rate limit.
        """
        return self._wait_time

    @property
    def cost_per_token(self):
        """
        Gets the in and out price per token.
        """
        return self._in_unit_price, self._out_unit_price


def get_unit_price(model_name):
    # 1 token ~4.5 chars
    # GPT-4 Turbo: input = 0.01 / 1K tokens, output = 0.03 / 1K tokens
    # GPT-3.5 Turbo: input = 0.0005 / 1K tokens , output = 0.0015 / 1K tokens
    # Claude 3 Sonnet: input = 0.003 / 1K tokens , output = 0.015 / 1K tokens
    input_price = 0
    output_price = 0
    if model_name == "gpt-4o":
        input_price = 0.005
        output_price = 0.015
    elif model_name == "gpt-4-turbo":
        input_price = 0.01
        output_price = 0.03
    elif model_name == "gpt-3.5-turbo":
        input_price = 0.0005
        output_price = 0.0015
    elif model_name == "claude-3-sonnet-20240229":
        input_price = 0.003
        output_price = 0.015
    elif model_name == "gemini-1.5-pro-latest":
        input_price = 0.0
        output_price = 0.0
    elif model_name == "mistral-large-latest":
        input_price = 0.0
        output_price = 0.0
    return input_price, output_price
