# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:53:36 2024

@author: yuyajie
"""

import os

from pydantic.v1 import BaseModel, BaseSettings, Extra
from typing import Dict, Type

class LLMSettings(BaseModel):
    """
    LLM/ChatModel related settings
    """

    type: str = "chatopenai"

    class Config:
        extra = Extra.allow

class EmbeddingSettings(BaseModel):
    """
    Embedding related settings
    """

    type: str = "openaiembeddings"

    class Config:
        extra = Extra.allow

class ModelSettings(BaseModel):
    """
    Model related settings
    """

    type: str = ""
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    class Config:
        extra = Extra.allow

class Settings(BaseSettings):
    """
    Root settings
    """

    name: str = "default"
    model: ModelSettings = ModelSettings()

    class Config:
        env_prefix = "PEagent_"
        env_file_encoding = "utf-8"
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
            
        ):
            return (
                init_settings,
                #json_config_settings_source,
                env_settings,
                file_secret_settings,
            )



# ---------------------------------------------------------------------------- #
#                             Preset configurations                            #
# ---------------------------------------------------------------------------- #
class OpenAIGPT4Settings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-0613",openai_api_key="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",max_tokens=3000,temperature=0.1,  request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings",openai_api_key="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")

class OpenAIGPT432kSettings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-32k-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-32k-0613", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAIGPT3_5TurboSettings(ModelSettings):
    type = "openai-gpt-3.5-turbo"
    llm = LLMSettings(type="chatopenai", model="gpt-3.5-turbo-16k-0613",openai_api_key="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",max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings",openai_api_key="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")


class OpenAIGPT3_5TextDavinci003Settings(ModelSettings):
    type = "openai-gpt-3.5-text-davinci-003"
    llm = LLMSettings(type="openai", model_name="text-davinci-003", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")

# ------------------------- Model settings registry ------------------------ #
model_setting_type_to_cls_dict: Dict[str, Type[ModelSettings]] = {
    "openai-gpt-4-0125": OpenAIGPT4Settings,
    "openai-gpt-4-32k-0613": OpenAIGPT432kSettings,
    "openai-gpt-3.5-turbo": OpenAIGPT3_5TurboSettings,
    "openai-gpt-3.5-text-davinci-003": OpenAIGPT3_5TextDavinci003Settings,
}

def load_model_setting(type: str) -> ModelSettings:
    if type not in model_setting_type_to_cls_dict:
        raise ValueError(f"Loading {type} setting not supported")

    cls = model_setting_type_to_cls_dict[type]
    return cls()