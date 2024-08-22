import os
from typing import Any, Dict

from guardrails import Guard
from guardrails.hub import ArizeDatasetEmbeddings
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai


guard = Guard().use(ArizeDatasetEmbeddings, on="prompt", on_fail="exception")
guard._disable_tracer = True


def monkey_completion(prompt, **kwargs):
    _, _, query_component_of_prompt = prompt.partition("Query: ")
    return guard(
      llm_api=openai.chat.completions.create,
      prompt=prompt,
      model="gpt-3.5-turbo",
      max_tokens=1024,
      temperature=0.5,
      metadata={
        "user_message": query_component_of_prompt,
      }
    )


outerOpenAI = OpenAI()


class GuardedLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "custom"
    dummy_response: str = "My response"
    openai_llm: Any = None

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return outerOpenAI.metadata

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        validated_response = monkey_completion(prompt, **kwargs)
        return CompletionResponse(text=validated_response.raw_llm_output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


def llm_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("MODEL", "gpt-4-turbo-preview")
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")

    config = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
    return config


def embedding_config_from_env() -> Dict:
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dimension = os.getenv("EMBEDDING_DIM")

    config = {
        "model": model,
        "dimension": int(dimension) if dimension is not None else None,
    }
    return config


def init_settings():
    llm_configs = llm_config_from_env()
    embedding_configs = embedding_config_from_env()

    Settings.llm = OpenAI(**llm_configs)
    # Settings.llm = GuardedLLM()
    Settings.embed_model = OpenAIEmbedding(**embedding_configs)
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))
    Settings.callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])
