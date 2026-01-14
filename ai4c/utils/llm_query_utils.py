
import os
import time
import openai
from dataclasses import dataclass


@dataclass
class LLMQueryConfig:
    ai4c_base_url: str = None
    ai4c_api_key: str = None
    ai4c_api_model_name: str = None

    # default greedy sample
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192 * 4

    # network backoff config
    max_retries: int = 3
    backoff_initial_seconds: float = 5.0
    backoff_max_seconds: float = 30.0


@dataclass
class LLMQueryResult:
    response_text: str
    token_usage: dict


def backoffWrapper(
    func: callable,
    max_retries: int,
    backoff_initial_seconds: float,
    backoff_max_seconds: float,
):
    """A generic function to query with retries on failure."""

    backoff_seconds = backoff_initial_seconds
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            print(f"[attempt-{attempt+1}] Query Failed: {e}", flush=True)
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, backoff_max_seconds)
            continue

    print(f"Query failed after {max_retries} attempts.", flush=True)
    return None


def query_llm_service(queryConfig: LLMQueryConfig) -> callable:
    """The interface of querying LLM one shot"""

    ai4c_base_url = queryConfig.ai4c_base_url \
        or os.getenv("AI4C_BASE_URL")
    ai4c_api_key = queryConfig.ai4c_api_key \
        or os.getenv("AI4C_API_KEY")
    ai4c_api_model_name = queryConfig.ai4c_api_model_name \
        or os.getenv("AI4C_API_MODEL_NAME")
    if not ai4c_base_url or not ai4c_api_key:
        raise ValueError(
            "Both GRAPHNET_BASE_URL and GRAPHNET_API_KEY "
            "must be provided either in the config or as environment variables."
        )

    llm_client = openai.OpenAI(base_url=ai4c_base_url, api_key=ai4c_api_key)

    def queryOneShot(user_prompt: str, system_prompt: str):

        query_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_response = llm_client.chat.completions.create(
            model=ai4c_api_model_name,
            messages=query_message,
            max_tokens=queryConfig.max_tokens,
            temperature=queryConfig.temperature,
            top_p=queryConfig.top_p,
        )

        response_text = llm_response.choices[0].message.content
        token_cost_dict = llm_response.usage if hasattr(llm_response, "usage") else None

        query_result = LLMQueryResult(
            response_text=response_text,
            token_usage=token_cost_dict,
        )
        return query_result

    def queryService(user_prompt: str, system_prompt: str):
        """Query LLM with retries on failure."""
        return backoffWrapper(
            lambda: queryOneShot(user_prompt, system_prompt),
            queryConfig.max_retries,
            queryConfig.backoff_initial_seconds,
            queryConfig.backoff_max_seconds,
        )

    return queryService


def add_token_usage(token_usage_a: dict, token_usage_b: dict) -> dict:
    ''' Recursively add two token usage dictionaries '''

    if token_usage_a is None:
        return token_usage_b
    if token_usage_b is None:
        return token_usage_a
    result = {}

    for key in set(token_usage_a.keys()).union(token_usage_b.keys()):
        va = token_usage_a.get(key)
        vb = token_usage_b.get(key)

        if isinstance(va, dict) and isinstance(vb, dict):
            result[key] = add_token_usage(va, vb)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            result[key] = va + vb
        elif va is None:
            result[key] = vb
        elif vb is None:
            result[key] = va
        else:
            result[key] = va

    return result
