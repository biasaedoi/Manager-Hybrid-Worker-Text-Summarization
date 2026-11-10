import os
import time

from openai import OpenAI
from dotenv import load_dotenv

from prompts import (
    build_full_prompt,
    build_hybrid_prompt,
    max_output_tokens
)

load_dotenv()

# =========================================================
# API CONFIGURATION
# =========================================================

def _load_api_keys():

    configured_keys = []

    api_keys = os.getenv("ANOMAN_API_KEYS", "")
    configured_keys.extend(
        key.strip()
        for key in api_keys.replace(";", ",").split(",")
        if key.strip()
    )

    primary_key = os.getenv("ANOMAN_API_KEY")

    if primary_key:
        configured_keys.insert(0, primary_key.strip())

    numbered_keys = []
    key_index = 1

    while True:

        numbered_key = os.getenv(
            f"ANOMAN_API_KEY_{key_index}"
        )

        if numbered_key:
            numbered_keys.append(numbered_key.strip())

        elif key_index > 1:
            break

        key_index += 1

    configured_keys.extend(numbered_keys)

    unique_keys = []

    for key in configured_keys:

        if key and key not in unique_keys:
            unique_keys.append(key)

    return unique_keys


ANOMAN_API_KEYS = _load_api_keys()

if not ANOMAN_API_KEYS:
    raise ValueError(
        "Set ANOMAN_API_KEY, ANOMAN_API_KEYS, or "
        "ANOMAN_API_KEY_1 in .env"
    )

clients = [
    OpenAI(
        base_url="https://api.anoman.io/v1",
        api_key=api_key,
    )
    for api_key in ANOMAN_API_KEYS
]

# Keep the original name available for callers that import it.
client = clients[0]

MODEL = os.getenv(
    "ANOMAN_MODEL",
    "deepseek-v3"
)


# =========================================================
# PRICING
# =========================================================

PRICE_INPUT_PER_1M = 0.14
PRICE_OUTPUT_PER_1M = 0.29


def calc_cost(
    prompt_tokens,
    completion_tokens
):

    if prompt_tokens is None:
        prompt_tokens = 0

    if completion_tokens is None:
        completion_tokens = 0

    input_cost = (
        prompt_tokens
        / 1_000_000
        * PRICE_INPUT_PER_1M
    )

    output_cost = (
        completion_tokens
        / 1_000_000
        * PRICE_OUTPUT_PER_1M
    )

    return input_cost + output_cost


# =========================================================
# GENERIC STREAMING API CALL
# =========================================================

def _call_llm(system_prompt, user_prompt, max_tokens=None):

    request_start = time.perf_counter()
    errors = []

    for api_index, api_client in enumerate(clients, start=1):

        attempt_start = time.perf_counter()
        first_token_time = None
        generated_text = ""
        prompt_tokens = None
        completion_tokens = None

        try:

            stream = api_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                stream=True,
                stream_options={
                    "include_usage": True
                }
            )

            for chunk in stream:

                now = time.perf_counter()

                if chunk.choices:

                    delta = chunk.choices[0].delta
                    content = getattr(
                        delta,
                        "content",
                        None
                    )

                    if content:

                        if first_token_time is None:
                            first_token_time = now

                        generated_text += content

                usage = getattr(
                    chunk,
                    "usage",
                    None
                )

                if usage:

                    prompt_tokens = getattr(
                        usage,
                        "prompt_tokens",
                        prompt_tokens
                    )

                    completion_tokens = getattr(
                        usage,
                        "completion_tokens",
                        completion_tokens
                    )

            if not generated_text.strip():
                raise RuntimeError("API returned an empty summary")

            end_time = time.perf_counter()
            total_latency = end_time - request_start

            if first_token_time is not None:
                ttft = first_token_time - attempt_start
            else:
                ttft = end_time - attempt_start

            if (
                first_token_time is not None
                and completion_tokens is not None
                and completion_tokens > 1
            ):

                generation_time = end_time - first_token_time
                tpot = generation_time / (completion_tokens - 1)

            else:
                tpot = None

            return {
                "summary": generated_text.strip(),
                "ttft": ttft,
                "tpot": tpot,
                "total_latency": total_latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": calc_cost(
                    prompt_tokens,
                    completion_tokens
                ),
                "api_index": api_index
            }

        except Exception as e:

            errors.append(
                f"API {api_index} {type(e).__name__}: {e}"
            )

            if api_index < len(clients):
                print(
                    f"[LLM FAILOVER] API {api_index} failed; "
                    f"trying API {api_index + 1}"
                )

    print("[LLM ERROR] All configured APIs failed")

    for error in errors:
        print(f"[LLM ERROR] {error}")

    return {
        "summary": "[LLM FAILED]",
        "ttft": None,
        "tpot": None,
        "total_latency": time.perf_counter() - request_start,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost": None
    }


# =========================================================
# FULL LLM BASELINE
# =========================================================

def summarize_with_llm(article, max_words=250):

    return _call_llm(
        build_full_prompt(max_words),
        article,
        max_tokens=max_output_tokens(max_words)
    )


# =========================================================
# HYBRID FINAL SYNTHESIS
# =========================================================

def summarize_hybrid_with_llm(hybrid_context, max_words=250):

    return _call_llm(
        build_hybrid_prompt(max_words),
        hybrid_context,
        max_tokens=max_output_tokens(max_words)
    )