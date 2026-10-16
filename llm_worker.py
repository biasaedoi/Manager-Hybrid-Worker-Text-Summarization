import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# API CONFIGURATION
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set in .env"
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

MODEL = "deepseek/deepseek-chat"


# =========================================================
# PRICING
# =========================================================

PRICE_INPUT_PER_1M = 0.2574
PRICE_OUTPUT_PER_1M = 1.029


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

def _call_llm(system_prompt, user_prompt):

    start_time = time.perf_counter()

    first_token_time = None

    generated_text = ""

    prompt_tokens = None
    completion_tokens = None

    try:

        stream = client.chat.completions.create(

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

            stream=True,

            stream_options={
                "include_usage": True
            }
        )

        for chunk in stream:

            now = time.perf_counter()

            # ==============================
            # TEXT
            # ==============================

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

            # ==============================
            # TOKEN USAGE
            # ==============================

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

        end_time = time.perf_counter()

        total_latency = (
            end_time - start_time
        )

        if first_token_time is not None:

            ttft = (
                first_token_time
                - start_time
            )

        else:

            ttft = total_latency

        if (
            first_token_time is not None
            and completion_tokens is not None
            and completion_tokens > 1
        ):

            generation_time = (
                end_time
                - first_token_time
            )

            tpot = (
                generation_time
                / (completion_tokens - 1)
            )

        else:

            tpot = None

        cost = calc_cost(
            prompt_tokens,
            completion_tokens
        )

        return {
            "summary": generated_text.strip(),
            "ttft": ttft,
            "tpot": tpot,
            "total_latency": total_latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost
        }

    except Exception as e:

        print(
            f"[LLM ERROR] {type(e).__name__}: {e}"
        )

        return {
            "summary": "[LLM FAILED]",
            "ttft": None,
            "tpot": None,
            "total_latency": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost": None
        }


# =========================================================
# FULL LLM BASELINE
# =========================================================

FULL_LLM_SYSTEM_PROMPT = """
You are a lightweight summarization worker.

Summarize the provided section of a patent document concisely.

Preserve the most important:
- technical concepts
- components
- processes
- inventions
- relationships
- important facts and numbers

The summary will later be used by another model to understand
this section of the original document.

Do not invent information.
Do not add explanations.
Do not repeat information.
Do not use bullet points.
Do not mention that this is a chunk or section.

Write approximately 3-5 concise sentences.

Output only the summary.
"""


def summarize_with_llm(article):

    return _call_llm(
        FULL_LLM_SYSTEM_PROMPT,
        article
    )


# =========================================================
# HYBRID FINAL SYNTHESIS
# =========================================================

HYBRID_SYSTEM_PROMPT = """
You are the final summarization model in a hybrid document
summarization system.

You will receive a reconstruction of a patent document.

Some parts are provided as ORIGINAL TEXT.
Other parts are provided as COMPRESSED TEXT generated from the
corresponding original text by a lightweight summarization model.

The order of all parts follows the original document.

Generate one coherent and concise summary representing the entire
document.

Treat compressed passages as representations of their corresponding
original passages and integrate them naturally with the surrounding
original information.

Preserve:
- the main technical problem
- the proposed invention or solution
- important components
- processes and relationships
- important technical facts
- important numbers when relevant

Do not ignore compressed passages merely because they are shorter.

Do not mention:
- chunks
- sections
- routing
- compression
- SLM
- LLM
- models

Do not invent information.
Do not repeat information.
Do not use bullet points.

Output only the final summary.
"""


def summarize_hybrid_with_llm(
    hybrid_context
):

    return _call_llm(
        HYBRID_SYSTEM_PROMPT,
        hybrid_context
    )