import time
import ollama

from prompts import (
    build_chunk_prompt,
    build_full_prompt,
    max_output_tokens
)


MODEL = "qwen2.5:1.5b"

def summarize_with_slm(
    text,
    max_words=250,
    is_chunk=False
):  

    if is_chunk:
        system_prompt = build_chunk_prompt(max_words)
    else:
        system_prompt = build_full_prompt(max_words)

    start_time = time.perf_counter()

    first_token_time = None
    generated_text = ""

    completion_tokens = None
    prompt_tokens = None
    eval_duration = None
    prompt_eval_duration = None

    try:

        # Qwen2.5 3B supports a context window up to 32K tokens.
        # A larger context is needed for full-document input.
        stream = ollama.chat(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            stream=True,

            options={
                "temperature": 0.2,
                "num_predict": max_output_tokens(max_words),
                "num_ctx": 32768
            }
        )

        for chunk in stream:

            message = chunk.get("message", {})
            content = message.get("content", "")

            if content:

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                generated_text += content

            if chunk.get("done", False):

                completion_tokens = chunk.get(
                    "eval_count"
                )

                prompt_tokens = chunk.get(
                    "prompt_eval_count"
                )

                eval_duration = chunk.get(
                    "eval_duration"
                )

                prompt_eval_duration = chunk.get(
                    "prompt_eval_duration"
                )

        end_time = time.perf_counter()

        total_latency = end_time - start_time

        ttft = (
            first_token_time - start_time
            if first_token_time is not None
            else total_latency
        )

        # TPOT based on Ollama's generation duration.
        if (
            eval_duration is not None
            and completion_tokens is not None
            and completion_tokens > 1
        ):

            tpot = (
                eval_duration / 1_000_000_000
            ) / (completion_tokens - 1)

        else:

            tpot = None

        summary = generated_text.strip()

        if not summary:

            return {
                "summary": "[SLM FAILED]",
                "ttft": ttft,
                "tpot": tpot,
                "total_latency": total_latency,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "prompt_eval_duration": prompt_eval_duration
            }

        return {
            "summary": summary,
            "ttft": ttft,
            "tpot": tpot,
            "total_latency": total_latency,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_eval_duration": prompt_eval_duration
        }

    except Exception as e:

        print(
            f"[SLM ERROR] {type(e).__name__}: {e}"
        )

        return {
            "summary": "[SLM FAILED]",
            "ttft": None,
            "tpot": None,
            "total_latency": None,
            "completion_tokens": None,
            "prompt_tokens": None,
            "prompt_eval_duration": None
        }