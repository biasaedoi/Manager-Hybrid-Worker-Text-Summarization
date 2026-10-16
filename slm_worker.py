import time
import ollama


MODEL = "qwen2.5:0.5b"


SYSTEM_PROMPT = """
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


def summarize_with_slm(text):

    start_time = time.perf_counter()

    first_token_time = None

    generated_text = ""

    completion_tokens = None

    try:

        stream = ollama.chat(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            stream=True,

            options={
                "temperature": 0.2,
                "num_predict": 1024
            }
        )

        for chunk in stream:

            now = time.perf_counter()

            message = chunk.get(
                "message",
                {}
            )

            content = message.get(
                "content",
                ""
            )

            if content:

                if first_token_time is None:
                    first_token_time = now

                generated_text += content

            if chunk.get("done", False):

                completion_tokens = chunk.get(
                    "eval_count",
                    None
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

        summary = generated_text.strip()

        if not summary:

            return {
                "summary": "[SLM FAILED]",
                "ttft": ttft,
                "tpot": tpot,
                "total_latency": total_latency,
                "completion_tokens": completion_tokens
            }

        return {
            "summary": summary,
            "ttft": ttft,
            "tpot": tpot,
            "total_latency": total_latency,
            "completion_tokens": completion_tokens
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
            "completion_tokens": None
        }