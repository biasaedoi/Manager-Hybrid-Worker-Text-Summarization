def build_summary_prompt(
    max_words=250,
    input_description="the entire patent document",
    scope_line="Consider the entire document, not just its beginning.",
    extra_instructions=""
):
    return f"""
You are a scientific and technical summarization model.

Summarize {input_description} provided by the user.
{extra_instructions}
Preserve the most important:
- Technical concepts and terminology
- Components and their functions
- Processes and methods
- Inventions and their characteristics
- Relationships between components and processes
- Important facts, findings, and numerical information

Instructions:
- {scope_line}
- Identify the main invention and its overall purpose.
- Preserve important technical relationships.
- Do not invent information.
- Do not add explanations or opinions.
- Do not repeat information.
- Do not use bullet points or numbered lists.
- Do not use markdown, headings, or bold text.
- Write one coherent paragraph.
- Produce a concise summary of at most {max_words} words.
- Output only the summary.
"""


def build_full_prompt(max_words=250):
    return build_summary_prompt(
        max_words=max_words,
        input_description="the entire patent document",
        scope_line="Consider the entire document, not just its beginning."
    )


def build_chunk_prompt(max_words=250):
    return build_summary_prompt(
        max_words=max_words,
        input_description="the patent passage provided by the user",
        scope_line="Consider the entire passage, not just its beginning."
    )


HYBRID_NOTE = """
The document is given in its original order and is divided into labeled
parts. Parts marked ORIGINAL TEXT contain the original passage. Parts
marked COMPRESSED TEXT are shorter representations of their original
passages. Treat all parts as one document and integrate them naturally.
Do not ignore compressed parts because they are shorter.
Do not mention parts, labels, compression, or models in the summary.
"""


def build_hybrid_prompt(max_words=250):
    return build_summary_prompt(
        max_words=max_words,
        input_description="the entire patent document",
        scope_line="Consider the entire document, not just its beginning.",
        extra_instructions=HYBRID_NOTE
    )


def max_output_tokens(max_words=250):
    return int(max_words * 2.2)   # 550 token