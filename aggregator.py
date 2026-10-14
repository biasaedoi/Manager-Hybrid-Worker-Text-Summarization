def build_hybrid_context(
    processed_chunks
):
    """
    processed_chunks example:

    [
        {
            "index": 0,
            "route": "LLM",
            "content": "original text..."
        },
        {
            "index": 1,
            "route": "SLM",
            "content": "compressed summary..."
        }
    ]

    The original document order is preserved.
    """

    processed_chunks = sorted(
        processed_chunks,
        key=lambda x: x["index"]
    )

    context_parts = []

    for item in processed_chunks:

        chunk_number = item["index"] + 1

        route = item["route"]

        content = item["content"]

        if route == "LLM":

            section = (
                f"[PART {chunk_number} - ORIGINAL TEXT]\n"
                f"{content}"
            )

        else:

            section = (
                f"[PART {chunk_number} - COMPRESSED TEXT]\n"
                f"{content}"
            )

        context_parts.append(section)

    return "\n\n".join(context_parts)