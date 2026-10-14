import time

from chunker import create_chunks
from manager import decide_route
from slm_worker import summarize_with_slm
from llm_worker import summarize_hybrid_with_llm
from aggregator import build_hybrid_context


MAX_WORDS = 1024


def run_hybrid_pipeline(article):

    pipeline_start = time.perf_counter()

    # =====================================================
    # CHUNKING
    # =====================================================

    chunks = create_chunks(
        article,
        max_words=MAX_WORDS
    )

    processed_chunks = []

    routing_results = []

    total_manager_latency = 0.0
    total_slm_latency = 0.0

    slm_ttfts = []
    slm_tpots = []

    slm_count = 0
    llm_count = 0

    original_words = len(
        article.split()
    )

    slm_original_words = 0
    slm_summary_words = 0

    # =====================================================
    # ROUTING
    # =====================================================

    for index, chunk in enumerate(chunks):

        manager_start = (
            time.perf_counter()
        )

        decision = decide_route(chunk)

        manager_end = (
            time.perf_counter()
        )

        manager_latency = (
            manager_end
            - manager_start
        )

        total_manager_latency += (
            manager_latency
        )

        route = decision["Route"]

        routing_results.append({
            "chunk": index + 1,
            "words": len(
                chunk.split()
            ),
            "route": route,
            "reason": decision["Reason"],
            "asl": decision["ASL"],
            "clause_density":
                decision["ClauseDensity"],
            "word_rarity":
                decision["WordRarity"],
            "manager_latency":
                manager_latency
        })

        # =================================================
        # LLM ROUTE
        #
        # IMPORTANT:
        # Do NOT call cloud LLM here.
        # Keep original text.
        # =================================================

        if route == "LLM":

            llm_count += 1

            processed_chunks.append({
                "index": index,
                "route": "LLM",
                "content": chunk
            })

        # =================================================
        # SLM ROUTE
        # =================================================

        else:

            slm_count += 1

            slm_original_words += len(
                chunk.split()
            )

            slm_result = (
                summarize_with_slm(
                    chunk
                )
            )

            summary = slm_result[
                "summary"
            ]

            # If SLM failed,
            # preserve original text instead
            if summary == "[SLM FAILED]":

                processed_chunks.append({
                    "index": index,
                    "route": "LLM",
                    "content": chunk
                })

                slm_count -= 1
                llm_count += 1

                continue

            processed_chunks.append({
                "index": index,
                "route": "SLM",
                "content": summary
            })

            slm_summary_words += len(
                summary.split()
            )

            latency = slm_result.get(
                "total_latency"
            )

            if latency is not None:

                total_slm_latency += (
                    latency
                )

            ttft = slm_result.get(
                "ttft"
            )

            if ttft is not None:
                slm_ttfts.append(ttft)

            tpot = slm_result.get(
                "tpot"
            )

            if tpot is not None:
                slm_tpots.append(tpot)

    # =====================================================
    # BUILD FINAL LLM CONTEXT
    # =====================================================

    hybrid_context = (
        build_hybrid_context(
            processed_chunks
        )
    )

    hybrid_input_words = len(
        hybrid_context.split()
    )

    # =====================================================
    # ONE CLOUD LLM CALL
    # =====================================================

    final_result = (
        summarize_hybrid_with_llm(
            hybrid_context
        )
    )

    pipeline_end = time.perf_counter()

    total_latency = (
        pipeline_end
        - pipeline_start
    )

    # =====================================================
    # ROUTING RATIO
    # =====================================================

    total_chunks = len(chunks)

    if total_chunks > 0:

        slm_ratio = (
            slm_count
            / total_chunks
        )

        llm_ratio = (
            llm_count
            / total_chunks
        )

    else:

        slm_ratio = 0.0
        llm_ratio = 0.0

    # =====================================================
    # SLM COMPRESSION
    # =====================================================

    if slm_original_words > 0:

        slm_compression_ratio = (
            1
            - (
                slm_summary_words
                / slm_original_words
            )
        )

    else:

        slm_compression_ratio = 0.0

    return {

        "summary":
            final_result["summary"],

        # Routing
        "total_chunks":
            total_chunks,

        "slm_chunks":
            slm_count,

        "llm_chunks":
            llm_count,

        "slm_ratio":
            slm_ratio,

        "llm_ratio":
            llm_ratio,

        "routing_results":
            routing_results,

        # Word reduction
        "original_words":
            original_words,

        "hybrid_input_words":
            hybrid_input_words,

        "slm_original_words":
            slm_original_words,

        "slm_summary_words":
            slm_summary_words,

        "slm_compression_ratio":
            slm_compression_ratio,

        # Latency
        "manager_latency":
            total_manager_latency,

        "slm_latency":
            total_slm_latency,

        "llm_latency":
            final_result[
                "total_latency"
            ],

        "total_latency":
            total_latency,

        # LLM metrics
        "ttft":
            final_result["ttft"],

        "tpot":
            final_result["tpot"],

        "prompt_tokens":
            final_result[
                "prompt_tokens"
            ],

        "completion_tokens":
            final_result[
                "completion_tokens"
            ],

        "cost":
            final_result["cost"]
    }