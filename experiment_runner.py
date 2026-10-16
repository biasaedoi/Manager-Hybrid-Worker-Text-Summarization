import statistics
import time
import csv
import os

from data_loader import load_bigpatent_sample
from llm_worker import summarize_with_llm
from slm_worker import summarize_with_slm
from pipeline import run_hybrid_pipeline
from evaluate import evaluate_summary, compression_ratio


# =========================================================
# CONFIGURATION
# =========================================================

NUM_DOCUMENTS = 50
START_INDEX = 0

# MODE CHOICES:
# "FULL_SLM"
# "FULL_LLM"
# "HYBRID"

MODES = [
    "HYBRID"
]


# =========================================================
# HELPERS
# =========================================================

def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.mean(values)


def safe_median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.median(values)


def safe_sum(values):
    values = [v for v in values if v is not None]
    if not values:
        return 0
    return sum(values)


def save_results_to_csv(results, mode):
    if not results:
        print("[CSV] No results to save.")
        return

    output_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(
        output_dir,
        f"results_{mode.lower()}.csv"
    )

    # Urutan fieldnames standar sesuai template kolom terstruktur
    standard_fields = [
        "document",
        "mode",
        "summary",
        "rouge_1",
        "rouge_l",
        "bertscore",
        "bge_m3",
        "compression",
        "ttft",
        "tpot",
        "total_latency",
        "prompt_tokens",
        "output_tokens",
        "cost",
        "total_chunks",
        "slm_chunks",
        "llm_chunks",
        "slm_ratio",
        "llm_ratio",
        "slm_compression_ratio"
    ]

    # Gabungkan dengan kunci lain jika ada tambahan dinamis
    all_keys = list(dict.fromkeys(
        key
        for result in results
        for key in result.keys()
    ))
    fieldnames = [f for f in standard_fields if f in all_keys] + [k for k in all_keys if k not in standard_fields]

    with open(
        filename,
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"[CSV] Results saved to: {filename}")


# =========================================================
# EXPERIMENT
# =========================================================

def run_experiment():

    samples = load_bigpatent_sample(
        n=NUM_DOCUMENTS,
        start=START_INDEX
    )

    print()
    print("=" * 70)
    print("DATASET SELECTION")
    print("=" * 70)

    print(
        f"Dataset range     : "
        f"{START_INDEX + 1}-{START_INDEX + len(samples)}"
    )

    print(
        f"Loaded samples    : "
        f"{len(samples)}"
    )

    print("=" * 70)

    selected_samples = []

    for index, sample in enumerate(
        samples,
        start=1
    ):
        actual_index = START_INDEX + index

        selected_samples.append(
            (actual_index, sample)
        )

    print()
    print(
        f"Selected documents: "
        f"{len(selected_samples)}"
    )

    print("=" * 70)

    for mode in MODES:

        print("\n")
        print("=" * 70)
        print(
            f"RUNNING MODE: {mode}"
        )
        print("=" * 70)

        results = []

        for experiment_index, (
            actual_index,
            sample
        ) in enumerate(
            selected_samples,
            start=1
        ):

            article = sample["article"]
            reference = sample["reference"]

            print("\n")
            print("=" * 70)
            print(
                f"DOCUMENT "
                f"{actual_index} "
                f"({experiment_index}/{len(selected_samples)})"
            )
            print("=" * 70)

            # =================================================
            # FULL SLM
            # =================================================

            if mode == "FULL_SLM":

                pipeline_start = time.perf_counter()

                try:
                    output = summarize_with_slm(
                        article
                    )

                except Exception as e:
                    print(
                        f"[SLM ERROR] "
                        f"{type(e).__name__}: {e}"
                    )

                    output = {
                        "summary": "[SLM FAILED]",
                        "ttft": None,
                        "tpot": None,
                        "total_latency": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "cost": 0
                    }

                pipeline_end = time.perf_counter()

                total_latency = (
                    pipeline_end
                    - pipeline_start
                )

                summary = output.get(
                    "summary",
                    "[SLM FAILED]"
                )

                result = {
                    "document": actual_index,
                    "mode": mode,
                    "summary": summary,
                    "ttft": output.get("ttft"),
                    "tpot": output.get("tpot"),
                    "total_latency": total_latency,
                    "prompt_tokens": output.get(
                        "prompt_tokens"
                    ),
                    "output_tokens": output.get(
                        "completion_tokens"
                    ),
                    "cost": output.get(
                        "cost",
                        0
                    ),
                    "total_chunks": None,
                    "slm_chunks": None,
                    "llm_chunks": None,
                    "slm_ratio": None,
                    "llm_ratio": None,
                    "slm_compression_ratio": None
                }

            # =================================================
            # FULL LLM
            # =================================================

            elif mode == "FULL_LLM":

                pipeline_start = time.perf_counter()

                try:
                    output = summarize_with_llm(
                        article
                    )

                except Exception as e:
                    print(
                        f"[LLM ERROR] "
                        f"{type(e).__name__}: {e}"
                    )

                    output = {
                        "summary": "[LLM FAILED]",
                        "ttft": None,
                        "tpot": None,
                        "total_latency": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "cost": None
                    }

                pipeline_end = time.perf_counter()

                total_latency = (
                    pipeline_end
                    - pipeline_start
                )

                summary = output.get(
                    "summary",
                    "[LLM FAILED]"
                )

                result = {
                    "document": actual_index,
                    "mode": mode,
                    "summary": summary,
                    "ttft": output.get("ttft"),
                    "tpot": output.get("tpot"),
                    "total_latency": total_latency,
                    "prompt_tokens": output.get(
                        "prompt_tokens"
                    ),
                    "output_tokens": output.get(
                        "completion_tokens"
                    ),
                    "cost": output.get(
                        "cost"
                    ),
                    "total_chunks": None,
                    "slm_chunks": None,
                    "llm_chunks": None,
                    "slm_ratio": None,
                    "llm_ratio": None,
                    "slm_compression_ratio": None
                }

            # =================================================
            # HYBRID
            # =================================================

            elif mode == "HYBRID":

                try:
                    output = run_hybrid_pipeline(
                        article
                    )

                except Exception as e:
                    print(
                        f"[HYBRID ERROR] "
                        f"{type(e).__name__}: {e}"
                    )

                    output = {
                        "summary": "[HYBRID FAILED]",
                        "ttft": None,
                        "tpot": None,
                        "total_latency": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "cost": None,
                        "total_chunks": 0,
                        "slm_chunks": 0,
                        "llm_chunks": 0,
                        "slm_ratio": 0,
                        "llm_ratio": 0,
                        "slm_compression_ratio": 0,
                        "routing_results": [],
                        "original_words": len(
                            article.split()
                        ),
                        "hybrid_input_words": 0,
                        "manager_latency": 0,
                        "slm_latency": 0,
                        "llm_latency": None
                    }

                summary = output["summary"]

                # Mapping key completion_tokens ke output_tokens jika ada di output pipeline
                if "completion_tokens" in output and "output_tokens" not in output:
                    output["output_tokens"] = output.pop("completion_tokens")

                result = {
                    "document": actual_index,
                    "mode": mode,
                    **output
                }

                print("\nROUTING:")

                for routing in output.get("routing_results", []):
                    print(
                        f"Chunk "
                        f"{routing['chunk']:>2} | "
                        f"{routing['words']:>4} words | "
                        f"{routing['route']:<3} | "
                        f"ASL="
                        f"{routing['asl']:.3f} | "
                        f"Clause="
                        f"{routing['clause_density']:.3f} | "
                        f"Rarity="
                        f"{routing['word_rarity']:.3f} | "
                        f"{routing['reason']}"
                    )

                print()

            else:
                print(
                    f"Unknown mode: {mode}"
                )
                return

            # =================================================
            # EVALUATION
            # =================================================

            print("\n[EVALUATION DEBUG]")

            try:
                evaluation = evaluate_summary(
                    reference,
                    summary
                )

                compression = compression_ratio(
                    article,
                    summary
                )

            except Exception as e:
                print(
                    f"[EVALUATION ERROR] "
                    f"{type(e).__name__}: {e}"
                )

                evaluation = {
                    "ROUGE1": None,
                    "ROUGEL": None,
                    "BERTScore": None,
                    "BGE_M3": None
                }

                compression = None

            # Menyimpan dengan nama kolom sesuai template template (rouge_1, rouge_l, bertscore, bge_m3)
            result.update({
                "rouge_1": evaluation.get("ROUGE1"),
                "rouge_l": evaluation.get("ROUGEL"),
                "bertscore": evaluation.get("BERTScore"),
                "bge_m3": evaluation.get("BGE_M3"),
                "compression": compression
            })

            results.append(
                result
            )

            # =================================================
            # DOCUMENT OUTPUT
            # =================================================

            print("\nMETRICS:")

            print(
                f"Document         : "
                f"{actual_index}"
            )

            print(
                f"ROUGE-1          : "
                f"{result['rouge_1']}"
            )

            print(
                f"ROUGE-L          : "
                f"{result['rouge_l']}"
            )

            print(
                f"BERTScore        : "
                f"{result['bertscore']}"
            )

            print(
                f"BGE-M3           : "
                f"{result['bge_m3']}"
            )

            print(
                f"Compression      : "
                f"{result['compression']}"
            )

            print(
                f"TTFT             : "
                f"{result.get('ttft')}"
            )

            print(
                f"TPOT             : "
                f"{result.get('tpot')}"
            )

            print(
                f"Total Latency    : "
                f"{result.get('total_latency')}"
            )

            print(
                f"Prompt Tokens    : "
                f"{result.get('prompt_tokens')}"
            )

            print(
                f"Output Tokens    : "
                f"{result.get('output_tokens')}"
            )

            print(
                f"Cost             : "
                f"${result.get('cost')}"
            )

            print("\nSUMMARY:")
            print(summary)

        # =====================================================
        # FINAL STATISTICS
        # =====================================================

        print("\n")
        print("=" * 70)
        print(
            f"{mode} — FINAL RESULTS"
        )
        print("=" * 70)

        print(
            f"Documents       : "
            f"{len(results)}"
        )

        print(
            "\n========== QUALITY =========="
        )

        rouge1_values = [
            r["rouge_1"]
            for r in results
        ]

        rougeL_values = [
            r["rouge_l"]
            for r in results
        ]

        bert_values = [
            r["bertscore"]
            for r in results
        ]

        bge_values = [
            r["bge_m3"]
            for r in results
        ]

        compression_values = [
            r["compression"]
            for r in results
        ]

        print(
            "ROUGE-1 mean    :",
            safe_mean(rouge1_values)
        )

        print(
            "ROUGE-1 median  :",
            safe_median(rouge1_values)
        )

        print(
            "ROUGE-L mean    :",
            safe_mean(rougeL_values)
        )

        print(
            "ROUGE-L median  :",
            safe_median(rougeL_values)
        )

        print(
            "BERTScore mean  :",
            safe_mean(bert_values)
        )

        print(
            "BERTScore median:",
            safe_median(bert_values)
        )

        print(
            "BGE-M3 mean     :",
            safe_mean(bge_values)
        )

        print(
            "BGE-M3 median   :",
            safe_median(bge_values)
        )

        print(
            "Compression mean:",
            safe_mean(compression_values)
        )

        save_results_to_csv(
            results,
            mode
        )


if __name__ == "__main__":
    run_experiment()