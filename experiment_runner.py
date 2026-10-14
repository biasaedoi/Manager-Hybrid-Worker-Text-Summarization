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

# Ambil 50 dokumen
NUM_DOCUMENTS = 50
START_INDEX = 0

# Pilihan mode:
# "FULL_SLM"
# "FULL_LLM"
# "HYBRID"

MODES = [
    "FULL_SLM"
]


# =========================================================
# HELPERS
# =========================================================

def safe_mean(values):
    values = [
        v for v in values
        if v is not None
    ]

    if not values:
        return None

    return statistics.mean(values)


def safe_median(values):
    values = [
        v for v in values
        if v is not None
    ]

    if not values:
        return None

    return statistics.median(values)


def safe_sum(values):
    values = [
        v for v in values
        if v is not None
    ]

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

    # Gabungkan semua kolom yang tersedia pada hasil eksperimen.
    fieldnames = list(dict.fromkeys(
        key
        for result in results
        for key in result.keys()
    ))

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

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    samples = load_bigpatent_sample(
        n=NUM_DOCUMENTS,
        start=START_INDEX
    )

    print()
    print("=" * 70)
    print("DATASET SELECTION")
    print("=" * 70)

    print(
        f"Dataset range    : "
        f"{START_INDEX + 1}-{START_INDEX + len(samples)}"
    )

    print(
        f"Loaded samples   : "
        f"{len(samples)}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # SELECT DOCUMENTS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RUN MODES
    # -----------------------------------------------------

    for mode in MODES:

        print("\n")
        print("=" * 70)
        print(
            f"RUNNING MODE: {mode}"
        )
        print("=" * 70)

        results = []

        # =================================================
        # DOCUMENT LOOP
        # =================================================

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
                    "mode": mode,
                    "summary": summary,
                    "ttft": output.get("ttft"),
                    "tpot": output.get("tpot"),
                    "total_latency": total_latency,
                    "prompt_tokens": output.get(
                        "prompt_tokens"
                    ),
                    "completion_tokens": output.get(
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
                    "mode": mode,
                    "summary": summary,
                    "ttft": output.get("ttft"),
                    "tpot": output.get("tpot"),
                    "total_latency": total_latency,
                    "prompt_tokens": output.get(
                        "prompt_tokens"
                    ),
                    "completion_tokens": output.get(
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

                result = {
                    "mode": mode,
                    **output
                }

                print("\nROUTING:")

                for routing in output[
                    "routing_results"
                ]:
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

                print(
                    f"Total Chunks     : "
                    f"{output['total_chunks']}"
                )

                print(
                    f"SLM Chunks       : "
                    f"{output['slm_chunks']}"
                )

                print(
                    f"LLM Chunks       : "
                    f"{output['llm_chunks']}"
                )

                print(
                    f"SLM Ratio        : "
                    f"{output['slm_ratio']:.2%}"
                )

                print(
                    f"LLM Ratio        : "
                    f"{output['llm_ratio']:.2%}"
                )

                print(
                    f"Original Words   : "
                    f"{output['original_words']}"
                )

                print(
                    f"Final LLM Words  : "
                    f"{output['hybrid_input_words']}"
                )

                print(
                    f"SLM Compression  : "
                    f"{output['slm_compression_ratio']:.2%}"
                )

            # =================================================
            # INVALID MODE
            # =================================================

            else:
                print(
                    f"Unknown mode: {mode}"
                )

                return

            # =================================================
            # EVALUATION
            # =================================================

            print("\n[EVALUATION DEBUG]")

            print(
                "Summary type   :",
                type(summary)
            )

            print(
                "Summary length :",
                len(summary)
                if summary
                else 0
            )

            print(
                "Summary start  :",
                repr(summary[:100])
                if summary
                else None
            )

            print(
                "Reference type :",
                type(reference)
            )

            print(
                "Reference len  :",
                len(reference)
                if reference
                else 0
            )

            print(
                "Reference start:",
                repr(reference[:100])
                if reference
                else None
            )

            # -------------------------------------------------
            # RUN EVALUATION
            # -------------------------------------------------

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

            # -------------------------------------------------
            # STORE EVALUATION
            # -------------------------------------------------

            result.update({
                "rouge1": evaluation.get(
                    "ROUGE1"
                ),
                "rougeL": evaluation.get(
                    "ROUGEL"
                ),
                "bert_score": evaluation.get(
                    "BERTScore"
                ),
                "bge_m3": evaluation.get(
                    "BGE_M3"
                ),
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
                f"{result['rouge1']}"
            )

            print(
                f"ROUGE-L          : "
                f"{result['rougeL']}"
            )

            print(
                f"BERTScore        : "
                f"{result['bert_score']}"
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
                f"{result['ttft']}"
            )

            print(
                f"TPOT             : "
                f"{result['tpot']}"
            )

            print(
                f"Total Latency    : "
                f"{result['total_latency']}"
            )

            print(
                f"Prompt Tokens    : "
                f"{result['prompt_tokens']}"
            )

            print(
                f"Output Tokens    : "
                f"{result['completion_tokens']}"
            )

            print(
                f"Cost             : "
                f"${result['cost']}"
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

        # =====================================================
        # QUALITY
        # =====================================================

        print(
            "\n========== QUALITY =========="
        )

        rouge1_values = [
            r["rouge1"]
            for r in results
        ]

        rougeL_values = [
            r["rougeL"]
            for r in results
        ]

        bert_values = [
            r["bert_score"]
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

        # =====================================================
        # LATENCY
        # =====================================================

        print(
            "\n========== LATENCY =========="
        )

        ttft_values = [
            r["ttft"]
            for r in results
        ]

        tpot_values = [
            r["tpot"]
            for r in results
        ]

        latency_values = [
            r["total_latency"]
            for r in results
        ]

        print(
            "TTFT mean       :",
            safe_mean(ttft_values)
        )

        print(
            "TTFT median     :",
            safe_median(ttft_values)
        )

        print(
            "TPOT mean       :",
            safe_mean(tpot_values)
        )

        print(
            "TPOT median     :",
            safe_median(tpot_values)
        )

        print(
            "Latency mean    :",
            safe_mean(latency_values)
        )

        print(
            "Latency median  :",
            safe_median(latency_values)
        )

        # =====================================================
        # TOKENS
        # =====================================================

        print(
            "\n========== TOKENS =========="
        )

        prompt_values = [
            r["prompt_tokens"]
            for r in results
        ]

        output_values = [
            r["completion_tokens"]
            for r in results
        ]

        print(
            "Prompt Tokens mean   :",
            safe_mean(prompt_values)
        )

        print(
            "Prompt Tokens median :",
            safe_median(prompt_values)
        )

        print(
            "Output Tokens mean   :",
            safe_mean(output_values)
        )

        print(
            "Output Tokens median :",
            safe_median(output_values)
        )

        # =====================================================
        # COST
        # =====================================================

        print(
            "\n========== COST =========="
        )

        cost_values = [
            r["cost"]
            for r in results
        ]

        print(
            "Cost mean       : $",
            safe_mean(cost_values)
        )

        print(
            "Cost median     : $",
            safe_median(cost_values)
        )

        print(
            "Total Cost      : $",
            safe_sum(cost_values)
        )

        # =====================================================
        # HYBRID ONLY
        # =====================================================

        if mode == "HYBRID":

            print(
                "\n========== ROUTING =========="
            )

            total_chunks = sum(
                r["total_chunks"]
                for r in results
            )

            total_slm = sum(
                r["slm_chunks"]
                for r in results
            )

            total_llm = sum(
                r["llm_chunks"]
                for r in results
            )

            print(
                "Total Chunks    :",
                total_chunks
            )

            print(
                "SLM Chunks      :",
                total_slm
            )

            print(
                "LLM Chunks      :",
                total_llm
            )

            if total_chunks > 0:

                print(
                    "SLM Percentage :",
                    f"{total_slm / total_chunks:.2%}"
                )

                print(
                    "LLM Percentage :",
                    f"{total_llm / total_chunks:.2%}"
                )

            slm_compressions = [
                r["slm_compression_ratio"]
                for r in results
            ]

            print(
                "SLM Compression :",
                safe_mean(
                    slm_compressions
                )
            )

        save_results_to_csv(
            results,
            mode
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_experiment()