from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
import numpy as np


# ============================================================
# MODELS (loaded once)
# ============================================================

bge_model = SentenceTransformer("BAAI/bge-m3")
bert_scorer = BERTScorer(lang="en")
rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"],
    use_stemmer=True
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_summary(reference, generated):

    if not generated or generated.startswith("["):
        return {
            "ROUGE1": None,
            "ROUGEL": None,
            "BERTScore": None,
            "BGE_M3": None
        }

    # ROUGE
    rouge_scores = rouge.score(reference, generated)
    rouge1 = rouge_scores["rouge1"].fmeasure
    rougeL = rouge_scores["rougeL"].fmeasure

    # BERTScore
    _, _, bert_f1 = bert_scorer.score([generated], [reference])
    bert_score = bert_f1.item()

    # BGE-M3
    embeddings = bge_model.encode(
        [generated, reference],
        normalize_embeddings=True
    )
    bge_score = float(np.dot(embeddings[0], embeddings[1]))

    return {
        "ROUGE1": rouge1,
        "ROUGEL": rougeL,
        "BERTScore": bert_score,
        "BGE_M3": bge_score
    }


# ============================================================
# COMPRESSION
# ============================================================

def compression_ratio(article, summary):

    article_words = len(article.split())
    summary_words = len(summary.split())

    if article_words == 0:
        return 0

    return 1 - summary_words / article_words