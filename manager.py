import re
import spacy

from wordfreq import iter_wordlist


nlp = spacy.load("en_core_web_sm")


# =========================================================
# COMMON WORD VOCABULARY
# =========================================================

TOP_K_WORDS = 5000

COMMON_WORDS = set()

for word in iter_wordlist("en"):

    if word.isalpha():
        COMMON_WORDS.add(word.lower())

    if len(COMMON_WORDS) >= TOP_K_WORDS:
        break


# =========================================================
# ROUTING THRESHOLDS
# =========================================================

ASL_THRESHOLD = 20.0
CLAUSE_THRESHOLD = 1.6
RARITY_THRESHOLD = 0.20


# =========================================================
# METRICS
# =========================================================

def calculate_asl(doc):

    sentences = list(doc.sents)

    if not sentences:
        return 0.0

    total_words = 0

    for sentence in sentences:

        words = [
            token
            for token in sentence
            if token.is_alpha
        ]

        total_words += len(words)

    return total_words / len(sentences)


def calculate_clause_density(doc):

    sentences = list(doc.sents)

    if not sentences:
        return 0.0

    total_clauses = 0

    for sentence in sentences:

        clause_indicators = [
            token
            for token in sentence
            if token.dep_ in (
                "advcl",
                "ccomp",
                "xcomp"
            )
        ]

        total_clauses += len(clause_indicators)

    return total_clauses / len(sentences)


def calculate_word_rarity(text):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    if not words:
        return 0.0

    rare_words = sum(
        1
        for word in words
        if word not in COMMON_WORDS
    )

    return rare_words / len(words)


# =========================================================
# MANAGER
# =========================================================

def decide_route(text):

    doc = nlp(text)

    asl = calculate_asl(doc)

    clause_density = calculate_clause_density(doc)

    word_rarity = calculate_word_rarity(text)

    if word_rarity >= RARITY_THRESHOLD:

        route = "LLM"
        reason = "High word rarity"

    elif (
        asl >= ASL_THRESHOLD
        and clause_density >= CLAUSE_THRESHOLD
    ):

        route = "LLM"
        reason = "High sentence and clause complexity"

    else:

        route = "SLM"
        reason = "Low document complexity"

    return {
        "ASL": round(asl, 5),
        "ClauseDensity": round(
            clause_density,
            5
        ),
        "WordRarity": round(
            word_rarity,
            5
        ),
        "Route": route,
        "Reason": reason
    }