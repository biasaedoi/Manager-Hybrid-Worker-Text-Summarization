import re
import spacy


# Lightweight tokenizer just for sentence segmentation
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def create_chunks(text, max_words=1024):
    """
    Split document into sentence-aware chunks.

    Each chunk contains approximately <= max_words words.
    Sentences are kept intact whenever possible.

    If one sentence itself exceeds max_words,
    the sentence will be split manually.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    doc = nlp(text)

    chunks = []

    current_sentences = []
    current_word_count = 0

    for sent in doc.sents:

        sentence = sent.text.strip()

        if not sentence:
            continue

        words = sentence.split()
        sentence_word_count = len(words)

        # Edge case:
        # one sentence is already longer than max_words
        if sentence_word_count > max_words:

            # Save previous accumulated chunk first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0

            # Split oversized sentence
            for i in range(0, sentence_word_count, max_words):
                piece = words[i:i + max_words]
                chunks.append(" ".join(piece))

            continue

        # Adding sentence would exceed max_words
        if (
            current_sentences
            and current_word_count + sentence_word_count > max_words
        ):

            chunks.append(" ".join(current_sentences))

            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    # Remaining text
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks