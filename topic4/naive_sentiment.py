# A hand-curated word list — you can expand this list
POSITIVE_WORDS = {
    "good", "great", "excellent", "happy", "love", "wonderful",
    "fantastic", "amazing", "best", "enjoy", "beautiful", "nice"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "horrible",
    "disgusting", "poor", "sad", "boring", "ugly", "dreadful"
}


def tokenize(text):
    """
    Split text into lowercase words, stripping punctuation.
    No libraries — just plain Python.
    """
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    table = str.maketrans("", "", punctuation)
    cleaned = text.translate(table)
    lowercase = cleaned.lower()
    tokens = lowercase.split()
    return tokens


def detect_sentiment(text):
    # A very naive sentiment detector that counts positive and negative words.
    # It doesn't handle negation, sarcasm, or any of the complexities of natural language.
    # But it's a start!

    tokens = tokenize(text)

    pos_count = 0
    neg_count = 0

    for token in tokens:
        if token in POSITIVE_WORDS:
            pos_count += 1
        elif token in NEGATIVE_WORDS:
            neg_count += 1

    print(f"  Tokens: {tokens}")
    print(f"  Positive count: {pos_count}, Negative count: {neg_count}")

    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


# --- Test sentences ---
sentences = [
    "The movie was great and I love the story.",
    "This is the worst, most horrible film I ever saw.",
    "The food was not good at all.",
    "I don't hate it.",
    "The movie was so bad it was actually good.",
    "Fine.",
    "It was an experience.",
]

for s in sentences:
    print(f"\nText: '{s}'")
    result = detect_sentiment(s)
    print(f"  Verdict: {result}")