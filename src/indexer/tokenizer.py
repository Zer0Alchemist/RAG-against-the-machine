import re
from typing import List


def camel_case(text: str | None) -> str:
    """
    this function separates camelCase and PascalCase words
    by adding spaces between lowercase and uppercase transitions

    args:
        text: str | None = text to normalize

    return:
        string with separated words, or an empty string if no text
    """
    if not text:
        return ""

    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    return text


def snake_case(text: str | None) -> str:
    """
    this function converts snake_case text into separated words
    by replacing underscores with spaces

    args:
        text: str | None = text to normalize

    return:
        string with underscores replaced by spaces,
        or an empty string if no text
    """
    if not text:
        return ""

    text = text.replace('_', ' ')

    return text


def punc_norm(text: str | None) -> str:
    """
    this function normalizes punctuation by replacing
    non-alphanumeric and non-whitespace characters with spaces

    args:
        text: str | None = text to normalize

    return:
        string with punctuation replaced by spaces,
        or an empty string if no text
    """
    if not text:
        return ""

    txt = ""

    for n in text:
        if n.isalnum() or n.isspace():
            txt += n
        else:
            txt += " "

    return txt


def tokenizer(text: str | None) -> List[str]:
    """
    this function tokenizes text by normalizing camelCase,
    snake_case and punctuation, converting text to lowercase,
    and removing stop words and single-character tokens

    args:
        text: str | None = text to tokenize

    return:
        list of normalized tokens, or an empty list if no text
"""
    if not text:
        return []

    stop_words: set[str] = {
        "an", "the", "and", "but", "or", "nor", "so", "yet", "if", "because",
        "as", "until", "while", "though", "although", "in", "on", "at", "to",
        "for", "with", "by", "from", "of", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "up",
        "down", "out", "off", "over", "under", "me", "my", "myself", "we",
        "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "would",
        "should", "could", "ought", "may", "might", "must", "shall", "will",
        "can", "very", "too", "also", "just", "then", "now", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "not",
        "only", "own", "same", "than"
    }

    text = camel_case(text)
    text = snake_case(text)
    text = punc_norm(text)

    text = text.lower()

    return [token for token in text.split()
            if token not in stop_words and len(token) > 1]
