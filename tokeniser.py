from string import punctuation
import re


def tokenise(sentence: str) -> list:
    """Tokenises a sentence into words and punctuation marks as separate tokens.

    Parameters
    ----------
    sentence : str
        A sentence to be tokenised.

    Returns
    -------
    list
        A list of tokens (words and punctuation) within that sentence.
    """
    sentence = sentence.strip()
    # Split words and punctuation as separate tokens
    tokens = re.findall(r"\w+|[{}]".format(re.escape(punctuation)), sentence)
    return tokens


def detokenise(tokens: list) -> str:
    """Reverses the tokenisation process, joining tokens into a sentence.

    Parameters
    ----------
    tokens : list
        A list of tokens (words and punctuation).

    Returns
    -------
    str
        The reconstructed sentence.
    """
    sentence = ""
    for token in tokens:
        if token in punctuation:
            sentence += token
        else:
            if sentence and sentence[-1] not in (" ", *punctuation):
                sentence += " "
            sentence += token
    return sentence


def parse_text(text: str) -> list:
    """Parses a text into a list of tokens. A very simple parser that splits on whitespace.

    Parameters
    ----------
    text : str
        A text to be parsed.

    Returns
    -------
    list
        A list of tokens within that text.
    """
    return tokenise(text)
