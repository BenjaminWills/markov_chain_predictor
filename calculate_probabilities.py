from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from tokeniser import tokenise


def calculate_n_grams(tokens: List[str], n: int) -> List[Tuple[str]]:
    """Calculates n-grams from a list of tokens, these are simply tuples of n consecutive tokens
    in a list of tokens.

    Parameters
    ----------
    tokens : List[str]
        A list of tokens to calculate n-grams from.
    n : int
        The size of the n-grams to calculate, e.g. 2 for bigrams, 3 for trigrams, etc.

    Returns
    -------
    List[Tuple[str]]
        A list of n-grams, where each n-gram is a tuple of n tokens.
    """

    # Create the list of n grams, to do this we look at the tokenisation n-wise and create a list of n grams
    n_grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    return n_grams


def calculate_n_gram_frequency(text_corpus: str, n: int) -> Counter:
    """Calculate the frequency of n-grams in a text corpus.

    Parameters
    ----------
    text_corpus : str
        The text corpus to calculate n-gram frequencies from.
    n : int
        The size of the n-grams to calculate, e.g. 2 for bigrams, 3 for trigrams, etc.

    Returns
    -------
    Dict[str, int]
        A dictionary where the keys are n-grams and the values are their frequencies in the text corpus.
    """

    # Tokenise the text corpus
    tokens = tokenise(text_corpus)

    # Create the list of n grams, to do this we look at the tokenisation n-wise and create a list of n grams
    n_grams = calculate_n_grams(tokens, n)

    # Count the frequency of each token
    n_gram_frequency = Counter(n_grams)

    return n_gram_frequency


def calculate_n_gram_probabilities(
    n_gram_frequency: Dict[str, int],
) -> Dict[str, float]:
    """Calculates the probability of each n-gram in a text corpus. The probability of each n-gram is
    calculated as the frequency of the n-gram divided by the total number of tokens in the text corpus.
    This is a simple way to calculate the probability of each n-gram.

    Parameters
    ----------
    n_gram_frequency : Dict[str, int]
        A dictionary where the keys are n-grams and the values are their frequencies in the text corpus.

    Returns
    -------
    Dict[str, float]
        A dictionary where the keys are n-grams and the values are their probabilities in the text corpus.
    """

    # Calculate the total number of tokens
    total_tokens = sum(n_gram_frequency.values())

    # Calculate the probability of each token
    word_probabilities = {
        word: freq / total_tokens for word, freq in n_gram_frequency.items()
    }

    return word_probabilities


def create_n_gram_graph(
    text_corpus: str,
    n: int,
) -> Dict[str, Dict[Tuple[str], float]]:
    """Creates a n-gram graph from a text corpus. The n-gram graph is a dictionary where the keys are n-grams
    and the values are dictionaries of n-grams and their probabilities of being the next state given the current state.
    This is what is known as a markov process, where the next state is dependent on the current state.

    Parameters
    ----------
    text_corpus : str
        The text corpus to create the n-gram graph from.
    n : int
        The size of the n-grams to calculate, e.g. 2 for bigrams, 3 for trigrams, etc.

    Returns
    -------
    Dict[str, Dict[Tuple[str], float]]
        A dictionary where the keys are n-grams and the values are dictionaries of n-grams and their probabilities
        of being the next state given the current state.
    """
    # Tokenise the text corpus
    tokens = tokenise(text_corpus)

    # Create the list of n grams, to do this we look at the tokenisation n-wise and create a list of n grams
    n_grams = calculate_n_grams(tokens, n)

    # Given some n_gram frequency, we want to calculate the probability distribution of a set of n-grams being after a given n-gram
    # An example of this could be {('hello', 'how'): {('how', 'are'): 0.5, ('how', 'is'): 0.5}}
    n_gram_graph = defaultdict(lambda: defaultdict(lambda: 0))

    for index in range(len(n_grams) - 1):
        # Get the n-gram and the next n-gram
        n_gram = n_grams[index]
        next_n_gram = n_grams[index + 1]

        # Add to the graph, calculate frequency
        n_gram_graph[n_gram][next_n_gram] += 1

    # Normalise the probabilities for each source token
    for source_token, target_tokens in n_gram_graph.items():
        total_count = sum(target_tokens.values())
        for target_token, count in target_tokens.items():
            n_gram_graph[source_token][target_token] = count / total_count

    return n_gram_graph
