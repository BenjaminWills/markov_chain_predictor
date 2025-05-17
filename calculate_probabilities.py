from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from tokeniser import tokenise, detokenise
from loader import load_in_text_corpus

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import random


def calculate_n_grams(tokens: List[str], n: int) -> List[Tuple[str]]:
    """_summary_

    Parameters
    ----------
    tokens : List[str]
        _description_
    n : int
        _description_

    Returns
    -------
    List[Tuple[str]]
        _description_
    """

    # Create the list of n grams, to do this we look at the tokenisation n-wise and create a list of n grams
    n_grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    return n_grams


def calculate_word_frequency(text_corpus: str, n: int) -> Counter:
    """_summary_

    Parameters
    ----------
    text_corpus : str
        _description_
    n : int
        _description_

    Returns
    -------
    Dict[str, int]
        _description_
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
    """_summary_

    Parameters
    ----------
    n_gram_frequency : Dict[str, int]
        _description_

    Returns
    -------
    Dict[str, float]
        _description_
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


def recombine_n_grams(n_grams: List[Tuple[str]]) -> str:
    """Recombine n-grams into a string.

    Parameters
    ----------
    n_grams : List[Tuple[str]]
        The list of n-grams to recombine.

    Returns
    -------
    str
        The recombined string.
    """
    # Initally take the first three tokens
    recombined_token_list = list(n_grams[0])
    # Loop through the n-grams and append the last token of each n-gram
    for n_gram in n_grams[1:]:
        # Append the last token of the n-gram
        recombined_token_list.append(n_gram[-1])
    return detokenise(recombined_token_list)


def sample_n_gram_graph(
    n_gram: Tuple[str], n_gram_graph: Dict[str, Dict[Tuple[str], float]]
) -> str:
    """Sample a token from the n-gram graph given a token.

    Parameters
    ----------
    token : str
        The token to sample from.
    n_gram_graph : List[Tuple[str, str, float]]
        The n-gram graph.

    Returns
    -------
    str
        The sampled token.
    """
    choices = list(n_gram_graph.get(n_gram, {}).keys())
    probabilities = list(n_gram_graph.get(n_gram, {}).values())

    # Sample a token based on the probabilities
    sampled_n_gram = random.choices(
        choices,
        weights=probabilities,
        k=1,
    )[0]

    return sampled_n_gram


if __name__ == "__main__":
    # Example usage
    text_corpus = load_in_text_corpus("corpus.txt")
    n = 5

    # Calculate n-grams
    n_grams = calculate_n_grams(tokenise(text_corpus), n)
    print(f"# of {n}-grams: {len(n_grams):,}")

    # Calculate n-gram frequency
    n_gram_frequency = calculate_word_frequency(text_corpus, n)
    # Print the first 5 most common n-grams
    print(f"Most common {n}-grams:")
    for n_gram, freq in n_gram_frequency.most_common(5):
        print(f"{n_gram}: {freq}")

    # Calculate n-gram probabilities
    n_gram_probabilities = calculate_n_gram_probabilities(n_gram_frequency)
    # Print the first 5 most common n-gram probabilities
    print(f"Most common {n}-gram probabilities:")
    for n_gram, prob in sorted(
        n_gram_probabilities.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{n_gram}: {prob:.2%}")
    # Create n-gram graph
    n_gram_graph = create_n_gram_graph(text_corpus, n)
    # Print the first 5 edges of the n-gram graph
    print(f"First 5 edges of the n-gram graph:")
    for source, targets in list(n_gram_graph.items())[:5]:
        for target, prob in targets.items():
            print(f"{source} -> {target}: {prob}")

    print("=" * 50)
    print("\n")

    n_gram = random.choice(list(n_gram_graph.keys()))
    print(f"Random n-gram: {n_gram}")
    n_grams = [n_gram]

    while len(n_grams) < 200:
        # Sample a token from the n-gram graph
        n_gram = sample_n_gram_graph(n_gram, n_gram_graph)
        # Append the token to the text
        n_grams.append(n_gram)
    print(recombine_n_grams(n_grams))
