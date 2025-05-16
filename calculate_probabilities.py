from collections import Counter
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
    n_gram_frequency: Dict[str, int],
) -> List[Tuple[str, str, float]]:
    """To do this we want to look at all the first words in the n-grams,
    and make a list of all second words that follow them and their probabilities.

    Parameters
    ----------
    n_gram_frequency : Dict[str, int]
        _description_

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
        _description_
    """

    # Create a graph of n-grams and their probabilities
    n_grams = n_gram_frequency.keys()

    # Group by the first token
    n_gram_graph = {}
    for n_gram in n_grams:
        first_token = n_gram[0]
        second_token = n_gram[1]

        if first_token not in n_gram_graph:
            n_gram_graph[first_token] = []

        # Calculate the probability of the second token given the first token
        probability = n_gram_frequency[n_gram]

        n_gram_graph[first_token].append((second_token, probability))

    # Normalise the probabilities
    for first_token, second_tokens in n_gram_graph.items():
        total_probability = sum(prob for _, prob in second_tokens)
        n_gram_graph[first_token] = [
            (second_token, prob / total_probability)
            for second_token, prob in second_tokens
        ]

    # Turn into a list of tuples
    list_graph = []
    for first_token, second_tokens in n_gram_graph.items():
        for second_token, prob in second_tokens:
            list_graph.append((first_token, second_token, prob))

    return list_graph


def sample_n_gram_graph(token: str, n_gram_graph: List[Tuple[str, str, float]]) -> str:
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
    # Filter the graph for the given token
    filtered_graph = [edge for edge in n_gram_graph if edge[0] == token]

    # Sample a token based on the probabilities
    sampled_token = random.choices(
        [edge[1] for edge in filtered_graph],
        weights=[edge[2] for edge in filtered_graph],
        k=1,
    )[0]

    return sampled_token


if __name__ == "__main__":
    # Example usage
    text_corpus = load_in_text_corpus("text.txt")
    n = 2

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
    n_gram_graph = create_n_gram_graph(n_gram_frequency)

    # Create a dataframe from the n-gram graph
    n_gram_graph_df = pd.DataFrame(
        n_gram_graph, columns=["source_token", "target_token", "probability"]
    )

    # Save the dataframe to a CSV file
    n_gram_graph_df.to_csv("n_gram_graph.csv", index=False)

    token = "the"
    tokens = [token]

    while len(tokens) < 500:
        # Sample a token from the n-gram graph
        token = sample_n_gram_graph(token, n_gram_graph)

        # Append the token to the text
        tokens.append(token)
    print(detokenise(tokens))
