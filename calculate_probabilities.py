from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from tokeniser import tokenise, detokenise
from loader import load_in_text_corpus
from generate_text import generate_text

import random
import numpy as np


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


if __name__ == "__main__":
    # Example usage
    text_corpus = load_in_text_corpus("frankenstein.txt")
    n = 5

    # Calculate n-grams
    n_grams = calculate_n_grams(tokenise(text_corpus), n)
    print(f"# of {n}-grams: {len(n_grams):,}")

    # Calculate n-gram frequency
    n_gram_frequency = calculate_n_gram_frequency(text_corpus, n)
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

    gnerated_text = generate_text(
        initial_state=n_gram,
        n_gram_graph=n_gram_graph,
        text_token_length=100,
    )
    print(f"Generated text: {gnerated_text}")

    # Find stationary states using Markov chain analysis

    # # Build a list of all unique n-grams (states)
    # states = list(n_gram_graph.keys())
    # state_indices = {state: i for i, state in enumerate(states)}
    # N = len(states)

    # # Build the transition matrix
    # P = np.zeros((N, N))
    # for i, source in enumerate(states):
    #     for target, prob in n_gram_graph[source].items():
    #         if target in state_indices:
    #             j = state_indices[target]
    #             P[i, j] = prob

    # # Find the stationary distribution: solve πP = π, sum(π)=1
    # eigvals, eigvecs = np.linalg.eig(P.T)
    # # Find the eigenvector corresponding to eigenvalue 1
    # stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    # stationary = stationary[:, 0]
    # stationary = stationary / stationary.sum()

    # # Print the top 5 stationary states
    # print("Top 5 stationary states:")
    # top_indices = np.argsort(stationary)[::-1][:5]
    # for idx in top_indices:
    #     print(f"{states[idx]}: {stationary[idx]:.4f}")
