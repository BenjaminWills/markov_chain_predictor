from typing import List, Tuple, Dict
import random

from tokeniser import detokenise


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


def generate_text(
    initial_state: Tuple[str],
    n_gram_graph: Dict[str, Dict[Tuple[str], float]],
    text_token_length: int = 100,
) -> str:
    """Generate text using the n-gram graph. This is a markov process where the
    next token is sampled from the n-gram graph given the current token.
    The process continues until the desired length is reached.

    Parameters
    ----------
    initial_state : Tuple[str]
        The initial state of the n-gram graph, this will be a tuple of n words that you
        would like to start the generation with.
    n_gram_graph : Dict[str, Dict[Tuple[str], float]]
        The n-gram graph, a dictionary where the keys are n-grams and the values are dictionaries
        of n-grams and their probabilities of being the next state given the current state.
    text_token_length : int, optional
        The length of the generated length in tokens, by default 100

    Returns
    -------
    str
        The generated text.
    """
    # Initialise the process
    token = initial_state

    # Track the number of n-grams generated
    n_grams = [token]

    # Generate n-grams until the desired length is reached
    while len(n_grams) < text_token_length:
        # Sample the next n-gram
        token = sample_n_gram_graph(token, n_gram_graph)

        # Append the sampled n-gram to the list
        n_grams.append(token)

    # Recombine the n-grams into a string
    recombined_string = recombine_n_grams(n_grams)
    return recombined_string
