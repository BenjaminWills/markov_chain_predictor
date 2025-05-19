from loader import load_in_text_corpus
from generate_text import generate_text
from tokeniser import tokenise, detokenise
from calculate_probabilities import (
    calculate_n_grams,
    calculate_n_gram_frequency,
    calculate_n_gram_probabilities,
    create_n_gram_graph,
)

import random

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
