import argparse
from loader import load_in_text_corpus
from generate_text import generate_text
from tokeniser import tokenise
from calculate_probabilities import (
    calculate_n_grams,
    calculate_n_gram_frequency,
    calculate_n_gram_probabilities,
    create_n_gram_graph,
)

import random


def main(n, text_path):
    # Example usage
    text_corpus = load_in_text_corpus(text_path)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markov Chain N-gram Text Predictor")
    parser.add_argument(
        "--n", type=int, help="n in n-grams (e.g. 2 for bigrams, 3 for trigrams)"
    )
    parser.add_argument("--text_path", type=str, help="Path to the text source file")
    args = parser.parse_args()
    main(args.n, args.text_path)
