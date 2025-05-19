import networkx as nx
import matplotlib.pyplot as plt

from calculate_probabilities import create_n_gram_graph
from loader import load_in_text_corpus
from tokeniser import tokenise

text_path = "corpus.txt"  # Path to your text file
# Load the text corpus
text_corpus = load_in_text_corpus(text_path)

# Tokenise the text corpus
tokens = tokenise(text_corpus)

# Create the n-gram graph
n = 2  # Size of n-grams
n_gram_graph = create_n_gram_graph(text_corpus, n)

# Turn this graph into a networkx graph
G = nx.DiGraph()
for source, targets in n_gram_graph.items():
    for target, prob in targets.items():
        G.add_edge(source, target, weight=prob)

print(f"Number of nodes: {G.number_of_nodes():,}")
print(f"Number of edges: {G.number_of_edges():,}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Density: {nx.density(G)}")
print(f"Average clustering coefficient: {nx.average_clustering(G)}")
print(
    f"Number of weakly connected components: {nx.number_weakly_connected_components(G)}"
)

# Visualise the degree distribution, with a histogram coupled with a boxplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

degrees = [degree for _, degree in G.degree()]

ax1.hist(
    degrees,
    bins=2000,
    alpha=0.5,
    color="blue",
    label="Degree Distribution",
    density=True,
)
# ax1.set_title(f"Degree Histogram for {n}-grams in Frankenstein")

ax2.boxplot(
    degrees,
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="orange", color="orange"),
    showfliers=False,
)
ax2.set_title("Degree Boxplot")
ax2.set_xlabel("Degree")
ax1.set_ylabel("Frequency density")

plt.tight_layout()
plt.xlim(0, 30)
plt.savefig(f"{n}_gram_frankenstein_degree_distribution.png")
plt.show()
