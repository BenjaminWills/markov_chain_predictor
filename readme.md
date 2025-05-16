# Markov chain text predictor

We can use mathematical objects called markov chains to predict the next word in a sentance.

## How does this work?

A markov chain is a collection of `states` connected by edges that denote the probability of going from state `i` to state `j`.

![image](markov_visual.png)

The image below shows an example of this in the context of text, this chain ranks the probability of the next token.

So the way that the text predictor works is by choosing an initial state, for example "the" and then randomly choosing (based on the connected edges) which token to choose next, then by repeating this process you do a `random walk` along the chain to create a chunk of text. Given enough text this should begin to make sense.

## N-grams

Of course in text we can predict one word at a time, i.e. looking at the probability that I say one word after another. But sometimes it is more helpful to look at chunks of words, these are called n-grams. Consider the following text

> The big red cat.

A 1-gram representation of this is:

```python
["The", "big", "red", "cat", "."]
```

A 2-gram representation of this is:

```python
[["The", "big"], ["big", "red"], ["red", "cat"], ["cat", "."]]
```

A 3-gram representation of this is:

```python
[["The", "big", "red"], ["big", "red", "cat"], ["red", "cat", "."]]
```

So an n-gram can represent the combinations of words in a sentance, and the number of n-grams will be the length of the text in tokens - n + 1.

## Using N-grams to predict accurately

As you can probably imagine, predicting sequences of words may be more useful to understanding the grammar of English than predicting just singular words. So we can define a state as being an n-gram of the data, and the transitions are from one n-gram to another. So when looking in the data, we look for n-grams that directly follow one another, for example in the above example we'd count the number of times "The big red" preceeded "big red cat" and calculate probabilies.

# Tokenisation

The tokeniser in this repo is rudimentary, it splits based on words and punctuation.

```py
"My name is Ben!" -> ["My", "name", "is", "Ben", "!"]
```

# Corpus generation

I used Project Gutenburg to collate a text corpus, see `scrape_books.py` to see how their `books` api was used to save an arbritrary amount of books.