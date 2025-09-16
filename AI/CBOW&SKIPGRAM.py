# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 20:07:09 2025

@author: Kshitij
"""

from gensim.models import Word2Vec

# Sample training data
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "cat", "climbed", "the", "tree"]
]

# ✅ CBOW model (sg=0 → CBOW)
cbow_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0)

# ✅ Skip-gram model (sg=1 → Skip-gram)
skipgram_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# Example: get vector for a word
print("CBOW vector for 'cat':")
print(cbow_model.wv['cat'])

print("\nSkip-gram vector for 'cat':")
print(skipgram_model.wv['cat'])

# Find most similar words
print("\nCBOW - Most similar to 'cat':", cbow_model.wv.most_similar('cat'))
print("Skip-gram - Most similar to 'cat':", skipgram_model.wv.most_similar('cat'))