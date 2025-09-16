# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 10:13:33 2025

@author: Kshitij
"""

import nltk
from nltk import pos_tag, word_tokenize, RegexpParser

# Download necessary NLTK data files (only need to do this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text
tokens = word_tokenize(text)

# Perform part-of-speech tagging
tagged_tokens = pos_tag(tokens)

# Define a chunk grammar
chunk_grammar = r"""
  NP: {<DT>?<JJ>*<NN>}   # Noun Phrase
  VP: {<VB.*><NP|PP>*}    # Verb Phrase
  PP: {<IN><NP>}          # Prepositional Phrase
"""

# Create a chunk parser
chunk_parser = RegexpParser(chunk_grammar)

# Parse the tagged tokens
chunked = chunk_parser.parse(tagged_tokens)

# Print the chunked output
print(chunked)

# Optionally, you can visualize the chunks
chunked.draw()
