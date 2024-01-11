# text_processing_util_mds24
Welcome to the repository for text processing, a part of the DSCI-524 course by Group 10 in the MDS-V Cohort 8 at UBC.

Empower your text analysis workflows with text processing package, a Python library designed for streamlined text processing tasks. This versatile package offers four key functions: text_clean for noise removal and text refinement, frequency_vectorizer to generate frequency-based vectors, tfidf_vectorizer for TF-IDF vectorization, and tokenizer_padding to assist in tokenization and padding of text sequences. By simplifying essential text preprocessing steps, this package facilitates efficient text-based analysis, providing an easy-to-use toolkit for natural language processing and text modeling endeavors.

# Contributors
Our team, in alphabetical order:

- **Allan Lee**
- **Jerry Yu**
- **Mo Norouzi**
- **Nasim Ghazanfari Nasrabadi**

This package provides functions to preprocess text documents for machine learning algorithms.

# Functions
1.  `text_clean`: 
Removes punctuation, make everything lower case and remove numbers in documents.
2.  `frequency_vectorizer`:
Calculates the frequency of each word in a list of text documents.
3.  `tfidf_vectorizer`:
Calculate TF-IDF scores for a list of documents.
4.  `tokenizer_padding`:
Converts each text document into a list of numerical tokens, and pads shorter sequences so that each tokenized document has the same length.

# Ecosystem
This package is intended to clean and transform texts into different representations to feed into machine learning algorithms.
Scikit-learn provides similar functionalities.

Frequency vectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

TF-IDF vectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html


Tokenizer + padding:
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
- https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences

# Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

# License

`text_processing_util_mds24` was created by Jerry Yu, Nasim Ghazanfari Nasrabadi, Mohammad Norouzi, Allan Lee. It is licensed under the terms of the MIT license.

# Credits

`text_processing_util_mds24` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
