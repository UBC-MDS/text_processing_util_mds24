# text_processing_util_mds24

This package provides functions to preprocess text documents for machine learning algorithms.

# functions
#### text_clean(): 
Removes punctuation, makes everything lower case and removes numbers from documents.
#### frequency_vectorizer():
Converts the document to word occurrence frequency representation.
#### tfidf_vectorizer()
Converts the document to TF-IDF representation.
#### tokenizer_padding()
?????WRITE?????


# Ecosystem
This package is intended to clean and transform texts into different representations to feed into machine learning algorithms.
Scikit-learn provides similar functionalities.

count vectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

TF-IDF vectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html




## Installation

```bash
$ pip install text_processing_util_mds24
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`text_processing_util_mds24` was created by Jerry Yu, Nasim Ghazanfari Nasrabadi, Mohammad Norouzi, Allan Lee. It is licensed under the terms of the MIT license.

## Credits

`text_processing_util_mds24` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
