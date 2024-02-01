---
editor_options: 
  markdown: 
    wrap: 72
---

# text_processing_util_mds24

[![Documentation
Status](https://readthedocs.org/projects/text-processing-util-mds24/badge/?version=latest)](https://text-processing-util-mds24.readthedocs.io/en/latest/?badge=latest)

Welcome to the repository for text processing, a part of the DSCI-524
course by Group 10 in the MDS-V Cohort 8 at UBC.

Empower your text analysis workflows with text processing package, a
Python library designed for streamlined text processing tasks. This
versatile package offers four key functions: `text_clean` for noise
removal and text refinement, `frequency_vectorizer` to generate
frequency-based vectors, `tfidf_vectorizer` for TF-IDF vectorization,
and `tokenizer_padding` to assist in tokenization and padding of text
sequences for usage in recurrent neural networks. By simplifying
essential text preprocessing steps, this package facilitates efficient
text-based analysis, providing an easy-to-use toolkit for natural
language processing and text modeling endeavors.

## Contributors

Our team, in alphabetical order:

-   **Allan Lee**
-   **Jerry Yu**
-   **Mo Norouzi**
-   **Nasim Ghazanfari Nasrabadi**

## Developer Notes

**Note:** For those who are looking to develop and/or test this package,
please follow the following instructions to install the package from
this GitHub repository.

### Installation

1.  First, please make sure that you have `poetry` and `conda` installed
    on your local computer. If not, please follow the official
    instructions for each respectively to install them.
    ([`poetry`](https://python-poetry.org/docs/),
    [`conda`](https://docs.conda.io/projects/miniconda/en/latest/))

2.  It is recommended to create a conda virtual environment to install
    the package by running the following commands:

``` bash
conda create --name text_processing_util_mds24 python=3.9 -y
conda activate text_processing_util_mds24
```

3.  Clone the repository to your local machine by running:

``` bash
git clone git@github.com:UBC-MDS/text_processing_util_mds24.git
```

4.  From the root of this repository, install the package using `poetry`
    by running the following command. It is recommended to use
    `poetry install` command inside your conda environment.

``` bash
poetry install
```


The package is also available for installation via:

```         
pip install text_processing_util_mds24
```

### Testing

**Note:** Every function in this package except for `text_clean` calls
`text_clean` in the first line of the code. Hence, testing for errors
that arise from unexpected inputs is only done for `text_clean`.
Integration testing is done for the other functions that call
`text_clean`.

To test this package, please first make sure you have activated the
`text_processing_util_mds24` conda environment that was created in the
previous section. Then, please run the following command from the root
directory of the repository:

``` bash
pytest tests/
```

If you would like to see line coverage, please run the following command
from the root directory of the repository:

``` bash
pytest --cov=text_processing_util_mds24
```

If you would like to see branch coverage, please run the following
command from the root directory of the repository:

``` bash
pytest --cov-branch --cov=text_processing_util_mds24
```

## Functions

1.  `text_clean`: Removes punctuation, turns all characters in each
    document lower case and removes numbers in documents.
2.  `frequency_vectorizer`: Calculates the frequency of each word in a
    list of text documents.
3.  `tfidf_vectorizer`: Calculates the Term Frequency-Inverse Document
    Frequency (TF-IDF) scores for a given list of documents, providing a
    numerical representation that highlights the importance of terms
    within the context of the entire document set.
4.  `tokenizer_padding`: Converts each word into an individual token
    represented by a number and pads shorter sequences, but keeps the
    order of the original sentence, which is important for RNNs.

## Usage

Here are some examples of usage of the functions in this package.

Example of using `text_clean`:

``` python
from text_processing_util_mds24 import (
    text_clean,
    tfidf_vectorizer,
    frequency_vectorizer,
    tokenizer_padding
)
docs = ["Here is document one.", "", "we have document 2"]
print(text_clean(docs))
```

``` text
[['here', 'is', 'document', 'one'], [], ['we', 'have', 'document']]
```

Example of using `frequency_vectorizer`:

``` python
docs = ["apple orange banana", "apple banana banana"]
result_tf_matrix, result_feature_names = frequency_vectorizer(docs)
print(result_tf_matrix)
print(result_feature_names)
```

``` text
[[0.33333333 0.33333333 0.33333333]
 [0.33333333 0.66666667 0.        ]]
['apple', 'banana', 'orange']
```

Example of using `tfidf_vectorizer`:

``` python
docs = ["machine learning is interesting", "machine learning is fascinating"]
tfidf_matrix, feature_names = tfidf_vectorizer(docs)
print(tfidf_matrix)
print(feature_names)
```

``` text
[[ 0.          0.         -0.10136628 -0.10136628 -0.10136628]
 [ 0.          0.         -0.10136628 -0.10136628 -0.10136628]]
['fascinating', 'interesting', 'is', 'learning', 'machine']
```

Example of using `tokenizer_padding`:

``` python
docs = ["one two three", "one three four five"]
tokenized_padded = tokenizer_padding(docs)
print(tokenized_padded)
```

``` text
[[1. 2. 3. 0.]
 [1. 3. 4. 5.]]
```

## Documentation

The official documentation for this package is hosted on Read the Docs:
<https://text-processing-util-mds24.readthedocs.io/en/latest/>.

## Ecosystem

This package is intended to clean and transform texts into different
representations to feed into machine learning algorithms. Scikit-learn
and Keras provide similar functionalities.

`frequency_vectorizer`:
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>

`tfidf_vectorizer`:
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>

`tokenizer_padding`:

-   <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer>
-   <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences>

Still, our functions are different in some ways. `frequency_vectorizer`
calculates the relative frequency of each word with regards to the total
number of words in its document rather than giving the raw counts.
`tokenizer_padding` combines what would be an otherwise two-step process
if one were to use Keras into one step. Both `tfidf_vectorizer` and
`tokenizer_padding` offer simpler functional APIs and implementations
compared to the implementations from scikit-learn and Keras
respectively.

## Contributing

Interested in contributing? Check out the contributing guidelines.
Please note that this project is released with a Code of Conduct. By
contributing to this project, you agree to abide by its terms.

## License

`text_processing_util_mds24` was created by Jerry Yu, Nasim Ghazanfari
Nasrabadi, Mohammad Norouzi, Allan Lee. It is licensed under the terms
of the MIT license.

## Credits

`text_processing_util_mds24` was created with
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the
`py-pkgs-cookiecutter`
[template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
