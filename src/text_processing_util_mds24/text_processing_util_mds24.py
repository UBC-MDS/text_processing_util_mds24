import string
import numpy as np


def text_clean(docs: list[str]) -> list[list[str]]:
    """Removes punctuation, turns all characters in each document lower case,
       removes numbers in documents, and splits each document into a list of words.

    Parameters
    ----------
    docs : list[str]
        Documents to be processed.
        Each item item in the list is a document.

    Returns
    -------
    list[list[str]]
        Cleaned documents.

    Examples
    --------
    >>> text_clean(["We are group 10.",
                    "We are the best!"])
    [["we", "are", "group"], ["we", "are", "the", "best"]]
    """

    if not isinstance(docs, list):
        raise TypeError('Input is not a list')
    for doc_i, doc in enumerate(docs):
        if not isinstance(doc, str):
            raise TypeError(f'Document {doc_i} is not string')

    cleaned_docs: list[str] = []

    for _ in docs:
        cleaned_docs.append('')

    # remove punctuation and number and lower case everything
    for doc_i, doc in enumerate(docs):
        for ch in doc:
            if not ch.isnumeric() and ch not in string.punctuation:
                cleaned_docs[doc_i] += ch.lower()

    out_docs: list[list[str]] = []
    # split by space
    for doc in cleaned_docs:
        out_docs.append([wrd for wrd in doc.split(' ') if wrd != ''])

    return out_docs


def frequency_vectorizer(docs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the frequency of each word in a list of text documents.

    Parameters
    ----------
    docs : list[str]
        A list of text documents.

    Returns
    -------
    np.ndarray of float
        A 2D array, where each dictionary contains each word
        and its frequency in a text document.
    np.ndarray of str
        Feature names

    Examples
    --------
    >>> documents = ["This is a sample document.", "Another document for testing."]
    >>> result = frequency_vectorizer(documents)
    >>> print(result)
    [{'This': 0.2, 'is': 0.2, 'a': 0.2, 'sample': 0.2, 'document.': 0.2},
    {'Another': 0.25, 'document': 0.25, 'for': 0.25, 'testing.': 0.25}]
    """
    pass


def tfidf_vectorizer(docs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates TF-IDF scores for a list of documents.

    Parameters
    ----------
    docs : list[str]: 
        List of documents (strings)

    Returns
    -------
    np.ndarray of float
        2D array containing TF-IDF scores for each term in each document.
    np.ndarray of str
        Feature names

    Examples
    --------
    >>> tfidf_vectorizer(["This is the first document."])
    [{'This': -0.13862943611198905, 'is': -0.13862943611198905, 
    'the': -0.13862943611198905, 'first': -0.13862943611198905, '
    'document.': -0.13862943611198905}]
    """
    pass


def tokenizer_padding(docs: list[str]) -> np.ndarray:
    """
    Converts each text document into a list of numerical tokens, and pads
    shorter sequences so that each tokenized document has the same length.

    Parameters
    ----------
    docs : list[str]
        A list of text documents.

    Returns
    -------
    np.ndarray
        2D array of tokenized and padded sequences of the input documents.

    Examples
    --------
    >>> tokenizer_padding(["the first sentence", "the second longer sentence"])
    [[1, 2, 3, 0], [1, 4, 5, 3]]
    >>> tokenizer_padding(["a sample text", "sample text two"])
    [[1, 2, 3], [2, 3, 4]]
    """
    cleaned = text_clean(docs)
    max_len = max([len(doc) for doc in cleaned])
    mapper = {}
    max_token = 1
    ret_array = np.zeros((len(cleaned), max_len))

    for i in range(len(cleaned)):
        for j in range(len(cleaned[i])):
            if cleaned[i][j] not in mapper:
                mapper[cleaned[i][j]] = max_token
                max_token += 1
            ret_array[i,j] = mapper[cleaned[i][j]]

    return ret_array
