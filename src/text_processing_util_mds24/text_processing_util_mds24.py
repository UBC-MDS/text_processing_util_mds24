def text_clean(docs: list[str]) -> list[str]:
    # remove punctuations
    # lowercase everything
    # remove numbers


def frequency_vectorizer(docs: list[str]) -> list[dict[str:float]]:
    """
    Calculates the frequency of each word in a list of text documents.

    Parameters
    ----------
    docs : list[str]
        A list of text documents.

    Returns
    -------
    list[dict[str: float]]
        A list of dictionaries, where each dictionary contains each word and its frequency in a text document.
        
    Examples
    --------
    >>> documents = ["This is a sample document.", "Another document for testing."]
    >>> result = frequency_vectorizer(documents)
    >>> print(result)
    [{'This': 0.2, 'is': 0.2, 'a': 0.2, 'sample': 0.2, 'document.': 0.2},
    {'Another': 0.25, 'document': 0.25, 'for': 0.25, 'testing.': 0.25}]
    """
    pass


def tfidf_vectorizer(docs: list[str]) -> list[dict[str:float]]:

    """
    Calculate TF-IDF scores for a list of documents.

    Parameters
    ----------
    docs : list[str]: 
        List of documents (strings)

    Returns
    -------
    list[dict[str:float]]
        List of dictionaries containing TF-IDF scores for each term in each document.

    Examples
    --------
    >>> tfidf_vectorizer(["This is the first document."])
    
    [{'This': -0.13862943611198905, 'is': -0.13862943611198905, 
    'the': -0.13862943611198905, 'first': -0.13862943611198905, '
    'document.': -0.13862943611198905}]
    """
    pass


def tokenizer_padding(docs: list[str]) -> list[list[int]]:
    """
    Converts each text document into a list of numerical tokens, and pads
    shorter sequences so that each tokenized document has the same length.

    Parameters
    ----------
    docs : list[str]
        A list of text documents.

    Returns
    -------
    list[list[int]]
        A list of tokenized and padded sequences of the input documents.

    Examples
    --------
    >>> tokenizer_padding(["the first sentence", "the second longer sentence"])
    [[1, 2, 3, 0], [1, 4, 5, 3]]
    >>> tokenizer_padding(["a sample text", "sample text two"])
    [[1, 2, 3], [2, 3, 4]]
    """
    pass
