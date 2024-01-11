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
        A list of dictionaries, where each dictionary contains the word and the word's frequency in a text document.
        
    Examples
    --------
    >>> documents = ["This is a sample document.", "Another document for testing."]
    >>> result = frequency_vectorizer(documents)
    >>> print(result)
    [{'This': 0.14, 'is': 0.14, 'a': 0.14, 'sample': 0.14, 'document.': 0.14},
    {'Another': 0.16, 'document': 0.16, 'for': 0.16, 'testing.': 0.16}]
    """
    pass


def tfidf_vectorizer(docs: list[str]) -> list[dict[str:float]]:
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


