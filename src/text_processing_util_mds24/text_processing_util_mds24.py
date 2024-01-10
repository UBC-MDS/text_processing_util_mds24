def text_clean(docs: list[str]) -> list[str]:
    # remove punctuations
    # lowercase everything
    # remove numbers


def frequency_vectorizer(docs: list[str]) -> list[dict[str:float]]:
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


