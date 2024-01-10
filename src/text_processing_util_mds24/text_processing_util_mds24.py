def text_clean(docs: list[str]) -> list[str]:
    """Removes punctuation, make everything lower case and remove numbers
       in documents.

    Parameters
    ----------
    docs : list[str]
        Documents to be processed.
        Each item item in the list is a document.

    Returns
    -------
    list[str]
        Cleaned documents.

    Examples
    --------
    >>> text_clean(["we are group 10",
                    "we are the best"])
    """
    pass


def frequency_vectorizer(docs: list[str]) -> list[dict[str:float]]:
    pass


def tfidf_vectorizer(docs: list[str]) -> list[dict[str:float]]:
    pass


def tokenizer_padding(docs: list[str]) -> list[list[int]]:
    pass


