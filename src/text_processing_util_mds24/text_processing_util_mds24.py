def text_clean(docs: list[str]) -> list[str]:
    # remove punctuations
    # lowercase everything
    # remove numbers


def frequency_vectorizer(docs: list[str]) -> list[dict[str:float]]:
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
    pass
