import string
import numpy as np
from collections import Counter



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


def tfidf_vectorizer(docs):
    """
    Calculate TF-IDF scores for a list of documents.

    Parameters
    ----------
    docs : List[str]
        List of documents (strings).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing two elements:
            - A 2D array containing TF-IDF scores for each term in each document.
            - An array of feature names corresponding to the columns in the TF-IDF matrix.

    Examples
    --------
    >>> calculate_tfidf(["Machine learning is interesting", "Python is widely used in machine learning"])
    (array([[0.        , 0.43550663, 0.43550663, 0.43550663, 0.43550663, 0.43550663],
           [0.57735027, 0.        , 0.        , 0.        , 0.        , 0.        ]]),
     array(['in', 'interesting', 'is', 'learning', 'machine', 'python'], dtype='<U11'))
    """

    
    # Clean the documents
    cleaned_docs = text_clean(docs)
    
    # Calculate term frequency (TF)
    tf = [{term: count / len(doc) for term, count in Counter(doc).items()} for doc in cleaned_docs]
    
    # Calculate document frequency (DF)
    df = Counter()
    
    for doc in cleaned_docs:
        df.update(set(doc))
    
    # Calculate inverse document frequency (IDF)
    idf = {term: np.log(len(docs) / (df[term] + 1)) for term in df}
    
    # Calculate TF-IDF
    tfidf_matrix = np.zeros((len(docs), len(idf)))
    feature_names = sorted(idf.keys())
    
    for i, doc in enumerate(cleaned_docs):
        for j, term in enumerate(feature_names):
            tfidf_matrix[i, j] = tf[i].get(term, 0) * idf[term]
    
    return tfidf_matrix, feature_names


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
    pass
docs_1= ["apple orange banana", "apple banana banana"]
tfidf_vectorizer(docs_1)