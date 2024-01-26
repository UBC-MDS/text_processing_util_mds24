import string
import numpy as np
from collections import Counter


def text_clean(docs: list[str]) -> list[list[str]]:
    """Removes punctuation, turns all characters in each document to lower case, \
       removes numbers in documents, and splits each document into a list of words.

    Parameters
    ----------
    docs : list[str]
        Documents to be processed.
        Each item in the list is a document.

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
    tuple[np.ndarray, np.ndarray]
        Tuple containing two elements:
            - A 2D array containing frequency scores for each term in each document.
            - An array of feature names corresponding to the columns in the frequency matrix.

    Examples
    --------
    >>> docs = ["This is a sample document.", "Another document for testing."]
    >>> result_tf_matrix, result_feature_names = frequency_vectorizer(documents)
    >>> print("Frequency Matrix:")
    >>> print(result_tf_matrix)
    Frequency Matrix:
    [[0.2  0.   0.2  0.   0.2  0.2  0.   0.2 ]
    [0.   0.25 0.25 0.25 0.   0.   0.25 0.  ]]
    >>> print("Feature Names:")
    >>> print(result_feature_names)
    Feature Names:
    ['a', 'another', 'document', 'for', 'is', 'sample', 'testing', 'this']
    """

    cleaned_docs = text_clean(docs)

    if len(docs) == 0:
        return np.array([]), np.array([])

    # Calculate frequency 
    tf_matrix = np.zeros((len(docs), len(set(term for doc in cleaned_docs for term in doc))))
    feature_names = sorted(set(term for doc in cleaned_docs for term in doc))

    for i, doc in enumerate(cleaned_docs):
        term_count = Counter(doc)
        total_terms = len(doc)

        for j, term in enumerate(feature_names):
            if total_terms == 0:
                tf_matrix[i, j] = 0
            else:
                tf_matrix[i, j] = term_count.get(term, 0) / total_terms

    return tf_matrix, feature_names


def tfidf_vectorizer(docs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates TF-IDF scores for a list of documents. The TF-IDF score measures \
    the importance of a word to its document, adjusted for the word's overall \
    frequency in all documents.

    Parameters
    ----------
    docs : list[str]
        A list of documents (strings).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing two elements:
            - A 2D array containing TF-IDF scores for each term in each document.
            - An array of feature names corresponding to the columns in the TF-IDF matrix.

    Examples
    --------
    >>> docs = ["Machine learning is interesting", "Python is widely used in machine learning"]
    >>> tdifd_matrix, feature_names = tfidf_vectorizer(docs)
    >>> print("TFIDF Matrix:")
    >>> print(tdifd_matrix)
    [[0.        , 0.43550663, 0.43550663, 0.43550663, 0.43550663, 0.43550663]
    [0.57735027, 0.        , 0.        , 0.        , 0.        , 0.        ]]
    >>> print(Feature Names:)
    >>> print(feature_names)
    ['in', 'interesting', 'is', 'learning', 'machine', 'python']
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
    Converts each text document into a list of numerical tokens, which are \
    numerical identifiers for each word, and pads shorter sequences so that \
    each tokenized document has the same length. These steps make it possible \
    for the transformed data to be accepted by deep learning libraries for \
    building recurrent neural networks.

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
    >>> tokenized_padded = tokenizer_padding(["the first sentence", "the second longer sentence"])
    >>> print(tokenized_padded)
    [[1, 2, 3, 0], [1, 4, 5, 3]]
    >>> tokenized_padded = tokenizer_padding(["a sample text", "sample text two"])
    >>> print(tokenized_padded)
    [[1, 2, 3], [2, 3, 4]]
    """
    cleaned = text_clean(docs)

    if len(docs) == 0:
        return np.array([])

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

