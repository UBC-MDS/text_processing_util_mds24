import pytest
import numpy as np
from text_processing_util_mds24.text_processing_util_mds24 import (
    text_clean, tfidf_vectorizer)

empty_list = []
list_empty_str = [""]
one_doc = ["This is a document with 1 string."]
list_invalid = [12, 33.5, None]
list_mixed_empty = ["Here is document one.", "", "we have document 2"]
list_num_punctuation = ["112.32", "!@#$", "795"]


# text_clean
def test_split_correct():
    test_str = ['hi!! wE are gRoup 30 . ',
                ' i at 12  bananas ']
    cleaned_doc = text_clean(test_str)
    assert len(cleaned_doc) == 2
    assert sorted(cleaned_doc[0]) == sorted(['hi', 'we', 'are', 'group'])

    assert sorted(cleaned_doc[1]) == sorted(['i', 'at', 'bananas'])

    return


def test_empty_list():
    assert text_clean(empty_list) == empty_list
    return


def test_list_empty_str():
    assert text_clean(list_empty_str) == [[]]
    return


def test_one_doc():
    cleaned_doc = text_clean(one_doc)
    assert len(cleaned_doc) == 1
    assert sorted(cleaned_doc[0]) \
           == sorted(['this', 'is', 'a', 'document', 'with', 'string'])

    return


def test_invalid_input():
    with pytest.raises(TypeError):
        text_clean(None)

    return


def test_invalid_doc():
    with pytest.raises(TypeError):
        text_clean(list_invalid)

    return


def test_mixed_empty():
    cleaned_doc = text_clean(list_mixed_empty)
    assert len(cleaned_doc) == 3
    assert sorted(cleaned_doc[0]) == sorted(['here', 'is', 'document', 'one'])
    assert cleaned_doc[1] == []
    assert sorted(cleaned_doc[2]) == sorted(['we', 'have', 'document'])

    return


def test_list_num_punctuation():
    cleaned_doc = text_clean(list_num_punctuation)
    assert cleaned_doc == [[], [], []]

    return


# frequency_vectorizer



# tfidf_vectorizer

def test_tfidf_vectorizer_single_doc():
    docs = ["python is a programming language"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 1
    assert len(feature_names) > 0

def test_tfidf_vectorizer_multiple_docs():
    docs = ["machine learning is interesting", "python is widely used in machine learning"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 2
    assert len(feature_names) > 0

def test_tfidf_vectorizer_negative_input():
    docs = ["negative words are not positive", "positive words are good"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 2
    assert len(feature_names) > 0

def test_tfidf_vectorizer_case_insensitive():
    docs = ["Python is a programming language", "python is widely used"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 2
    assert len(feature_names) > 0

def test_tfidf_vectorizer_empty_input():
    docs = []
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 0
    assert len(feature_names) == 0




# tokenizer_padding



