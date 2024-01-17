import pytest
import numpy as np
from text_processing_util_mds24.text_processing_util_mds24 import (
    text_clean,
    tokenizer_padding)

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



# tokenizer_padding
def test_tokenizer_empty_list():
    tokenized_padded = tokenizer_padding(empty_list)
    assert tokenized_padded == np.array([])

def test_tokenizer_list_empty_str():
    tokenized_padded = tokenizer_padding(list_empty_str)
    assert tokenized_padded == np.array([[]])

def test_tokenizer_one_doc():
    tokenized_padded = tokenizer_padding(one_doc)
    assert tokenized_padded == np.array([[1, 2, 3, 4, 5, 6]])

def test_tokenizer_invalid():
    with pytest.raises(TypeError):
        tokenizer_padding(123)

def test_tokenizer_list_invalid():
    with pytest.raises(TypeError):
        tokenizer_padding(list_invalid)

def test_tokenizer_list_mixed_empty():
    tokenized_padded = tokenizer_padding(list_mixed_empty)
    assert tokenized_padded == np.array([[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 3, 0]])

def test_tokenizer_list_num_punctuation():
    tokenized_padded = tokenizer_padding(list_num_punctuation)
    assert tokenized_padded == np.array([[], [], []])

def test_tokenizer_equal_lists():
    equal_lists = ["My first document 1.", "My first 5 documents!", "the Great doc"]
    tokenized_padded = tokenizer_padding(equal_lists)
    assert tokenized_padded == np.array([[1, 2, 3], [1, 2, 4], [5, 6, 7]])

def test_tokenizer_unequal_lists():
    unequal_lists = ["This is by far the longest doc in this list of docs.",
                     "I am the shortest",
                     "Is last doc the second longest?"]
    tokenized_padded = tokenizer_padding(unequal_lists)
    assert tokenized_padded == np.array([[1, 2, 3, 4, 5, 6, 7, 8, 1, 9, 10, 11],
                                         [12, 13, 5, 14, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [15, 16, 7, 5, 17, 6, 0, 0, 0, 0, 0, 0]])

def test_tokenizer_repeats():
    repeats = ["one two three", "one three four five"]
    tokenized_padded = tokenizer_padding(repeats)
    assert tokenized_padded == np.array([[1, 2, 3, 0], [1, 3, 4, 5]])

def test_tokenizer_no_repeats():
    no_repeats = ["one two three", "four five six"]
    tokenized_padded = tokenizer_padding(no_repeats)
    assert tokenized_padded == np.array([[1, 2, 3], [4, 5, 6]])


