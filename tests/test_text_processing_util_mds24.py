import pytest
import numpy as np
from text_processing_util_mds24.text_processing_util_mds24 import (
    text_clean,
    tfidf_vectorizer,
    tokenizer_padding,
    frequency_vectorizer)


# Some common test data for edge cases and errors
empty_list = []
list_empty_str = [""]
one_doc = ["This is a document with 1 string."]
list_invalid = [12, 33.5, None]
list_mixed_empty = ["Here is document one.", "", "we have document 2"]
list_num_punctuation = ["112.32", "!@#$", "795"]


# Tests for text_clean
# Testing for correctly cleaning text for text_clean
def test_text_clean_split_correct():
    test_str = ['hi!! wE are gRoup 30 . ',
                ' i at 12  bananas ']
    cleaned_doc = text_clean(test_str)
    assert len(cleaned_doc) == 2
    assert sorted(cleaned_doc[0]) == sorted(['hi', 'we', 'are', 'group'])

    assert sorted(cleaned_doc[1]) == sorted(['i', 'at', 'bananas'])


# Testing an empty list for text_clean
def test_text_clean_empty_list():
    assert text_clean(empty_list) == empty_list


# Testing a list of one empty string for text_clean
def test_text_clean_list_empty_str():
    assert text_clean(list_empty_str) == [[]]


# Testing a list with one document for text_clean
def test_text_clean_one_doc():
    cleaned_doc = text_clean(one_doc)
    assert len(cleaned_doc) == 1
    assert sorted(cleaned_doc[0]) \
           == sorted(['this', 'is', 'a', 'document', 'with', 'string'])


# Testing an invalid input for text_clean
def test_text_clean_invalid_input():
    with pytest.raises(TypeError):
        text_clean(None)


# Testing a list with invalid input for text_clean
def test_text_clean_invalid_doc():
    with pytest.raises(TypeError):
        text_clean(list_invalid)


# Testing a list with one document being empty string for text_clean
def test_text_clean_mixed_empty():
    cleaned_doc = text_clean(list_mixed_empty)
    assert len(cleaned_doc) == 3
    assert sorted(cleaned_doc[0]) == sorted(['here', 'is', 'document', 'one'])
    assert cleaned_doc[1] == []
    assert sorted(cleaned_doc[2]) == sorted(['we', 'have', 'document'])


# Testing a list with only numbers and punctuations for text_clean
def test_text_clean_list_num_punctuation():
    cleaned_doc = text_clean(list_num_punctuation)
    assert cleaned_doc == [[], [], []]


# Tests for frequency_vectorizer
# Testing an empty list for frequency_vectorizer
def test_frequency_vectorizer_empty_docs():
    result_tf_matrix, result_feature_names = frequency_vectorizer(empty_list)

    assert np.array_equal(result_tf_matrix, np.array([]))
    assert np.array_equal(result_feature_names, np.array([]))


# Testing a list with one document for frequency_vectorizer
def test_frequency_vectorizer_single_doc():
    result_tf_matrix, result_feature_names = frequency_vectorizer(one_doc)

    expected_matrix = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]])
    expected_feature_names = np.array(['a', 'document', 'is', 'string', 'this', 'with'])

    np.testing.assert_array_almost_equal(result_tf_matrix, expected_matrix)
    np.testing.assert_array_equal(result_feature_names, expected_feature_names)


# Testing a list of one empty string for frequency_vectorizer
def test_frequency_vectorizer_list_empty_str():
    result_tf_matrix, result_feature_names = frequency_vectorizer(list_empty_str)

    np.testing.assert_array_equal(result_tf_matrix, np.empty((1,0)))
    np.testing.assert_array_equal(result_feature_names, np.array([]))


# Testing a list with only numbers and punctuations for frequency_vectorizer
def test_frequency_vectorizer_list_num_punctuation():
    result_tf_matrix, result_feature_names = frequency_vectorizer(list_num_punctuation)

    np.testing.assert_array_equal(result_tf_matrix, np.empty((3,0)))
    np.testing.assert_array_equal(result_feature_names, np.array([]))


# Testing a list with one document being empty string for frequency_vectorizer
def test_frequency_vectorizer_list_mixed_empty():
    result_tf_matrix, result_feature_names = frequency_vectorizer(list_mixed_empty)

    expected_matrix = np.array([[0.25, 0, 0.25, 0.25, 0.25, 0],
                                [0, 0, 0, 0, 0, 0],
                                [1/3, 1/3, 0, 0, 0, 1/3]])
    expected_feature_names = np.array(['document', 'have', 'here', 'is', 'one', 'we'])

    np.testing.assert_array_almost_equal(result_tf_matrix, expected_matrix)
    np.testing.assert_array_equal(result_feature_names, expected_feature_names)


# Testing multiple documents for frequency_vectorizer
def test_frequency_vectorizer_multiple_docs():
    docs = ["This is a sample document.", "Another document for testing."]
    result_tf_matrix, result_feature_names = frequency_vectorizer(docs)

    expected_matrix = np.array([[0.2, 0.,   0.2 , 0. ,  0.2 , 0.2 , 0. ,  0.2 ], [0. ,  0.25 ,0.25 ,0.25, 0. ,  0.,   0.25, 0.]])
    expected_feature_names = np.array(['a', 'another', 'document', 'for', 'is', 'sample', 'testing', 'this'])

    np.testing.assert_array_almost_equal(result_tf_matrix, expected_matrix)
    np.testing.assert_array_equal(result_feature_names, expected_feature_names)


# Testing an additional case of multiple documents for frequency_vectorizer
def test_frequency_vectorizer_additional_case():
    docs = ["apple orange banana", "apple banana banana"]
    result_tf_matrix, result_feature_names = frequency_vectorizer(docs)

    expected_matrix = np.array([[1/3, 1/3, 1/3], [1/3, 2/3, 0]])
    expected_feature_names = np.array(['apple', 'banana', 'orange'])

    np.testing.assert_array_almost_equal(result_tf_matrix, expected_matrix)
    np.testing.assert_array_equal(result_feature_names, expected_feature_names)


# Tests for tfidf_vectorizer
# Testing an empty list for tfidf_vectorizer
def test_tfidf_vectorizer_empty_list():
    tfidf_matrix, feature_names = tfidf_vectorizer(empty_list)
    assert tfidf_matrix.shape == (0, 0)
    assert len(feature_names) == 0


# Testing a list of one empty string for tfidf_vectorizer
def test_tfidf_vectorizer_list_empty_str():
    tfidf_matrix, feature_names = tfidf_vectorizer(list_empty_str)
    np.testing.assert_array_equal(tfidf_matrix, np.empty((1,0)))
    np.testing.assert_array_equal(feature_names, np.array([]))


# Testing a list with one document for tfidf_vectorizer
def test_tfidf_vectorizer_one_document():
    docs = ["python is a programming language"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)

    expected_matrix = np.array([
        [-0.13862944, -0.13862944, -0.13862944, -0.13862944, -0.13862944]
    ])
    assert len(tfidf_matrix) == 1
    assert len(feature_names) > 0
    np.testing.assert_array_almost_equal(tfidf_matrix, expected_matrix[:, :5], decimal=6)
    np.testing.assert_array_equal(feature_names, np.array(['a', 'is', 'language', 'programming', 'python']))


# Testing a list with one document being empty string for tfidf_vectorizer
def test_tfidf_vectorizer_list_mixed_empty():
    tfidf_matrix, feature_names = tfidf_vectorizer(list_mixed_empty)
    expected_matrix = np.array([
        [0., 0., 0.10136628, 0.10136628, 0.10136628, 0.],  
        [0., 0., 0., 0., 0., 0.],
        [0., 0.13515504, 0., 0., 0., 0.13515504]
    ])
    np.testing.assert_array_almost_equal(tfidf_matrix, expected_matrix, decimal=6)
    np.testing.assert_array_equal(feature_names, np.array(['document', 'have', 'here', 'is', 'one', 'we']))


# Testing a list with only numbers and punctuations for tfidf_vectorizer
def test_tfidf_vectorizer_list_num_punctuation():
    tfidf_matrix, feature_names = tfidf_vectorizer(list_num_punctuation)
    np.testing.assert_array_equal(tfidf_matrix, np.empty((3,0)))
    np.testing.assert_array_equal(feature_names, np.array([]))


# Testing a list with multiple documents for tfidf_vectorizer
def test_tfidf_vectorizer_multiple_docs():
    docs = ["machine learning is interesting", "python is widely used in machine learning"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 2
    assert len(feature_names) > 0


# Testing a list with upper and lower case characters for tfidf_vectorizer
def test_tfidf_vectorizer_case_insensitive():
    docs = ["Python is a programming language", "python is widely used"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)
    assert len(tfidf_matrix) == 2
    assert len(feature_names) > 0


# Testing document with repeated words for tfidf_vectorizer
def test_tfidf_vectorizer_repeated_words():
    docs = ["apple orange banana", "apple banana banana"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)

    expected_matrix = np.array([
        [-0.13515504, -0.13515504,  0.],  
        [-0.13515504, -0.27031007,  0.] 
    ])
    np.testing.assert_array_almost_equal(tfidf_matrix, expected_matrix, decimal=6)
    np.testing.assert_array_equal(feature_names, np.array(['apple', 'banana', 'orange']))


# Testing a list with similar documents for tfidf_vectorizer
def test_tfidf_vectorizer_similar_content():
    docs = ["machine learning is interesting", "machine learning is fascinating"]
    tfidf_matrix, feature_names = tfidf_vectorizer(docs)

    expected_matrix = np.array([
        [ 0.,  0., -0.10136628, -0.10136628, -0.10136628],  
        [ 0.,  0., -0.10136628, -0.10136628, -0.10136628] 
    ])
    np.testing.assert_array_almost_equal(tfidf_matrix, expected_matrix[:, :5], decimal=6)
    np.testing.assert_array_equal(feature_names, np.array(['fascinating', 'interesting', 'is', 'learning', 'machine']))


# Tests for tokenizer_padding
# Testing an empty list for tokenizer_padding
def test_tokenizer_padding_empty_list():
    tokenized_padded = tokenizer_padding(empty_list)
    np.testing.assert_array_equal(tokenized_padded, np.array([]),
                                  err_msg="Function should return an empty array.")


# Testing a list of one empty string for tokenizer_padding
def test_tokenizer_padding_list_empty_str():
    tokenized_padded = tokenizer_padding(list_empty_str)
    np.testing.assert_array_equal(tokenized_padded, np.array([[]]),
                                  err_msg="Function should return an empty 2D array.")


# Testing a list with one document for tokenizer_padding
def test_tokenizer_padding_one_doc():
    tokenized_padded = tokenizer_padding(one_doc)
    np.testing.assert_array_equal(tokenized_padded, np.array([[1, 2, 3, 4, 5, 6]]),
                                  err_msg="Function returns incorrect output.")


# Testing a list with one document being empty string for tokenizer_padding
def test_tokenizer_padding_list_mixed_empty():
    tokenized_padded = tokenizer_padding(list_mixed_empty)
    np.testing.assert_array_equal(tokenized_padded, np.array([[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 3, 0]]),
                                  err_msg="Function returns incorrect output.")


# Testing a list with only numbers and punctuations for tokenizer_padding
def test_tokenizer_padding_list_num_punctuation():
    tokenized_padded = tokenizer_padding(list_num_punctuation)
    np.testing.assert_array_equal(tokenized_padded, np.array([[], [], []]),
                                  err_msg="Function should return an empty 2D array of length 3.")


# Testing a list of documents of equal length for tokenizer_padding
def test_tokenizer_padding_equal_lists():
    equal_lists = ["My first document 1.", "My first 5 documents!", "the Great doc"]
    tokenized_padded = tokenizer_padding(equal_lists)
    np.testing.assert_array_equal(tokenized_padded, np.array([[1, 2, 3], [1, 2, 4], [5, 6, 7]]),
                                  err_msg="Function returns incorrect output.")


# Testing a list of documents of unequal lengths for tokenizer_padding
def test_tokenizer_padding_unequal_lists():
    unequal_lists = ["This is by far the longest doc in this list of docs.",
                     "I am the shortest",
                     "Is last doc the second longest?"]
    tokenized_padded = tokenizer_padding(unequal_lists)
    np.testing.assert_array_equal(tokenized_padded,
                                  np.array([[1, 2, 3, 4, 5, 6, 7, 8, 1, 9, 10, 11],
                                            [12, 13, 5, 14, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [2, 15, 7, 5, 16, 6, 0, 0, 0, 0, 0, 0]]),
                                  err_msg="Function returns incorrect output.")


# Testing a list of documents with repeated words for tokenizer_padding
def test_tokenizer_padding_repeats():
    repeats = ["one two three", "one three four five"]
    tokenized_padded = tokenizer_padding(repeats)
    np.testing.assert_array_equal(tokenized_padded, np.array([[1, 2, 3, 0], [1, 3, 4, 5]]),
                                  err_msg="Function returns incorrect output.")


# Teseting a list of documents without repeated words for tokenizer_padding
def test_tokenizer_padding_no_repeats():
    no_repeats = ["one two three", "four five six"]
    tokenized_padded = tokenizer_padding(no_repeats)
    np.testing.assert_array_equal(tokenized_padded, np.array([[1, 2, 3], [4, 5, 6]]),
                                  err_msg="Function returns incorrect output.")

