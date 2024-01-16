import unittest
from text_processing_util_mds24.text_processing_util_mds24 import (
    text_clean)

empty_list = []
list_empty_str = [""]
one_doc = ["This is a document with 1 string."]
list_invalid = [12, 33.5, None]
list_mixed_empty = ["Here is document one.", "", "we have document 2"]
list_num_punctuation = ["112.32", "!@#$", "795"]


# text_clean
class TestTextClean(unittest.TestCase):

    def test_split_correct(self):
        test_str = ['hi!! wE are gRoup 30 . ',
                    ' i at 12  bananas ']
        cleaned_doc = text_clean(test_str)
        self.assertEqual(len(cleaned_doc), 2)
        self.assertCountEqual(cleaned_doc[0],
                              ['hi', 'we', 'are', 'group'])
        self.assertCountEqual(cleaned_doc[1],
                              ['i', 'at', 'bananas'])

        return

    def test_empty_list(self):
        self.assertEqual(text_clean(empty_list), empty_list)
        return

    def test_list_empty_str(self):
        self.assertEqual(text_clean(list_empty_str), [[]])
        return

    def test_one_doc(self):
        cleaned_doc = text_clean(one_doc)
        self.assertEqual(len(cleaned_doc), 1)
        self.assertCountEqual(cleaned_doc[0],
                              ['this', 'is', 'a', 'document', 'with',
                               'string'])

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            text_clean(None)

        return

    def test_invalid_doc(self):
        with self.assertRaises(TypeError):
            text_clean(list_invalid)

        return

    def test_mixed_empty(self):
        cleaned_doc = text_clean(list_mixed_empty)
        self.assertEqual(len(cleaned_doc), 3)
        self.assertCountEqual(cleaned_doc[0],
                              ['here', 'is', 'document', 'one'])
        self.assertEqual(cleaned_doc[1], [])
        self.assertCountEqual(cleaned_doc[2],
                              ['we', 'have', 'document'])

        return

    def test_list_num_punctuation(self):
        cleaned_doc = text_clean(list_num_punctuation)
        self.assertEqual(cleaned_doc, [[], [], []])

        return


# frequency_vectorizer



# tfidf_vectorizer



# tokenizer_padding



