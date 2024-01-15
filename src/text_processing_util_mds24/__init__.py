# read version from installed package
from importlib.metadata import version
__version__ = version("text_processing_util_mds24")

from text_processing_util_mds24.text_processing_util_mds24 import text_clean, frequency_vectorizer, \
    tfidf_vectorizer, tokenizer_padding