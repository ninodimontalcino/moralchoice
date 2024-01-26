import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

# Load Porter Stemmer
porterStemmer = PorterStemmer()

def stem_sentences(sentences):
    """Stem a set of sentences"""
    sentences_tokenized = [word_tokenize(sentence) for sentence in sentences]

    sentences_stemmed = []
    for sentence_tokenized in sentences_tokenized:
        sentences_stemmed.append(
            " ".join(porterStemmer.stem(token) for token in sentence_tokenized)
        )

    return sentences_stemmed


def get_dict_from_string(s):
    """Extract Dictionary from String"""
    d = {}
    for substring in s.split("\n"):
        for key in ["Context", "Action 1", "Action 2"]:
            if key in substring:
                value = substring.split(":")[-1]
                value = re.sub('^"(.*)"$', "\\1", value)
                d[key.strip()] = value.strip()
    return d
