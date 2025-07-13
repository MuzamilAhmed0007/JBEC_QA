# features/semantic_features.py
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api

nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
glove = api.load("glove-wiki-gigaword-50")

def lemmatize_word(word, pos='n'):
    return lemmatizer.lemmatize(word, pos)

def get_wordnet_synsets(word):
    return wordnet.synsets(word)

def compute_synset_similarity(syn1, syn2):
    lcs = syn1.lowest_common_hypernyms(syn2)
    if not lcs:
        return 0.0
    return 2 * lcs[0].max_depth() / (syn1.max_depth() + syn2.max_depth())

def get_embedding(word):
    return glove[word] if word in glove else None
