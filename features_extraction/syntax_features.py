# features/syntax_features.py
import nltk
import re
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def regex_tokenize(text):
    pattern = r"\b\w+\b"
    return re.findall(pattern, text)

def extract_pos_tree(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    grammar = r"""
      NP: {<DT>?<JJ>*<NN>}
      VP: {<VB.*><NP|PP|CLAUSE>+$}
      PP: {<IN><NP>}
      CLAUSE: {<NP><VP>}
    """
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged)
    return tree

def visualize_tree(tree: Tree):
    tree.pretty_print()
