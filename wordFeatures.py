__author__ = 'Manvendra Dattatrey'
__date__   = 'Apr 19, 2019'

from importer import *
from wordShape import getWordShapes
from nltk import LancasterStemmer, PorterStemmer

lancaster_st = LancasterStemmer()
porter_st = PorterStemmer()

def feature_word(word):
    return {('word',word.lower()): 1}

def feature_stem_lancaster(word):
    return {('stem_lancaster',lancaster_st.stem(word.lower())): 1}

def feature_generic(word):
    generic = re.sub('[0-9]','0',word)
    return {('Generic#',generic): 1}

def feature_last_two_letters(word):
    return {('last_two_letters',word[-2:]): 1}

def feature_length(word):
    return {('length', ''): len(word)}

def feature_stem_porter(word):
    try:
        return {('stem_porter', porter_st.stem(word)): 1}
    except Exception as e:
        return {}

def feature_mitre(word):
    features = {}
    for f in mitre_features:
        if re.search(mitre_features[f], word):
            features[('mitre', f)] = 1
    return features

def feature_word_shape(word):
    features = {}
    wordShapes = getWordShapes(word)
    for shape in wordShapes:
        features[('word_shape', shape)] = 1
    return features

def feature_metric_unit(word):
    unit = ''
    if is_weight(word):
        unit = 'weight'
    elif is_size(word):
        unit = 'size'
    elif is_volume(word):
        unit = 'volume'
    return {('metric_unit', unit): 1}


enabled_IOB_prose_word_features = frozenset( [feature_generic, feature_last_two_letters, feature_word, feature_length, feature_stem_porter, feature_mitre, feature_stem_lancaster, feature_word_shape, feature_metric_unit] )

def IOBProseFeatures(word):
    features = {('dummy', ''): 1}
    for feature in enabled_IOB_prose_word_features:
        current_feat = feature(word)
        features.update(current_feat)

    return features

mitre_features = {
    "INITCAP": r"^[A-Z].*$",
    "ALLCAPS": r"^[A-Z]+$",
    "CAPSMIX": r"^[A-Za-z]+$",
    "HASDIGIT": r"^.*[0-9].*$",
    "SINGLEDIGIT": r"^[0-9]$",
    "DOUBLEDIGIT": r"^[0-9][0-9]$",
    "FOURDIGITS": r"^[0-9][0-9][0-9][0-9]$",
    "NATURALNUM": r"^[0-9]+$",
    "REALNUM": r"^[0-9]+.[0-9]+$",
    "ALPHANUM": r"^[0-9A-Za-z]+$",
    "HASDASH": r"^.*-.*$",
    "PUNCTUATION": r"^[^A-Za-z0-9]+$",
    "PHONE1": r"^[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
    "PHONE2": r"^[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
    "FIVEDIGIT": r"^[0-9][0-9][0-9][0-9][0-9]",
    "NOVOWELS": r"^[^AaEeIiOoUu]+$",
    "HASDASHNUMALPHA": r"^.*[A-z].*-.*[0-9].*$ | *.[0-9].*-.*[0-9].*$",
    "DATESEPERATOR": r"^[-/]$",
}


def is_volume(word):
    regex = r"^[0-9]*( )?(ml|mL|dL)$"
    return re.search(regex, word)

def is_weight(word):
    regex = r"^[0-9]*( )?(mg|g|mcg|milligrams|grams)$"
    return re.search(regex, word)

def is_size(word):
    regex = r"^[0-9]*( )?(mm|cm|millimeters|centimeters)$"
    return re.search(regex, word)