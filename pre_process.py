stop_words = []
with open('stop_words.txt', "r") as f:
    for line in f:
        stop_words.extend(line.split())

import re
import nltk
from nltk.tokenize import word_tokenize
from hazm import *
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def is_number(word):
    return word.isnumeric()


def pre_process_message(message):
    normalizer = Normalizer()
    norm_doc = normalizer.normalize(message)

    doc_tokens = word_tokenize(norm_doc)

    tagger = POSTagger(model='resource/postagger.model')
    pos = tagger.tag(word_tokenize(message))
    verbs = []
    for (key, val) in pos:
        if (val == 'V'):
            verbs.append(key)
    stemmer = Stemmer()
    pre_doc = [word for word in doc_tokens if (stemmer.stem(word) not in stop_words and word not in stop_words) and (
            stemmer.stem(word) not in verbs and word not in verbs) and not is_number(word)]
    pre_doc = ' '.join(pre_doc)

    return pre_doc


def extract_keyword(message, keyword_count):
    processed_message = pre_process_message(message)
    keywords = []
    if (len(processed_message) < 1):
        return keywords
    n_gram_range = (1, 1)

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range).fit([processed_message])
    candidates = count.get_feature_names()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([processed_message])
    candidate_embeddings = model.encode(candidates)
    top_n = keyword_count
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    keywords = check_static_keyword(processed_message, keywords)
    return keywords


def extract_hashtags(message):
    return re.findall(r"#(\w+)", message)


def extract_emails(message):
    return re.findall(r'[\w\.-]+@[\w\.-]+', message)
