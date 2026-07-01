# src/predict.py
from gensim import models, corpora
import os
lda = models.LdaModel.load("models/lda_model.gensim")
dictionary = corpora.Dictionary.load("models/dictionary.dict")

def get_topics(tokens):
    bow = dictionary.doc2bow(tokens)
    return lda.get_document_topics(bow)

def get_all_topics():
    return [lda.print_topic(i) for i in range(lda.num_topics)]

def get_model_and_dict():
    return lda, dictionary
def get_corpus():
    from gensim import corpora

    texts = []
    for file in os.listdir("data/raw")[:500]:   # limit for speed
        with open(os.path.join("data/raw", file), 'r', encoding='latin1') as f:
            tokens = f.read().split()
            texts.append(tokens)

    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus