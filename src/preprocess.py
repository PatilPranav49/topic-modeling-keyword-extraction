# src/preprocess.py

import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def preprocess_text(text):
    text = text.lower()

    # remove urls and emails
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # remove non-letters
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = simple_preprocess(text)

    # 🔥 dataset-specific stopwords (AG News)
    custom_stopwords = {
        "said","reuters","today","yesterday",
        "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
        "year","years","week","weeks","month","months",
        "new","one","two","three",
        "would","could","also","first","last",
        "company","companies","inc","corp",
        "news","report","reported"
    }

    tokens = [
        w for w in tokens
        if w not in STOPWORDS
        and w not in custom_stopwords
        and len(w) > 3
    ]

    return tokens# src/preprocess.py

import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def preprocess_text(text):
    text = text.lower()

    # remove urls and emails
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove non-letters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # tokenize
    tokens = simple_preprocess(text, deacc=True)

    # 🔥 dataset-specific stopwords (AG News)
    custom_stopwords = {
        # news noise
        "said","reuters","today","yesterday","report","reported","news",

        # days / time
        "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
        "year","years","week","weeks","month","months",

        # generic useless words
        "new","one","two","three","also","would","could","first","last",
        "make","made","take","taken","come","came","going","goes",

        # business filler
        "company","companies","inc","corp","group","share","shares",

        # dataset-specific noise
        "people","like","know","just","time","good","need"
    }

    # final filtering
    tokens = [
        w for w in tokens
        if w not in STOPWORDS
        and w not in custom_stopwords
        and len(w) > 3
        and not w.isdigit()
    ]

    return tokens