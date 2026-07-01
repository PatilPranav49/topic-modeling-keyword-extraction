# src/train.py

import os
from gensim import corpora, models
from src.preprocess import preprocess_text

DATA_PATH = "data/raw"

texts = []

print("Loading dataset...")

# 🔥 use more data but still controlled
count = 0
MAX_DOCS = 10000   # strong training (will take time)

for file in os.listdir(DATA_PATH):
    if count >= MAX_DOCS:
        break

    file_path = os.path.join(DATA_PATH, file)

    with open(file_path, 'r', encoding='latin1') as f:
        tokens = preprocess_text(f.read())

        if tokens:
            texts.append(tokens)
            count += 1

print("Documents loaded:", len(texts))


# 🔧 dictionary
print("Creating dictionary...")
dictionary = corpora.Dictionary(texts)

print("Filtering words...")
dictionary.filter_extremes(
    no_below=15,   # remove rare noise
    no_above=0.4   # remove overly common words
)

# corpus
print("Creating corpus...")
corpus = [dictionary.doc2bow(text) for text in texts]

# 🔥 PROPER LDA TRAINING
print("Training LDA... (this may take a few minutes)")

lda = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=12,      # fewer → clearer topics
    passes=25,          # deeper training
    alpha='auto',       # adaptive distribution
    eta='auto',
    chunksize=2000,     # stable training
    random_state=42
)

# save
os.makedirs("models", exist_ok=True)

lda.save("models/lda_model.gensim")
dictionary.save("models/dictionary.dict")

# 🔍 print topics
print("\nSample Topics:")
for i in range(8):
    print(f"Topic {i}: {lda.print_topic(i)}")

print("\n✅ DONE")