import pickle
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import numpy as np

source_path = "data/nlp/Patio.csv"
df_clean = pd.read_csv(source_path)

print("Setup completed.\n\n")

sent = [row.split() for row in df_clean['text']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

print("Data preprocessed.\n\n")

w2v_model = Word2Vec(sentences=sentences, vector_size=200)

w2v_model.build_vocab(sentences, progress_per=10000)

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)

print("Model trained.\n\n")

for i in range(len(sent)):
    word_tokens = sent[i]
    words_mean = np.mean(w2v_model.wv.vectors_for_all(word_tokens).vectors, axis=0).reshape(-1, 200)
    if i == 0:
        x_train = words_mean
    else:
        x_train = np.concatenate((x_train, words_mean))

x_train = np.c_[np.transpose(df_clean["score"].values), x_train]

with open("data/train_data.pkl", "wb") as f:
    pickle.dump(x_train, f)
