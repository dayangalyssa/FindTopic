import pandas as pd
import numpy as np
import json
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

with open("data/processed_articles.json", "r") as f:
    df = pd.DataFrame(json.load(f))

corpus = df["Cleaned_Title"].fillna("")

documents = corpus.tolist()

vectorizer = TfidfVectorizer(max_features=500)  # Batasi ke 500 kata fitur terpenting
tfidf_matrix = vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

tfidf_df.to_csv("data/tfidf_results.csv", index=False)

print("\nHasil TF-IDF telah disimpan ke 'data/tfidf_results.csv'.")

word_scores = tfidf_df.sum(axis=0).sort_values(ascending=False)

print("\n10 Kata Paling Penting Berdasarkan TF-IDF:")
print(word_scores.head(10))
