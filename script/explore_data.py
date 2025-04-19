import json
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    doc = nlp(str(text))
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and token.text.lower() not in stop_words
    ]
    return " ".join(tokens)

def preprocess_data():
    with open("data_api/data_scrape.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    df["Cleaned_Title"] = df["Judul"].apply(preprocess)

    cleaned_data = df[["Cleaned_Title"]].to_dict(orient="records")

    with open("data_api/data_clean.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    return cleaned_data
