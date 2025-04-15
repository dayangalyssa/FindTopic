from fastapi import FastAPI, Query
from typing import List, Optional
import pandas as pd
import json
import os

app = FastAPI(
    title="Topic Modeling API",
    description="API untuk menampilkan hasil topic modeling dari judul-judul artikel.",
    version="1.0"
)

DATA_PATH = "../data/data_with_topics.json"

# Load data saat startup
@app.on_event("startup")
def load_data():
    global df
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Tidak menemukan file di {DATA_PATH}")
    df = pd.read_json(DATA_PATH)

@app.get("/")
def read_root():
    return {"message": "Selamat datang di Topic Modeling API 🎉"}

@app.get("/articles")
def get_all_articles(limit: Optional[int] = Query(default=10, description="Jumlah maksimum artikel yang ditampilkan")):
    return df.head(limit).to_dict(orient="records")

@app.get("/topics")
def get_all_topics():
    return df["Predicted_Topic"].unique().tolist()

@app.get("/articles/by-topic")
def get_articles_by_topic(topic: int):
    filtered = df[df["Predicted_Topic"] == topic]
    return filtered.to_dict(orient="records")
