# script/main.py
import json
import os
from fastapi import FastAPI
from script.scrape import scrape_data
from script.explore_data import preprocess_data

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Server berjalan dengan baik"}

@app.get("/scrape")
def run_scraping():
    data = scrape_data()
    return {"message": "Scraping selesai", "jumlah_data": len(data)}

@app.get("/results")
def show_preprocessed():
    cleaned = preprocess_data()
    return {"message": "Hasil preprocessing", "data": cleaned[:5]}  # tampilkan 5 data awal saja

@app.get("/view")
def view_scraped_data():
    path = "data/data_scraping.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"message": "Data hasil scraping", "jumlah_data": len(data), "data": data[:5]}
    else:
        return {"error": "File data_scraping.json tidak ditemukan"}

