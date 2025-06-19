import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from prometheus_client import start_http_server, Counter, Gauge, Summary
import threading

# Prometheus metrics
SCRAPING_DURATION = Summary('scraping_duration_seconds', 'Total waktu yang dihabiskan untuk proses scraping')
ARTICLE_COUNTER = Counter('articles_scraped_total', 'Jumlah total artikel yang berhasil di-scrape')
SCRAPING_IN_PROGRESS = Gauge('scraping_in_progress', 'Gauge yang menunjukkan apakah proses scraping sedang berjalan')

# Konstanta
URL = "https://dspace.mit.edu/discover"
OUTPUT_FILENAME = "data_api/data_v2/data_scrapev2.json"
LAST_PAGE_FILE = "data_api/data_v2/last_pagev2.json"
MAX_ARTICLES = 500

os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

def extract_field_after_h5(detail_soup, field_name):
    divs = detail_soup.find_all("div", class_=re.compile(r"item-page-field-wrapper"))
    for div in divs:
        h5 = div.find("h5")
        if h5 and field_name.lower() in h5.text.strip().lower():
            for sibling in h5.next_siblings:
                if isinstance(sibling, str):
                    text = sibling.strip()
                    if text:
                        return text
                elif sibling.name:
                    text = sibling.get_text(strip=True)
                    if text:
                        return text
    return ""

def extract_year(detail_soup):
    raw_date = extract_field_after_h5(detail_soup, "Date issued")
    match = re.search(r'\b(19|20)\d{2}\b', raw_date)
    return match.group(0) if match else ""

def load_last_page():
    if os.path.exists(LAST_PAGE_FILE):
        with open(LAST_PAGE_FILE, "r") as f:
            return json.load(f).get("last_page", 1)
    return 1

def save_last_page(page):
    with open(LAST_PAGE_FILE, "w") as f:
        json.dump({"last_page": page}, f)

def load_existing_data():
    if os.path.exists(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@SCRAPING_DURATION.time()
def scrape_data():
    SCRAPING_IN_PROGRESS.set(1)  
    
    try:
        all_data = load_existing_data()
        initial_count = len(all_data)
        page = load_last_page()

        while len(all_data) < MAX_ARTICLES:
            print(f"[INFO] Scraping page {page}...")
            params = {"page": page}
            headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

            response = requests.get(URL, params=params, headers=headers)
            if response.status_code != 200:
                print("[ERROR] Failed to fetch page.")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='row ds-artifact-item')
            if not articles:
                print("[INFO] No more articles found.")
                break

            for art in articles:
                title_tag = art.find('h4', class_='artifact-title')
                title = title_tag.get_text(strip=True) if title_tag else "Tidak ada judul"

                url_tag = art.find('a')
                url = "https://dspace.mit.edu" + url_tag['href'] if url_tag else ""

                author_tag = art.find('span', class_='ds-dc_contributor_author-authority')
                author = [author_tag.get_text(strip=True)] if author_tag else []

                detail_resp = requests.get(url, headers=headers)
                time.sleep(1)

                abstract = publisher = year = ""
                if detail_resp.status_code == 200:
                    detail_soup = BeautifulSoup(detail_resp.text, 'html.parser')
                    abstract = extract_field_after_h5(detail_soup, "Abstract")
                    publisher = extract_field_after_h5(detail_soup, "Publisher")
                    year = extract_year(detail_soup)

                item = {
                    "Judul": title,
                    "Penulis": author,
                    "URL": url,
                    "Abstrak": abstract,
                    "Publisher": publisher,
                    "Tahun": year
                }
                all_data.append(item)
                ARTICLE_COUNTER.inc() 

                if len(all_data) >= MAX_ARTICLES:
                    break

            page += 1
            save_last_page(page)
            time.sleep(2)

        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)

        articles_scraped = len(all_data) - initial_count
        print(f"[DONE] Total artikel disimpan: {len(all_data)}")
        print(f"[STATS] Articles scraped this run: {articles_scraped}")
        
        return all_data
    finally:
        SCRAPING_IN_PROGRESS.set(0)  # Set gauge ke 0 (tidak aktif)

def start_metrics_server(port=8001):
    print(f"🚀 Starting Prometheus metrics server on port {port}")
    start_http_server(port)

def start_scraping_thread():
    thread = threading.Thread(target=scrape_data)
    thread.daemon = True
    thread.start()
    return thread

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_metrics_server()
    
    print("""
    ┌────────────────────────────────────────────────┐
    │  MIT DSpace Scraper with Prometheus Metrics    │
    └────────────────────────────────────────────────┘
    
    • Metrics server running at: http://localhost:8001/metrics
    • Press Ctrl+C to stop the metrics server
    • Press 's' + Enter to start scraping
    • Press 'q' + Enter to quit
    """)
    
    try:
        scraping_thread = None
        while True:
            command = input("Command (s=scrape, q=quit): ").lower()
            
            if command == 's':
                if scraping_thread and scraping_thread.is_alive():
                    print("Scraping is already in progress...")
                else:
                    print("Starting scraping...")
                    scraping_thread = start_scraping_thread()
            elif command == 'q':
                print("Shutting down...")
                break
            else:
                print("Unknown command. Use 's' to scrape or 'q' to quit.")
    except KeyboardInterrupt:
        print("\nShutting down...")