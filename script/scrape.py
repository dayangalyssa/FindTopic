# script/scrape.py
import requests
from bs4 import BeautifulSoup
import json
import time
import os

URL = "https://dspace.mit.edu/discover"
OUTPUT_FILENAME = "data/data_scraping.json"
MAX_ARTICLES = 2000

if not os.path.exists('data'):
    os.makedirs('data')

def scrape_data():
    all_data = []
    page = 1

    while len(all_data) < MAX_ARTICLES:
        params = {"page": page}
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(URL, params=params, headers=headers)
            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='row ds-artifact-item')

            if not articles:
                break

            for article in articles:
                title_tag = article.find('h4', class_='artifact-title')
                title = title_tag.get_text(strip=True) if title_tag else "Tidak ada judul"

                url_tag = article.find('a')
                url = "https://dspace.mit.edu" + url_tag['href'] if url_tag and url_tag.has_attr('href') else "Tidak ada URL"

                author_tag = article.find('span', class_='ds-dc_contributor_author-authority')
                author = author_tag.get_text(strip=True) if author_tag else "Tidak ada penulis"

                all_data.append({
                    "Judul": title,
                    "Author": author,
                    "URL": url
                })

                if len(all_data) >= MAX_ARTICLES:
                    break

            page += 1
            time.sleep(1)

        except requests.RequestException as e:
            print(f"Error: {e}")
            break

    # Simpan data ke file
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    return all_data
