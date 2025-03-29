import requests
import json
import os
import time
from bs4 import BeautifulSoup

topik_list = ["komputer", "sosial budaya", "pertanian"]
headers = {"User-Agent": "Mozilla/5.0"}
max_data_per_topic = 100
start_page = 1
reverse_mode = False
session = requests.Session()

filename = "data_scraping.json"

# Cek apakah file sudah ada dan baca data lama
if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        try:
            all_scraped_data = json.load(f)
        except json.JSONDecodeError:
            all_scraped_data = []  # Jika file kosong atau rusak, mulai dari awal
else:
    all_scraped_data = []

# Loop untuk setiap topik
for topik in topik_list:
    print(f"Memulai scraping untuk topik: {topik}")
    scraped_data = []
    base_url = f"https://garuda.kemdikbud.go.id/documents?page={{}}&q={topik}&select=title"

    while len(scraped_data) < max_data_per_topic:
        url = base_url.format(start_page)

        try:
            response = session.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.find_all("div", class_="article-item")

                if not articles:
                    print(f"Tidak ada artikel di halaman {start_page}, berhenti scraping untuk {topik}.")
                    break

                for article in articles:
                    title_tag = article.find("a", class_="title-article")
                    title = title_tag.get_text(strip=True) if title_tag else "Tidak ada judul"

                    author_tag = article.find("a", class_="author-article")
                    author = author_tag.get_text(strip=True) if author_tag else "Tidak ada author"

                    subtitle_tag = article.find("xmp", class_="subtitle-article")
                    year = "Tidak ada tahun"
                    if subtitle_tag:
                        try:
                            year = subtitle_tag.text.strip().split("(")[1].split(")")[0]
                        except IndexError:
                            pass

                    scraped_data.append({
                        "Judul": title,
                        "Tahun": year,
                        "Author": author
                    })

                    if len(scraped_data) >= max_data_per_topic:
                        break

                print(f"Halaman {start_page} selesai untuk topik '{topik}', total data: {len(scraped_data)}")

                start_page = start_page - 1 if reverse_mode else start_page + 1

                time.sleep(5)

            else:
                print(f"Gagal mengakses halaman {start_page} untuk topik '{topik}', status: {response.status_code}")
                break

        except requests.RequestException as e:
            print(f"Error saat mengakses {url}: {e}")
            break

    if scraped_data:
        # Cek apakah topik sudah ada dalam data lama
        existing_topic = next((item for item in all_scraped_data if item["topik"] == topik), None)

        if existing_topic:
            existing_topic["data"].extend(scraped_data)  # Tambahkan data baru ke topik lama
        else:
            all_scraped_data.append({
                "topik": topik,
                "data": scraped_data
            })

# Simpan hasil scraping ke JSON
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_scraped_data, f, indent=4, ensure_ascii=False)

print(f"Scraping selesai! Semua data tersimpan di '{filename}'")
