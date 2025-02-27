import os
from scholarly import scholarly
import pandas as pd

# Membuat folder data jika belum ada
os.makedirs("data", exist_ok=True)  # Menggunakan jalur relatif yang lebih aman

# Mengambil publikasi hanya berdasarkan rentang tahun 2020-2025
search_results = scholarly.search_pubs_custom_url("/scholar?as_ylo=2020&as_yhi=2025")

# Menyiapkan list untuk menyimpan data
titles = []

# Menampilkan 10 hasil pertama dan menyimpan ke list
for i in range(10):
    try:
        paper = next(search_results)
        title = paper["bib"].get("title", "No Title")
        year = paper["bib"].get("pub_year", "No Year")
        authors = ", ".join(paper["bib"].get("author", ["No Authors"]))
        url = paper.get("pub_url", "No URL")
        
        # Tampilkan di terminal dengan format yang lebih rapi
        print(f"Title: {title}")
        print(f"Author(s): {authors}")
        print(f"Year: {year}")
        print(f"URL: {url}")
        print("-" * 50)

        # Simpan ke list untuk CSV dengan nama kolom yang konsisten
        titles.append({
            "title": title, 
            "year": year, 
            "authors": authors, 
            "url": url
        })
    
    except StopIteration:
        print("Tidak ada lagi hasil yang ditemukan.")
        break

# Simpan hasil scraping ke dalam CSV dengan urutan kolom yang konsisten
df = pd.DataFrame(titles, columns=["title", "authors", "year", "url"])
# Ganti nama file untuk menghindari masalah permission
df.to_csv("data/research_titles_new.csv", index=False, encoding='utf-8')


print("✅ Scraping selesai! Data disimpan di data/research_titles.csv")
