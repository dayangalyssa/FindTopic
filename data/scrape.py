import os
from scholarly import scholarly
import pandas as pd

# Nama file CSV untuk menyimpan data
csv_file = "data/research_titles.csv"

# Membuat folder data jika belum ada
os.makedirs("data", exist_ok=True)

# 🟢 1. Cek apakah file CSV sudah ada
if os.path.exists(csv_file):
    # Baca data lama dari CSV
    df_existing = pd.read_csv(csv_file)
    print(f"📂 File ditemukan. {len(df_existing)} data lama dimuat.")
else:
    # Jika belum ada, buat DataFrame kosong
    df_existing = pd.DataFrame(columns=["Title", "Year", "Authors", "URL"])
    print("📁 File tidak ditemukan. Membuat file baru.")

# 🟢 2. Lakukan scraping 50 jurnal baru
search_results = scholarly.search_pubs_custom_url("/scholar?as_ylo=2020&as_yhi=2025")
new_titles = []

for i in range(50):
    try:
        paper = next(search_results)
        title = paper["bib"]["title"]
        year = paper["bib"].get("pub_year", "Unknown")
        authors = ", ".join(paper["bib"].get("author", []))
        url = paper.get("pub_url", "No URL")
        
        # Tampilkan di terminal
        print(f"Title: {title}")
        print(f"Author(s): {authors}")
        print(f"Year: {year}")
        print(f"URL: {url}")
        print("-" * 50)

        # Simpan ke list untuk CSV
        new_titles.append({"Title": title, "Year": year, "Authors": authors, "URL": url})
    
    except StopIteration:
        print("❌ Tidak ada lagi hasil yang ditemukan.")
        break

# 🟢 3. Gabungkan data lama dan data baru
df_new = pd.DataFrame(new_titles)
df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()

# 🟢 4. Simpan kembali ke file CSV
df_combined.to_csv(csv_file, index=False, encoding='utf-8')

print(f"✅ Scraping selesai! Data tersimpan di {csv_file} dengan total {len(df_combined)} entri.")
