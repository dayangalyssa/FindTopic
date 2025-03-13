import os
import time
import pandas as pd
from scholarly import scholarly

# Nama file CSV
csv_file = "data/research_articles.csv"

# Buat folder data jika belum ada
os.makedirs("data", exist_ok=True)

# Cek file lama
if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
    print(f"File ditemukan. {len(df_existing)} data lama dimuat.")
else:
    df_existing = pd.DataFrame(columns=["Title", "Year", "Authors", "URL"])
    print("File tidak ditemukan. Membuat file baru.")

# Mulai pencarian
query = "Artificial Intelligence OR IoT OR Machine Learning"
search_results = scholarly.search_pubs_custom_url(f"/scholar?hl=en&as_ylo=2020&as_yhi=2025&q={query}")

new_articles = []
count = 0
max_articles = 1000

while count < max_articles:
    try:
        paper = next(search_results)
        title = paper["bib"]["title"]
        year = paper["bib"].get("pub_year", "Unknown")
        authors = ", ".join(paper["bib"].get("author", []))
        url = paper.get("pub_url", "No URL")

        print(f"{count+1}. {title} ({year})")
        
        new_articles.append({"Title": title, "Year": year, "Authors": authors, "URL": url})
        count += 1

        # Hindari pemblokiran dengan delay
        time.sleep(1)

    except StopIteration:
        print("Tidak ada lagi hasil yang ditemukan.")
        break
    except Exception as e:
        print(f"⚠ Error: {e}")
        break

# Gabungkan dan simpan ke CSV
df_new = pd.DataFrame(new_articles)
df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
df_combined.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Scraping selesai! Data tersimpan di {csv_file} dengan total {len(df_combined)} entri.")