import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import spacy

# Load model NLP spaCy untuk lemmatisasi
nlp = spacy.load("en_core_web_sm")

# Membaca file JSON
with open("data/data_scraping.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Normalisasi data agar bisa diproses dalam DataFrame
df = pd.json_normalize(raw_data, record_path=["data"], meta=["topik"])

# Tambahkan kolom panjang teks sebelum preprocessing
df["text_length"] = df["Judul"].apply(lambda x: len(str(x).split()))

# Fungsi untuk membersihkan teks
def preprocess(text):
    doc = nlp(str(text))
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Terapkan preprocessing ke judul
df["Cleaned_Title"] = df["Judul"].apply(preprocess)

# Tambahkan kolom panjang teks setelah preprocessing
df["cleaned_text_length"] = df["Cleaned_Title"].apply(lambda x: len(x.split()))

# Simpan hasil preprocessing ke file JSON
df[["topik", "Cleaned_Title"]].to_json("data/data_preprocessed.json", orient="records", indent=4)
print("\nHasil preprocessing telah disimpan ke 'data/data_preprocessed.json'.")

# Visualisasi WordCloud
text = " ".join(df["Cleaned_Title"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud dari Judul Artikel (Setelah Preprocessing)")
plt.show()

# Visualisasi Histogram Panjang Judul
plt.figure(figsize=(8,5))
plt.hist(df["cleaned_text_length"], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Jumlah Kata per Judul (Setelah Preprocessing)")
plt.ylabel("Frekuensi")
plt.title("Distribusi Panjang Judul Artikel (Setelah Preprocessing)")
plt.show()

# Visualisasi Scatter Plot Frekuensi Kata
all_words = " ".join(df["Cleaned_Title"]).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(50)
words, counts = zip(*common_words)

plt.figure(figsize=(12, 6))
plt.scatter(range(len(words)), counts, color='red', alpha=0.6)
plt.xticks(range(len(words)), words, rotation=90)
plt.xlabel("Kata")
plt.ylabel("Frekuensi Kemunculan")
plt.title("Sebaran Frekuensi Kata dalam Judul Artikel")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()