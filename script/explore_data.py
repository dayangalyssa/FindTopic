import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import spacy

# Load model NLP spaCy untuk lemmatisasi
nlp = spacy.load("en_core_web_sm")

# Membaca Data dari JSON
df = pd.read_json("data/research_articles.json")

# Analisis Statistik Dasar
print(df.head())  # Tampilkan 5 data pertama
print(f"\nJumlah data: {len(df)}")

# Tambahkan kolom panjang teks judul
df["text_length"] = df["Title"].apply(lambda x: len(str(x).split()))
print(df["text_length"].describe())

# Fungsi untuk membersihkan teks (lemmatisasi + hapus stopwords)
def preprocess(text):
    doc = nlp(str(text))  
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Terapkan preprocessing ke judul
df["Cleaned_Title"] = df["Title"].apply(preprocess)

# Kata Paling Sering Muncul (setelah preprocessing)
all_words = " ".join(df["Cleaned_Title"]).split()
word_counts = Counter(all_words)
print("\n10 Kata Paling Umum:", word_counts.most_common(10))

# Visualisasi WordCloud setelah preprocessing
text = " ".join(df["Cleaned_Title"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud dari Judul Artikel (Setelah Preprocessing)")
plt.show()

# Visualisasi Histogram Panjang Judul setelah preprocessing
df["cleaned_text_length"] = df["Cleaned_Title"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
plt.hist(df["cleaned_text_length"], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Jumlah Kata per Judul (Setelah Preprocessing)")
plt.ylabel("Frekuensi")
plt.title("Distribusi Panjang Judul Artikel (Setelah Preprocessing)")
plt.show()