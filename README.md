# 📌 MLOps Data Scraping and Cleaning for Topic Modeling

## 📖 Deskripsi Proyek
 **Proyek ini dirancang untuk mengotomatisasi pengumpulan dan pembersihan data penelitian menggunakan pendekatan MLOps, sehingga mempermudah analisis dan penerapan topic modeling secara efisien!** 📚📋

## 📂 Struktur Direktori
```
mlops-scrapping/
│── data/                 # Folder untuk menyimpan dataset
│   ├── research_titles.csv  # Hasil scraping
│   ├── dataset-cleaned.csv  # Data yang bersih setelah preprocessing
│   ├── clean_data.py        # Script untuk preprocessing
│   ├── scrape.py            # Script untuk scraping
│── models/               # Model hasil training
│── notebooks/            # Notebook eksplorasi data
│── scrap/                # Folder tambahan untuk proses scraping
│── venv/                 # Virtual environment
│── .gitignore            # File untuk mengabaikan file tertentu di Git
│── README.md             # Dokumentasi proyek
│── requirements.txt      # Dependencies yang dibutuhkan
```

## 🔧 Cara Instalasi
1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/dayangalyssa/mlops-scrapping.git
   ```
2. **Masuk ke direktori proyek:**
   ```bash
   cd mlops-scrapping
   ```
3. **Buat environment virtual (opsional tetapi disarankan):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Cara Menjalankan
### **1️⃣ Scraping Data**
Untuk menjalankan scraping dan menyimpan hasilnya di `data/research_titles.csv`, jalankan:
```bash
python data/scrape.py
```
### **2️⃣ Membersihkan Data**
Untuk membersihkan data yang sudah di-scrape dan menyimpannya di `data/dataset-cleaned.csv`, jalankan:
```bash
python data/clean_data.py
```

## 🛠️ Teknologi yang Digunakan
- **Python** 🐍 → Bahasa utama dalam pengolahan data dan otomasi workflow
- **Scholarly** 🔍 → Untuk scraping publikasi akademik dari Google Scholar
- **pandas** 📊 → Untuk manipulasi dan analisis data
- **spaCy** 📝 → Untuk pembersihan teks seperti lemmatization dan stopword removal
- **MLOps workflow** ⚙️ → Untuk otomatisasi pipeline

## 📄 Penjelasan
### **`scrape.py` (Scraping Judul Penelitian)**
- Melakukan pencarian publikasi akademik menggunakan Scholarly.
- Mengambil judul, tahun, penulis, dan URL publikasi.
- Menyimpan hasil dalam `data/research_titles.csv`.

### **`clean_data.py` (Pembersihan Data)**
- Membaca data dari `data/research_titles.csv`.
- Menggunakan spaCy untuk lemmatization dan stopword removal.
- Menyimpan hasil bersih ke `data/dataset-cleaned.csv`.

## 📊 Hasil
✅ **Hasil scraping** disimpan dalam **`data/research_titles.csv`**.  
✅ **Hasil preprocessing** tersimpan dalam **`data/dataset-cleaned.csv`**.  