# AgriSentiment

AgriSentiment adalah aplikasi analisis sentimen komentar YouTube bertema pertanian yang dirancang untuk membantu peneliti, akademisi, dan pelaku sektor pertanian memahami persepsi publik secara lebih cepat, visual, dan terukur.

Melalui aplikasi ini, komentar YouTube tidak hanya diklasifikasikan menjadi sentimen positif, netral, dan negatif, tetapi juga dipetakan berdasarkan topik pertanian seperti pemupukan, irigasi, hidroponik, budidaya organik, pengendalian hama, dan isu pertanian lainnya. Hasil analisis kemudian disajikan dalam bentuk dashboard interaktif, visualisasi kata, ringkasan statistik, serta evaluasi performa model machine learning.

## Mengapa Penelitian Ini Penting

Pertanian modern tidak hanya berkembang melalui teknologi di lapangan, tetapi juga melalui percakapan publik di ruang digital. Komentar pada video YouTube menyimpan banyak opini, pengalaman, keluhan, dan harapan masyarakat terhadap praktik pertanian.

AgriSentiment hadir untuk menjawab kebutuhan tersebut dengan mengubah komentar yang tidak terstruktur menjadi insight yang dapat dibaca, dibandingkan, dan digunakan sebagai dasar evaluasi.

## Fitur Utama

- Dashboard interaktif untuk melihat ringkasan sentimen, distribusi data, metadata video, dan tren pembahasan.
- Sentiment Analysis untuk mengeksplorasi hasil sentimen berdasarkan video, topik, dan kategori pertanian.
- Realtime Analysis untuk menganalisis komentar YouTube secara langsung menggunakan URL atau ID video.
- Model Evaluation untuk membandingkan performa Naive Bayes, Support Vector Machine, dan LSTM.
- Word cloud dan frekuensi kata untuk menemukan kata kunci dominan pada setiap topik atau video.
- Contoh komentar representatif untuk membantu interpretasi hasil sentimen secara lebih manusiawi.
- Integrasi PostgreSQL agar data penelitian dapat dikelola lebih rapi dan terpusat.

## Fokus Penelitian

Penelitian ini berfokus pada analisis sentimen komentar YouTube dalam domain pertanian. Data komentar diproses melalui tahapan text preprocessing, pemetaan topik, klasifikasi sentimen, dan evaluasi model.

Topik yang dianalisis mencakup:

- Pemupukan
- Irigasi
- Hidroponik
- Budidaya organik
- Pengendalian hama
- Produk organik
- Feedback penonton
- Kategori turunan lain yang berkaitan dengan isu pertanian

## Metodologi Singkat

Alur analisis pada AgriSentiment meliputi:

1. Pengumpulan komentar YouTube.
2. Cleaning untuk membersihkan teks dari karakter atau pola yang tidak relevan.
3. Case folding untuk menyeragamkan huruf.
4. Tokenizing untuk memecah teks menjadi token kata.
5. Normalization untuk menyesuaikan kata tidak baku.
6. Stopword removal untuk menghapus kata umum yang kurang bermakna.
7. Stemming untuk mengubah kata ke bentuk dasar.
8. Identifikasi topik dan kategori pertanian.
9. Klasifikasi sentimen positif, netral, dan negatif.
10. Evaluasi model menggunakan akurasi, presisi, recall, dan F1-score.

## Model yang Digunakan

AgriSentiment membandingkan beberapa pendekatan klasifikasi:

- Naive Bayes sebagai model probabilistik yang efisien untuk klasifikasi teks.
- Support Vector Machine sebagai model yang kuat dalam memisahkan kelas sentimen berdasarkan pola fitur teks.
- Long Short-Term Memory sebagai pendekatan deep learning untuk menangkap pola bahasa yang lebih kompleks.

Perbandingan model membantu menunjukkan pendekatan mana yang paling sesuai untuk data komentar YouTube bertema pertanian.

## Struktur Data dan File Penting

Beberapa file utama dalam repositori ini:

- `app.py` - aplikasi utama Streamlit.
- `database.py` - koneksi database PostgreSQL.
- `data/` - dataset hasil preprocessing, pemetaan, dan ringkasan analisis.
- `csv_per_category/` - data komentar berdasarkan kategori.
- `topic_keywords.csv` - daftar kata kunci untuk identifikasi topik.
- `kategori_keywords.csv` - daftar kata kunci kategori.
- `sentiment_keywords.csv` - daftar kata kunci sentimen.
- `kamuskatabaku.xlsx` - kamus normalisasi kata.
- `wordcloud_*.png` dan `frekuensi_*.png` - visualisasi kata hasil analisis.

## Teknologi

Proyek ini dibangun menggunakan:

- Python
- Streamlit
- Pandas
- Plotly
- Matplotlib
- Scikit-learn
- TextBlob
- WordCloud
- PostgreSQL
- SQLAlchemy
- YouTube Data API

## Cara Menjalankan Aplikasi

Clone repositori:

```bash
git clone https://github.com/puputayus/AgriSentiment.git
cd AgriSentiment
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows:

```powershell
.venv\Scripts\activate
```

Install dependency yang dibutuhkan:

```bash
pip install streamlit pandas matplotlib plotly psycopg2-binary python-dotenv sqlalchemy textblob scikit-learn wordcloud
```

Buat file `.env` berdasarkan kebutuhan lokal:

```env
DB_HOST=localhost
DB_NAME=agrisentiment
DB_USER=postgres
DB_PASSWORD=password_database
DB_PORT=5432
YOUTUBE_API_KEY=api_key_youtube
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

## Catatan Keamanan

File `.env` tidak disertakan ke GitHub karena berisi konfigurasi sensitif seperti password database dan API key. Gunakan `.env` hanya di lingkungan lokal atau server deployment.

## Kontributor

Penelitian dan aplikasi ini dikembangkan oleh:

**Puput Ayu Setiawati**  
Politeknik Elektronika Negeri Surabaya

## Tujuan Akhir

AgriSentiment diharapkan dapat menjadi media analisis yang membantu pembaca memahami bagaimana masyarakat merespons isu pertanian di ruang digital. Dengan pendekatan machine learning dan visualisasi interaktif, penelitian ini memperlihatkan bahwa opini publik dapat diolah menjadi insight yang bernilai bagi pengembangan pertanian modern.
