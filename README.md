# 🧠 Aplikasi LLM dengan RAG (Retrieval-Augmented Generation)

Repositori ini berisi aplikasi NLP berbasis Large Language Model (LLM) yang menggunakan teknik RAG (Retrieval-Augmented Generation) untuk meningkatkan kualitas jawaban berdasarkan dokumen eksternal. Aplikasi ini dibuat menggunakan Python 3.10 dan dilengkapi dengan antarmuka menggunakan Streamlit.

## 📦 Fitur Utama
- Ekstraksi dokumen menggunakan Docling
- Penyimpanan dokumen dalam PostgreSQL dengan ekstensi pgvector
- Pencarian vektor dengan PostgreSQL Vector
- Manage collection untuk embedding
- Evaluasi jawaban model menggunakan metrik **ROUGE** dan **BLEU**
- Antarmuka pengguna sederhana berbasis Streamlit

---

## 🔧 Instalasi

### 1. Clone Repo

```bash
git clone https://github.com/Dhaman81/llm-rag-app.git
cd llm-rag-app
```

### 2. Install Ollama
**Ollama** untuk menjalankan LLM secara lokal.

#### Untuk Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Setelah instalasi, jalankan:
```bash
ollama run llama3
```

### 3. Siapkan Virtual Environment (Python 3.10)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 4. Install Library
```bash
pip install -r requirements.txt
```



### 5. Setup PostgreSQL 16 dan pgvector
Ubuntu / Debian:
```bash
# Tambahkan repo PostgreSQL 16
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
> /etc/apt/sources.list.d/pgdg.list'
wget -qO - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Install PostgreSQL dan pgvector
sudo apt update
sudo apt install postgresql-16 postgresql-server-dev-16

```
Aktifkan pgvector:
```bash
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5. Setting Environment untuk connect ke Database

Buat file **.env** yang berisi:
```bash
DB_USER=user_db
DB_PASS=password_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nama_db
TOP_K=3
```

### 6. Jalankan Aplikasi
```bash
streamlit run app.py
```

## 📊 Evaluasi Model
Aplikasi mendukung evaluasi hasil model dengan:

**ROUGE**: Mengukur kemiripan berdasarkan n-gram, longest common subsequence.

**BLEU**: Evaluasi teks berbasis precision pada n-gram (melalui nltk.translate.bleu_score).

---
## 👤 Author
**Dhaman, S.T.**

## 👤 Special Thanks to

**Dr. SAJARWO ANGGAI, S.ST., M.T.** : Kaprodi Fakultas Magister Teknis Informatika - Universitas Pamulang
**DR ARYA ADHYAKSA WASKITA, S.SI.,M.SI** : Dosen di Fakultas Magister Teknis Informatika - Universitas Pamulang

**Rekan-rekan peneliti**
1. Rafi Mahmud Zein
2. Septian
3. Adrian
4. Dahlan
5. Dandi
6. Agus Salim
