# file: app_rouge_rag_ollama_pgvector.py

import streamlit as st
import pandas as pd
from rouge_score import rouge_scorer

# LangChain imports untuk RAG + pgvector
from langchain.chains import RetrievalQA
from langchain.vectorstores import PGVector
from langchain_ollama import ChatOllama, OllamaEmbeddings

st.set_page_config(page_title="Evaluasi ROUGE RAG (Ollama + pgvector)", layout="wide")

@st.cache_resource
def init_rag_chain():
    """
    Inisialisasi pipeline RAG dengan:
      - ChatOllama (Llama3.2 via Ollama)
      - OllamaEmbeddings (untuk embedding)
      - PGVector (sebagai vectorstore, terhubung ke tabel PostgreSQL dengan ekstensi pgvector)

    Pastikan:
    1. Ollama server sudah berjalan (ollama serve).
    2. Model llama3.2 sudah diunduh: ollama pull llama3.2.
    3. Anda sudah membuat tabel rag_documents di PostgreSQL dengan ekstensi pgvector, dan sudah mengisi embedding.
    4. Ganti POSTGRES_CONNECTION_STRING sesuai kredensial Anda.
    """

    # -----------------------------
    # 1. Inisialisasi ChatOllama (LLM)
    # -----------------------------
    llm = ChatOllama(
        model="llama3.2",  # Varian Llama3.2 3B
        temperature=0.3,  # Pengaturan deterministik (rendah)
    )

    # -----------------------------
    # 2. Inisialisasi OllamaEmbeddings (Embedding Function)
    # -----------------------------
    embeddings = OllamaEmbeddings(
        model="llama3.2",  # Model untuk embed; pastikan model embedding tersedia
        base_url="http://localhost:11434",  # default endpoint Ollama
    )

    # -----------------------------
    # 3. Inisialisasi PGVector (vectorstore via LangChain)
    # -----------------------------
    # Ganti string di bawah dengan koneksi PostgreSQL Anda:
    POSTGRES_CONNECTION_STRING = "postgresql+psycopg2://dhaman:panikem01@localhost/odoo18c"

    try:
        # Nama tabel: rag_documents
        # Kolom embedding di tabel: embedding (VECTOR)
        # Kolom konten: content (TEXT)
        vectordb = PGVector.from_existing_index(
            embedding=embeddings,
            connection_string=POSTGRES_CONNECTION_STRING,
            table_name="langchain_pg_embedding",
            columns={"document": "document", "embedding": "embedding"},
        )
    except Exception as e:
        st.error(
            "❌ Gagal menginisialisasi PGVector. "
            "Pastikan:\n"
            "1. Tabel 'rag_documents' ada di database dan sudah diisi embedding (pgvector).\n"
            "2. Koneksi (connection_string) benar.\n"
            f"Detail error: {e}"
        )
        return None

    # Buat retriever dengan mengambil 5 dokumen teratas
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # -----------------------------
    # 4. Buat chain RetrievalQA
    # -----------------------------
    # Chain ini akan melakukan:
    #  - mengambil 5 dokumen teratas dari PGVector (pertanyaan dijadikan kueri embedding)
    #  - kemudian memanggil Llama3.2 (ChatOllama) dengan konteks dokumen + pertanyaan
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,  # Tidak menampilkan dokumen sumber di output
    )

    return chain


def generate_answer(chain, question: str) -> str:
    """
    Jalankan RAG chain untuk menghasilkan jawaban berdasarkan sebuah pertanyaan.
    Jika terjadi error (misalnya koneksi PGVector gagal, atau LLM gagal), kembalikan string error.
    """
    try:
        # chain.run(question) biasanya langsung memberikan string hasil jawaban
        # Jika pada versi LangChain Anda chain.run mengembalikan dict, sesuaikan ambilan field-nya.
        return chain.run(question)
    except Exception as e:
        return f"ERROR: {e}"


def compute_rouge_scores(pred: str, ref: str):
    """
    Hitung metrik ROUGE-1, ROUGE-2, dan ROUGE-L (precision, recall, f-measure)
    antara prediksi model (pred) dan referensi (ref).
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(ref, pred)
    return {
        "rouge1_precision": scores["rouge1"].precision,
        "rouge1_recall": scores["rouge1"].recall,
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougel_precision": scores["rougeL"].precision,
        "rougel_recall": scores["rougeL"].recall,
        "rougel_f1": scores["rougeL"].fmeasure,
    }


def main():
    st.title("🔍 Evaluasi ROUGE RAG (Ollama + pgvector)")

    st.markdown(
        """
        Aplikasi ini melakukan evaluasi metrik ROUGE secara masif terhadap data `question` (pertanyaan) 
        dan `reference` (jawaban acuan) yang Anda unggah dalam file CSV, menggunakan pipeline RAG:
          1. RAG dibangun dengan **Ollama (Llama3.2)** sebagai LLM.
          2. **pgvector** (PostgreSQL) sebagai vector database yang menyimpan embedding dokumen.

        **Langkah penggunaan:**  
        1. Pastikan Anda sudah memiliki:
           - Ollama server berjalan, dan model `llama3.2` telah di-pull.  
           - Tabel `rag_documents` di PostgreSQL berisi kolom `content` (teks dokumen) dan `embedding` (VECTOR).  
        2. Siapkan file CSV dengan dua kolom:  
           - `question` (pertanyaan yang akan dievaluasi)  
           - `reference` (jawaban acuan / ground truth)  
        3. Upload file CSV melalui widget di bawah.  
        4. Klik **"▶️ Mulai Evaluasi ROUGE"** untuk menjalankan pipeline RAG & menghitung skor ROUGE.  
        5. Setelah selesai, Anda akan melihat tabel hasil skor ROUGE per baris, dan bisa mengunduhnya sebagai file CSV.
        """
    )

    # -----------------------------
    # 1. Upload file CSV
    # -----------------------------
    uploaded_file = st.file_uploader(
        "📂 Upload file CSV (harus ada kolom 'question' dan 'reference')", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file, usecols=["question", "reference"])
        except Exception as e:
            st.error(
                "❌ Gagal membaca CSV. Pastikan file memiliki kolom 'question' dan 'reference'.\n"
                f"Detail: {e}"
            )
            return

        st.markdown("**🗂️ Contoh data (5 baris pertama):**")
        st.dataframe(df_input.head())

        # Tombol untuk mulai evaluasi
        if st.button("▶️ Mulai Evaluasi ROUGE"):
            with st.spinner("Sedang melakukan evaluasi, mohon tunggu..."):
                # Inisialisasi pipeline RAG (Ollama + pgvector)
                chain = init_rag_chain()
                if chain is None:
                    # Jika gagal inisialisasi, hentikan eksekusi selanjutnya
                    st.stop()

                # Siapkan list untuk menyimpan hasil
                results = []
                total = len(df_input)
                my_bar = st.progress(0, text="Memproses 0/{}".format(total))

                # Iterasi setiap baris (question, reference)
                for idx, row in enumerate(df_input.itertuples(index=False), start=1):
                    question = row.question
                    reference = row.reference

                    # 1. Generate jawaban model (RAG)
                    predicted = generate_answer(chain, question)

                    # 2. Hitung skor ROUGE
                    rouge_scores = compute_rouge_scores(predicted, reference)

                    # 3. Simpan hasil setiap baris
                    hasil_baris = {
                        "question": question,
                        "reference": reference,
                        "predicted": predicted,
                        **rouge_scores,
                    }
                    results.append(hasil_baris)

                    # 4. Update progress bar
                    my_bar.progress(min(idx / total, 1.0), text=f"Memproses {idx}/{total} baris")

                # Setelah loop selesai, konversi hasil ke DataFrame
                df_results = pd.DataFrame(results)

                st.success("✅ Evaluasi selesai!")
                st.markdown("**📊 Tabel Hasil Evaluasi:**")
                st.dataframe(df_results)

                # Tombol unduh file CSV hasil evaluasi
                csv_buffer = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Unduh Hasil Evaluasi (CSV)",
                    data=csv_buffer,
                    file_name="evaluasi_rouge_rag_pgvector.csv",
                    mime="text/csv",
                )

    else:
        st.info("📥 Silakan upload file CSV terlebih dahulu.")


if __name__ == "__main__":
    main()
