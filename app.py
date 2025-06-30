import streamlit as st
import pandas as pd
from libs.chat_agent import chat_with_agent
from libs import rag
from libs.models import get_ollama_models
import libs.models as md
import tempfile
import os
import libs.logging_txt as log
import libs.eval_rouge as rouge
import libs.eval_bleu as bleu
import altair as alt

from libs.rag import (
    load_and_split_pdf,
    load_and_split_pdf_docling,
    store_embeddings,
    search_similar_docs
)

# ===== Sidebar =====
st.set_page_config(layout="wide")
st.sidebar.title("LLM Navigator")
models = md.get_ollama_models()
collections = md.get_collection()

default_model = 'llama3.2:latest'
default_index = models.index(default_model)
selected_model = st.sidebar.selectbox("Model", models, index=default_index)

collection_name = ""
selected_collection = st.sidebar.selectbox("Collection", collections)
if selected_collection:
    collection_name = selected_collection.split("-")[0]

menu = st.sidebar.radio("Menu", [
    "Chatbot RAG",
    "Upload and Embedding PDF",
    "Test RAG With Data QA",
    "ROUGE Evaluation"
])

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history.clear()

#1
if menu == "Upload and Embedding PDF":
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    show_input = st.checkbox("Save embedding vector to new collection")
    if show_input:
        col1, col2 = st.columns([1, 3])
        new_collection_name = col1.text_input("new collection name")
        if new_collection_name:
            collection_name = new_collection_name

    if pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            path = tmp.name

        st.info("Load and split documents...") # ['A','B','C']
        docs = load_and_split_pdf_docling(path)
        for chunk in docs:
            # st.success(f"✅ {(len(chunk.page_content))} - {(chunk.page_content)} ")
            st.success(f"{(chunk)}")
        st.success(f"✅ {len(docs)} chunks created.")

        if st.button("Save to vector DB"):
            store_embeddings(collection_name,docs)
            st.success("✅ Chunks has been saved to pgvector!")
#2
elif menu == "Chatbot RAG":
    eval_score_rouge = {}
    eval_score_bleu = {}
    rouge1 = {}
    ref_input = ""
    st.title("Customer Service")
    user_input = st.chat_input("Write your question...")
    # user_input = st.text_input("Write your question...", key="chat")
    is_evaluation = st.sidebar.checkbox("I want to evaluate LLM response")
    if is_evaluation:
        ref_input = st.text_input("Reference")

    # initial session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input:
        with st.spinner("Answering ..."):
            st.session_state.retriever = rag.load_retriever_from_pgvector(collection_name)
            response = chat_with_agent(user_input, st.session_state.retriever, reference=ref_input, model=selected_model, collection=collection_name)

            log.write_log("log.txt", "session", st.session_state.retriever)
            eval_score_rouge = rouge.get_rouge_score(ref_input,response)
            eval_score_bleu = bleu.get_bleu_score(ref_input, response)
            rouge1 =  rouge.get_rouge_stat(ref_input, response)

            log.write_log("log.txt", "ROUGE", eval_score_rouge)
            log.write_log("log.txt", "BLEU", eval_score_bleu)

            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("agent", response))

            # Melihat Similarity Score
            sim_chunk = rag.search_similar_docs(collection_name,user_input)
            # log.write_log("log.txt", "Chunks", sim_chunk)
            for doc, score in sim_chunk:
                log.write_log("log.txt", "Top k Chunks", f"Score: {score} \n {doc.page_content}")
                # print(doc.page_content)
                # print(f"Score: {score}")

    chat_placeholder = st.container()
    for role, msg in st.session_state.chat_history:
        if role == "user":
            with chat_placeholder.chat_message("user"):
                st.markdown(msg)
        else:
            with chat_placeholder.chat_message("assistant"):
                st.markdown(msg)

    if eval_score_rouge and is_evaluation:
        st.dataframe(rouge.view_rouge_score(eval_score_rouge).style.format("{:.2%}"), width=500)
    # if eval_score_bleu and is_evaluation:
    #     st.write("BLEU Score:", eval_score_bleu)
    # if rouge1 and is_evaluation:
    #     st.write("ROUGE1:", rouge1)

#3
elif menu == "Test RAG With Data QA":
    uploaded_file = st.file_uploader(
        "Please upload CSV file format (must have column 'question' and 'reference')", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file, usecols=["question", "reference"])
        except Exception as e:
            st.error(
                "❌ Failed read CSV. Make sure you have column 'question' and 'reference'.\n"
                f"Detail: {e}"
            )
            # return

        st.markdown("**Top 5 data:**")
        st.dataframe(df_input.head())
        total = len(df_input)
        st.info("Question total: ",total)
        if st.button("Start ROUGE Evaluation"):
            with st.spinner("Evaluating, Please wait ... "):
                my_bar = st.progress(0, text="Processing 0/{}".format(total))
                for idx, row in enumerate(df_input.itertuples(index=False), start=1):
                    question = row.question
                    reference = row.reference
                    st.session_state.retriever = rag.load_retriever_from_pgvector(collection_name)
                    response = chat_with_agent(question, st.session_state.retriever, reference=reference,
                                           model=selected_model, collection=collection_name)
                    eval_score_rouge = rouge.get_rouge_score(question, response)
                    my_bar.progress(min(idx / total, 1.0), text=f"Progress {idx}/{total} lines")
                    # st.dataframe(rouge.view_rouge_score(eval_score_rouge).style.format("{:.2%}"), width=500)
                st.success("Finished.")


    else:
        st.info("Please upload your QA data.")

#4
elif menu == "ROUGE Evaluation":
    st.title("ROUGE Evaluation Metrik")
    # try:
    #     df = rouge.fetch_data_eval_rouge()
    #     if df.empty:
    #         st.warning("Tidak ada data untuk ditampilkan.")
    #     else:
    #         st.dataframe(df, use_container_width=True)
    # except Exception as e:
    #     st.error(f"Terjadi error saat mengambil data: {e}")

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.info("Tabel Perbandingan ROUGE Precision by Model")
            df_prec = rouge.query_precision()
            if df_prec.empty:
                st.warning("Tidak ada data untuk ditampilkan.")
            else:
                st.dataframe(df_prec, use_container_width=True)
                df_long = df_prec.melt(
                    id_vars="model",
                    value_vars=[
                        "rag_rouge1_precision",
                        "rag_rouge2_precision",
                        "rag_rougel_precision",
                        "nonrag_rouge1_precision",
                        "nonrag_rouge2_precision",
                        "nonrag_rougel_precision"
                    ],
                    var_name="Tipe Score",
                    value_name="Nilai"
                )

                ordered_scores = [
                    "rag_rouge1_precision",
                    "rag_rouge2_precision",
                    "rag_rougel_precision",
                    "nonrag_rouge1_precision",
                    "nonrag_rouge2_precision",
                    "nonrag_rougel_precision"
                ]

                chart = alt.Chart(df_long).mark_bar(size=20).encode(
                    x=alt.X('Tipe Score:N', title='Jenis Score', sort=ordered_scores),
                    xOffset=alt.XOffset('model:N'),
                    y=alt.Y('Nilai:Q', title='Nilai Precision'),
                    color=alt.Color('model:N', title='Model'),
                    tooltip=['model', 'Tipe Score', 'Nilai']
                ).properties(
                    width=700,
                    height=400,
                    title='Perbandingan Precision ROUGE (Grouped by Model)'
                )

                st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error saat mengambil data: {e}")
    with col2:
        try:
            st.info("Tabel Perbandingan ROUGE Recall by Model")
            df_recall = rouge.query_recall()
            if df_recall.empty:
                st.warning("Tidak ada data untuk ditampilkan.")
            else:
                st.dataframe(df_recall, use_container_width=True)

                df_long = df_recall.melt(
                    id_vars="model",
                    value_vars=[
                        "rag_rouge1_recall",
                        "rag_rouge2_recall",
                        "rag_rougel_recall",
                        "nonrag_rouge1_recall",
                        "nonrag_rouge2_recall",
                        "nonrag_rougel_recall"
                    ],
                    var_name="Tipe Score",
                    value_name="Nilai"
                )

                ordered_scores = [
                    "rag_rouge1_recall",
                    "rag_rouge2_recall",
                    "rag_rougel_recall",
                    "nonrag_rouge1_recall",
                    "nonrag_rouge2_recall",
                    "nonrag_rougel_recall"
                ]

                chart = alt.Chart(df_long).mark_bar(size=20).encode(
                    x=alt.X('Tipe Score:N', title='Jenis Score', sort=ordered_scores),
                    xOffset=alt.XOffset('model:N'),
                    y=alt.Y('Nilai:Q', title='Nilai Recall'),
                    color=alt.Color('model:N', title='Model'),
                    tooltip=['model', 'Tipe Score', 'Nilai']
                ).properties(
                    width=700,
                    height=400,
                    title='Perbandingan Recall ROUGE (Grouped by Model)'
                )

                st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error saat mengambil data: {e}")

    try:
        st.info("Tabel Perbandingan ROUGE F1 Score by Model")
        df_f1score = rouge.query_f1score()
        if df_f1score.empty:
            st.warning("Tidak ada data untuk ditampilkan.")
        else:
            st.dataframe(df_f1score, use_container_width=True)

            df_long = df_f1score.melt(
                id_vars="model",
                value_vars=[
                    "rag_rouge1_f1score",
                    "rag_rouge2_f1score",
                    "rag_rougel_f1score",
                    "nonrag_rouge1_f1score",
                    "nonrag_rouge2_f1score",
                    "nonrag_rougel_f1score"
                ],
                var_name="Tipe Score",
                value_name="Nilai"
            )

            ordered_scores = [
                "rag_rouge1_f1score",
                "rag_rouge2_f1score",
                "rag_rougel_f1score",
                "nonrag_rouge1_f1score",
                "nonrag_rouge2_f1score",
                "nonrag_rougel_f1score"
            ]

            chart = alt.Chart(df_long).mark_bar(size=20).encode(
                x=alt.X('Tipe Score:N', title='Jenis Score', sort=ordered_scores),
                xOffset=alt.XOffset('model:N'),
                y=alt.Y('Nilai:Q', title='Nilai F1 Score'),
                color=alt.Color('model:N', title='Model'),
                tooltip=['model', 'Tipe Score', 'Nilai']
            ).properties(
                width=700,
                height=400,
                title='Perbandingan F1 Score ROUGE (Grouped by Model)'
            )

            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi error saat mengambil data: {e}")

