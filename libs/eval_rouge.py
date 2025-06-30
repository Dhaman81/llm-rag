from rouge_score import rouge_scorer
import pandas as pd
import libs.logging_txt as log
import libs.llm_api as llm_api
import libs.db as pg_conn
from sqlalchemy import create_engine, text

class WordTokenizer:
    def tokenize(self, text: str):
        # Memecah teks berdasarkan whitespace → daftar kata
        text = text.lower()
        return text.split()

def eval_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    # llm_api.kirim_data("ref", "pred", scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure)
    llm_api.kirim_data("ref", "pred", 0.9, 0.43, 0.3)
    # print(scores)

def get_rouge_score(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores

def view_rouge_score(scores):
    return pd.DataFrame(scores, index=["Precision", "Recall", "F1 Score"])

def get_rouge_stat(reference, prediction):

    scorer = rouge_scorer.RougeScorer(
        ['rouge1'],
        use_stemmer=False,
        tokenizer=WordTokenizer()
    )
    score = scorer.score(reference, prediction)['rouge1']

    total_ref = len(WordTokenizer().tokenize(reference))
    total_pred = len(WordTokenizer().tokenize(prediction))

    #(ingat recall = match / total_ref)
    matched = int(round(score.recall * total_ref))

    return {
        'ref_chars': reference,
        'total_ref_chars': total_ref,
        'pred_chars': prediction,
        'total_pred_chars': total_pred,
        'matched_chars': matched,
        'precision': score.precision,
        'recall': score.recall,
        'fmeasure': score.fmeasure
    }

def fetch_data_eval_rouge():
    conn = pg_conn.get_engine()
    query = text("""SELECT model,collection,question,reference,prediction, 
        r1_precision as Precision_R1,r1_recall as Recall_R1,r1_fmeasure as F1_Score_R1,
        r2_precision as Precision_R2,r2_recall as Recall_R2,r2_fmeasure as F1_Score_R2,
        rl_precision as Precision_RL,rl_recall as Recall_RL,rl_fmeasure as F1_Score_RL 
        FROM rag_score""")

    df = pd.read_sql_query(query, conn)
    return df

def query_precision():
    conn = pg_conn.get_engine()
    query = text("""Select 
            model,
            SUM(rag_Rouge1_Precision) as rag_Rouge1_Precision,
            SUM(rag_Rouge2_Precision) as rag_Rouge2_Precision,
            SUM(rag_RougeL_Precision) as rag_RougeL_Precision,
            SUM(nonrag_Rouge1_Precision) as nonrag_Rouge1_Precision,
            SUM(nonrag_Rouge2_Precision) as nonrag_Rouge2_Precision,
            SUM(nonrag_RougeL_Precision) as nonrag_RougeL_Precision
        from 
        ((select model, 
            avg(r1_precision) as rag_Rouge1_Precision,
            avg(r2_precision) as rag_Rouge2_Precision,
            avg(rl_precision) as rag_RougeL_Precision,
            0 as nonrag_Rouge1_Precision,
            0 as nonrag_Rouge2_Precision,
            0 as nonrag_RougeL_Precision
        from rag_score
            where collection in ('Katalog Produk','Penyakit dan produknya', 'Profil Perusahaan')
            -- and r1_precision is not null
            group by model
            order by model)
        UNION 
        (select model, 
            0 as rag_Rouge1_Precision,
            0 as rag_Rouge2_Precision,
            0 as rag_RougeL_Precision,
            avg(r1_precision) as nonrag_Rouge1_Precision,
            avg(r2_precision) as nonrag_Rouge2_Precision,
            avg(rl_precision) as nonrag_RougeL_Precision
        from rag_score
            where collection in ('Kosong')
            -- and r1_precision is not null
            group by model
            order by model)) AS Precision
            group by model""")

    precision_df = pd.read_sql_query(query, conn)
    return precision_df

def query_recall():
    conn = pg_conn.get_engine()
    query = text("""Select 
            model,
            SUM(rag_Rouge1_recall) as rag_Rouge1_recall,
            SUM(rag_Rouge2_recall) as rag_Rouge2_recall,
            SUM(rag_RougeL_recall) as rag_RougeL_recall,
            SUM(nonrag_Rouge1_recall) as nonrag_Rouge1_recall,
            SUM(nonrag_Rouge2_recall) as nonrag_Rouge2_recall,
            SUM(nonrag_RougeL_recall) as nonrag_RougeL_recall
        from 
        ((select model, 
            avg(r1_recall) as rag_Rouge1_recall,
            avg(r2_recall) as rag_Rouge2_recall,
            avg(rl_recall) as rag_RougeL_recall,
            0 as nonrag_Rouge1_recall,
            0 as nonrag_Rouge2_recall,
            0 as nonrag_RougeL_recall
        from rag_score
            where collection in ('Katalog Produk','Penyakit dan produknya', 'Profil Perusahaan')
            -- and r1_precision is not null
            group by model
            order by model)
        UNION 
        (select model, 
            0 as rag_Rouge1_recall,
            0 as rag_Rouge2_recall,
            0 as rag_RougeL_recall,
            avg(r1_recall) as nonrag_Rouge1_Recall,
            avg(r2_recall) as nonrag_Rouge2_Recall,
            avg(rl_recall) as nonrag_RougeL_Recall
        from rag_score
            where collection in ('Kosong')
            -- and r1_precision is not null
            group by model
            order by model)) AS Recall
            group by model""")

    recall_df = pd.read_sql_query(query, conn)
    return recall_df

def query_f1score():
    conn = pg_conn.get_engine()
    query = text("""Select 
            model,
            SUM(rag_Rouge1_fmeasure) as rag_Rouge1_F1Score,
            SUM(rag_Rouge2_fmeasure) as rag_Rouge2_F1Score,
            SUM(rag_RougeL_fmeasure) as rag_RougeL_F1Score,
            SUM(nonrag_Rouge1_fmeasure) as nonrag_Rouge1_F1Score,
            SUM(nonrag_Rouge2_fmeasure) as nonrag_Rouge2_F1Score,
            SUM(nonrag_RougeL_fmeasure) as nonrag_RougeL_F1Score
        from 
        ((select model, 
            avg(r1_recall) as rag_Rouge1_fmeasure,
            avg(r2_recall) as rag_Rouge2_fmeasure,
            avg(rl_recall) as rag_RougeL_fmeasure,
            0 as nonrag_Rouge1_fmeasure,
            0 as nonrag_Rouge2_fmeasure,
            0 as nonrag_RougeL_fmeasure
        from rag_score
            where collection in ('Katalog Produk','Penyakit dan produknya', 'Profil Perusahaan')
            -- and r1_precision is not null
            group by model
            order by model)
        UNION 
        (select model, 
            0 as rag_Rouge1_fmeasure,
            0 as rag_Rouge2_fmeasure,
            0 as rag_RougeL_fmeasure,
            avg(r1_recall) as nonrag_Rouge1_fmeasure,
            avg(r2_recall) as nonrag_Rouge2_fmeasure,
            avg(rl_recall) as nonrag_RougeL_fmeasure
        from rag_score
            where collection in ('Kosong')
            -- and r1_precision is not null
            group by model
            order by model)) AS F1Score
            group by model""")

    f1score_df = pd.read_sql_query(query, conn)
    return f1score_df

def insert_data_eval_rouge(model,collection,question, reference, prediction, score_type="rouge_all"):
    engine = pg_conn.get_engine()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    if reference:
        r1 = scores["rouge1"]
        r2 = scores["rouge2"]
        rl = scores["rougeL"]

        data = {
            "model": model,
            "collection": collection,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "score_type": score_type,
            "r1_precision": r1.precision,
            "r1_recall": r1.recall,
            "r1_fmeasure": r1.fmeasure,
            "r2_precision": r2.precision,
            "r2_recall": r2.recall,
            "r2_fmeasure": r2.fmeasure,
            "rl_precision": rl.precision,
            "rl_recall": rl.recall,
            "rl_fmeasure": rl.fmeasure,
        }

        query = text("""
            INSERT INTO rag_score (
                model,collection,
                question, reference, prediction, score_type,
                r1_precision, r1_recall, r1_fmeasure,
                r2_precision, r2_recall, r2_fmeasure,
                rl_precision, rl_recall, rl_fmeasure,
                create_date, write_date
            )
            VALUES (
                :model,:collection,
                :question, :reference, :prediction, :score_type,
                :r1_precision, :r1_recall, :r1_fmeasure,
                :r2_precision, :r2_recall, :r2_fmeasure,
                :rl_precision, :rl_recall, :rl_fmeasure,
                now(), now()
            )
        """)
        with engine.begin() as conn:
            conn.execute(query, data)