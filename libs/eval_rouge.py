from rouge_score import rouge_scorer
import pandas as pd
import libs.logging_txt as log
import libs.llm_api as llm_api
import libs.db as pg_conn
from sqlalchemy import create_engine, text

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

def fetch_data_eval_rouge():
    conn = pg_conn.get_engine()
    query = text("""SELECT model,collection,question,reference,prediction, 
        r1_precision as Precision_R1,r1_recall as Recall_R1,r1_fmeasure as F1_Score_R1,
        r2_precision as Precision_R2,r2_recall as Recall_R2,r2_fmeasure as F1_Score_R2,
        rl_precision as Precision_RL,rl_recall as Recall_RL,rl_fmeasure as F1_Score_RL 
        FROM rag_score""")

    df = pd.read_sql_query(query, conn)
    return df

def insert_data_eval_rouge(model,collection,question, reference, prediction, score_type="rouge_all"):
    engine = pg_conn.get_engine()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

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