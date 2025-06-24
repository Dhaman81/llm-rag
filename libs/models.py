import subprocess
import libs.db as pg_conn
from sqlalchemy import create_engine, text
import pandas as pd

def get_ollama_models():
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
        models = [line.split()[0] for line in output.splitlines()[1:] if line.strip()]
        return models
    except Exception as e:
        return []

def get_collection():
    conn = pg_conn.get_engine()
    query = text("""select b.name,count(b.name) as count 
        from langchain_pg_embedding a
        inner join langchain_pg_collection b on a.collection_id=b.uuid
        group by b.name""")

    df = pd.read_sql_query(query, conn)
    collection_val = df["name"] + "-" + df["count"].astype(str) + " chunks"
    return collection_val.tolist()