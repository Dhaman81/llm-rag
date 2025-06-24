import requests
import json

# Konfigurasi API
API_URL = "http://localhost:8069/v1/llm/create_rag_score"
API_KEY = "ba41bad742d5ed51ea65aa51cb53e66be2fbe03"

# Fungsi untuk mengirim POST request
def kirim_data(ref, pred, precision, recall, fmeasure):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "jsonrpc": "2.0",
        "params": {
            "ref": ref,
            "pred": pred,
            "r1_precision": precision,
            "r1_recall": recall,
            "r1_fmeasure": fmeasure
        }
    }

    try:
        # response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response = requests.post(API_URL, headers=headers, json=payload)
        print(headers, "----", payload)
        print("RESPONSE: ",response.json())
        return response.status_code, response.json()
    except Exception as e:
        print(str(e))
        return None, {"error": str(e)}