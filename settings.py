import streamlit as st
import requests

# URL API (Ganti sesuai endpoint aplikasi Anda)
GET_API_URL = "http://example.com/api/get_data"
POST_API_URL = "http://example.com/api/send_data"

st.title("API Integration dengan Streamlit")

# --- GET SECTION ---
st.header("GET Data dari API")
if st.button("Ambil Data"):
    try:
        response = requests.get(GET_API_URL)
        if response.status_code == 200:
            st.success("Data berhasil diambil!")
            st.json(response.json())
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")

# --- POST SECTION ---
st.header("Kirim Data (POST) ke API")

agent_prompt = st.text_input("Masukkan Agent Prompt")
prompt_size = st.number_input("Masukkan Prompt Size", min_value=1, step=1)

if st.button("Kirim Data"):
    payload = {
        "agent_prompt": agent_prompt,
        "prompt_size": prompt_size
    }

    try:
        response = requests.post(POST_API_URL, json=payload)
        if response.status_code in (200, 201):
            st.success("Data berhasil dikirim!")
            st.json(response.json())
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Gagal mengirim data: {e}")