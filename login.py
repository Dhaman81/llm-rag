import streamlit as st
from libs.db import check_user
import streamlit.components.v1 as components

st.set_page_config(page_title="Login", page_icon="🔐")

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

st.title("🔐 Login Page")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    user = check_user(username, password)
    if user:
        st.session_state.is_authenticated = True
        st.session_state.username = username
        st.success("Login berhasil! Mengalihkan ke halaman utama...")
        st.switch_page("modules/1_Main.py")
        # st.rerun()
    else:
        st.error("Username atau password salah!")

# Redirect to main page if already logged in
# if st.session_state.is_authenticated:
#     st.switch_page("1_Main.py")