import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Question Answering Demo")

question = st.text_input("Question", "What is the main topic?")
context = st.text_area("Context", "Paste a paragraph here...", height=200)
max_len = st.slider("Max answer length", 10, 200, 80)

if st.button("Get answer"):
    payload = {"question": question, "context": context, "max_answer_len": max_len}
    r = requests.post(f"{API_URL}/qa", json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    st.success(out["answer"])
    st.caption(f"start={out['start']} end={out['end']}")