from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Extractive QA Demo",
    page_icon="QA",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            padding-top: 0rem;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.2;
            margin-top: 0;
            margin-bottom: 0.4rem;
        }
        .app-subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-title">Extractive Question Answering</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Professional demo UI for a FastAPI-based extractive QA service.</div>',
    unsafe_allow_html=True,
)

top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    st.info(f"Backend API: {API_URL}")

with top_col2:
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            st.success(f"Status: {health['status']}")
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.markdown("### Input")

    context = st.text_area(
        "Context",
        value=(
            "France is a country in Europe. Paris is the capital of France. "
            "It is known for its history, art, and culture."
        ),
        height=260,
        placeholder="Paste the context paragraph here...",
    )

    question = st.text_input(
        "Question",
        value="What is the capital of France?",
        placeholder="Enter your question...",
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
    )
    overlap = st.slider(
        "Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
    )

    action_col1, action_col2 = st.columns(2)
    qa_clicked = action_col1.button("Get Answer", use_container_width=True)
    chunk_clicked = action_col2.button("Generate Chunks", use_container_width=True)

mode = "default"
result: dict | None = None
error_message: str | None = None

if qa_clicked:
    if not context.strip() or not question.strip():
        error_message = "Please provide both context and question."
    else:
        try:
            response = requests.post(
                f"{API_URL}/qa",
                json={
                    "context": context,
                    "question": question,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            mode = "qa"
        except requests.RequestException as exc:
            error_message = f"Request failed: {exc}"

elif chunk_clicked:
    if not context.strip():
        error_message = "Please provide context."
    else:
        try:
            response = requests.post(
                f"{API_URL}/chunks",
                json={
                    "context": context,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            mode = "chunks"
        except requests.RequestException as exc:
            error_message = f"Request failed: {exc}"

with right_col:
    st.markdown("### Output")

    with st.container(border=True):
        if mode == "qa" and result is not None:
            st.markdown("#### Answer")
            answer = result.get("answer", "").strip()
            if answer:
                st.success(answer)
            else:
                st.warning("No answer found.")
        elif mode == "chunks" and result is not None:
            st.markdown("#### Chunk Summary")
            st.success(f"Generated {result.get('num_chunks', 0)} chunks.")
        else:
            st.markdown("#### Answer")
            st.write("Your answer will appear here.")

    with st.container(border=True):
        if mode == "qa" and result is not None:
            st.markdown("#### Response Details")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Score", f"{result.get('score', 0.0):.3f}")
            metric_col2.metric("Start", result.get("start", -1))
            metric_col3.metric("End", result.get("end", -1))

            st.markdown("##### Full JSON")
            st.json(result)

        elif mode == "chunks" and result is not None:
            st.markdown("#### Chunk Details")
            st.json(result)

        else:
            st.markdown("#### Response Details")
            st.write("Scores, positions, and chunk info will appear here.")

    if error_message:
        st.error(error_message)
