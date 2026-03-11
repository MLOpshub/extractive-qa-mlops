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
            padding-top: 1.5rem;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        .section-card {
            padding: 1rem 1rem 0.5rem 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            background-color: #ffffff;
            margin-bottom: 1rem;
        }
        .metric-box {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background-color: #fafafa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Extractive Question Answering</div>', unsafe_allow_html=True)
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

    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=300, step=50)
    overlap = st.slider("Overlap", min_value=0, max_value=200, value=50, step=10)

    action_col1, action_col2 = st.columns(2)

    qa_clicked = action_col1.button("Get Answer", use_container_width=True)
    chunk_clicked = action_col2.button("Generate Chunks", use_container_width=True)

with right_col:
    st.markdown("### Output")

    answer_container = st.container(border=True)
    meta_container = st.container(border=True)

    with answer_container:
        st.markdown("#### Answer")
        st.write("Your answer will appear here.")

    with meta_container:
        st.markdown("#### Response Details")
        st.write("Scores, positions, and chunk info will appear here.")

if qa_clicked:
    if not context.strip() or not question.strip():
        st.warning("Please provide both context and question.")
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

            with right_col:
                answer_container = st.container(border=True)
                meta_container = st.container(border=True)

                with answer_container:
                    st.markdown("#### Answer")
                    answer = result.get("answer", "").strip()
                    if answer:
                        st.success(answer)
                    else:
                        st.warning("No answer found.")

                with meta_container:
                    st.markdown("#### Response Details")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Score", f"{result.get('score', 0.0):.3f}")
                    metric_col2.metric("Start", result.get("start", -1))
                    metric_col3.metric("End", result.get("end", -1))

                    st.markdown("##### Full JSON")
                    st.json(result)

        except requests.RequestException as exc:
            with right_col:
                st.error(f"Request failed: {exc}")

if chunk_clicked:
    if not context.strip():
        st.warning("Please provide context.")
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

            with right_col:
                answer_container = st.container(border=True)
                meta_container = st.container(border=True)

                with answer_container:
                    st.markdown("#### Chunk Summary")
                    st.success(f"Generated {result.get('num_chunks', 0)} chunks.")

                with meta_container:
                    st.markdown("#### Chunk Details")
                    st.json(result)

        except requests.RequestException as exc:
            with right_col:
                st.error(f"Request failed: {exc}")