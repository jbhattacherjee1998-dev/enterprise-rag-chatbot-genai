import streamlit as st

from utils.rag_engine import ask_question

st.title("Enterprise RAG Assistant")

question = st.text_input(
    "Ask Question"
)

if st.button("Submit"):

    answer = ask_question(
        question
    )

    st.write(answer)