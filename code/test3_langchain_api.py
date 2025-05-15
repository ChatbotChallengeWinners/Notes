from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama-3.1-8b-instruct",
    # model="meta-llama-3.1-8b-instruct",
    # model="qwen3-235b-a22b",
    # model="qwen3-32b",
    openai_api_base="https://chat-ai.academiccloud.de/v1",
)

api_key = st.sidebar.text_input("NYI: Enter your API key")

def generate_response(input: str):
    res = llm.invoke(input)
    print(res)
    # st.markdown(res["content"])
    st.markdown(res.content)

with st.form("test_form"):
    text = st.text_area(
        "Enter Prompt:",
        "What's the best way to learn how to code?"
    )

    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
