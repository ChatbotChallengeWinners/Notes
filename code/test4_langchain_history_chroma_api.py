# from langchain_community.llms import Ollama
from dotenv import load_dotenv
# from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
import streamlit as st
from test4_chroma import get_context_retriever_chain, get_vectorstore_from_url, get_conversational_rag_chain

from langchain_core.messages import AIMessage, HumanMessage
from streamlit_markdown import st_markdown

load_dotenv()

model_name = "MFDoom/deepseek-r1-tool-calling:7b"
model_name = "qwen2:7b-instruct"
# model_name = "llama3.2"
# model_name = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:IQ4_NL"
with st.spinner("Loading model"):
    # llm = OllamaLLM(model=model_name)
    llm = ChatOpenAI(
        # model="meta-llama-3.1-8b-instruct",
        model="llama-3.3-70b-instruct",
        openai_api_base="https://chat-ai.academiccloud.de/v1",
    )


# Create new array to store history (first run)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = get_vectorstore_from_url('https://python.langchain.com/docs/integrations/vectorstores/chroma/')

with st.sidebar:
    web_url = st.text_input("Enter URL")
    st.write(st.session_state.chat_history)

# Streamlit re-runs the script. Show all messages
for message in st.session_state.chat_history:
    role = isinstance(message, AIMessage) and "assistant" or "user"
    with st.chat_message(role):
        st.markdown(message.content)

# retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
# conv_rag_chain = get_conversational_rag_chain(retriever_chain)
# Draw a Mermaid diagram showing major animal families


def get_response(prompt: str):
    # print("get_response")
    # not very optimal (move to backend)
    with st.spinner("Loading vectorstore"):
        # vector_store = get_vectorstore_from_url('https://python.langchain.com/docs/integrations/vectorstores/chroma/')
        vector_store = get_vectorstore_from_url([
            'https://gist.githubusercontent.com/ChristopherA/bffddfdf7b1502215e44cec9fb766dfd/raw/e1e7fcd99cad49c8b9effdb6f876f3ad44239c1a/Mermaid_on_Github_Examples.md',
            # 'https://mermaid.js.org/intro/syntax-reference.html',
            # 'https://raw.githubusercontent.com/mermaid-js/mermaid/refs/heads/develop/docs/syntax/flowchart.md',
            'https://en.wikipedia.org/wiki/List_of_animal_classes',
            # 'https://raw.githubusercontent.com/mermaid-js/mermaid/refs/heads/develop/docs/syntax/classDiagram.md'
            # 'https://python.langchain.com/docs/integrations/vectorstores/chroma/'
            ])
    with st.spinner("getting context"):
        retriever_chain = get_context_retriever_chain(vector_store)
        conv_rag_chain = get_conversational_rag_chain(retriever_chain)
    with st.spinner("running query"):
        # print(st.session_state.chat_history)
        response = conv_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": prompt
        })
        print(response)
        return response["answer"]

# Print user prompt and save to history
if prompt := st.chat_input("Enter your prompt..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # response = conv_rag_chain.invoke({
    #     "chat_history": st.session_state.chat_history,
    #     "input": prompt
    # })

    response = get_response(prompt)

    st_markdown(response)
    # print(response)
    st.session_state.chat_history.append(HumanMessage(prompt))
    st.session_state.chat_history.append(AIMessage(response))

    # Get docs relevant to conversation
    # retrieved_docs = retriever_chain.invoke({
    #     "chat_history": st.session_state.chat_history,
    #     "input": prompt
    # })

    # st.write(retrieved_docs)


# def generate_response(input: str):
#     res = llm.invoke(input)
#     print(res)
#     st.info(res)

# with st.form("test_form"):
#     text = st.text_area(
#         "Enter Prompt:",
#         "What's the best way to learn how to code?"
#     )

#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         generate_response(text)
