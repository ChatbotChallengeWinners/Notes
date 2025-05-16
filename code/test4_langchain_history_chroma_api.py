from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from test4_chroma_api import get_context_retriever_chain, get_vectorstore_from_url, get_conversational_rag_chain

from langchain_core.messages import AIMessage, HumanMessage
from streamlit_markdown import st_markdown

load_dotenv()

# with st.spinner("Loading model"):
#     # llm = OllamaLLM(model=model_name)
#     llm = ChatOpenAI(
#         # model="meta-llama-3.1-8b-instruct",
#         model="llama-3.3-70b-instruct",
#         openai_api_base="https://chat-ai.academiccloud.de/v1",
#     )


# Create new array to store history (first run)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = get_vectorstore_from_url('https://python.langchain.com/docs/integrations/vectorstores/chroma/')

with st.sidebar:
    web_url = st.text_input("Enter URL")
    # st.write(st.session_state.chat_history)

# Streamlit re-runs the script. Show all messages
for message in st.session_state.chat_history:
    role = isinstance(message, AIMessage) and "assistant" or "user"
    with st.chat_message(role):
        st_markdown(message.content)

# retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
# conv_rag_chain = get_conversational_rag_chain(retriever_chain)
# Draw a Mermaid diagram showing major animal families


def get_response(prompt: str):
    # print("get_response")
    # not very optimal (move to backend)
    with st.spinner("Creating vector DB"):
        vector_store = get_vectorstore_from_url([
            'https://gist.githubusercontent.com/ChristopherA/bffddfdf7b1502215e44cec9fb766dfd/raw/e1e7fcd99cad49c8b9effdb6f876f3ad44239c1a/Mermaid_on_Github_Examples.md',
            'https://de.wikipedia.org/wiki/Systematik_der_vielzelligen_Tiere',
            'https://en.wikipedia.org/wiki/List_of_animal_classes',
            'https://en.wikipedia.org/wiki/Acanthocephala',
            'https://en.wikipedia.org/wiki/Acoelomorpha',
            'https://en.wikipedia.org/wiki/Annelida',
            'https://en.wikipedia.org/wiki/Chordate',
            'https://en.wikipedia.org/wiki/Echinodermata',
            'https://en.wikipedia.org/wiki/Tardigrada',
            ])
    with st.spinner("Setting up Chains"):
        retriever_chain = get_context_retriever_chain(vector_store)
        conv_rag_chain = get_conversational_rag_chain(retriever_chain)
    with st.spinner("Executing Prompt"):
        # print(st.session_state.chat_history)
        response = conv_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": prompt
        })
        print(response)
        return response

# Print user prompt and save to history
if prompt := st.chat_input("Enter your prompt..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt)

    st_markdown(response["answer"])
    # print(response)
    st.session_state.chat_history.append(HumanMessage(prompt))
    st.session_state.chat_history.append(AIMessage(response["answer"]))
    with st.sidebar:
        for idx, c in enumerate(response["context"]):
            if c.metadata:
                st.markdown(f'{idx}: [{c.metadata['title']}]({c.metadata['source']})')
