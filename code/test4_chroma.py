
import pprint as pp
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains import create_history_aware_retriever

mbed_model = "mxbai-embed-large"
# db_loc = "Streamlit/test4_chroma_db"

embeddings = OllamaEmbeddings(model=mbed_model)
model_name = "llama3.2"
model_name = "MFDoom/deepseek-r1-tool-calling:7b"
model_name = "qwen2:7b-instruct"
# model_name = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:IQ4_NL"


def get_vectorstore_from_url(url):
    # Parse website with beautifulsoup4
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    doc_chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(doc_chunks, embeddings)

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = OllamaLLM(model=model_name)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You use Markdown features and Mermaid diagram in your replies. Answer the user's questions based on the below context:\n\n{context}"),        
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


if __name__ == "__main__":
    res = get_vectorstore_from_url('https://python.langchain.com/docs/integrations/vectorstores/chroma/')
    # pp.pp(res, indent=2)

