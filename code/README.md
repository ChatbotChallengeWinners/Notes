## Langchain example code

Create a `.env` file in this directory and store your API key there:
```bash
OPENAI_API_KEY=<key>
```

Don't forget the [requirements.txt](requirements.txt)

### [get_models.py](get_models.py)
- grabs a list of models from Academic Cloud
- you can copy-paste it into example files
    <details>
    <summary>List of Models (16) as of 15.05</summary>


    ```
    codestral-22b
    deepseek-r1
    deepseek-r1-distill-llama-70b
    gemma-3-27b-it
    internvl2.5-8b
    llama-3.1-sauerkrautlm-70b-instruct
    llama-3.3-70b-instruct
    llama-4-scout-17b-16e-instruct
    meta-llama-3.1-8b-instruct
    meta-llama-3.1-8b-rag
    mistral-large-instruct
    qwen2.5-coder-32b-instruct
    qwen2.5-vl-72b-instruct
    qwen3-235b-a22b
    qwen3-32b
    qwq-32b
    ```
    </details>

### [test3_langchain_api.py](test3_langchain_api.py)
- Basic Example
- Streamlit gui
- API key field is just an example


### [test4_langchain_history_chroma_api.py](test4_langchain_history_chroma_api.py)
- Example of RAG (Retrieval Augmented Generation)
- uses websites for context
- I'm using streamlit-markdown here to enable mermaid diagrams
    - some models are terrible at this
- [test4_chroma_api.py](test4_chroma_api.py) handles the RAG chain
- context chains are generated on every response rn

    ![Screenshot 2025-05-16 034607](https://github.com/user-attachments/assets/ccfad48d-e7df-46d7-9f07-e62375c4fccf)
