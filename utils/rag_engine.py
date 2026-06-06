from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

def ask_question(question):

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    docs = db.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatOllama(
        model="llama3"
    )

    prompt = f"""
You are an enterprise assistant.

Answer using context only.

Context:
{context}

Question:
{question}
"""
print("your Question -> "+question)
print("your context -> "+context)
    print("Going to invoke LLM, please wait !! it took time")
    response = llm.invoke(prompt)
    print("Response got")
    return response.content