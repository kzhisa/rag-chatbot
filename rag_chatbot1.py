import os

import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME")

# Retriever settings
TOP_K_VECTOR = 10

# 既存のChromaDBを読み込みVector Retrieverを作成
def vector_retriever(top_k: int = TOP_K_VECTOR):
    """Create base vector retriever from ChromaDB

    Returns:
        Vector Retriever
    """

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )

    # base retriever (vector retriever)
    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k},
    )

    return vector_retriever


# プロンプトテンプレート
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 実際の応答生成の例
def chat_with_bot(session_id: str):

    # LLM
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

    # Vector Retriever
    retriever = vector_retriever()

    # RAG Chain
    basic_qa_chain = create_stuff_documents_chain(
        llm = chat_model,
        prompt = prompt_template,
    )
    rag_chain = create_retrieval_chain(retriever, basic_qa_chain)

    count = 0
    while True:
        print("---")
        input_message = input(f"[{count}]あなた: ")
        if input_message.lower() == "終了":
            break

        # プロンプトテンプレートに基づいて応答を生成
        response = rag_chain.invoke(
            {"input": input_message},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"AI: {response['answer']}")
        count += 1


if __name__ == "__main__":

    # チャットセッションの開始
    session_id = "example_session"
    chat_with_bot(session_id)
