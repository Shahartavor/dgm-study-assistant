from operator import itemgetter

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def load_transcripts(transcript_dir="/rag/data/gen_transcripts"):
    base_dir = Path(__file__).parent / "data" / "gen_transcripts"
    print("Looking for transcripts in:", base_dir)
    docs = []
    for file in base_dir.glob("*.txt"):
        print("Loading:", file)
        docs.extend(TextLoader(str(file)).load())
    print("Loaded", len(docs), "documents.")
    return docs

def embed_documents_in_batches(embedding_model, docs, batch_size=32):
    vectors = []
    for i in range(0, len(docs), batch_size):
        batch = [d.page_content for d in docs[i:i+batch_size]]
        vectors.extend(embedding_model.embed_documents(batch))
    return vectors

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def build_vectorstore(docs, embedding_model, save_path="faiss_index", batch_size=32):
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # Batch embed documents
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors.extend(embedding_model.embed_documents(batch))

    # Pair texts with embeddings
    text_embeddings = list(zip(texts, vectors))

    # Build FAISS index
    vs = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model,
        metadatas=metadatas
    )

    vs.save_local(save_path)
    return vs


def load_vectorstore(embeddings, save_path="faiss_index"):
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        k=4
    )

def build_rag_chain(llm):
    embedding_model = get_embeddings()
    try:
        print("Loading existing FAISS index...")
        vectorstore = load_vectorstore(embedding_model)
    except Exception:
        print("FAISS not found, building...")
        docs = load_transcripts()
        chunks = split_docs(docs)
        vectorstore = build_vectorstore(chunks, embedding_model)

    retriever = create_retriever(vectorstore)
    #create_retriever
    context_prompt  = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant specialized in Deep Generative Models (DGMs).
    Answer using ONLY the provided transcript context unless asked otherwise.
    Question:
    {question}
    Context:
    {context}
    Answer clearly and helpfully.
    """)

    chain = (
        {
            'context': itemgetter("question") | retriever,
            'question': itemgetter("question")

        }
    | context_prompt
    | llm
    | StrOutputParser()
    )
    return chain

