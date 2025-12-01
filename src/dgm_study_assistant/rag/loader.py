from operator import itemgetter

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableMap


def load_slides():
    slides_dir = Path(__file__).parent / "data" / "gen_slides"
    print("Looking for slides in:", slides_dir)

    docs = []

    for pdf_path in slides_dir.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        pdf_stem = pdf_path.stem  # e.g. "DGM_L7_Normalizing_Flows"
        print("Loading:", pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            cleaned = text.strip()
            if not cleaned:
                continue

            docs.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "source": "slides",
                        "pdf_stem": pdf_stem,  # matches folder & filename stem
                        "page": page_num,      # matches _page_<page>.png
                    },
                )
            )

    print(f"Loaded {len(docs)} slide documents.")
    return docs


def load_transcripts():
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
        k=7
    )

def format_docs(docs):
    """Convert retrieved Document objects into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_slide_image_from_metadata(metadata):
    if metadata.get("source") != "slides":
        return None

    pdf_stem = metadata.get("pdf_stem")
    page = metadata.get("page")

    if pdf_stem is None or page is None:
        return None

    slides_dir = Path(__file__).parent / "data" / "gen_slides_images"

    img_path = slides_dir / pdf_stem / f"{pdf_stem}_page_{page}.png"

    if img_path.exists():
        return str(img_path)
    return None



def build_rag_chain(llm):
    embedding_model = get_embeddings()
    #save_path = "faiss_index_transcripts"
    save_path = "faiss_index"
    try:
        print("Loading existing FAISS index...")
        vectorstore = load_vectorstore(embedding_model,save_path)
    except Exception:
        print("FAISS not found, building...")
        transcript_docs = load_transcripts()
        slides_docs = load_slides()
        docs = transcript_docs + slides_docs
        #docs = transcript_docs
        chunks = split_docs(docs)
        vectorstore = build_vectorstore(chunks, embedding_model,save_path)

    retriever = create_retriever(vectorstore)
    #create_retriever
    context_prompt = ChatPromptTemplate.from_template(
        """
        You are a Deep Generative Models (DGM) tutor.

        YOU MUST ANSWER USING ONLY THE CONTEXT BELOW.
        If the context does not contain the answer, say:
        "The provided course materials do not contain this information."

        QUESTION:
        {question}

        CONTEXT:
        {context}

        RULES:
        - USE EXACT TERMINOLOGY FROM THE CONTEXT.
        - IF THE CONTEXT CONTAINS EQUATIONS, RETURN THEM EXACTLY AS THEY APPEAR.
        - DO NOT USE PRIOR KNOWLEDGE.
        - DO NOT ADD EXTRA INFORMATION.
        """
    )

    chain = (
            RunnableMap({
                "docs": itemgetter("question") | retriever,
                "question": itemgetter("question")
            })
            |
            RunnableMap({
                "context": lambda x:format_docs(x["docs"]),
                "docs": itemgetter("docs"),
                "question": itemgetter("question")
            })
            |
            RunnableMap({
                "answer": context_prompt | llm | StrOutputParser(),
                "docs": itemgetter("docs")
            })

    )
    return chain

