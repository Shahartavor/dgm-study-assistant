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
    """Load transcript files."""
    base_dir = Path(__file__).parent / "data" / "gen_transcripts"
    docs = []
    
    for file in base_dir.glob("*.txt"):
        try:
            loader = TextLoader(str(file), encoding='utf-8')
            docs.extend(loader.load())
        except UnicodeDecodeError:
            try:
                # Try with latin-1 as fallback
                loader = TextLoader(str(file), encoding='latin-1')
                docs.extend(loader.load())
            except Exception as e:
                print(f"Skipping {file}: {e}")
        except Exception as e:
            print(f"Skipping {file}: {e}")
    
    print(f"Loaded {len(docs)} transcript documents")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
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
        search_kwargs={"k": 5}
    )

def format_docs(docs):
    """Convert retrieved Document objects into a single text block with source labels."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source = metadata.get("source", "unknown")
        
        if source == "slides":
            pdf_stem = metadata.get("pdf_stem", "Unknown")
            page = metadata.get("page", "Unknown")
            header = f"[Slide {pdf_stem} - Page {page}]"
        elif source == "transcripts":
            header = f"[Transcript - Document {i+1}]"
        else:
            header = f"[Source {i+1}]"
        
        formatted_docs.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted_docs)

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
    """
    Build the RAG chain for question answering.
    
    Args:
        llm: Language model to use
    """
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
    
    print("Using standard retrieval...")
    retriever = create_retriever(vectorstore)
    
    context_prompt = ChatPromptTemplate.from_template(
        """You are a Deep Generative Models (DGM) tutor. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Instructions:
First, determine if this is a DGM-related question. DGM topics include: VAEs, GANs, Diffusion Models, EM algorithm, GMMs, Normalizing Flows, ELBO, variational inference, latent variables, likelihood, posterior, sampling, generation, autoregressive models, pixelCNN, score-based models.

If NOT a DGM question, respond with exactly: "I can only help with questions about Deep Generative Models and related topics."

If it IS a DGM question:
- Use the context to provide the best answer possible
- Extract and explain relevant information
- Use quotes for important definitions
- Cite sources as [Source 1], [Slide X] etc.
- If context has limited information, provide what you can from the context
- Do NOT add the refusal message at the end

Remember: For DGM questions, provide helpful answers. For non-DGM questions, refuse politely once.

ANSWER:"""
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

