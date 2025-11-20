from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.config import settings


def get_llm():
    provider = settings.llm_provider.lower()

    # --- Ollama ---
    if provider == "ollama":
        return ChatOllama(model=settings.llm_model)

    raise ValueError(f"Unknown LLM provider: {provider}")
