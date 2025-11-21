from langchain_ollama import ChatOllama
from dgm_study_assistant.config import settings


def get_llm():
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(model=settings.llm_model)

    raise ValueError(f"Unknown LLM provider: {provider}")
