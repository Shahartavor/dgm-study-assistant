from langchain_ollama import ChatOllama
from dgm_study_assistant.config import settings

def get_llm():
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url="http://localhost:11434", #ollama port
            temperature=0.1,
            keep_alive=300,
            timeout=300,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
