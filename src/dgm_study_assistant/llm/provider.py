from dgm_study_assistant.config import settings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_llm():
    provider = settings.llm_provider.lower()

    # === OLLAMA ===
    if provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            keep_alive=300,
        )

    elif provider == "openai":
        return ChatNVIDIA(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

    # === HUGGINGFACE INFERENCE API ===
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
