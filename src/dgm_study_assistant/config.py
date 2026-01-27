from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    #llm_model: str = "granite4:micro"
    #llm_model: str  = "phi3:mini"
    llm_temperature: float = 0.1
    llm_timeout: int = 60
    llm_max_tokens: int = 2048

    openai_api_key: str | None = None
    nvidia_api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434"

    class Config:
        env_file = ".env"

settings = Settings()
