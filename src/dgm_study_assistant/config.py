from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "ollama"
    llm_model: str = "granite4:micro"

    llm_temperature: float = 0.1
    llm_timeout: int = 300
    llm_max_tokens: int = 4096

    openai_api_key: str | None = None
    nvidia_api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434"

    class Config:
        env_file = ".env"

settings = Settings()
