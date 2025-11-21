from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "ollama"
    llm_model: str = "granite4:micro"

    class Config:
        env_file = ".env"

settings = Settings()
