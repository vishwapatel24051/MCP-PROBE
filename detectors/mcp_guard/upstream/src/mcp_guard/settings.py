from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Timeouts and thresholds
    LLM_TIMEOUT_SECONDS: int = 60
    LEARNABLESHIELD_UNSURE_THRESHOLD: float = 0.45
    
    # Service timeouts
    DEFAULT_TIMEOUT: int = 2
    VERIFY_TIMEOUT: float = 8.0
    
    # Model configuration
    DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
    
    class Config:
        env_prefix = "MCP_GUARD_"


settings = Settings()

