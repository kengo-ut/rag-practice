from pydantic_settings import BaseSettings, SettingsConfigDict


class PipeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # モデル設定
    LLM_MODEL_NAME: str = "SakanaAI/TinySwallow-1.5B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 1024
    SENTENCE_TRANSFORMER_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMER_MAX_SEQ_LENGTH: int = 8192
