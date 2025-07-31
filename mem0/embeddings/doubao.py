import os
import warnings
from typing import Literal, Optional
from dotenv import load_dotenv

from volcenginesdkarkruntime import Ark

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

load_dotenv()

class DoubaoEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        self.config.model = self.config.model or "doubao-embedding-text-240715"
        self.config.embedding_dims = self.config.embedding_dims or 2560
        api_key = self.config.api_key or os.getenv("ARK_API_KEY_DOBAO_1_6")
        self.client = Ark(api_key=api_key)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.config.model, encoding_format="float")
            .data[0]
            .embedding
        )
