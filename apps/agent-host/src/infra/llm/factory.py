"""
Fábrica del modelo de lenguaje.

Centraliza qué proveedor responde para que ninguna parte del grafo tenga el
proveedor escrito a mano. El proveedor se elige con la variable de entorno
LLM_PROVIDER o, si no existe, con `llm.provider` de config/settings.yaml.

- deepseek: el proveedor original de la tesis (API compatible con OpenAI).
- bedrock: modelos de AWS Bedrock (p. ej. Claude), sin que los datos salgan
  de la cuenta de AWS. El modelo se indica con BEDROCK_MODEL_ID y la región
  con AWS_REGION.
"""

import os
import logging

from infra.config.loader import ConfigLoader

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("deepseek", "bedrock")


def current_provider() -> str:
    """Devuelve el proveedor activo, priorizando la variable de entorno."""
    settings = ConfigLoader.load_settings() or {}
    provider = os.getenv("LLM_PROVIDER") or (settings.get("llm") or {}).get("provider") or "deepseek"
    provider = str(provider).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Proveedor de LLM no soportado: '{provider}'. Opciones: {', '.join(SUPPORTED_PROVIDERS)}")
    return provider


def get_chat_model(temperature: float = 0.0):
    """
    Construye el modelo de chat del proveedor activo.

    La temperatura por defecto es 0: las habilidades del agente deben dar la
    misma respuesta ante la misma entrada (requisito de la tesis y del
    importador de catálogo).
    """
    provider = current_provider()

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        model_id = os.getenv("BEDROCK_MODEL_ID")
        if not model_id:
            raise ValueError("Falta BEDROCK_MODEL_ID: indica el modelo de Bedrock habilitado en la cuenta.")
        region = os.getenv("AWS_REGION", "us-east-2")
        logger.info(f"LLM: Bedrock '{model_id}' en {region}")
        return ChatBedrockConverse(model=model_id, region_name=region, temperature=temperature)

    from langchain_openai import ChatOpenAI

    settings = ConfigLoader.load_settings() or {}
    model = (settings.get("llm") or {}).get("model") or "deepseek-chat"
    logger.info(f"LLM: DeepSeek '{model}'")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
