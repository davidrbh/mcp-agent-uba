"""
Cargador de la feature de Resolución Semántica (Entity Matching).

Este módulo gestiona el contexto de negocio y construye los prompts
del sistema específicos para comparar productos de e-commerce.
"""

import os
import yaml
import logging
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Configuración de rutas
DOCKER_CONFIG_PATH = Path("/app/config")
if DOCKER_CONFIG_PATH.exists():
    BASE_DIR = Path("/app")
    CONFIG_DIR = DOCKER_CONFIG_PATH
else:
    try:
        BASE_DIR = Path(__file__).resolve().parents[5]
    except IndexError:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
    CONFIG_DIR = BASE_DIR / "config"

# ============================================================================
# 1. HERRAMIENTA OBLIGATORIA
# ============================================================================
@tool
def emit_semantic_prediction(reasoning: str, is_match: int) -> str:
    """
    USAR ESTA HERRAMIENTA OBLIGATORIAMENTE PARA EMITIR EL VEREDICTO FINAL.
    Analiza si Producto A y Producto B son el mismo ítem físico.
    
    Args:
        reasoning: Explicación paso a paso (marca, modelo, especificaciones críticas).
        is_match: 1 si son exactamente el mismo ítem, 0 si son diferentes.
    """
    return f"PREDICTION_EMITTED: {is_match} | REASONING: {reasoning}"

async def get_tools() -> list:
    """Retorna las herramientas locales de esta feature."""
    return [emit_semantic_prediction]

# ============================================================================
# 2. CARGA DE CONFIGURACIONES (YAML)
# ============================================================================
def load_business_context() -> str:
    path = CONFIG_DIR / "business_context.yaml"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Sin equivalencias semánticas definidas."

def load_agent_prompts() -> dict:
    path = CONFIG_DIR / "prompts.yaml"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        return {}

# ============================================================================
# 3. CONSTRUCTOR DEL PROMPT COGNITIVO
# ============================================================================
def get_semantic_system_prompt(channel: str = "batch_processing") -> str:
    catalog = load_agent_prompts()
    
    persona = catalog.get("persona", "Eres un Agente experto en e-commerce.")
    channel_config = catalog.get("channels", {}).get(channel, {})
    style_formatting = channel_config.get("formatting", "")
    
    semantic_skill = catalog.get("skills", {}).get("entity_matching", {})
    instructions = semantic_skill.get("instructions", "Compara Producto A y B.")
    strict_rules = semantic_skill.get("strict_rules", "Diferentes specs = 0.")
    
    examples_str = ""
    for ex in semantic_skill.get("examples", []):
        examples_str += f"Prod A: {ex.get('product_a')}\nProd B: {ex.get('product_b')}\nRazón: {ex.get('reasoning')}\nVeredicto: {ex.get('is_match')}\n\n"
    
    context = load_business_context()
    
    return f"""{persona}

--- MODO DE RESPUESTA ---
{style_formatting}

--- HABILIDAD: RESOLUCIÓN SEMÁNTICA (ENTITY MATCHING) ---
{instructions}

⚠️ REGLAS ESTRICTAS:
{strict_rules}

EJEMPLOS DE EVALUACIÓN:
{examples_str}

📘 CONTEXTO DE NEGOCIO Y EQUIVALENCIAS:
```yaml
{context}
OBLIGATORIO: Al finalizar tu análisis, DEBES invocar la herramienta 'emit_semantic_prediction' para retornar el veredicto (1 o 0).```
"""