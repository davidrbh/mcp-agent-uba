"""
Cargador de la feature de importación de catálogo.

Igual que `entity_matching`: las herramientas de salida estructurada y los
prompts salen del catálogo `config/prompts.yaml` (skills.import_mapping y
skills.categorize).
"""

import json
from typing import Dict, List, Sequence

from langchain_core.tools import StructuredTool

from features.catalog_import.schema import FIELDS, CategoryPlan, ImportPlan
from features.entity_matching.loader import load_agent_prompts

IMPORT_TOOL = "emit_import_plan"
CATEGORY_TOOL = "emit_category_plan"


# ============================================================================
# 1. HERRAMIENTAS OBLIGATORIAS (su esquema es el contrato con el modelo)
# ============================================================================
def _emit(**kwargs) -> str:
    return "PLAN_EMITTED"


emit_import_plan = StructuredTool.from_function(
    func=_emit,
    name=IMPORT_TOOL,
    description="USAR OBLIGATORIAMENTE para entregar el plan de importación: una decisión por columna, separador decimal y moneda.",
    args_schema=ImportPlan,
)

emit_category_plan = StructuredTool.from_function(
    func=_emit,
    name=CATEGORY_TOOL,
    description="USAR OBLIGATORIAMENTE para entregar la categoría elegida para cada elemento.",
    args_schema=CategoryPlan,
)


# ============================================================================
# 2. PROMPTS
# ============================================================================
def _skill(name: str) -> Dict:
    return (load_agent_prompts().get("skills") or {}).get(name, {})


def get_import_system_prompt() -> str:
    skill = _skill("import_mapping")
    fields = "\n".join(
        f"- {k}: {v['label']}{' (obligatorio)' if v['required'] else ''}{' (admite varias columnas)' if v['multi'] else ''}"
        for k, v in FIELDS.items()
    )
    examples = "\n".join(
        f"Títulos: {ex.get('headers')}\nFila: {ex.get('rows')}\nPlan: {ex.get('plan')}\n"
        for ex in skill.get("examples", [])
    )
    return f"""{skill.get('persona', 'Eres un agente de importación de catálogo.')}

--- HABILIDAD: EMPAREJAR COLUMNAS DE INVENTARIO ---
{skill.get('instructions', '')}

CAMPOS DE KATHENIX:
{fields}

⚠️ REGLAS ESTRICTAS:
{skill.get('strict_rules', '')}

EJEMPLOS:
{examples}
OBLIGATORIO: responde llamando a '{IMPORT_TOOL}'."""


def build_import_input(headers: Sequence[str], rows: Sequence[Sequence[str]], branches: Sequence[str]) -> str:
    cols = " | ".join(f"{i}: {h}" for i, h in enumerate(headers))
    sample = "\n".join(json.dumps(list(r), ensure_ascii=False) for r in rows[:5])
    return f"""Sucursales del comercio: {', '.join(branches) or 'una sola'}.
Columnas del archivo (índice: título): {cols}
Filas de muestra (mismo orden de columnas):
{sample}"""


def get_category_system_prompt(categories: Sequence[Sequence[str]]) -> str:
    skill = _skill("categorize")
    cats = "\n".join(f"- {slug}: {name}" for slug, name in categories)
    return f"""{skill.get('persona', 'Eres un agente de catálogo.')}

--- HABILIDAD: CATEGORIZAR ---
{skill.get('instructions', '')}

LISTA CERRADA DE CATEGORÍAS:
{cats}

⚠️ REGLAS ESTRICTAS:
{skill.get('strict_rules', '')}

OBLIGATORIO: responde llamando a '{CATEGORY_TOOL}'."""


def build_category_input(items: Sequence[str], kind: str) -> str:
    what = "categorías propias de un comercio" if kind == "categorias" else "nombres de productos"
    listing = "\n".join(f"{i}: {t}" for i, t in enumerate(items))
    return f"Asigna una categoría a cada uno de estos {what}:\n{listing}"


async def get_tools() -> List[StructuredTool]:
    return [emit_import_plan, emit_category_plan]
