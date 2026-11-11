"""
Orquestación de la habilidad de importación de catálogo.

Orden fijo, de lo barato y seguro a lo caro:
  1. Diccionario de sinónimos (sin IA).
  2. Agente (solo si quedan columnas sin resolver o hay ambigüedad de precios).
  3. Validador sin IA del plan del agente.
  4. Fusión: el diccionario manda salvo que el agente marque un riesgo con 'doubt'.

El resultado son sugerencias para el tablero del comercio: nada se importa
sin su confirmación.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.application.workflows.graph import build_graph
from features.catalog_import.dictionary import SINGLE, map_headers, normalize
from features.catalog_import.loader import (
    CATEGORY_TOOL, IMPORT_TOOL, build_category_input, build_import_input,
    emit_category_plan, emit_import_plan, get_category_system_prompt, get_import_system_prompt,
)
from features.catalog_import.validator import looks_numeric, parse_amount, validate_categories, validate_plan

logger = logging.getLogger(__name__)

PRICE_HINT = re.compile(r"precio|pvp|costo|valor|monto", re.IGNORECASE)


def detect_decimal(rows: Sequence[Sequence[str]]) -> str:
    text = " ".join(str(c) for r in rows[:10] for c in r)
    comma = len(re.findall(r"\d,\d{2}\b", text))
    point = len(re.findall(r"\d\.\d{2}\b", text))
    return "coma" if comma >= point else "punto"


def detect_currency(headers: Sequence[str], rows: Sequence[Sequence[str]], decimal: str) -> str:
    """
    Moneda por marcas explícitas («$», «USD», «Bs»). Sin marcas, se deduce del
    separador decimal (coma → bolívares), y el tablero pide confirmarla.
    """
    raw = " ".join(headers) + " " + " ".join(str(c) for r in rows[:5] for c in r)
    text = normalize(raw)
    if "$" in raw or re.search(r"\b(usd|dolares)\b", text):
        return "USD"
    if re.search(r"\b(bs|ves|bolivares)\b", text):
        return "VES"
    return "VES" if decimal == "coma" else "USD"


def needs_agent(headers: Sequence[str], rows: Sequence[Sequence[str]], mapping: Dict[int, str]) -> bool:
    """El agente solo se llama si el diccionario no resolvió todo o hay dos columnas con pinta de precio."""
    if len(mapping) < len(headers):
        return True
    priced = [i for i, h in enumerate(headers) if PRICE_HINT.search(h) and mapping.get(i) not in ("precio_mayor",)]
    return len(priced) >= 2 and not ({"precio", "costo"} <= set(mapping.values()))


async def propose_import_plan(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    branches: Sequence[str] = (),
    llm: Optional[Any] = None,
    use_agent: bool = True,
) -> Dict[str, Any]:
    """
    Propone el emparejamiento de columnas.

    Returns:
        {
          "columns": {indice: {"field", "branch", "source": "dic"|"ia", "doubt"}},
          "decimal", "currency", "agent_used", "issues": [...], "notes"
        }
    """
    dic = map_headers(headers)
    columns: Dict[int, Dict[str, Any]] = {
        i: {"field": f, "branch": _branch_from_header(headers[i], branches) if f == "existencia" else None,
            "source": "dic", "doubt": None}
        for i, f in dic.items()
    }
    decimal = detect_decimal(rows)
    result = {
        "columns": columns,
        "decimal": decimal,
        "currency": detect_currency(headers, rows, decimal),
        "agent_used": False,
        "issues": [],
        "notes": "",
    }
    if not (use_agent and needs_agent(headers, rows, dic)):
        return result

    graph = build_graph(
        tools=[emit_import_plan],
        system_prompt=get_import_system_prompt(),
        llm=llm,
        forced_tool=IMPORT_TOOL,
        terminal_tools=[IMPORT_TOOL],
    )
    try:
        final = await graph.ainvoke({
            "task": "CATALOG_IMPORT",
            "task_input": build_import_input(headers, rows, branches),
            "messages": [],
        })
    except Exception as e:  # El importador sigue funcionando sin agente.
        logger.error(f"Agente de importación falló: {e!r}")
        result["issues"].append("El agente no respondió; se usa solo el diccionario.")
        return result

    raw = (final.get("outputs") or {}).get(IMPORT_TOOL)
    if not raw:
        reason = "El agente no respondió" if final.get("error") else "El agente no emitió un plan"
        result["issues"].append(f"{reason}; se usa solo el diccionario.")
        return result
    plan, issues = validate_plan(raw, headers, rows, branches)
    result["issues"].extend(issues)
    result["agent_used"] = True
    if not plan:
        return result

    _merge(columns, plan)
    result["decimal"], result["currency"], result["notes"] = plan.decimal, plan.currency, plan.notes
    _price_sanity(columns, rows, result)
    return result


def _branch_from_header(header: str, branches: Sequence[str]) -> Optional[str]:
    n = normalize(header)
    return next((b for b in branches if normalize(b) and normalize(b) in n), None)


def _merge(columns: Dict[int, Dict[str, Any]], plan) -> None:
    """
    El diccionario manda; el agente completa lo vacío o corrige cuando explica
    un riesgo en 'doubt'. Dos pasadas para no depender del orden del modelo:
    primero las correcciones explicadas (liberan campos), después el resto.
    """
    corrections = [c for c in plan.columns if c.doubt and c.index in columns and columns[c.index]["field"] != c.field]
    others = [c for c in plan.columns if c not in corrections]
    for col in corrections + others:
        current = columns.get(col.index)
        if current and current["field"] == col.field:
            current["branch"] = current["branch"] or col.branch
            current["doubt"] = current["doubt"] or col.doubt
            continue
        if current and not col.doubt:
            continue  # Sin explicación, no se pisa una decisión del diccionario.
        if col.field in SINGLE:
            holder = next((j for j, c in columns.items() if j != col.index and c["field"] == col.field), None)
            if holder is not None:
                if columns[holder]["source"] == "dic" and not col.doubt:
                    continue
                del columns[holder]
        columns[col.index] = {"field": col.field, "branch": col.branch, "source": "ia", "doubt": col.doubt}


def _price_sanity(columns: Dict[int, Dict[str, Any]], rows: Sequence[Sequence[str]], result: Dict[str, Any]) -> None:
    """Comprobación sin IA: si 'costo' supera de media a 'precio', algo está al revés."""
    idx = {c["field"]: i for i, c in columns.items() if c["field"] in ("precio", "costo")}
    if {"precio", "costo"} - set(idx):
        return

    def avg(i: int) -> float:
        vals = [parse_amount(r[i], result["decimal"]) for r in rows if i < len(r) and looks_numeric(r[i])]
        vals = [v for v in vals if v]
        return sum(vals) / len(vals) if vals else 0.0

    if avg(idx["costo"]) > avg(idx["precio"]):
        columns[idx["precio"]]["doubt"] = "El costo sale mayor que el precio de venta: revisa si están al revés."
        result["issues"].append("Costo medio mayor que precio medio.")


async def categorize(
    items: Sequence[str],
    categories: Sequence[Sequence[str]],
    kind: str = "productos",
    llm: Optional[Any] = None,
    batch: int = 100,
) -> Dict[int, Dict[str, str]]:
    """
    Asigna una categoría de la lista cerrada a cada elemento, por tandas.
    Lo que el agente no ubique (o si falla) queda en 'otros'.
    """
    allowed = [slug for slug, _ in categories]
    out: Dict[int, Dict[str, str]] = {i: {"slug": "otros", "source": "none"} for i in range(len(items))}
    graph = build_graph(
        tools=[emit_category_plan],
        system_prompt=get_category_system_prompt(categories),
        llm=llm,
        forced_tool=CATEGORY_TOOL,
        terminal_tools=[CATEGORY_TOOL],
    )
    for start in range(0, len(items), batch):
        chunk = list(items[start:start + batch])
        try:
            final = await graph.ainvoke({
                "task": "CATALOG_CATEGORIZE",
                "task_input": build_category_input(chunk, kind),
                "messages": [],
            })
        except Exception as e:
            logger.error(f"Agente de categorías falló en la tanda {start}: {e!r}")
            continue
        raw = (final.get("outputs") or {}).get(CATEGORY_TOOL)
        for i, slug in validate_categories(raw, len(chunk), allowed).items():
            out[start + i] = {"slug": slug, "source": "ia"}
    return out
