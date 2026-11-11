"""
Validador sin IA del plan emitido por el modelo.

La IA propone; este módulo decide qué se acepta. Descarta lo que no cuadra
con el archivo real y nunca deja pasar un plan mal formado.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from features.catalog_import.dictionary import SINGLE, normalize
from features.catalog_import.schema import ImportPlan, CategoryPlan

_NUM = re.compile(r"^[\s$]*(bs\.?|usd)?\s*-?[\d.,]+\s*(bs\.?|usd|\$)?$", re.IGNORECASE)


def looks_numeric(value: str) -> bool:
    s = str(value or "").strip()
    return bool(s) and bool(re.search(r"\d", s)) and bool(_NUM.match(s))


def parse_amount(raw: str, decimal: str) -> Optional[float]:
    s = re.sub(r"(?i)bs\.?|usd|\$|\s", "", str(raw or ""))
    if not s:
        return None
    s = s.replace(".", "").replace(",", ".") if decimal == "coma" else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def validate_plan(
    raw: Any,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    branches: Sequence[str],
) -> Tuple[Optional[ImportPlan], List[str]]:
    """
    Valida el plan del modelo contra el archivo.

    Returns:
        (plan limpio o None si es inservible, lista de problemas encontrados)
    """
    issues: List[str] = []
    try:
        plan = ImportPlan.model_validate(raw)
    except ValidationError as e:
        return None, [f"El plan no tiene la forma esperada: {e.error_count()} errores de formato."]

    clean = []
    seen_single: Dict[str, int] = {}
    branch_by_norm = {normalize(b): b for b in branches}
    for col in plan.columns:
        if not 0 <= col.index < len(headers):
            issues.append(f"Columna {col.index} no existe en el archivo: descartada.")
            continue
        if col.field in ("precio", "precio_mayor", "costo"):
            values = [r[col.index] for r in rows if col.index < len(r) and str(r[col.index]).strip()]
            if values and not all(looks_numeric(v) for v in values):
                issues.append(f"'{headers[col.index]}' no tiene solo números: no puede ser {col.field}.")
                continue
        if col.field in SINGLE:
            if col.field in seen_single:
                issues.append(f"'{col.field}' asignado a dos columnas; se conserva la primera.")
                continue
            seen_single[col.field] = col.index
        if col.field == "existencia":
            col.branch = branch_by_norm.get(normalize(col.branch or "")) if col.branch else None
        else:
            col.branch = None
        clean.append(col)
    plan.columns = clean
    return plan, issues


def validate_categories(raw: Any, n_items: int, allowed: Sequence[str]) -> Dict[int, str]:
    """Solo acepta códigos de la lista cerrada; lo demás se descarta."""
    try:
        plan = CategoryPlan.model_validate(raw)
    except ValidationError:
        return {}
    allowed_set = set(allowed)
    return {a.i: a.slug for a in plan.assignments if 0 <= a.i < n_items and a.slug in allowed_set}
