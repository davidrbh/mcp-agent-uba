"""
Diccionario de sinónimos (capa sin IA).

Resuelve lo obvio gratis y siempre igual. El agente solo interviene en lo que
este diccionario no reconoce o en lo que detecta como ambiguo.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Sequence

SYNONYMS: Dict[str, List[str]] = {
    "nombre": ["nombre", "descripcion", "descripcion art", "desc", "desc art", "articulo", "producto", "item", "detalle"],
    "precio": ["pvp", "p v p", "precio", "precio venta", "precio de venta", "precio detal", "p venta", "p unit", "precio unitario", "pvp bs", "pvp usd"],
    "precio_mayor": ["p mayor", "precio mayor", "precio al mayor", "mayor", "mayorista", "precio mayorista"],
    "costo": ["costo", "costo unitario", "precio costo", "p costo", "costo bs", "costo usd", "precio compra"],
    "sku": ["ref", "referencia", "codigo", "cod", "sku", "codigo interno", "cod int", "item code", "codigo item"],
    "codigo_barras": ["cod barras", "codigo de barras", "cod barra", "barras", "ean", "upc", "gtin"],
    "existencia": ["existencia", "existencias", "stock", "cantidad", "cant", "inventario", "exist", "disponible", "unidades", "qty on hand"],
    "categoria": ["categoria", "departamento", "depto", "familia", "linea", "rubro", "seccion"],
    "descripcion": ["descripcion larga", "detalle largo", "observacion", "observaciones"],
    "atributos": ["presentacion", "talla", "color", "tamano", "modelo", "peso", "medida"],
}

# Campos que solo admiten una columna.
SINGLE = {"nombre", "precio", "precio_mayor", "costo", "sku", "codigo_barras", "categoria", "descripcion"}


def normalize(text: str) -> str:
    """Minúsculas, sin tildes ni signos, espacios simples."""
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def match_header(header: str) -> Optional[str]:
    """Campo que corresponde al título, o None si el diccionario no lo reconoce."""
    n = normalize(header)
    if not n:
        return None
    # El sinónimo más largo gana: «cod barras» es código de barras, no «cod».
    for field, w in _BY_LENGTH:
        if n == w or n.startswith(w + " "):
            return field
    return None


_BY_LENGTH = sorted(((f, w) for f, ws in SYNONYMS.items() for w in ws), key=lambda fw: -len(fw[1]))


def map_headers(headers: Sequence[str]) -> Dict[int, str]:
    """
    Empareja los títulos con campos. Un campo de una sola columna se queda
    con la primera columna que lo reclame; las siguientes quedan sin asignar
    para que el agente las resuelva.
    """
    taken = set()
    result: Dict[int, str] = {}
    for i, h in enumerate(headers):
        field = match_header(h)
        if not field:
            continue
        if field in SINGLE and field in taken:
            continue
        taken.add(field)
        result[i] = field
    return result
