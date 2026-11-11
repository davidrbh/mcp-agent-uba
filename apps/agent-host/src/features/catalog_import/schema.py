"""
Esquemas de la habilidad de importación de catálogo.

Define los campos destino (los del importador de Kathenix) y la forma rígida
del plan que el modelo está obligado a emitir con `emit_import_plan`.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# Campos destino del inventario. `multi` = admite varias columnas (una por
# sucursal en existencia, varias en atributos o en ignorar).
FIELDS: Dict[str, Dict] = {
    "nombre": {"label": "Nombre del producto", "required": True, "multi": False},
    "precio": {"label": "Precio de venta al detal (lo que paga el cliente)", "required": True, "multi": False},
    "precio_mayor": {"label": "Precio al mayor", "required": False, "multi": False},
    "costo": {"label": "Costo de compra del comercio (no es precio de venta)", "required": False, "multi": False},
    "sku": {"label": "Código interno del producto (referencia)", "required": False, "multi": False},
    "codigo_barras": {"label": "Código de barras EAN/UPC", "required": False, "multi": False},
    "existencia": {"label": "Existencia o stock; una columna por sucursal", "required": False, "multi": True},
    "categoria": {"label": "Categoría o departamento propio del comercio", "required": False, "multi": False},
    "descripcion": {"label": "Descripción larga", "required": False, "multi": False},
    "atributos": {"label": "Atributos de variante: talla, color, presentación, peso", "required": False, "multi": True},
    "ignorar": {"label": "Columna que no se importa", "required": False, "multi": True},
}

FieldName = Literal[
    "nombre", "precio", "precio_mayor", "costo", "sku", "codigo_barras",
    "existencia", "categoria", "descripcion", "atributos", "ignorar",
]


class ColumnDecision(BaseModel):
    """Decisión sobre una columna del archivo del comercio."""

    index: int = Field(description="Índice de la columna en el archivo (empieza en 0).")
    field: FieldName = Field(description="Campo de Kathenix al que corresponde la columna.")
    branch: Optional[str] = Field(
        default=None,
        description="Solo para 'existencia': nombre exacto de la sucursal a la que pertenece esa columna, o null.",
    )
    doubt: Optional[str] = Field(
        default=None,
        description="Si la decisión no es segura, explica por qué en español sencillo; si es segura, null.",
    )


class ImportPlan(BaseModel):
    """Plan completo para importar el archivo. Es lo único que el modelo devuelve."""

    columns: List[ColumnDecision] = Field(description="Una decisión por cada columna que puedas clasificar.")
    decimal: Literal["coma", "punto"] = Field(description="Separador decimal de los números del archivo.")
    currency: Literal["VES", "USD"] = Field(description="Moneda de los precios del archivo.")
    notes: str = Field(default="", description="Una frase con lo más importante que notaste.")


class CategoryAssignment(BaseModel):
    i: int = Field(description="Índice del elemento en la lista recibida.")
    slug: str = Field(description="Código de la categoría elegida; solo de la lista cerrada, o 'otros'.")


class CategoryPlan(BaseModel):
    """Asignación de categorías. Es lo único que el modelo devuelve."""

    assignments: List[CategoryAssignment] = Field(description="Una asignación por elemento.")
