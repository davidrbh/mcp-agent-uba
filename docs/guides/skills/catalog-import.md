# Habilidad: importación de catálogo (`import_mapping` + `categorize`)

Reutiliza el motor de la tesis (grafo LangGraph, temperatura 0, salida obligatoria por herramienta e interceptor) para una tarea de comercio real: entender el archivo de inventario de un comercio y proponer cómo encaja en el importador de catálogo de Kathenix.

## Qué hace

| Paso | Quién | Qué |
|---|---|---|
| 1 | Diccionario (sin IA) | Empareja títulos conocidos: «PVP» → precio, «Exist.» → existencia, «Cód. barras» → código de barras. Gana el sinónimo más largo. |
| 2 | Agente (solo si hace falta) | Si quedan columnas sin reconocer o hay dos columnas con pinta de precio, recibe **solo los títulos y 5 filas** y emite `emit_import_plan`: campo por columna, sucursal de cada existencia, decimal, moneda y dudas. |
| 3 | Validador (sin IA) | Descarta índices inexistentes, precios que no son números, campos únicos repetidos y sucursales que no existen. Un plan mal formado se ignora. |
| 4 | Fusión | El diccionario manda; el agente completa lo vacío o corrige cuando explica el riesgo en `doubt` (p. ej. «parece costo, no precio de venta»). |
| 5 | Comprobación (sin IA) | Si el costo medio supera al precio medio, marca la duda. |

`categorize` asigna categorías de una **lista cerrada** por tandas de 100; lo que no se ubique o lo que el modelo invente queda en `otros`.

El resultado son sugerencias para el tablero del comercio: nada se importa sin su confirmación.

## Cambios en el núcleo (compatibles con la tesis)

- `infra/llm/factory.py`: el proveedor ya no está escrito en el grafo. `LLM_PROVIDER=deepseek|bedrock` (o `llm.provider` en `settings.yaml`); para Bedrock, `BEDROCK_MODEL_ID` y `AWS_REGION`.
- `build_graph(..., llm=, forced_tool=, terminal_tools=)`: modelo inyectable (pruebas sin red), herramienta obligatoria y herramientas tras las cuales no se vuelve a llamar al modelo.
- Estado: `task` (evita la llamada del clasificador), `task_input` y `outputs` (lo emitido por cualquier `emit_*`).
- La tesis no cambia: sin `terminal_tools` conserva su segunda vuelta y `prediction`/`reasoning` siguen llenándose (lo comprueba una prueba).

## Cómo probar

```bash
python3.12 -m venv .venv && .venv/bin/pip install langchain-core langgraph langchain-openai langchain-aws pydantic pyyaml pytest python-dotenv
cd apps/agent-host && ../../.venv/bin/python -m pytest tests -q        # 10 pruebas, sin red
cd ../.. && .venv/bin/python scripts/run_import_evaluation.py --offline  # referencia sin IA
LLM_PROVIDER=bedrock BEDROCK_MODEL_ID=<modelo> AWS_REGION=us-east-2 \
  .venv/bin/python scripts/run_import_evaluation.py                       # con agente
```

Referencia sin IA sobre los 12 casos sintéticos de `data/import_eval/cases.jsonl`: **50/59 columnas (85 %)**, 8/12 casos con nombre y precio correctos, decimal 12/12, moneda 12/12. Los fallos que quedan (títulos genéricos «Col1», «Precio A/B», «Valor») son justo los que el agente debe resolver; la medición con agente queda pendiente de credenciales.

## Hoja de ruta: habilidades que complementan

Ordenadas por valor para un comercio que carga su catálogo. Todas siguen el mismo patrón: diccionario o reglas primero, agente con salida `emit_*` y validador sin IA.

| Habilidad | Qué resuelve | Salida | Reutiliza |
|---|---|---|---|
| `extract_attributes` | Saca talla, color, presentación y gramaje del nombre («Franela Polo Roja L», «Harina PAN 1kg») y agrupa variantes de un mismo producto. Shopify, Walmart y Amazon normalizan variantes así. | `emit_attributes` con valores normalizados (color canónico, unidades) | Grafo, interceptor, evaluación |
| `entity_matching` en español | Detecta productos repetidos al importar o al unir con el catálogo del proveedor. Hace falta reescribir reglas y ejemplos para comercio venezolano (presentación y gramaje deciden, «bulto» ≠ unidad) y añadir un paso previo de candidatos (embeddings) para no comparar todo contra todo. | `emit_semantic_prediction` (ya existe) | Todo el flujo de la tesis |
| `receive_invoice` | Convierte la factura o nota de entrega del proveedor en una entrada de existencias: OCR, extracción de líneas, emparejamiento con productos y borrador para confirmar. Lightspeed lanzó algo equivalente en su POS. | `emit_receipt_lines` | OCR + `entity_matching` |
| `describe_product` | Redacta título limpio y descripción para la tienda en línea a partir del nombre y los atributos, con reglas de marca. | `emit_product_copy` | Catálogo de prompts |
| `photo_to_product` | Crea una ficha borrador desde una foto del producto (modelo con visión). | `emit_product_draft` | Grafo + validador |

Lo que no conviene hacer con IA: los límites de precio, las existencias y la validez de los códigos de barras (dígito de control GS1) son reglas deterministas.

## Privacidad y proveedor

El agente recibe títulos y unas filas de muestra, nunca el archivo completo. Para datos de comercios reales, usar Bedrock (los datos no salen de la cuenta de AWS) en lugar de un proveedor externo.
