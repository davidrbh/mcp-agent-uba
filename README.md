# MCP Semantic Orchestrator (Tesis UBA)

**Agente Orquestador para la Resolución Semántica de Productos en Comercio Electrónico**

_Una plataforma de Inteligencia Artificial Agéntica diseñada para resolver el problema de Entity Matching en catálogos heterogéneos, utilizando el Model Context Protocol (MCP) y flujos cognitivos deterministas._

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cognitive_Engine-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![MCP](https://img.shields.io/badge/MCP-Protocol_v1.0-green.svg)](https://modelcontextprotocol.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-F7931E.svg)](https://scikit-learn.org/)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Propósito del Proyecto

Este repositorio contiene el código fuente, la arquitectura y los scripts de validación empírica desarrollados como Trabajo de Grado para la **Maestría en Informática Aplicada, Mención Inteligencia Artificial**.

El sistema abandona los algoritmos de emparejamiento léxico tradicionales (como la distancia de Levenshtein) y propone un agente autónomo capaz de comprender el contexto técnico de los productos (hardware, voltajes, números de parte) para unificar catálogos de e-commerce con una precisión matemática.

## 🚀 Arquitectura y Capacidades

El proyecto implementa un diseño _Domain-Driven_ estructurado bajo el patrón **Hybrid Slice Architecture**:

- **🧠 Motor Semántico (LangGraph):** Orquestador de estados que evalúa la intención, maneja el historial y ejecuta herramientas MCP sin caer en bucles infinitos.
- **⚖️ Inferencia Determinista:** Configuración estricta de temperatura (T=0) y enrutamiento forzado (`emit_semantic_prediction`) para transformar el razonamiento de un LLM en salidas binarias (1 = Iguales, 0 = Diferentes).
- **🔌 Extensibilidad MCP:** Preparado para interactuar de forma segura con Sidecars (microservicios) para la obtención de contexto técnico externo mediante el estándar _Model Context Protocol_.
- **📊 Pipeline de Evaluación:** Scripts integrados para la extracción secuencial del _WDC Product Data Corpus_ (archivos de +3.9GB) y generación de métricas científicas (Matriz de Confusión, F1-Score).

## 🏗️ Estructura del Ecosistema

| Directorio                      | Rol            | Descripción                                                              |
| :------------------------------ | :------------- | :----------------------------------------------------------------------- |
| `apps/agent-host/src/core/`     | **Cerebro**    | Orquestador LangGraph (`graph.py`, `state.py`).                          |
| `apps/agent-host/src/features/` | **Dominio**    | Lógica de negocio para `entity_matching` y Prompts YAML.                 |
| `data/`                         | **Corpus**     | Almacenamiento local del dataset crudo y las muestras procesadas en CSV. |
| `scripts/`                      | **Validación** | Rutinas de experimentación cuantitativa y extracción de datos.           |

## 🚀 Guía de Reproducibilidad (Experimento Piloto)

Para auditar y reproducir los resultados obtenidos en la investigación:

### 1. Prerrequisitos

- Python 3.11+ y Poetry (Gestor de dependencias).
- Acceso a la API de DeepSeek.

### 2. Configuración del Entorno

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/dbarrera/mcp-agent-uba.git](https://github.com/dbarrera/mcp-agent-uba.git)
    cd mcp-agent-uba
    ```
2.  Instala las dependencias:
    ```bash
    poetry install
    ```
3.  Configura las variables de entorno (`.env`):
    ```env
    DEEPSEEK_API_KEY=tu_clave_aqui
    ```

### 3. Preparación de Datos (WDC Corpus)

Descarga el _Gold Standard_ (`gs_computers.txt`) y el _Training Corpus_ (`offers_english.json.gz`) desde la página oficial del WDC y colócalos en `data/raw/`.
Luego, extrae la muestra representativa:

```bash
poetry run python scripts/prepare_wdc_sample.py
```

### 4. Ejecución de la Inferencia

Lanza el agente orquestador en modo lote (Batch Processing) para evaluar los productos y generar el reporte estadístico:

```bash
poetry run python scripts/run_pilot_evaluation.py

```

_Salida esperada: Matriz de Confusión y Reporte de Clasificación (Precision, Recall, F1-Score)._
