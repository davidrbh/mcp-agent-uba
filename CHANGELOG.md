# Changelog

All notable changes to the **SQL Agent OSS** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🧩 Habilidades de catálogo (importación y categorías)

- **`import_mapping`**: empareja las columnas de un inventario con los campos del importador de catálogo. Diccionario primero, agente con `emit_import_plan` solo si hace falta, validador sin IA y fusión que no pisa al diccionario sin una duda explicada.
- **`categorize`**: asigna categorías de una lista cerrada por tandas; lo inventado o no ubicado queda en `otros`.
- **Fábrica de LLM** (`infra/llm/factory.py`): DeepSeek o Bedrock por `LLM_PROVIDER`; el grafo ya no tiene el proveedor escrito.
- **Grafo**: modelo inyectable, `forced_tool`, `terminal_tools`, `task`/`task_input` y `outputs` genéricos para cualquier `emit_*`. La tesis conserva su comportamiento.
- **Evaluación** (`scripts/run_import_evaluation.py`) con 12 casos sintéticos y **pruebas** (`apps/agent-host/tests`, 10, sin red).

## [v2.2.0] - 2026-01-11

### 🚀 WhatsApp Integration & Memory Enhancements

- **Migración de Evolution API a WAHA**:
  - Reemplazado Evolution API con WAHA (WhatsApp HTTP API) para mayor estabilidad y ligereza.
  - Actualizada arquitectura de contenedores: `waha` (motor WhatsApp) y `agent-bridge` (FastAPI intermedio).
  - Mejorada configuración de sesiones, QR scanning y webhooks.
- **Memoria de Conversaciones Persistente**:
  - Implementado LangGraph MemorySaver para mantener contexto entre mensajes en WhatsApp.
  - Agregado reset de estado de trabajo para evitar contaminación entre consultas.
  - Inyección de historial de chat en prompts de SQL para mayor precisión (e.g., referencias como "y los activos?").
- **Indicadores de Escritura (Typing)**:
  - Agregado soporte para "escribiendo..." en WhatsApp durante procesamiento del agente.
  - Mejora la experiencia de usuario simulando respuestas humanas.
- **Filtros de Seguridad**:
  - Bloqueo de mensajes de status@broadcast para evitar respuestas automáticas a stories.
  - Webhook protegido con secrets para mayor seguridad.

### ✨ New Features

- **Self-Healing Contextual**: El generador SQL ahora recibe historial de conversación para resolver ambigüedades.
- **Optimización de Respuestas**: Truncado de datos largos en prompts para reducir latencia del LLM.

### 📚 Documentation

- Reescrita `docs/05_whatsapp_integration.md` para WAHA.
- Actualizado `README.md` con nueva característica de WhatsApp y roadmap.
- Agregadas referencias cruzadas en `overview.md` e `infrastructure_spec.md`.

## [v2.1.0] - 2026-01-10

### 🚀 Major Performance & Architecture Updates ("Fast Agent")

- **Singleton Pattern Implementation**:
  - Moved tool loading and graph compilation to `__init__`.
  - Reduced agent initialization time from ~3s to <0.1s per request.
  - Implemented thread-safe reuse of the compilation scope.
- **API "Light Mode"**:
  - Replaced heavy LangChain Toolkit loading with a text-based "Swagger Summary" cache.
  - Reduces LLM token consumption by avoiding full JSON schema injection for every query.
- **AsyncIO Core**:
  - Full restructuring of the `Graph` and `Nodes` to operate natively with `async/await`.
  - Fixed `RuntimeError: Event loop is closed` by using `atexit` and explicit session management.

### ✨ New Features

- **Self-Healing SQL Loop**:
  - The agent now captures `pymysql.err` exceptions.
  - Inject DB error messages back into the prompt for an automatic "Constraint/Syntax Correction" retry pass.
- **Smart URL Rewriting**:
  - Added `BaseUrlRequestsWrapper` middleware.
  - Automatically prepends the base domain (e.g., `https://api.myapp.com`) to relative paths (e.g., `/users`) hallucinated by LLMs.
- **Semantic Layer v2.5**:
  - Introduced `business_context.yaml` for defining Logical Busines Models over raw physical tables.

### 🐛 Bug Fixes

- Fixed SSL certificate verification warnings for local development environments.
- Resolved inheritance issues in `RequestsToolkit` by defining a proper Pydantic subclass wrapper.
- Corrected database connection leakage in `generate_dictionary.py`.

### 📚 Documentation

- Updated `01_architecture/` specs to reflect the Hybrid Router and Light Mode.
- Documented the distinction between "Physical Schema" and "Logical Models" in `semantic_spec.md`.

## [v1.0.0] - Initial Release

- Basic Text-to-SQL functionality.
- Simple LangGraph implementation.
- Support for MySQL and PostgreSQL.
