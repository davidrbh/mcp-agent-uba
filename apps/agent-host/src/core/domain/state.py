"""
Definición del estado del agente para Resolución Semántica.

Este módulo define la estructura de datos que representa la 'Memoria de Trabajo'
del agente, fusionando la capacidad conversacional con las métricas 
necesarias para la evaluación del Trabajo de Grado.
"""

from typing import Any, Dict, TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    """
    Representa el Estado del Agente Orquestador MCP.
    Utilizamos total=False para permitir actualizaciones parciales del estado.
    """
    # ---------------------------------------------------------
    # 1. BASE CONVERSACIONAL (Mantiene vivas tus interfaces UI/API)
    # ---------------------------------------------------------
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str  # Ej: 'SEMANTIC_MATCHING', 'GENERAL_CHAT'
    
    # ---------------------------------------------------------
    # 2. ENTRADAS DE LA TESIS (Datos extraídos del WDC Corpus)
    # ---------------------------------------------------------
    product_a: str
    product_b: str
    
    # ---------------------------------------------------------
    # 3. CONTEXTO MCP (Memoria RAG)
    # ---------------------------------------------------------
    # Guarda la info técnica recuperada por tus herramientas MCP
    external_context: Annotated[List[str], operator.add] 
    
    # ---------------------------------------------------------
    # 4. SALIDAS DEL MODELO (Para la Matriz de Confusión)
    # ---------------------------------------------------------
    reasoning: str       # La cadena de pensamiento (Chain-of-Thought)
    prediction: int      # 1 si son el mismo producto, 0 si son diferentes
    
    # ---------------------------------------------------------
    # 5. CONTROL DE ERRORES
    # ---------------------------------------------------------
    error: str

    # ---------------------------------------------------------
    # 6. HABILIDADES INVOCADAS POR CÓDIGO (importador de catálogo, categorías…)
    # ---------------------------------------------------------
    task: str                 # Intención declarada; evita la llamada del clasificador
    task_input: str           # Entrada ya redactada para el modelo
    outputs: Dict[str, Any]   # Lo emitido por cada herramienta emit_* (nombre -> argumentos)