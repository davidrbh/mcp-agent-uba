'''
Orquestador del flujo cognitivo (LangGraph) - Versión Tesis UBA.

Este módulo define la máquina de estados que gobierna el comportamiento del agente,
incluyendo la clasificación de intención, la invocación de DeepSeek y la 
ejecución concurrente de herramientas MCP y evaluación semántica.
'''

import os
import re
import uuid
import logging
import asyncio
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from core.domain.state import AgentState

logger = logging.getLogger(__name__) 


def intent_classifier_node(state: AgentState) -> dict:
    '''Clasifica la intención del usuario para la Tesis de E-commerce.'''
    conversation_history = []
    
    # Si estamos inyectando productos directamente para evaluación en lote (Batch)
    if state.get("product_a") and state.get("product_b") and not state.get("messages"):
        return {"intent": "SEMANTIC_MATCHING"}
        
    for msg in state.get("messages", [])[-5:]:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        content = str(msg.content)[:200]
        conversation_history.append(f"{role}: {content}")
    
    context_str = "\n".join(conversation_history)

    prompt = ChatPromptTemplate.from_template(
        """Eres un clasificador de intenciones experto. Analiza la conversación y clasifica la ÚLTIMA petición en:
        - SEMANTIC_MATCHING: Peticiones para comparar productos, resolver entidades o limpiar catálogos.
        - GENERAL: Saludos o charla casual.
        
        Historial:
        {context}
        
        Responde ÚNICAMENTE con una palabra: SEMANTIC_MATCHING o GENERAL.
        """
    )
    
    llm = ChatOpenAI(
        model="deepseek-chat", 
        temperature=0, 
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com"
    )
    
    intent = "GENERAL"
    try:
        if context_str:
            chain = prompt | llm
            intent_raw = chain.invoke({"context": context_str}).content.upper()
            if "SEMANTIC" in intent_raw or "MATCH" in intent_raw: intent = "SEMANTIC_MATCHING"
    except Exception as e:
        logger.error(f"Error en clasificación de intención: {e}")
        
    logger.info(f"Intención detectada: {intent}")
    return {"intent": intent}


def parse_deepseek_xml(content: str) -> list:
    '''Parsea llamadas a herramientas en formato XML raw de DeepSeek (robusto a pipes).'''
    tool_calls = []
    invoke_pattern = r"<[|｜]DSML[|｜]invoke name=\"(.*?)\">(.*?)</[|｜]DSML[|｜]invoke>"
    invokes = re.findall(invoke_pattern, content, re.DOTALL)
    
    for name, body in invokes:
        args = {}
        param_pattern = r"<[|｜]DSML[|｜]parameter name=\"(.*?)\".*?>(.*?)</[|｜]DSML[|｜]parameter>"
        params = re.findall(param_pattern, body, re.DOTALL)
        for param_name, param_value in params:
            args[param_name] = param_value.strip()
            
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "name": name,
            "args": args,
            "type": "tool_call"
        })
    return tool_calls


def sanitize_history_for_llm(messages: List[BaseMessage]) -> List[BaseMessage]:
    '''Asegura que el historial de mensajes sea válido para la API de OpenAI/DeepSeek.'''
    sanitized = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, AIMessage) and msg.tool_calls:
            has_tool_response = False
            if i + 1 < len(messages) and isinstance(messages[i+1], ToolMessage):
                has_tool_response = True
            
            if not has_tool_response:
                logger.warning(f"Sanitizando AIMessage con tool_calls sin respuesta (ID: {msg.id})")
                new_msg = AIMessage(content=msg.content or "Procesando...", id=msg.id)
                sanitized.append(new_msg)
            else:
                sanitized.append(msg)
        else:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                text = "".join([b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text"])
                sanitized.append(ToolMessage(content=text, tool_call_id=msg.tool_call_id, name=msg.name))
            else:
                sanitized.append(msg)
        i += 1
    return sanitized


def agent_node(state: AgentState, llm_with_tools: dict, system_prompt: str) -> dict:
    '''Invoca al LLM con refuerzo de instrucciones y sanitización de historial.'''
    llm_runnable = llm_with_tools.get(state.get("intent", "GENERAL"), llm_with_tools["GENERAL"])

    full_system_content = system_prompt
    if state.get("intent") == "SEMANTIC_MATCHING":
        full_system_content += (
            "\n\n[INSTRUCCIÓN DINÁMICA]: Evalúa los productos usando tus herramientas MCP "
            "y finaliza obligatoriamente llamando a 'emit_semantic_prediction'."
        )

    # Si estamos en modo batch (Tesis), inyectamos los productos como si fueran un mensaje del usuario
    messages = state.get("messages", [])
    if not messages and state.get("product_a") and state.get("product_b"):
        messages = [HumanMessage(content=f"Compara estos productos:\nProducto A: {state['product_a']}\nProducto B: {state['product_b']}")]

    if messages and isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=full_system_content)] + messages[1:]
    else:
        messages = [SystemMessage(content=full_system_content)] + messages
        
    sanitized_messages = sanitize_history_for_llm(messages)
    
    try:
        response = llm_runnable.invoke(sanitized_messages)
    except Exception as e:
        logger.error(f"Error invocando LLM: {e}")
        return {"messages": [AIMessage(content="Error técnico al procesar solicitud.")], "error": str(e)}
    
    content_str = str(response.content)
    if not response.tool_calls and ("<|DSML|" in content_str or "<｜DSML｜" in content_str):
        parsed = parse_deepseek_xml(content_str)
        if parsed:
            response.tool_calls = parsed
            clean_content = re.sub(r"<[|｜]DSML[|｜].*?>", "", content_str, flags=re.DOTALL).strip()
            response.content = clean_content
            
    return {"messages": [response]}


def build_graph(
    tools: List[BaseTool], 
    system_prompt: str, 
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> StateGraph:
    '''Construye el grafo de ejecución semántica.'''
    tool_map = {tool.name: tool for tool in tools}
    
    llm = ChatOpenAI(
        model="deepseek-chat", 
        temperature=0,  # CRÍTICO PARA LA TESIS: Temperatura 0 para determinismo
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com"
    )
    
    llm_with_tools_map = {
        "SEMANTIC_MATCHING": llm.bind_tools(tools),
        "GENERAL": llm 
    }

    async def validated_tool_node(state: AgentState):
        '''Ejecuta herramientas y captura la predicción matemática para el State.'''
        last_message = state["messages"][-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        state_updates = {}

        async def run_single_tool(tool_call):
            name = tool_call["name"]
            args = tool_call["args"]
            tid = tool_call["id"]
            
            logger.info(f"Iniciando ejecución de herramienta: {name}")
            
            # --- INTERCEPTOR DE LA TESIS ---
            if name == "emit_semantic_prediction":
                state_updates["prediction"] = int(args.get("is_match", 0))
                state_updates["reasoning"] = str(args.get("reasoning", ""))

            # AQUÍ ESTÁ LA MAGIA: Usamos la lista 'tools' que le pasamos al build_graph
            # ¡Ya no llamamos al Container ni a la red MCP!
            tool = next((t for t in tools if t.name == name), None)
            
            if not tool:
                return ToolMessage(content=f"Error: Herramienta '{name}' no disponible.", tool_call_id=tid, name=name)

            try:
                # Ejecutamos la herramienta localmente
                output = await asyncio.wait_for(tool.ainvoke(args), timeout=70.0)
                return ToolMessage(content=str(output), tool_call_id=tid, name=name)
            except Exception as e:
                err_info = repr(e)
                logger.error(f"Fallo crítico en '{name}': {err_info}")
                return ToolMessage(content=f"❌ Error en '{name}': {err_info}", tool_call_id=tid, name=name)

        # Ejecución paralela
        results = await asyncio.gather(*(run_single_tool(tc) for tc in last_message.tool_calls))
        
        state_updates["messages"] = list(results)
        return state_updates

    workflow = StateGraph(AgentState)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("agent", lambda s: agent_node(state=s, llm_with_tools=llm_with_tools_map, system_prompt=system_prompt))
    workflow.add_node("tools", validated_tool_node)

    workflow.set_entry_point("intent_classifier")
    workflow.add_edge("intent_classifier", "agent")

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)