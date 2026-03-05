"""
Script de Evaluación Cuantitativa - Agente Orquestador MCP.
Ejecuta la muestra piloto del WDC Corpus, genera la Matriz de Confusión,
y exporta un reporte de Análisis de Errores.
"""

import sys
import asyncio
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "apps" / "agent-host" / "src"))

from core.application.workflows.graph import build_graph
from features.entity_matching.loader import get_tools, get_semantic_system_prompt

async def run_evaluation():
    print("🚀 Iniciando Evaluación Piloto y Análisis de Errores...")
    csv_path = PROJECT_ROOT / "data" / "processed" / "muestra_piloto_200.csv"
    errores_path = PROJECT_ROOT / "data" / "processed" / "analisis_errores_piloto.csv"
    
    if not csv_path.exists():
        print(f"❌ Error: No se encontró el dataset en {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Dataset cargado con {len(df)} pares de productos.")

    tools = await get_tools()
    system_prompt = get_semantic_system_prompt(channel="batch_processing")
    agent_app = build_graph(tools=tools, system_prompt=system_prompt, checkpointer=None)

    y_true = []
    y_pred = []
    errores_detectados = []

    # SEMÁFORO: Controla cuántos productos procesamos en paralelo (ajustar según Rate Limit de DeepSeek)
    semaphore = asyncio.Semaphore(10) 

    async def process_item(index, row):
        async with semaphore:
            prod_a = row['title_1']
            prod_b = row['title_2']
            label = int(row['label_real'])
            
            initial_state = {
                "product_a": prod_a,
                "product_b": prod_b,
                "messages": [] 
            }

            try:
                # El grafo ya es asíncrono, lo aprovechamos al máximo
                final_state = await agent_app.ainvoke(initial_state)
                pred = int(final_state.get("prediction", 0))
                reasoning = final_state.get("reasoning", "Sin razonamiento")
            except Exception as e:
                pred = 0
                reasoning = f"Error técnico: {str(e)}"
            
            status = "✅ ACIERTO" if label == pred else "❌ FALLO"
            print(f"[{index+1}/{len(df)}] {status} | Real: {label} | Pred: {pred}")
            
            return {
                "label": label,
                "pred": pred,
                "error": {
                    "id_fila": index + 1,
                    "producto_a": prod_a,
                    "producto_b": prod_b,
                    "etiqueta_real": label,
                    "prediccion_agente": pred,
                    "tipo_de_error": "Falso Positivo" if (pred == 1 and label == 0) else "Falso Negativo",
                    "razonamiento_llm": reasoning
                } if label != pred else None
            }

    print(f"🧠 Iniciando inferencia semántica concurrente (Lote de {len(df)})...")
    
    # Creamos todas las tareas y las ejecutamos en paralelo
    tasks = [process_item(index, row) for index, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    # Recomponemos los resultados para las métricas
    for res in results:
        y_true.append(res["label"])
        y_pred.append(res["pred"])
        if res["error"]:
            errores_detectados.append(res["error"])
            
    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES DE LA MATRIZ DE CONFUSIÓN")
    print("="*50)
    
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, digits=4))

    # NUEVO: Exportar los errores a un CSV
    if errores_detectados:
        df_errores = pd.DataFrame(errores_detectados)
        df_errores.to_csv(errores_path, index=False)
        print("\n" + "="*50)
        print(f"🕵️‍♂️ ¡ANÁLISIS DE ERRORES EXPORTADO!")
        print(f"Se encontraron {len(errores_detectados)} fallos.")
        print(f"Revisa el archivo: {errores_path}")
        print("Lee la columna 'razonamiento_llm' para entender por qué falló.")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(run_evaluation())