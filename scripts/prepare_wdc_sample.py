"""
Script de Extracción de Datos - Tesis UBA
Extrae una muestra representativa de 200 pares del WDC Product Data Corpus
y los prepara en formato CSV para la evaluación del Agente MCP.
"""

import gzip
import json
from pathlib import Path

import pandas as pd


def extraer_valor(lista_propiedades, clave):
    """
    Función auxiliar para leer el extraño formato del WDC.
    Ej: [{"/name":"[ celular samsung ]"}] -> "celular samsung"
    """
    if not isinstance(lista_propiedades, list):
        return None

    for prop in lista_propiedades:
        if isinstance(prop, dict) and f"/{clave}" in prop:
            valor = prop[f"/{clave}"]
            # Limpiamos los corchetes y comillas raras del WDC
            if isinstance(valor, str):
                return valor.strip("[]\"' ")
    return None


def preparar_muestra_piloto():
    print("🚀 Iniciando el pipeline de preparación de datos del WDC...")

    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    gs_file = RAW_DIR / "gs_computers.txt"
    corpus_file = RAW_DIR / "offers_english.json.gz"
    output_file = PROCESSED_DIR / "muestra_piloto_200.csv"

    if not gs_file.exists():
        raise FileNotFoundError(f"Falta: {gs_file}")
    if not corpus_file.exists():
        raise FileNotFoundError(f"Falta: {corpus_file}")

    print("✅ Archivos encontrados. Leyendo el Gold Standard...")

    with open(gs_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    positives, negatives = [], []
    for line in lines:
        parts = line.strip().split("#####")
        if len(parts) == 3:
            id1, id2, label = parts
            id1, id2 = id1.strip(), id2.strip()

            if label == "1" and len(positives) < 100:
                positives.append((id1, id2, label))
            elif label == "0" and len(negatives) < 100:
                negatives.append((id1, id2, label))

            if len(positives) == 100 and len(negatives) == 100:
                break

    all_pairs = positives + negatives
    needed_ids = set([p[0] for p in all_pairs] + [p[1] for p in all_pairs])

    print(
        f"🔍 Buscando {len(needed_ids)} descripciones únicas en el corpus de 3.9GB..."
    )

    extracted_data = {}

    with gzip.open(corpus_file, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not needed_ids:
                break

            try:
                obj = json.loads(line)

                node_id = str(obj.get("nodeID", "")).strip()
                url = str(obj.get("url", "")).strip()
                current_id = f"{node_id} {url}"

                if current_id in needed_ids:
                    props = obj.get("schema.org_properties", [])
                    parent_props = obj.get("parent_schema.org_properties", [])

                    # 1. Intentamos sacar el nombre de la oferta
                    name = extraer_valor(props, "name")

                    # 2. Si no hay, intentamos sacar el nombre del producto padre
                    if not name:
                        name = extraer_valor(parent_props, "name")

                    # 3. Si no, a veces el padre usa 'title' (como vimos en tu JSON)
                    if not name:
                        name = extraer_valor(parent_props, "title")

                    if not name:
                        name = "PRODUCTO_SIN_NOMBRE_EN_JSON"

                    extracted_data[current_id] = str(name)
                    needed_ids.remove(current_id)

            except json.JSONDecodeError:
                continue

    if needed_ids:
        print(
            f"⚠️ Aún faltaron {len(needed_ids)} IDs (Probablemente no están en el subset English)."
        )

    print("🏗️ Construyendo el CSV final...")

    csv_data = []
    for id1, id2, label in all_pairs:
        name1 = extracted_data.get(id1, "SIN_DATOS")
        name2 = extracted_data.get(id2, "SIN_DATOS")
        csv_data.append({"title_1": name1, "title_2": name2, "label_real": int(label)})

    df = pd.DataFrame(csv_data)
    df["title_1"] = df["title_1"].replace(r"\n|\r", " ", regex=True)
    df["title_2"] = df["title_2"].replace(r"\n|\r", " ", regex=True)
    df.to_csv(output_file, index=False)

    print(f"🎉 ¡Dataset generado exitosamente en: {output_file}!")


if __name__ == "__main__":
    preparar_muestra_piloto()
