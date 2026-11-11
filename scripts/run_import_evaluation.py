"""
Evaluación de la habilidad de importación de catálogo.

Mide, sobre casos con respuesta conocida (data/import_eval/cases.jsonl), cuántas
columnas empareja bien:
  - solo con el diccionario (--offline, sin IA, siempre reproducible), o
  - con diccionario + agente (requiere LLM_PROVIDER y sus credenciales).

Exporta los errores con el razonamiento del agente, igual que la evaluación
de la tesis, para entender por qué falló.

Uso:
    python scripts/run_import_evaluation.py --offline
    LLM_PROVIDER=bedrock BEDROCK_MODEL_ID=... python scripts/run_import_evaluation.py
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "apps" / "agent-host" / "src"))

from features.catalog_import.pipeline import propose_import_plan  # noqa: E402

CASES = PROJECT_ROOT / "data" / "import_eval" / "cases.jsonl"
ERRORS = PROJECT_ROOT / "data" / "import_eval" / "errores.csv"


async def evaluate(use_agent: bool) -> None:
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = ok = 0
    required_ok = price_swaps = decimal_ok = currency_ok = branch_ok = branch_total = agent_calls = 0
    errors = []
    semaphore = asyncio.Semaphore(5)

    async def one(case):
        async with semaphore:
            return case, await propose_import_plan(case["headers"], case["rows"], case.get("branches", []), use_agent=use_agent)

    for case, res in await asyncio.gather(*(one(c) for c in cases)):
        got = {str(i): c["field"] for i, c in res["columns"].items()}
        accept = case.get("accept", {})
        agent_calls += int(res["agent_used"])
        case_required = True
        for idx, expected in case["expected"].items():
            total += 1
            field = got.get(idx, "ignorar" if expected == "ignorar" else "(sin asignar)")
            good = field == expected or field in accept.get(idx, [])
            ok += good
            if expected in ("nombre", "precio") and not good:
                case_required = False
            if {expected, field} == {"precio", "costo"}:
                price_swaps += 1
            if not good:
                col = res["columns"].get(int(idx), {})
                errors.append({
                    "caso": case["id"], "columna": case["headers"][int(idx)], "esperado": expected,
                    "obtenido": field, "fuente": col.get("source", "-"), "duda": col.get("doubt") or "",
                    "notas_agente": res.get("notes", ""),
                })
        required_ok += case_required
        decimal_ok += res["decimal"] == case["decimal"]
        currency_ok += res["currency"] == case["currency"]
        for idx, br in case.get("expected_branches", {}).items():
            branch_total += 1
            branch_ok += (res["columns"].get(int(idx)) or {}).get("branch") == br

    n = len(cases)
    mode = "diccionario + agente" if use_agent else "solo diccionario"
    print("=" * 56)
    print(f"EVALUACIÓN DEL IMPORTADOR · {mode} · {n} casos")
    print("=" * 56)
    print(f"Columnas bien emparejadas:      {ok}/{total} ({ok / total:.0%})")
    print(f"Casos con nombre y precio bien: {required_ok}/{n}")
    print(f"Costo y precio confundidos:     {price_swaps}")
    print(f"Separador decimal correcto:     {decimal_ok}/{n}")
    print(f"Moneda correcta:                {currency_ok}/{n}")
    if branch_total:
        print(f"Sucursal de existencia:         {branch_ok}/{branch_total}")
    print(f"Llamadas al agente:             {agent_calls}/{n}")

    with ERRORS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["caso", "columna", "esperado", "obtenido", "fuente", "duda", "notas_agente"])
        w.writeheader()
        w.writerows(errors)
    print(f"\nErrores exportados: {ERRORS.relative_to(PROJECT_ROOT)} ({len(errors)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="solo diccionario, sin llamar al modelo")
    asyncio.run(evaluate(use_agent=not parser.parse_args().offline))
