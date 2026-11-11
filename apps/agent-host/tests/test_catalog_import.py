"""Pruebas de la habilidad de importación de catálogo (sin red ni credenciales)."""

import asyncio

from conftest import FailingLLM, StubLLM

from core.application.workflows.graph import build_graph
from features.catalog_import.dictionary import map_headers
from features.catalog_import.pipeline import categorize, propose_import_plan
from features.catalog_import.validator import validate_plan
from features.entity_matching.loader import emit_semantic_prediction

BRANCHES = ["Chacao", "Valencia"]
HEADERS = ["Ref", "Descripción art.", "Precio A", "Precio B", "Existencia Chacao", "Existencia Valencia"]
ROWS = [
    ["HP-001", "Harina PAN 1kg", "224,00", "283,50", "120", "30"],
    ["AV-1L", "Aceite Vatel 1L", "690,00", "780,00", "40", "0"],
]


def run(coro):
    return asyncio.run(coro)


def fields(result):
    return {i: c["field"] for i, c in result["columns"].items()}


def test_diccionario_reconoce_titulos_comunes():
    m = map_headers(["Código", "Artículo", "PVP Bs", "Exist.", "Departamento", "Cód. barras"])
    assert m == {0: "sku", 1: "nombre", 2: "precio", 3: "existencia", 4: "categoria", 5: "codigo_barras"}


def test_diccionario_no_asigna_dos_veces_un_campo_unico():
    m = map_headers(["Precio A", "Precio B"])
    assert m == {0: "precio"}


def test_sin_ambiguedad_no_se_llama_al_agente():
    headers = ["Código", "Artículo", "PVP", "Costo", "Existencia"]
    rows = [["A1", "Arroz", "1,45", "1,10", "4"]]
    r = run(propose_import_plan(headers, rows, BRANCHES, llm=FailingLLM()))
    assert r["agent_used"] is False
    assert fields(r) == {0: "sku", 1: "nombre", 2: "precio", 3: "costo", 4: "existencia"}
    assert r["decimal"] == "coma"


def test_agente_corrige_costo_y_precio_sin_depender_del_orden():
    plan = {
        "columns": [
            {"index": 3, "field": "precio"},  # primero el que necesita el campo libre
            {"index": 2, "field": "costo", "doubt": "Valores menores que 'Precio B': parece costo."},
            {"index": 4, "field": "existencia", "branch": "Chacao"},
            {"index": 5, "field": "existencia", "branch": "Valencia"},
        ],
        "decimal": "coma", "currency": "VES", "notes": "",
    }
    llm = StubLLM([[("emit_import_plan", plan)]])
    r = run(propose_import_plan(HEADERS, ROWS, BRANCHES, llm=llm))
    assert r["agent_used"] is True
    assert fields(r)[2] == "costo" and fields(r)[3] == "precio"
    assert r["columns"][2]["doubt"] and r["columns"][2]["source"] == "ia"
    assert r["columns"][4]["branch"] == "Chacao" and r["columns"][5]["branch"] == "Valencia"
    assert llm.calls == 1, "la herramienta terminal evita una segunda llamada al modelo"
    assert llm.bound_tool_choice == "emit_import_plan"


def test_agente_sin_explicacion_no_pisa_al_diccionario():
    plan = {"columns": [{"index": 0, "field": "nombre"}], "decimal": "coma", "currency": "VES"}
    r = run(propose_import_plan(HEADERS, ROWS, BRANCHES, llm=StubLLM([[("emit_import_plan", plan)]])))
    assert fields(r)[0] == "sku"


def test_si_el_agente_falla_queda_el_diccionario():
    r = run(propose_import_plan(HEADERS, ROWS, BRANCHES, llm=StubLLM([RuntimeError("timeout")])))
    assert fields(r)[1] == "nombre"
    assert any("diccionario" in i for i in r["issues"])


def test_validador_descarta_lo_que_no_cuadra():
    raw = {
        "columns": [
            {"index": 99, "field": "nombre"},
            {"index": 1, "field": "precio"},                       # texto, no números
            {"index": 3, "field": "precio"},
            {"index": 2, "field": "precio"},                       # repetido
            {"index": 4, "field": "existencia", "branch": "Marte"},  # sucursal inexistente
        ],
        "decimal": "coma", "currency": "VES",
    }
    plan, issues = validate_plan(raw, HEADERS, ROWS, BRANCHES)
    assert [(c.index, c.field) for c in plan.columns] == [(3, "precio"), (4, "existencia")]
    assert plan.columns[1].branch is None
    assert len(issues) == 3


def test_validador_rechaza_plan_mal_formado():
    plan, issues = validate_plan({"columns": [{"index": 0, "field": "inventado"}]}, HEADERS, ROWS, BRANCHES)
    assert plan is None and issues


def test_categorias_solo_de_la_lista_cerrada():
    cats = [["viveres", "Víveres"], ["bebidas", "Bebidas"], ["otros", "Otros"]]
    llm = StubLLM([[("emit_category_plan", {"assignments": [{"i": 0, "slug": "viveres"}, {"i": 1, "slug": "inventada"}]})]])
    r = run(categorize(["Harina PAN 1kg", "Cosa rara"], cats, llm=llm))
    assert r[0] == {"slug": "viveres", "source": "ia"}
    assert r[1] == {"slug": "otros", "source": "none"}


def test_la_tesis_sigue_funcionando_igual():
    llm = StubLLM([[("emit_semantic_prediction", {"reasoning": "mismo modelo", "is_match": 1})]])
    graph = build_graph(tools=[emit_semantic_prediction], system_prompt="x", llm=llm)
    final = run(graph.ainvoke({"product_a": "a", "product_b": "b", "messages": []}))
    assert final["prediction"] == 1
    assert final["outputs"]["emit_semantic_prediction"]["is_match"] == 1
    assert llm.calls == 2, "sin herramientas terminales, la tesis conserva su segunda vuelta"
