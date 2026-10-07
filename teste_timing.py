"""
teste_timing.py — Script de profiling do pipeline RAG sem Streamlit.

Espelha nucleo.carregar_pipeline() (sem @st.cache_resource) e executa os
mesmos passos de retrieval que nucleo.responder(), na mesma ordem:
  [1] embed + retrieve semântico
  [2] pós-filtro de cotas
  [3] hybrid tabelas   (_buscar_tabelas_por_keyword)
  [4] hybrid cronograma (_buscar_cronograma_por_keyword)
  [5] montagem de contexto
  [6] LLM

Uso:
    python teste_timing.py
    python teste_timing.py --modelo qwen3:0.6b
    python teste_timing.py --pergunta "Quando são as provas?"
    DEBUG_CONTEXTO=1 python teste_timing.py --pergunta "Quais cursos superiores?"

Com DEBUG_CONTEXTO=1 imprime cada chunk na ordem exata em que vai para o
prompt, precedido de "--- CHUNK i (origem: semantico|tabela|cronograma) ---".
"""
import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from nucleo import (
    PASTA_CHROMA, MODELO_EMB, TOP_K, EMBEDDER_DEVICE,
    MODELOS, MODELO_PADRAO_SIMPLES,
    perguntar_llm,
    _OllamaOffline, _MSG_OLLAMA_OFFLINE,
    _PADRAO_COTA, _PADRAO_DATA_RECENTE,
    _buscar_tabelas_por_keyword,
    _buscar_cronograma_por_keyword,
)

_DEBUG_CONTEXTO = os.getenv("DEBUG_CONTEXTO", "").strip() not in ("", "0", "false", "no")

PERGUNTAS = [
    "Quais são os cursos técnicos disponíveis no campus Ibirubá?",
    "Qual é a duração dos cursos de Ciência da Computação e Matemática?",
    "Quais são os cursos superiores oferecidos no campus Ibirubá?",
]


def carregar_pipeline():
    """Espelha nucleo.carregar_pipeline() sem o @st.cache_resource do Streamlit."""
    chroma_path = Path(PASTA_CHROMA)
    if not chroma_path.exists():
        print(f"ERRO: índice Chroma não encontrado em {chroma_path}")
        print("Execute  python indexar.py  primeiro.")
        sys.exit(1)

    print(f"[setup] Abrindo ChromaDocumentStore em {chroma_path} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=str(chroma_path))
    n_chunks = document_store.count_documents()
    print(f"OK — {n_chunks} chunks em {time.perf_counter() - t0:.3f}s")

    print("[setup] Pré-carregando chunks de tabela e cronograma ...", end=" ", flush=True)
    t0 = time.perf_counter()
    todos = document_store.filter_documents()
    tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
    cronograma_docs = [
        d for d in todos
        if (d.meta or {}).get("tipo") != "tabela"
        and (
            "cronograma" in (d.content or "").lower()
            or len(_PADRAO_DATA_RECENTE.findall(d.content or "")) >= 3
        )
    ]
    print(f"OK — {len(tabela_docs)} tabelas, {len(cronograma_docs)} cronograma "
          f"em {time.perf_counter() - t0:.3f}s")

    print(f"[setup] EMBEDDER_DEVICE={EMBEDDER_DEVICE!r}")
    print("[setup] Construindo pipeline ...", end=" ", flush=True)
    t0 = time.perf_counter()
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(
            model=MODELO_EMB,
            device=ComponentDevice.from_str(EMBEDDER_DEVICE),
        ),
    )
    pipeline.add_component(
        "retriever",
        ChromaEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
    )
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    print(f"OK em {time.perf_counter() - t0:.3f}s")
    return pipeline, tabela_docs, cronograma_docs


def perguntar(pipeline, tabela_docs: list, cronograma_docs: list, pergunta: str, modelo_cfg: dict):
    """Replica os passos de nucleo.responder() expondo timings e debug de contexto."""

    # ── [1] embed + retrieve ─────────────────────────────────────────
    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
    t_retrieval = time.perf_counter() - t0
    print(f"[timing] embed+retrieve:      {t_retrieval:.3f}s  ({len(docs)} docs recuperados)")

    # ── [2] pós-filtro de cota ───────────────────────────────────────
    docs_antes = len(docs)
    docs = [
        d for d in docs
        if (d.meta or {}).get("tipo") == "tabela" or not _PADRAO_COTA.search(d.content or "")
    ]
    if len(docs) < docs_antes:
        print(f"[timing] pos_filtro_cota:     {docs_antes - len(docs)} chunk(s) removido(s)")

    # ── [3] hybrid tabelas ───────────────────────────────────────────
    t0_hib = time.perf_counter()
    docs_tabela_extra = _buscar_tabelas_por_keyword(pergunta, tabela_docs, docs)
    ids_tabela_extra = {d.id for d in docs_tabela_extra if d.id}

    if docs_tabela_extra:
        docs_tabela_ja   = [d for d in docs if (d.meta or {}).get("tipo") == "tabela"]
        docs_texto_final = [d for d in docs if (d.meta or {}).get("tipo") != "tabela"][:6]
        docs = docs_tabela_ja + docs_tabela_extra + docs_texto_final
        print(f"[timing] hibrido_tabela:      {time.perf_counter() - t0_hib:.6f}s  "
              f"({len(docs_tabela_extra)} tabelas injetadas, {len(docs)} docs total)")
    else:
        print(f"[timing] hibrido_tabela:      {time.perf_counter() - t0_hib:.6f}s  (0 injetados)")

    # ── [4] hybrid cronograma ────────────────────────────────────────
    t0_crono = time.perf_counter()
    docs_crono_extra = _buscar_cronograma_por_keyword(pergunta, cronograma_docs, docs)
    ids_crono_extra = {d.id for d in docs_crono_extra if d.id}

    if docs_crono_extra:
        docs = docs_crono_extra + docs
        print(f"[timing] hibrido_cronograma:  {time.perf_counter() - t0_crono:.6f}s  "
              f"({len(docs_crono_extra)} cronograma injetados, {len(docs)} docs total)")
    else:
        print(f"[timing] hibrido_cronograma:  {time.perf_counter() - t0_crono:.6f}s  (0 injetados)")

    if not docs:
        print("[timing] TOTAL (sem docs):    —")
        return

    # ── [5] montagem do contexto ─────────────────────────────────────
    t0 = time.perf_counter()
    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join(d.content for d in docs)
    t_ctx = time.perf_counter() - t0
    n_tabela = sum(1 for d in docs if (d.meta or {}).get("tipo") == "tabela")
    print(f"[timing] montagem_contexto:   {t_ctx:.6f}s  ({len(contexto)} chars, "
          f"{n_tabela} tabela(s) + {len(docs)-n_tabela} texto(s))")

    if _DEBUG_CONTEXTO:
        print()
        for i, d in enumerate(docs, 1):
            if d.id in ids_crono_extra:
                origem = "cronograma"
            elif d.id in ids_tabela_extra:
                origem = "tabela"
            else:
                origem = "semantico"
            print(f"--- CHUNK {i} (origem: {origem}) ---")
            print(d.content or "")
        print()

    # ── [6] LLM ──────────────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        r = perguntar_llm(pergunta, contexto, modelo_cfg)
    except _OllamaOffline:
        t_llm = time.perf_counter() - t1
        print(f"[timing] llm_total:           {t_llm:.3f}s  (FALHA — Ollama offline)")
        return

    t_llm = time.perf_counter() - t1
    print(f"[timing] llm_total:           {t_llm:.3f}s")
    print(f"[timing] TOTAL:               {t_retrieval + t_ctx + t_llm:.3f}s  ════════════════")
    print()
    print("Resposta:", r["texto"][:300], "..." if len(r["texto"]) > 300 else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--pergunta", default=None,
                        help="Roda apenas esta pergunta em vez da lista padrão")
    args = parser.parse_args()

    if args.modelo:
        modelo_cfg = {"modelo": args.modelo}
    else:
        modelo_cfg = MODELOS[MODELO_PADRAO_SIMPLES]

    print(f"[setup] Modelo : {modelo_cfg['modelo']}")
    print(f"[setup] Top-K  : {TOP_K}")
    print()

    pipeline, tabela_docs, cronograma_docs = carregar_pipeline()
    print()

    perguntas = [args.pergunta] if args.pergunta else PERGUNTAS

    for i, pergunta in enumerate(perguntas, 1):
        print(f"{'='*60}")
        print(f"PERGUNTA {i}: {pergunta}")
        print(f"{'='*60}")
        perguntar(pipeline, tabela_docs, cronograma_docs, pergunta, modelo_cfg)
        print()


if __name__ == "__main__":
    main()
