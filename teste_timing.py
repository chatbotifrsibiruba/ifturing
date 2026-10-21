"""
teste_timing.py — Script de profiling do pipeline RAG sem Streamlit.

Espelha nucleo.carregar_pipeline() (sem @st.cache_resource) e chama a MESMA função de retrieval de
nucleo.responder() (nucleo.recuperar_contexto — não há mais cópia do fluxo aqui):
  [1] embed + retrieve semântico (RERANKER_CANDIDATOS chunks se RERANKER_ATIVO=1)
  [1b] reranker opcional (linha "[timing] reranker: Xs (aplicado / NÃO aplicado)")
  [2] pós-filtro de cotas
  [3] hybrid tabelas   (_buscar_tabelas_por_keyword)
  [4] hybrid cronograma (_buscar_cronograma_por_keyword)
  [5] montagem de contexto
  [6] LLM
preaquecer() roda antes da 1ª pergunta (embedder e reranker já carregados nas medições).

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
    PASTA_CHROMA, MODELO_EMB, TOP_K, EMBEDDER_DEVICE, PREFIXO_CONSULTA,
    MODELOS, MODELO_PADRAO_SIMPLES,
    perguntar_llm,
    _OllamaOffline, _MSG_OLLAMA_OFFLINE,
    selecionar_cronograma,
    recuperar_contexto, preaquecer,
    RERANKER_ATIVO, RERANKER_MODELO, RERANKER_DEVICE, RERANKER_CANDIDATOS,
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
    cronograma_docs = selecionar_cronograma(todos)
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
            prefix=PREFIXO_CONSULTA,
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
    """Mesmo retrieval de nucleo.responder() (nucleo.recuperar_contexto), expondo timings e debug de contexto."""

    # ── [1]-[5] retrieval: o MESMO caminho de nucleo.responder() (busca, reranker opcional, pós-filtro de
    #    cotas, híbrido de tabelas e de cronograma, montagem do contexto) ──────────────────────────────
    rc = recuperar_contexto(pergunta, pipeline, tabela_docs, cronograma_docs)
    docs, contexto = rc["docs"], rc["contexto"]
    t_retrieval, t_ctx = rc["t_retrieval"] + rc["t_rerank"], rc["t_contexto"]
    ids_tabela_extra, ids_crono_extra = rc["ids_tabela_extra"], rc["ids_crono_extra"]

    if not docs:
        print("[timing] TOTAL (sem docs):    —")
        return

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
    print(f"[setup] RERANKER_ATIVO={int(RERANKER_ATIVO)}"
          + (f" modelo={RERANKER_MODELO} device={RERANKER_DEVICE} candidatos={RERANKER_CANDIDATOS}" if RERANKER_ATIVO else ""))
    print(f"[setup] preaquecer: {preaquecer(pipeline)}")
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
