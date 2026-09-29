"""
teste_timing.py — Script de profiling do pipeline RAG sem Streamlit.

Carrega o índice, roda as perguntas padrão e imprime o breakdown completo
de tempo de cada etapa via os [timing] logs instrumentados em nucleo.py.

Uso:
    python teste_timing.py
    python teste_timing.py --modelo qwen3:0.6b
"""
import sys
import time
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from nucleo import (
    PASTA_FAISS, MODELO_EMB, TOP_K,
    MODELOS, MODELO_PADRAO_SIMPLES,
    perguntar_llm,
    _OllamaOffline, _MSG_OLLAMA_OFFLINE,
)
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

PERGUNTAS = [
    "Quais são os cursos técnicos disponíveis no campus Ibirubá?",
    "Qual é a duração dos cursos de Ciência da Computação e Matemática?",
    "Quais são os cursos superiores oferecidos no campus Ibirubá?",
]


def carregar_pipeline():
    store_path = Path(PASTA_FAISS) / "store.pkl"
    if not store_path.exists():
        print(f"ERRO: índice não encontrado em {store_path}")
        print("Execute  python indexar.py  primeiro.")
        sys.exit(1)

    print(f"[setup] Carregando índice de {store_path} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    with open(store_path, "rb") as f:
        documentos = pickle.load(f)
    t_pickle = time.perf_counter() - t0
    print(f"OK — {len(documentos)} chunks em {t_pickle:.3f}s")

    print("[setup] Construindo InMemoryDocumentStore + pipeline ...", end=" ", flush=True)
    t0 = time.perf_counter()
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documentos)
    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
    )
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    t_pipeline = time.perf_counter() - t0
    print(f"OK em {t_pipeline:.3f}s")
    return pipeline


def perguntar(pipeline, pergunta: str, modelo_cfg: dict):
    # ── embed + retrieve ─────────────────────────────────────────────
    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
    t_retrieval = time.perf_counter() - t0
    print(f"[timing] embed+retrieve:      {t_retrieval:.3f}s  ({len(docs)} docs recuperados)")

    if not docs:
        print("[timing] TOTAL (sem docs):    —")
        return

    # ── montagem do contexto ─────────────────────────────────────────
    t0 = time.perf_counter()
    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join(d.content for d in docs)
    t_ctx = time.perf_counter() - t0
    print(f"[timing] montagem_contexto:   {t_ctx:.6f}s  ({len(contexto)} chars)")

    # ── prompt + Ollama (detalhado dentro de perguntar_llm/chamar_ollama) ─
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
    args = parser.parse_args()

    if args.modelo:
        modelo_cfg = {"modelo": args.modelo}
    else:
        modelo_cfg = MODELOS[MODELO_PADRAO_SIMPLES]

    print(f"[setup] Modelo : {modelo_cfg['modelo']}")
    print(f"[setup] Top-K  : {TOP_K}")
    print()

    pipeline = carregar_pipeline()
    print()

    for i, pergunta in enumerate(PERGUNTAS, 1):
        print(f"{'='*60}")
        print(f"PERGUNTA {i}: {pergunta}")
        print(f"{'='*60}")
        perguntar(pipeline, pergunta, modelo_cfg)
        print()


if __name__ == "__main__":
    main()
