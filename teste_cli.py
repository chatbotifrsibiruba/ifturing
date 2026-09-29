"""
teste_cli.py — Testa o pipeline RAG direto no terminal, sem Streamlit.

Uso:
    python teste_cli.py
    python teste_cli.py --modelo "llama3:latest"
"""
import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Importa constantes e funções puras de nucleo (sem o carregar_pipeline decorado com
# @st.cache_resource, que só funciona dentro de uma sessão Streamlit)
from nucleo import (
    PASTA_CHROMA, MODELO_EMB, TOP_K,
    MODELOS, MODELO_PADRAO_SIMPLES,
    perguntar_llm,
    _OllamaOffline, _MSG_OLLAMA_OFFLINE,
)

from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever


def _carregar_pipeline():
    chroma_path = Path(PASTA_CHROMA)
    if not chroma_path.exists():
        print(f"ERRO: índice Chroma não encontrado em {chroma_path}")
        print("Execute  python indexar.py  primeiro.")
        sys.exit(1)

    print("Carregando índice...", end=" ", flush=True)
    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=str(chroma_path))
    n_chunks = document_store.count_documents()

    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component(
        "retriever",
        ChromaEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
    )
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    print(f"OK ({n_chunks} chunks)")
    return pipeline


def _perguntar(pipeline, pergunta: str, modelo_cfg: dict) -> dict:
    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
    t_retrieval = time.perf_counter() - t0

    if not docs:
        return {
            "resposta": "Nenhum trecho relevante encontrado nos documentos.",
            "t_retrieval": round(t_retrieval, 3),
            "t_llm": 0.0,
            "n_docs": 0,
            "score_max": 0.0,
            "tokens": "0→0",
        }

    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join(d.content for d in docs)

    t1 = time.perf_counter()
    try:
        r = perguntar_llm(pergunta, contexto, modelo_cfg)
    except _OllamaOffline:
        return {
            "resposta": _MSG_OLLAMA_OFFLINE,
            "t_retrieval": round(t_retrieval, 3),
            "t_llm": 0.0,
            "n_docs": len(docs),
            "score_max": round(max(scores), 4) if scores else 0.0,
            "tokens": "0→0",
        }
    t_llm = time.perf_counter() - t1

    return {
        "resposta": r["texto"],
        "t_retrieval": round(t_retrieval, 3),
        "t_llm": round(t_llm, 3),
        "n_docs": len(docs),
        "score_max": round(max(scores), 4) if scores else 0.0,
        "tokens": f"{r['tokens_entrada']}→{r['tokens_saida']}",
    }


def main():
    parser = argparse.ArgumentParser(description="Testa o RAG do IF Turing no terminal.")
    parser.add_argument(
        "--modelo",
        default=None,
        help="ID do modelo Ollama (ex: llama3:latest). Padrão: modelo simples configurado em nucleo.py",
    )
    args = parser.parse_args()

    if args.modelo:
        modelo_cfg = {"modelo": args.modelo}
        nome_modelo = args.modelo
    else:
        nome_modelo = MODELO_PADRAO_SIMPLES
        modelo_cfg = MODELOS[nome_modelo]

    print(f"Modelo : {modelo_cfg['modelo']}")
    print(f"Top-K  : {TOP_K}")
    print()

    pipeline = _carregar_pipeline()

    print("\nPipeline pronto. Digite 'sair' para encerrar.\n")
    print("-" * 60)

    while True:
        try:
            pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if pergunta.lower() in ("sair", "exit", "quit", "q"):
            print("Encerrando.")
            break
        if not pergunta:
            continue

        print()
        m = _perguntar(pipeline, pergunta, modelo_cfg)

        print(m["resposta"])
        print(
            f"\n[retrieval {m['t_retrieval']}s | LLM {m['t_llm']}s | "
            f"{m['n_docs']} trechos | score máx {m['score_max']} | tokens {m['tokens']}]"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
