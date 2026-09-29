"""
debug_retrieval.py — Inspeciona os TOP_K documentos retornados pelo retriever
para uma pergunta específica, sem chamar o LLM.

Uso:
    python debug_retrieval.py
    python debug_retrieval.py --pergunta "sua pergunta aqui" --top_k 10
"""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from nucleo import PASTA_CHROMA, MODELO_EMB
from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

import re
_PADRAO_COTA = re.compile(r"C\d{1,2}:")

PERGUNTA_PADRAO = "Quais são os cursos técnicos disponíveis no campus Ibirubá?"


def carregar_pipeline(top_k: int):
    chroma_path = Path(PASTA_CHROMA)
    if not chroma_path.exists():
        print(f"ERRO: índice Chroma não encontrado em {chroma_path}")
        sys.exit(1)
    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=str(chroma_path))
    n = document_store.count_documents()
    print(f"[setup] {n} chunks no índice | top_k={top_k}\n")
    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=top_k))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    return pipeline


def inspecionar(pergunta: str, top_k: int):
    pipeline = carregar_pipeline(top_k)

    print(f"Pergunta: {pergunta!r}\n")
    print("=" * 70)

    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]

    for i, doc in enumerate(docs, 1):
        score = doc.score if doc.score is not None else 0.0
        content_preview = (doc.content or "")[:200].replace("\n", " ")
        meta = doc.meta or {}
        tipo = meta.get("tipo", "texto")
        source = meta.get("source", meta.get("file_path", "?"))
        n_cota = len(_PADRAO_COTA.findall(doc.content or ""))

        flag = ""
        if tipo == "texto" and n_cota >= 4:
            flag = "  ⚠️  DEVERIA TER SIDO FILTRADO (>=4 C\\d:)"
        elif tipo == "texto" and n_cota > 0:
            flag = f"  ⚠️  contém {n_cota} ocorrência(s) de C\\d: (abaixo do limiar=4)"

        print(f"[{i:02d}] score={score:.4f}  tipo={tipo}  source={source}{flag}")
        print(f"      {content_preview!r}")
        print()

    print("=" * 70)
    print(f"\nResumo: {len(docs)} docs retornados")
    print(f"  • tabela : {sum(1 for d in docs if (d.meta or {}).get('tipo') == 'tabela')}")
    print(f"  • texto  : {sum(1 for d in docs if (d.meta or {}).get('tipo') != 'tabela')}")
    n_contaminados = sum(1 for d in docs if _PADRAO_COTA.search(d.content or ""))
    print(f"  • com qualquer C\\d: no content : {n_contaminados}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pergunta", default=PERGUNTA_PADRAO)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    inspecionar(args.pergunta, args.top_k)


if __name__ == "__main__":
    main()
