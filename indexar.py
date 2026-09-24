"""
indexar.py — Indexa os PDFs e salva o store.pkl

Execute este script UMA VEZ (ou toda vez que atualizar os PDFs):
    python indexar.py
"""

import os
import pickle
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from dotenv import load_dotenv

load_dotenv()


def indexar_documentos(
    pasta_docs: str = "./documentos",
    pasta_faiss: str = "./faiss_index",
) -> dict:
    """
    Processa todos os PDFs de pasta_docs e gera o índice em pasta_faiss.
    Retorna {"n_arquivos": int, "n_chunks": int}.
    Levanta exceção se não houver PDFs ou se algo falhar.
    """
    pasta_docs  = str(pasta_docs)
    pasta_faiss = str(pasta_faiss)

    os.makedirs(pasta_faiss, exist_ok=True)

    arquivos = list(Path(pasta_docs).glob("*.pdf"))

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {pasta_docs}\n"
            "Coloque os PDFs na pasta e rode novamente."
        )

    print(f"📄 {len(arquivos)} PDF(s) encontrado(s):")
    for a in arquivos:
        print(f"   • {a.name}")

    print("\n⏳ Indexando... (pode demorar alguns minutos na primeira vez)")

    document_store = InMemoryDocumentStore()

    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner",   DocumentCleaner())
    pipeline.add_component("splitter",  DocumentSplitter(
        split_by="word",
        split_length=150,
        split_overlap=20,
    ))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
        model="intfloat/multilingual-e5-base"
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner",   "splitter")
    pipeline.connect("splitter",  "embedder")
    pipeline.connect("embedder",  "writer")

    pipeline.run({"converter": {"sources": arquivos}})

    all_documents = document_store.filter_documents()
    n_chunks = len(all_documents)

    store_path = Path(pasta_faiss) / "store.pkl"
    with open(store_path, "wb") as f:
        pickle.dump(all_documents, f)

    print(f"\n✅ Indexação concluída! {n_chunks} chunks salvos em {store_path}")

    return {"n_arquivos": len(arquivos), "n_chunks": n_chunks}


if __name__ == "__main__":
    _pasta_docs  = os.getenv("PASTA_DOCS",  "./documentos")
    _pasta_faiss = os.getenv("PASTA_FAISS", "./faiss_index")
    resultado = indexar_documentos(_pasta_docs, _pasta_faiss)
    print(f"   Agora execute: streamlit run main.py")
