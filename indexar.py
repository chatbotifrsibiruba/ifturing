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
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from dotenv import load_dotenv

load_dotenv()

PASTA_DOCS  = os.getenv("PASTA_DOCS",  "./documentos")
PASTA_FAISS = os.getenv("PASTA_FAISS", "./faiss_index")

os.makedirs(PASTA_FAISS, exist_ok=True)

arquivos = list(Path(PASTA_DOCS).glob("*.pdf"))

if not arquivos:
    print(f"❌ Nenhum PDF encontrado em: {PASTA_DOCS}")
    print("   Coloque os PDFs na pasta 'documentos/' e rode novamente.")
    exit(1)

print(f"📄 {len(arquivos)} PDF(s) encontrado(s):")
for a in arquivos:
    print(f"   • {a.name}")

print("\n⏳ Indexando... (pode demorar alguns minutos na primeira vez)")

document_store = ChromaDocumentStore(persist_path=PASTA_FAISS)  # reaproveita a mesma env var

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

store_path = Path(PASTA_FAISS) / "store.pkl"
with open(store_path, "wb") as f:
    pickle.dump(all_documents, f)

print(f"\n✅ Indexação concluída! {len(all_documents)} chunks salvos em {store_path}")
print("   Agora execute: streamlit run app.py")
