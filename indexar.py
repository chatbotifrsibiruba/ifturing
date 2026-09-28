"""
indexar.py — Indexa os PDFs e salva o store.pkl

Execute este script UMA VEZ (ou toda vez que atualizar os PDFs):
    python indexar.py
"""

import os
import pickle
from pathlib import Path
import pdfplumber
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses import Document
from dotenv import load_dotenv

load_dotenv()


def extrair_tabelas_como_texto(caminho_pdf: str) -> list[str]:
    linhas: list[str] = []
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tabelas = page.extract_tables()
                    for tabela in tabelas:
                        if not tabela:
                            continue
                        cabecalho = tabela[0]
                        cols_validas = [c for c in cabecalho if c and str(c).strip()]
                        usar_cabecalho = len(cols_validas) >= 2

                        for linha in tabela[1:]:
                            if usar_cabecalho:
                                partes = []
                                for col, val in zip(cabecalho, linha):
                                    col_s = str(col).strip() if col else ""
                                    val_s = str(val).strip() if val else ""
                                    if col_s and val_s:
                                        partes.append(f"{col_s}: {val_s}")
                                if partes:
                                    linhas.append(" | ".join(partes))
                            else:
                                partes = [str(v).strip() for v in linha if v and str(v).strip()]
                                if partes:
                                    linhas.append(" | ".join(partes))
                except Exception as e:
                    print(f"⚠️  Aviso: erro ao processar página {i + 1} de {caminho_pdf}: {e}")
    except Exception as e:
        print(f"⚠️  Aviso: erro ao abrir {caminho_pdf} com pdfplumber: {e}")
    return linhas


def indexar_documentos(
    pasta_docs: str = "./documentos",
    pasta_faiss: str = "./faiss_index",
) -> dict:
    """
    Processa todos os PDFs de pasta_docs e gera o índice em pasta_faiss.
    Retorna {"n_arquivos": int, "n_chunks": int, "n_linhas_tabela": int}.
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

    # Extrai documentos de texto via PyPDFToDocument
    converter = PyPDFToDocument()
    text_docs = converter.run(sources=arquivos)["documents"]

    # Extrai linhas de tabelas de todos os PDFs
    table_docs: list[Document] = []
    n_linhas_tabela = 0
    for arq in arquivos:
        linhas = extrair_tabelas_como_texto(str(arq))
        n_linhas_tabela += len(linhas)
        for linha in linhas:
            table_docs.append(Document(
                content=linha,
                meta={"source": arq.name, "tipo": "tabela"},
            ))

    # Combina texto normal + linhas de tabela e passa pelo pipeline de limpeza/split/embed
    todos_docs = text_docs + table_docs

    document_store = InMemoryDocumentStore()

    cleaner  = DocumentCleaner()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=150,
        split_overlap=20,
    )
    embedder = SentenceTransformersDocumentEmbedder(
        model="intfloat/multilingual-e5-base"
    )
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    embedder.warm_up()

    cleaned  = cleaner.run(documents=todos_docs)
    split    = splitter.run(documents=cleaned["documents"])
    embedded = embedder.run(documents=split["documents"])
    writer.run(documents=embedded["documents"])

    all_documents = document_store.filter_documents()
    n_chunks = len(all_documents)

    store_path = Path(pasta_faiss) / "store.pkl"
    with open(store_path, "wb") as f:
        pickle.dump(all_documents, f)

    print(f"\n✅ Indexação concluída! {n_chunks} chunks salvos em {store_path}")
    print(f"📊 {n_linhas_tabela} linhas de tabela extraídas de {len(arquivos)} PDFs")

    return {"n_arquivos": len(arquivos), "n_chunks": n_chunks, "n_linhas_tabela": n_linhas_tabela}


if __name__ == "__main__":
    _pasta_docs  = os.getenv("PASTA_DOCS",  "./documentos")
    _pasta_faiss = os.getenv("PASTA_FAISS", "./faiss_index")
    resultado = indexar_documentos(_pasta_docs, _pasta_faiss)
    print(f"   Agora execute: streamlit run main.py")
