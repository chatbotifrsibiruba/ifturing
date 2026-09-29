"""
indexar.py — Indexa os PDFs e salva o store.pkl

Execute este script UMA VEZ (ou toda vez que atualizar os PDFs):
    python indexar.py
"""

import os
import re
import shutil
from pathlib import Path
import pdfplumber
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses import Document
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from dotenv import load_dotenv

load_dotenv()


_LABELS = ("Curso", "Turnos", "Duração (semestres)", "Total de Vagas")

# Cabeçalho de seção Ibirubá: "Campus Ibirubá:" (cota) ou "Campus Ibirubá (" (resumo)
_RE_CAMPUS_IBIRUBA = re.compile(r"campus\s+ibirub[aá]\s*[:(]", re.IGNORECASE)
# Cabeçalho de qualquer outro campus — mesma dualidade de separador
_RE_CAMPUS_OUTRO   = re.compile(r"campus\s+\w+(?:\s+\w+)*\s*[:(]", re.IGNORECASE)


def extrair_tabelas_como_texto(caminho_pdf: str) -> list[str]:
    """Extrai linhas de tabela de um PDF como texto estruturado.

    Usa posição fixa das 4 primeiras colunas (Curso, Turnos, Duração,
    Total de Vagas). Colunas 4+ (cotas C1-C10) são sempre descartadas.

    Filtro de campus: só inclui tabelas de páginas dentro da seção do
    Campus Ibirubá. Ativa ao encontrar 'Campus Ibirubá:' e desativa ao
    encontrar o cabeçalho de qualquer outro campus. Continuações de tabela
    em páginas sem cabeçalho explícito são mantidas enquanto o estado estiver
    ativo.
    """
    linhas: list[str] = []
    n_mantidas = 0
    n_descartadas = 0
    ibiruba_ativo = False
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    texto = page.extract_text() or ""

                    if _RE_CAMPUS_IBIRUBA.search(texto):
                        ibiruba_ativo = True
                    elif _RE_CAMPUS_OUTRO.search(texto):
                        ibiruba_ativo = False

                    tabelas = page.extract_tables()
                    if not tabelas:
                        continue

                    if not ibiruba_ativo:
                        n_descartadas += len(tabelas)
                        continue

                    n_mantidas += len(tabelas)
                    for tabela in tabelas:
                        if not tabela:
                            continue
                        ultimo_col0 = ""  # resetado a cada tabela
                        for linha in tabela[1:]:  # tabela[0] é sempre o cabeçalho
                            if not linha:
                                continue

                            col0_raw = " ".join(str(linha[0]).replace("\n", " ").split()) if linha[0] else ""

                            # Verifica se há dados nas colunas 1-3 (Turnos, Duração, Vagas)
                            tem_dados = any(
                                linha[idx] and str(linha[idx]).strip()
                                for idx in range(1, min(4, len(linha)))
                            )

                            if col0_raw:
                                # Col 0 preenchida: atualiza referência e usa diretamente
                                ultimo_col0 = col0_raw
                                col0 = col0_raw
                            elif tem_dados:
                                # Col 0 vazia mas há dados → célula mesclada, herda último curso
                                col0 = ultimo_col0
                            else:
                                # Col 0 vazia e sem dados → sub-cabeçalho ou linha vazia, descarta
                                continue

                            if not col0:
                                continue

                            # Pula rodapés: começa com "Observação" ou texto corrido (>15 palavras)
                            if col0.lower().startswith("observa") or len(col0.split()) > 15:
                                continue

                            partes = []
                            for idx, label in enumerate(_LABELS):
                                if idx >= len(linha):
                                    break
                                val = linha[idx]
                                val_s = " ".join(str(val).replace("\n", " ").split()) if val else ""
                                if val_s:
                                    partes.append(f"{label}: {val_s}")
                                elif idx == 0:
                                    # Garante que o nome do curso (herdado) sempre aparece
                                    partes.append(f"{label}: {col0}")
                            if partes:
                                linhas.append(" | ".join(partes))
                except Exception as e:
                    print(f"⚠️  Aviso: erro ao processar página {i + 1} de {caminho_pdf}: {e}")
    except Exception as e:
        print(f"⚠️  Aviso: erro ao abrir {caminho_pdf} com pdfplumber: {e}")

    nome = Path(caminho_pdf).name
    print(f"   📋 {nome}: {n_mantidas} tabela(s) Ibirubá mantida(s), "
          f"{n_descartadas} de outros campi descartada(s)")
    return linhas


def indexar_documentos(
    pasta_docs: str = "./documentos",
    pasta_chroma: str = "./chroma_data",
) -> dict:
    """
    Processa todos os PDFs de pasta_docs e persiste o índice em pasta_chroma
    via ChromaDocumentStore (embedded, sem servidor).
    Retorna {"n_arquivos": int, "n_chunks": int, "n_linhas_tabela": int}.
    Levanta exceção se não houver PDFs ou se algo falhar.
    """
    pasta_docs   = str(pasta_docs)
    pasta_chroma = str(pasta_chroma)

    # Remove índice anterior para garantir reindexação limpa
    chroma_path = Path(pasta_chroma)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f"   🗑️  Índice anterior em {pasta_chroma} removido")
    chroma_path.mkdir(parents=True, exist_ok=True)

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

    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=pasta_chroma)

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

    # Documentos de texto normal: limpar + split + filtra chunks de cotas
    _PADRAO_COTA = re.compile(r"C\d{1,2}:")
    cleaned_text     = cleaner.run(documents=text_docs)
    split_text       = splitter.run(documents=cleaned_text["documents"])
    chunks_texto_raw = split_text["documents"]
    chunks_texto = [
        c for c in chunks_texto_raw
        if len(_PADRAO_COTA.findall(c.content or "")) < 4
    ]
    n_descartados = len(chunks_texto_raw) - len(chunks_texto)
    if n_descartados:
        print(f"   🗑️  {n_descartados} chunk(s) de texto descartado(s) (parágrafos de cotas)")

    # Documentos de tabela: apenas limpar, cada linha já é um chunk final
    cleaned_table = cleaner.run(documents=table_docs)
    chunks_tabela = cleaned_table["documents"]

    # Chroma não suporta listas/dicts em metadata — serializa _split_overlap para string
    todos_chunks = chunks_texto + chunks_tabela
    for doc in todos_chunks:
        if "_split_overlap" in (doc.meta or {}):
            doc.meta["_split_overlap"] = str(doc.meta["_split_overlap"])
    embedded = embedder.run(documents=todos_chunks)
    writer.run(documents=embedded["documents"])

    n_chunks = document_store.count_documents()

    n_chunks_texto  = len(chunks_texto)
    n_chunks_tabela = len(chunks_tabela)
    print(f"\n✅ Indexação concluída! {n_chunks} chunks persistidos em {pasta_chroma}/")
    print(f"   • Chunks de texto normal : {n_chunks_texto}")
    print(f"   • Chunks de tabela       : {n_chunks_tabela} ({n_linhas_tabela} linhas extraídas)")

    return {"n_arquivos": len(arquivos), "n_chunks": n_chunks, "n_linhas_tabela": n_linhas_tabela}


if __name__ == "__main__":
    _pasta_docs   = os.getenv("PASTA_DOCS",   "./documentos")
    _pasta_chroma = os.getenv("PASTA_CHROMA", "./chroma_data")
    resultado = indexar_documentos(_pasta_docs, _pasta_chroma)
    print(f"   Agora execute: streamlit run main.py")
