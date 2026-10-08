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
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses import Document
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from dotenv import load_dotenv

load_dotenv()


_LABELS = ("Curso", "Turnos", "Duração (semestres)", "Total de Vagas")

# Cabeçalho de seção Ibirubá: "Campus Ibirubá:" (corpo/Annexo 2) ou "Campus Ibirubá (" (Annexo 1)
_RE_CAMPUS_IBIRUBA = re.compile(r"campus\s+ibirub[aá]\s*[:(]", re.IGNORECASE)

# Nomes dos campi do IFRS que NÃO são Ibirubá — usados nos regex abaixo.
# Usar nomes específicos (em vez de \w+ genérico) evita falsos positivos como
# "campus Ibirubá (RS)", "campus Osório é na modalidade EaD (EaD)", e
# "Instituto Federal...do Rio Grande do Sul".
_OUTROS_CAMPI_NOMES = (
    r"alvorada|bento\s+gon[cç]alves|canoas|caxias\s+do\s+sul|erechim|"
    r"farroupilha|feliz|os[oó]rio|porto\s+alegre|restinga|rio\s+grande|"
    r"rolante|sert[aã]o|vacaria|veran[oó]polis|viam[aã]o"
)

# Para páginas de TEXTO: detecta cabeçalho de seção de outro campus nos dois formatos —
#   "Campus Farroupilha: A descrição da tabela..."  → usa ':'
#   "Campus Restinga (a descrição das vagas..."     → usa '('
# Usar nomes específicos evita os falsos positivos que o padrão genérico \w+ causava.
_RE_CAMPUS_OUTRO_TEXTO = re.compile(
    rf"campus\s+(?:{_OUTROS_CAMPI_NOMES})\s*[:(]",
    re.IGNORECASE,
)

# Para páginas de TABELA: mantido com o mesmo comportamento anterior (nomes específicos
# garantem a mesma precisão que _RE_CAMPUS_OUTRO_TEXTO).
_RE_CAMPUS_OUTRO_TABELA = re.compile(
    rf"campus\s+(?:{_OUTROS_CAMPI_NOMES})\s*[:(]",
    re.IGNORECASE,
)

# Cabeçalho institucional repetido em todas as páginas do PDF.
# Removido do texto extraído para não poluir os chunks.
_RE_CABECALHO_PAGINA = re.compile(
    r"MINIST[EÉ]RIO\s+DA\s+EDUCA[CÇ][AÃ]O\s*\n"
    r"Secretaria\s+de\s+Educa[cç][aã]o\s+Profissional\s+e\s+Tecnol[oó]gica\s*\n"
    r"Instituto\s+Federal\s+de\s+Educa[cç][aã]o[^\n]*\n"
    r"(?:Gabinete\s+do\s+Reitor[^\n]*\n?)?",
    re.IGNORECASE,
)

# Nomes de outros campi para auditoria pós-indexação (busca inline, sem prefixo "campus").
_RE_OUTROS_NOMES_AUDIT = re.compile(
    rf"\b(?:{_OUTROS_CAMPI_NOMES})\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# ATENÇÃO: atualizar este conjunto a cada novo edital do IFRS – Campus Ibirubá.
# Serve como rede de segurança (Opção C) contra falsos positivos do detector de
# campus por regex: qualquer linha de tabela extraída cujo campo 'Curso:' não
# corresponda a um curso real de Ibirubá é descartada com aviso de auditoria.
# ─────────────────────────────────────────────────────────────────────────────
_CURSOS_IBIRUBA: frozenset = frozenset({
    # Técnico integrado ao Ensino Médio
    "Técnico em Agropecuária",
    "Técnico em Informática",
    "Técnico em Mecânica",
    # Técnico concomitante/subsequente
    "Técnico em Eletrotécnica",
    # Cursos superiores (Bacharelado e Licenciatura)
    "Bacharelado em Agronomia",
    "Bacharelado em Ciência da Computação",
    "Bacharelado em Engenharia Mecânica",
    "Licenciatura em Matemática",
})


def extrair_texto_ibiruba(caminho_pdf: str) -> tuple[list[Document], int, int]:
    """Extrai texto de páginas relevantes ao Campus Ibirubá (e páginas gerais).

    Três estados de rastreamento por página (persiste entre páginas):
    - 'general': antes de qualquer cabeçalho de campus → inclui (regras gerais)
    - 'ibiruba': dentro da seção Campus Ibirubá      → inclui
    - 'outro':   dentro de seção de outro campus     → descarta

    O estado persiste entre páginas: páginas sem cabeçalho de campus herdam
    o estado da página anterior (cobre continuações de annexo sem cabeçalho).

    Retorna (docs, n_paginas_mantidas, n_paginas_descartadas).
    """
    docs: list[Document] = []
    n_mantidas = 0
    n_descartadas = 0
    estado = "general"
    nome = Path(caminho_pdf).name
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    texto = page.extract_text() or ""

                    # Remove cabeçalho institucional repetido antes de checar estado
                    texto_limpo = _RE_CABECALHO_PAGINA.sub("", texto)

                    if _RE_CAMPUS_IBIRUBA.search(texto):
                        estado = "ibiruba"
                    elif _RE_CAMPUS_OUTRO_TEXTO.search(texto):
                        estado = "outro"

                    if estado in ("general", "ibiruba") and texto_limpo.strip():
                        n_mantidas += 1
                        docs.append(Document(
                            content=texto_limpo,
                            meta={"file_path": caminho_pdf, "page_number": i + 1},
                        ))
                    elif texto_limpo.strip():
                        n_descartadas += 1
                except Exception as e:
                    print(f"⚠️  Aviso: erro ao ler texto da página {i + 1} de {nome}: {e}")
    except Exception as e:
        print(f"⚠️  Aviso: erro ao abrir {nome} para extração de texto: {e}")

    print(f"   📝 {nome}: {n_mantidas} pág(s) mantida(s) "
          f"(geral/Ibirubá), {n_descartadas} de outros campi descartada(s)")
    return docs, n_mantidas, n_descartadas


def extrair_tabelas_como_texto(caminho_pdf: str) -> list[str]:
    """Extrai linhas de tabela de um PDF como texto estruturado.

    Usa posição fixa das 4 primeiras colunas (Curso, Turnos, Duração,
    Total de Vagas). Colunas 4+ (cotas C1-C10) são sempre descartadas.

    Filtro de campus: só inclui tabelas de páginas dentro da seção do
    Campus Ibirubá. Ativa ao encontrar 'Campus Ibirubá:' ou 'Campus Ibirubá ('
    e desativa ao encontrar o cabeçalho de qualquer outro campus em qualquer
    dos dois formatos de annexo presentes no PDF. Aplica validação cruzada
    final contra _CURSOS_IBIRUBA como rede de segurança adicional.
    """
    linhas: list[str] = []
    n_mantidas = 0
    n_descartadas = 0        # tabelas de outros campi
    n_descartadas_curso = 0  # linhas fora da whitelist _CURSOS_IBIRUBA
    ibiruba_ativo = False
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    texto = page.extract_text() or ""

                    if _RE_CAMPUS_IBIRUBA.search(texto):
                        ibiruba_ativo = True
                    elif _RE_CAMPUS_OUTRO_TABELA.search(texto):
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

    # Opção C — validação cruzada: descarta linhas cujo curso não está na lista conhecida
    # de Ibirubá, sinalizando possíveis falsos positivos do detector de campus por regex.
    _re_curso = re.compile(r"Curso:\s*(.+?)(?:\s*\||\s*$)")
    linhas_ok: list[str] = []
    for linha in linhas:
        m = _re_curso.search(linha)
        # Strip de marcadores de rodapé ('*', '**') antes de comparar com o frozenset
        nome_curso = m.group(1).strip().rstrip("*").strip() if m else ""
        if nome_curso and nome_curso not in _CURSOS_IBIRUBA:
            print(f"   ⚠️  [AUDITORIA] Curso não reconhecido descartado: {nome_curso!r}")
            n_descartadas_curso += 1
        else:
            linhas_ok.append(linha)
    linhas = linhas_ok

    nome = Path(caminho_pdf).name
    print(f"   📋 {nome}: {n_mantidas} tabela(s) Ibirubá mantida(s), "
          f"{n_descartadas} de outros campi descartada(s)")
    if n_descartadas_curso:
        print(f"   📋 {nome}: {n_descartadas_curso} linha(s) de tabela descartada(s) por whitelist de cursos")
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

    # Extrai texto filtrando páginas por campus (geral + Ibirubá, descarta outros)
    text_docs: list[Document] = []
    n_pags_mantidas_total = 0
    n_pags_descartadas_total = 0
    for arq in arquivos:
        docs_pag, n_m, n_d = extrair_texto_ibiruba(str(arq))
        text_docs.extend(docs_pag)
        n_pags_mantidas_total += n_m
        n_pags_descartadas_total += n_d
    if n_pags_descartadas_total:
        print(f"   📋 Total texto: {n_pags_descartadas_total} página(s) de outros campi descartada(s)")

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
        if not _PADRAO_COTA.search(c.content or "")
    ]
    n_descartadas_cota = len(chunks_texto_raw) - len(chunks_texto)
    if n_descartadas_cota:
        print(f"   🗑️  {n_descartadas_cota} chunk(s) de texto descartado(s) por parágrafo de cota")

    # Documentos de tabela: apenas limpar, cada linha já é um chunk final
    cleaned_table = cleaner.run(documents=table_docs)
    chunks_tabela = cleaned_table["documents"]

    # ── Auditoria pós-indexação ───────────────────────────────────────────
    # Varre todos os chunks e avisa se algum ainda contém cabeçalho ou nome
    # de outro campus do IFRS que não seja Ibirubá.
    print("\n🔍 Auditoria pós-indexação...")
    n_audit_falhas = 0
    for chunk in chunks_texto + chunks_tabela:
        txt = chunk.content or ""
        m_hdr = _RE_CAMPUS_OUTRO_TEXTO.search(txt)
        m_nome = _RE_OUTROS_NOMES_AUDIT.search(txt)
        if m_hdr:
            cid = (getattr(chunk, "id", None) or "?")[:8]
            print(f"   [AUDITORIA] cabeçalho id={cid} "
                  f"campus={m_hdr.group().strip()!r} | {txt[:100]!r}")
            n_audit_falhas += 1
        elif m_nome:
            cid = (getattr(chunk, "id", None) or "?")[:8]
            print(f"   [AUDITORIA] nome     id={cid} "
                  f"campus={m_nome.group().strip()!r} | {txt[:100]!r}")
            n_audit_falhas += 1

    # Conta chunks com perfil de cronograma (subset dos de texto)
    _PADRAO_DATA_RECENTE = re.compile(r"\d{1,2}/\d{2}/202[4-9]")
    n_chunks_cronograma = sum(
        1 for c in chunks_texto
        if "cronograma" in (c.content or "").lower()
        or len(_PADRAO_DATA_RECENTE.findall(c.content or "")) >= 3
    )
    # ── ──────────────────────────────────────────────────────────────────

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
    n_descartes = n_pags_descartadas_total + n_descartadas_cota
    print(f"\n✅ Indexação concluída! {n_chunks} chunks persistidos em {pasta_chroma}/")
    print(f"   • Chunks de texto      : {n_chunks_texto}")
    print(f"     └─ cronograma-like   : {n_chunks_cronograma}")
    print(f"   • Chunks de tabela     : {n_chunks_tabela} ({n_linhas_tabela} linhas extraídas)")
    print(f"   • Descartes (páginas)  : {n_pags_descartadas_total}")
    print(f"   • Descartes (cota)     : {n_descartadas_cota}")
    print(f"   • Descartes total      : {n_descartes}")
    if n_audit_falhas:
        print(f"   ⚠️  Alertas de auditoria: {n_audit_falhas} (revisar acima)")
    else:
        print(f"   ✅ Auditoria: nenhum chunk com campus externo detectado")

    return {"n_arquivos": len(arquivos), "n_chunks": n_chunks, "n_linhas_tabela": n_linhas_tabela}


if __name__ == "__main__":
    _pasta_docs   = os.getenv("PASTA_DOCS",   "./documentos")
    _pasta_chroma = os.getenv("PASTA_CHROMA", "./chroma_data")
    resultado = indexar_documentos(_pasta_docs, _pasta_chroma)
    print(f"   Agora execute: streamlit run main.py")
