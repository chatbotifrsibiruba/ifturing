"""
indexar.py — Indexa os PDFs no ChromaDB (store embutido, sem servidor).

Execute este script toda vez que atualizar os PDFs:
    python indexar.py

Fase 2 da auditoria (ver auditoria_indexacao.md):
  • Escopo de campus por PÁGINA (corpo / anexo de tabela do Ibirubá / anexo de tabela de outro
    campus / anexo geral) em vez de um interruptor global, e por ITEM nas listas "N.N.N Campus X:".
  • Páginas de anexo de tabela do Ibirubá: tabelas estruturadas + "legenda" (notas, descrição das
    vagas) extraída com y_tolerance=1; o texto bruto embaralhado não é indexado.
  • Cronograma (págs. do item 2) reconstruído a partir da tabela (linhas "Data | Evento").
  • Portão de legibilidade de chunks e portão de cobertura: o indexador FALHA (exit code 1) se
    alguma seção de nível 1 ficar sem chunk. Nada é gravado em disco antes de os portões passarem.
  • Metadados reais: secao, secao_titulo, secoes, pagina_ini, pagina_fim, page_number, tipo,
    escopo_campus.
"""

import collections
import dataclasses
import hashlib
import os
import re
import shutil
import sys
import unicodedata
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


def _env_flag(nome: str, padrao: str = "0") -> bool:
    return os.getenv(nome, padrao).strip().lower() not in ("", "0", "false", "no", "nao", "não")


# ── Variantes da Fase 3 (cada uma controlada por variável de ambiente; ver auditoria_indexacao.md) ──
#   Padrões = melhor configuração medida na Fase 3 (chunker item_titulo, cabeçalho de tabela ligado, ids estáveis,
#   prefixos E5 DESLIGADOS: só ligue E5_PREFIXOS ao reindexar e consultar com o mesmo valor).
#   IDX_CHUNKER           atual | item | item_titulo   chunker do CORPO (seções 1 a 20): janela de 150 palavras por
#                         página (atual) × um chunk por item numerado "N.N." (item) × idem com o título da seção
#                         prefixado em cada chunk ("[Seção 11 — REALIZAÇÃO DA PROVA]")
#   IDX_TABELA_CABECALHO  1 = cada chunk de tabela do Ibirubá ganha uma linha de cabeçalho (anexo, título do anexo e
#                         rótulo da tabela, todos lidos do PDF)
#   IDX_IDS               estavel | legado             ids = hash(nome do arquivo + posição) × hash do Haystack
#                         (conteúdo + metadados, inclusive o CAMINHO do arquivo → muda entre Windows e Linux)
#   E5_PREFIXOS           1 = "passage: " no texto embedado de cada chunk (o texto gravado não muda); a consulta
#                         usa "query: " com a mesma variável em nucleo.py. Precisa dos dois lados.
IDX_CHUNKER = os.getenv("IDX_CHUNKER", "item_titulo").strip().lower()
IDX_TABELA_CABECALHO = _env_flag("IDX_TABELA_CABECALHO", "1")
IDX_IDS = os.getenv("IDX_IDS", "estavel").strip().lower()
E5_PREFIXOS = _env_flag("E5_PREFIXOS")
_CHUNKERS = ("atual", "item", "item_titulo")
_CORPO_TETO_PALAVRAS, _CORPO_SOBREPOSICAO = 200, 20       # chunker por item: teto e sobreposição (só ao subdividir)


class IndexacaoIncompleta(RuntimeError):
    """Levantada quando o portão de cobertura detecta seção de nível 1 sem chunk."""


_LABELS = ("Curso", "Turnos", "Duração (semestres)", "Total de Vagas")

# ── Parâmetros do chunker (inalterados na fase 2) ────────────────────────────
_SPLIT_BY, _SPLIT_LENGTH, _SPLIT_OVERLAP = "word", 150, 20
_CLEANER = DocumentCleaner()
_SPLITTER = DocumentSplitter(split_by=_SPLIT_BY, split_length=_SPLIT_LENGTH, split_overlap=_SPLIT_OVERLAP)

# ── Portão de legibilidade ───────────────────────────────────────────────────
# Rejeita chunk de texto (corpo/legenda/anexo_geral) quando:
#   • >= 40% dos tokens têm 1 ou 2 caracteres (tokens só-dígitos não contam: "30", "2" são
#     legítimos em tabelas; texto embaralhado produz letras soltas), ou
#   • a fração de tokens de 4+ caracteres que aparecem em >= 3 chunks do corpus é < _FREQ_MIN.
_CURTOS_MAX = 0.40
_FREQ_MIN = 0.30
_MIN_TOKENS_PORTAO = 30      # chunks menores (título, item curto do chunker por item) são pequenos demais para avaliar
_TIPOS_COM_PORTAO = ("corpo", "legenda", "anexo_geral")

# ── Campus ───────────────────────────────────────────────────────────────────
# Cabeçalho de seção de anexo de tabela, nos dois formatos do edital:
#   "Campus X (a descrição das vagas está após a tabela)"      "Campus Canoas(a descrição…"
#   "Campus X: A descrição da tabela encontra-se ao final…"    "Campus X: (a descrição das vagas…"
_RE_CAMPUS_CAB_ANEXO = re.compile(r"^Campus\s+(?P<nome>.+?)\s*:?\s*\(?\s*a\s+descri", re.IGNORECASE)

# Item de lista de campi (corpo): "9.5.3 Campus Canoas:", "20.1.8 Campus Feliz:" ou sem número
# ("Campus Zona Norte (Porto Alegre):").
_RE_ITEM_CAMPUS = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})+\.?\s+)?Campus\s+(?P<nome>[^:()]+?(?:\s*\([^)]*\))?)\s*:\s*$",
    re.IGNORECASE,
)
# Fim de lista: novo item numerado com ponto (9.6, 20.1), título de nível 1 ou ANEXO.
_RE_FIM_LISTA = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2})+\.?\s+\S|\d{1,2}\s*\.?\s+[A-ZÀ-Ý]{3,}|ANEXO\s+[IVXL]+\s*$)")

# Nomes de outros campi (para a auditoria pós-indexação e o filtro de linhas do Anexo IX).
_OUTROS_CAMPI_NOMES = (
    r"alvorada|bento\s+gon[cç]alves|canoas|caxias\s+do\s+sul|erechim|"
    r"farroupilha|feliz|os[oó]rio|porto\s+alegre|restinga|rio\s+grande|"
    r"rolante|sert[aã]o|vacaria|veran[oó]polis|viam[aã]o"
)
_RE_CAMPUS_OUTRO_TEXTO = re.compile(rf"campus\s+(?:{_OUTROS_CAMPI_NOMES})\s*[:(]", re.IGNORECASE)
_RE_OUTROS_NOMES_AUDIT = re.compile(rf"\b(?:{_OUTROS_CAMPI_NOMES})\b", re.IGNORECASE)

# Anexos cujas linhas de tabela começam pelo nome do campus (Anexo IX: pesos por curso).
_ANEXOS_LINHAS_POR_CAMPUS = frozenset({"ANEXO IX"})
_RE_INICIO_LINHA_CAMPUS = re.compile(
    r"^(Alvorada|Bento|Canoas|Caxias|Erechim|Farroupilha|Feliz|Gramado|Ibirub[aá]|Os[oó]rio|Porto|"
    r"Restinga|Rio|Rolante|Sert[aã]o|Vacaria|Veran[oó]polis|Viam[aã]o|Zona)\b",
    re.IGNORECASE,
)

# ── Estrutura do edital ──────────────────────────────────────────────────────
_RE_ANEXO = re.compile(r"^ANEXO\s+([IVXL]+)\s*$")
_RE_TITULO_N1 = re.compile(r"^(\d{1,2})\s*\.?\s+([A-ZÀ-Ý][A-ZÀ-Ý0-9 ,/()\-–ÇÃÕÉÊÍÓÚÂÔ&]{6,})$")
_RE_DATA_COMPLETA = re.compile(r"\d{1,2}/\d{2}/20\d{2}")

# Cabeçalho institucional repetido em todas as páginas e rodapé (endereço, telefone, nº da página).
_RE_CABECALHO_PAGINA = re.compile(
    r"MINIST[EÉ]RIO\s+DA\s+EDUCA[CÇ][AÃ]O\s*\n"
    r"Secretaria\s+de\s+Educa[cç][aã]o\s+Profissional\s+e\s+Tecnol[oó]gica\s*\n"
    r"Instituto\s+Federal\s+de\s+Educa[cç][aã]o[^\n]*\n"
    r"(?:Gabinete\s+do\s+Reitor[^\n]*\n?)?",
    re.IGNORECASE,
)
_RE_RODAPE = re.compile(r"^(?:Rua\s+Gen\.\s+Os[óo]rio,\s*348|Telefone:\s*\(54\)\s*3449\.3371)", re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# ATENÇÃO: atualizar este conjunto a cada novo edital do IFRS – Campus Ibirubá.
# Rede de segurança: linha de tabela cujo 'Curso:' não esteja aqui é descartada com aviso.
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

_PADRAO_COTA = re.compile(r"C\d{1,2}:")                 # só estatística (o filtro de cotas foi removido)
_TOKEN = re.compile(r"\w+", re.UNICODE)


# ═════════════════════════════════════════════════════════════════════════════
# Utilidades de texto
# ═════════════════════════════════════════════════════════════════════════════

def _sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def _eh_ibiruba(nome: str) -> bool:
    return _sem_acento(nome) == "ibiruba"


def _remover_moldura(texto: str) -> str:
    """Remove cabeçalho institucional, linhas de rodapé (endereço/telefone) e o nº da página."""
    t = _RE_CABECALHO_PAGINA.sub("", texto or "")
    linhas = [l for l in t.split("\n") if not _RE_RODAPE.match(l.strip())]
    while linhas and not linhas[-1].strip():
        linhas.pop()
    if linhas and re.fullmatch(r"\d{1,3}", linhas[-1].strip()):
        linhas.pop()
    return "\n".join(linhas)


def _limpar(texto: str) -> str:
    """Moldura + DocumentCleaner (linhas vazias e espaços extras)."""
    t = _remover_moldura(texto)
    if not t.strip():
        return ""
    return _CLEANER.run(documents=[Document(content=t)])["documents"][0].content


def _eh_caixa_alta(linha: str) -> bool:
    s = linha.strip()
    return 5 <= len(s) <= 120 and any(c.isalpha() for c in s) and not any(c.islower() for c in s)


_TOKEN_COMPOSTO = re.compile(r"\w+(?:/\w+)*", re.UNICODE)     # "da/do", "candidata/candidato" = 1 token


def _frac_curtos(texto: str, contar_digitos: bool, compostos: bool = False) -> float:
    """Fração de tokens de 1-2 caracteres. `compostos`: formas com barra da escrita inclusiva do edital
    ("a/o", "da/do") contam como UM token; sem isso, itens curtos como "11.15. É … da/do candidata/candidato…"
    parecem texto embaralhado (44% de tokens curtos). Só o portão de legibilidade usa compostos=True."""
    toks = (_TOKEN_COMPOSTO if compostos else _TOKEN).findall(texto)
    if not toks:
        return 1.0
    curtos = sum(1 for t in toks if len(t) <= 2 and (contar_digitos or not t.isdigit()))
    return curtos / len(toks)


def _linha_ilegivel(linha: str) -> bool:
    """Linha de cabeçalho rotacionado: muitos tokens, quase todos com 1-2 caracteres."""
    toks = _TOKEN.findall(linha)
    if toks and all(len(t) <= 2 and not t.isdigit() for t in toks):
        return True                      # "e", "a", "s g": letras soltas de texto rotacionado
    return len(toks) >= 6 and _frac_curtos(linha, True) >= 0.7


_CONECTIVOS_TITULO = frozenset({"E", "DE", "DA", "DO", "DAS", "DOS", "PARA", "COM", "EM", "OU", "-", "–"})


def _titulo_anexo(linhas: list, j: int) -> str:
    """Título do anexo = linha em caixa alta depois de "ANEXO X" (linhas[j]). Quando a linha termina em conectivo
    ("…CURSOS INTEGRADOS, CONCOMITANTES E"), o título continua na linha seguinte em caixa alta."""
    if j + 1 >= len(linhas) or not _eh_caixa_alta(linhas[j + 1]):
        return ""
    tit, k = linhas[j + 1].strip(), j + 2
    while k < len(linhas) and _eh_caixa_alta(linhas[k]) and not _RE_CAMPUS_CAB_ANEXO.match(linhas[k].strip()) \
            and (tit.split()[-1] in _CONECTIVOS_TITULO or tit.endswith((",", "-", "–"))):
        tit += " " + linhas[k].strip()
        k += 1
    return tit


def _intervalos(paginas) -> str:
    ps = sorted(set(paginas))
    if not ps:
        return "—"
    partes, ini, ant = [], ps[0], ps[0]
    for p in ps[1:]:
        if p == ant + 1:
            ant = p
            continue
        partes.append(f"{ini}" if ini == ant else f"{ini}-{ant}")
        ini = ant = p
    partes.append(f"{ini}" if ini == ant else f"{ini}-{ant}")
    return ",".join(partes)


# ═════════════════════════════════════════════════════════════════════════════
# Parte A — classificação de páginas e escopo de campus
# ═════════════════════════════════════════════════════════════════════════════

def _classificar_paginas(textos: list) -> list:
    """Classifica cada página em: 'corpo', 'anexo_tabela_ibiruba', 'anexo_tabela_outro' ou 'anexo_geral'.

    • corpo: páginas antes do primeiro "ANEXO …".
    • Anexo de tabela = anexo em que alguma página tem cabeçalho de campus (os dois formatos acima).
      O campus vale até o próximo cabeçalho e continua nas páginas seguintes do mesmo anexo
      (páginas de continuação sem cabeçalho). Uma página pode conter trechos de dois campi.
    • anexo_geral: anexo sem nenhum cabeçalho de campus (legendas, regras, formulários, declarações).
    Páginas de corpo e de anexo geral nunca são descartadas por escopo de campus.
    """
    linhas_pag = [[l.strip() for l in _remover_moldura(t).split("\n") if l.strip()] for t in textos]
    primeiro_anexo = next((i for i, L in enumerate(linhas_pag) if any(_RE_ANEXO.match(l) for l in L)), None)

    info = []
    anexo, anexo_tit, campus_atual = None, "", None
    for i, L in enumerate(linhas_pag):
        d = {"pagina": i + 1, "anexo": None, "anexo_titulo": "", "campi": [], "categoria": "corpo", "mista": False}
        if primeiro_anexo is None or i < primeiro_anexo:
            info.append(d)
            continue
        campi, viu_cab = [], False
        for j, l in enumerate(L):
            ma = _RE_ANEXO.match(l)
            if ma:
                anexo = f"ANEXO {ma.group(1)}"
                anexo_tit = _titulo_anexo(L, j)
                campus_atual = None
                continue
            mc = _RE_CAMPUS_CAB_ANEXO.match(l)
            if mc:
                campus_atual = mc.group("nome").strip()
                viu_cab = True
                campi.append(_sem_acento(campus_atual))
            elif not viu_cab and campus_atual is not None:
                campi.append(_sem_acento(campus_atual))   # continuação do campus da página anterior
        d["anexo"], d["anexo_titulo"] = anexo, anexo_tit
        d["campi"] = list(dict.fromkeys(campi))
        d["categoria"] = "?"
        info.append(d)

    # Anexo de tabela = algum cabeçalho de campus dentro do anexo
    tem_cab = collections.defaultdict(bool)
    for d in info:
        if d["categoria"] == "?" and d["campi"]:
            tem_cab[d["anexo"]] = True
    for d in info:
        if d["categoria"] != "?":
            continue
        if not tem_cab[d["anexo"]]:
            d["categoria"] = "anexo_geral"
        elif "ibiruba" in d["campi"]:
            d["categoria"] = "anexo_tabela_ibiruba"
            d["mista"] = len(d["campi"]) > 1
        else:
            d["categoria"] = "anexo_tabela_outro"
    return info


def _filtrar_itens_campus(linhas: list, estado: str) -> tuple:
    """Filtra POR ITEM listas "N.N.N Campus X:": mantém a introdução e o item do Ibirubá,
    descarta os itens dos outros campi. `estado` ('todos'|'ibiruba'|'outro') é herdado entre páginas.
    Retorna (linhas_mantidas, flags_ibiruba, estado_final, n_descartadas)."""
    saida, flags, descartadas = [], [], 0
    for l in linhas:
        s = l.strip()
        m = _RE_ITEM_CAMPUS.match(s)
        if m:
            estado = "ibiruba" if _eh_ibiruba(m.group("nome")) else "outro"
        elif estado != "todos" and _RE_FIM_LISTA.match(s):
            estado = "todos"
        if estado == "outro":
            descartadas += 1
            continue
        saida.append(l)
        flags.append(estado == "ibiruba")
    return saida, flags, estado, descartadas


def _filtrar_linhas_por_campus(linhas: list) -> tuple:
    """Anexo IX (tabela de pesos): cada linha começa pelo nome do campus. Mantém título, cabeçalho
    legível e linhas do Ibirubá; descarta as dos outros campi (e suas linhas de continuação)."""
    saida, flags, manter, descartadas = [], [], True, 0
    for l in linhas:
        s = l.strip()
        m = _RE_INICIO_LINHA_CAMPUS.match(s)
        if m:
            manter = _eh_ibiruba(m.group(1))
        elif _linha_ilegivel(s):
            descartadas += 1
            continue
        if not manter:
            descartadas += 1
            continue
        saida.append(l)
        flags.append(bool(m) and manter)
    return saida, flags, descartadas


def _montar_pagina(linhas: list, flags: list) -> tuple:
    """Junta as linhas mantidas e devolve (texto, offsets_das_linhas, spans_ibiruba)."""
    offs, spans, off = [], [], 0
    for l, f in zip(linhas, flags):
        offs.append(off)
        if f:
            spans.append((off, off + len(l)))
        off += len(l) + 1
    return "\n".join(linhas), offs, spans


def _detectar_titulos(linhas: list, offs: list, corpo: bool) -> list:
    """Títulos de nível 1 da página: [(offset, rótulo, título)]. 'ANEXO X' em qualquer página;
    "N. TÍTULO EM CAIXA ALTA" apenas no corpo."""
    res = []
    for i, l in enumerate(linhas):
        s = l.strip()
        ma = _RE_ANEXO.match(s)
        if ma:
            res.append((offs[i], f"ANEXO {ma.group(1)}", _titulo_anexo(linhas, i)))
        elif corpo:
            m1 = _RE_TITULO_N1.match(s)
            if m1 and not any(c.islower() for c in s) and int(m1.group(1)) <= 25:
                res.append((offs[i], m1.group(1), m1.group(2).strip()))
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Tabelas (anexos de tabela do Ibirubá)
# ═════════════════════════════════════════════════════════════════════════════

def _linhas_tabela_pagina(page) -> list:
    """Linhas "Curso: … | Turnos: … | Duração (semestres): … | Total de Vagas: …" de uma página."""
    return [l for l, _ in _linhas_tabela_pagina_cab(page)]


def _linhas_tabela_pagina_cab(page) -> list:
    """[(linha, rótulo da tabela)] de uma página. O rótulo é a 1ª célula do cabeçalho da tabela no PDF
    ("Cursos Técnicos Integrados ao Ensino Médio", "Cursos Superiores"…).
    Posição fixa das 4 primeiras colunas; colunas de cotas (C1-C10) descartadas."""
    linhas: list = []
    for tabela in page.extract_tables() or []:
        if not tabela:
            continue
        rotulo = " ".join(str(tabela[0][0]).replace("\n", " ").split()) if tabela[0] and tabela[0][0] else ""
        ultimo_col0 = ""  # resetado a cada tabela
        for linha in tabela[1:]:  # tabela[0] é sempre o cabeçalho
            if not linha:
                continue
            col0_raw = " ".join(str(linha[0]).replace("\n", " ").split()) if linha[0] else ""
            tem_dados = any(linha[idx] and str(linha[idx]).strip() for idx in range(1, min(4, len(linha))))
            if col0_raw:
                ultimo_col0 = col0_raw
                col0 = col0_raw
            elif tem_dados:
                col0 = ultimo_col0     # célula mesclada: herda o último curso
            else:
                continue               # sub-cabeçalho ou linha vazia
            if not col0:
                continue
            if col0.lower().startswith("observa") or len(col0.split()) > 15:
                continue               # rodapé
            partes = []
            for idx, label in enumerate(_LABELS):
                if idx >= len(linha):
                    break
                val = linha[idx]
                val_s = " ".join(str(val).replace("\n", " ").split()) if val else ""
                if val_s:
                    partes.append(f"{label}: {val_s}")
                elif idx == 0:
                    partes.append(f"{label}: {col0}")
            if partes:
                linhas.append((" | ".join(partes), rotulo))
    return linhas


_RE_CURSO_LINHA = re.compile(r"Curso:\s*(.+?)(?:\s*\||\s*$)")


def _validar_cursos(linhas: list, nome_pdf: str = "") -> tuple:
    """Opção C — descarta linhas cujo curso não está em _CURSOS_IBIRUBA. Retorna (ok, n_descartadas)."""
    ok, n = [], 0
    for linha in linhas:
        m = _RE_CURSO_LINHA.search(linha)
        nome_curso = m.group(1).strip().rstrip("*").strip() if m else ""
        if nome_curso and nome_curso not in _CURSOS_IBIRUBA:
            print(f"   ⚠️  [AUDITORIA] Curso não reconhecido descartado: {nome_curso!r}")
            n += 1
        else:
            ok.append(linha)
    return ok, n


_FORMACAO_POR_PREFIXO = (("Bacharelado em", "bacharelado"), ("Licenciatura em", "licenciatura"),
                         ("Tecnologia em", "tecnologia"))


def _cabecalho_tabela(anexo: str, anexo_titulo: str, rotulo: str, linha: str) -> str:
    """Linha descritiva prefixada em cada chunk de tabela (IDX_TABELA_CABECALHO). Tudo vem do PDF: número e
    título do anexo, rótulo da tabela (1ª célula do cabeçalho) e a formação, lida do prefixo do próprio curso.
    Ex.: "Campus Ibirubá — Anexo V, Quadros de vagas - cursos de graduação: Cursos Superiores (bacharelado)."
    Não usa "Técnico em"/"Bacharelado em"/… (nucleo._tipo_tabela_chunk classifica o chunk por esses prefixos)."""
    m = _RE_CURSO_LINHA.search(linha)
    curso = m.group(1).strip() if m else ""
    formacao = next((f for pref, f in _FORMACAO_POR_PREFIXO if curso.startswith(pref)), "")
    cab = f"Campus Ibirubá — Anexo {anexo.split()[-1]}" if anexo else "Campus Ibirubá"
    if anexo_titulo:
        cab += f", {anexo_titulo.lower().capitalize()}"
    if rotulo:
        cab += f": {rotulo}"
    if formacao:
        cab += f" ({formacao})"
    return cab + "."


def _texto_legenda(page) -> str:
    """Legendas e notas de uma página de anexo de tabela do Ibirubá.

    • Texto FORA das tabelas (cabeçalho "Campus X…", notas "*", "**", parágrafo "descrição das vagas"):
      o parágrafo é texto de fonte ~1 pt; com y_tolerance=3 (padrão) as linhas se fundem e ficam
      embaralhadas, com y_tolerance=1 saem legíveis. Mantêm-se as linhas com >= 3 tokens e < 40% de
      tokens de 1-2 caracteres, mais os cabeçalhos "Campus X…".
    • Rodapés DENTRO da tabela ("Observação: …"): lidos das células. Células de cabeçalho rotacionado
      e linhas de dados (já estruturadas em tipo="tabela") não entram."""
    tabelas = page.find_tables()
    fora = page
    for tb in tabelas:
        fora = fora.outside_bbox(tb.bbox)
    t = _remover_moldura(fora.extract_text(y_tolerance=1) or "")
    mantidas = []
    for l in t.split("\n"):
        s = l.strip()
        if not s:
            continue
        toks = _TOKEN.findall(s)
        if (_RE_CAMPUS_CAB_ANEXO.match(s) or _sem_acento(s) == "campus ibiruba"
                or (len(toks) >= 3 and _frac_curtos(s, True) < _CURTOS_MAX)):
            mantidas.append(s)
    for tb in tabelas:
        for row in tb.extract():
            c0 = _norm_ws(row[0]) if row and row[0] else ""
            if c0.lower().startswith("observa") or len(c0.split()) > 15:
                mantidas.append(" ".join(_norm_ws(c) for c in row if c))
    if not mantidas:
        return ""
    return _CLEANER.run(documents=[Document(content="\n".join(mantidas))])["documents"][0].content


# ═════════════════════════════════════════════════════════════════════════════
# Parte B — cronograma estruturado
# ═════════════════════════════════════════════════════════════════════════════

def _norm_ws(s) -> str:
    return " ".join(str(s or "").replace("\n", " ").split())


def _extrair_cronograma(pdf, textos: list, info: list) -> dict:
    """Reconstrói as linhas "Data | Evento" do item 2 a partir da tabela (find_tables).

    Cada linha da tabela é uma célula de data + uma célula de evento na mesma faixa vertical, o que
    associa a data ao evento certo. Linha com data vazia = continuação (inclusive através da quebra
    de página) do evento anterior. Retorna {"linhas": [(data, evento, pagina)], "bboxes": {idx: [bbox]}}
    ou linhas=[] se a tabela não for detectada."""
    pag_ini = pag_fim = None
    for i, t in enumerate(textos):
        if info[i]["categoria"] != "corpo":
            continue
        for l in _remover_moldura(t).split("\n"):
            m1 = _RE_TITULO_N1.match(l.strip())
            if m1 and not any(c.islower() for c in l):
                if m1.group(1) == "2" and pag_ini is None:
                    pag_ini = i
                elif m1.group(1) == "3" and pag_ini is not None and pag_fim is None:
                    pag_fim = i
    if pag_ini is None or pag_fim is None:
        return {"linhas": [], "bboxes": {}, "avisos": ["títulos das seções 2 e 3 não encontrados"]}

    linhas, bboxes, avisos, dentro = [], {}, [], False
    for i in range(pag_ini, pag_fim + 1):
        for tb in pdf.pages[i].find_tables():
            rows = tb.extract()
            if not rows or max(len(r) for r in rows) != 2:
                continue
            cab = _norm_ws(rows[0][0]).lower().startswith("data")
            if not dentro and not cab:
                continue
            dentro = True
            bboxes.setdefault(i, []).append(tb.bbox)
            for c0, c1 in rows[1:] if cab else rows:
                data, ev = _norm_ws(c0), _norm_ws(c1)
                if not data:
                    if linhas:
                        d0, e0, p0 = linhas[-1]
                        linhas[-1] = (d0, f"{e0} {ev}".strip(), p0)
                    continue
                if not _RE_DATA_COMPLETA.search(data):
                    avisos.append(f"pág. {i + 1}: célula de data sem data completa: {data!r}")
                linhas.append((data, ev, i + 1))
    return {"linhas": linhas, "bboxes": bboxes, "avisos": avisos}


_CRONO_LINHAS, _CRONO_MAX_PALAVRAS, _CRONO_MIN_LINHAS = 6, 180, 3


def _agrupar_cronograma(linhas: list) -> list:
    """Agrupa as linhas em chunks de até _CRONO_LINHAS linhas, fechando o chunk antes de passar de
    _CRONO_MAX_PALAVRAS palavras (~2,3 tokens/palavra; o E5 trunca em 512 tokens)."""
    grupos, atual, palavras = [], [], 0
    for lin in linhas:
        n = len(f"Data: {lin[0]} | Evento: {lin[1]}".split())
        if atual and (len(atual) >= _CRONO_LINHAS or palavras + n > _CRONO_MAX_PALAVRAS):
            grupos.append(atual)
            atual, palavras = [], 0
        atual.append(lin)
        palavras += n
    if atual:
        grupos.append(atual)
    # Nenhum chunk de cronograma fica com menos de _CRONO_MIN_LINHAS linhas (evita um último chunk minúsculo).
    if len(grupos) > 1 and len(grupos[-1]) < _CRONO_MIN_LINHAS:
        falta = _CRONO_MIN_LINHAS - len(grupos[-1])
        grupos[-1] = grupos[-2][-falta:] + grupos[-1]
        grupos[-2] = grupos[-2][:-falta]
    return grupos


# ═════════════════════════════════════════════════════════════════════════════
# Processamento de um PDF
# ═════════════════════════════════════════════════════════════════════════════

def _chunkar(texto: str, meta_base: dict, secao_entrada: tuple, titulos: list, spans_ibiruba: list,
             escopo_padrao: str) -> list:
    """Divide o texto de UMA página com o splitter (150/20 palavras) e devolve Documents com os
    metadados reais. O splitter sobrescreve meta['page_number'] (=1), por isso a página é
    reaplicada aqui. `titulos` = [(offset, rótulo, título)] da página."""
    if not texto.strip():
        return []
    chunks = _SPLITTER.run(documents=[Document(content=texto)])["documents"]
    docs = []
    for ch in chunks:
        s = int(ch.meta.get("split_idx_start", 0))
        e = s + len(ch.content)
        secao = secao_entrada
        for off, rot, tit in titulos:
            if off <= s:
                secao = (rot, tit)
        cobertas = [secao[0]] + [rot for off, rot, _ in titulos if s < off < e and rot != secao[0]]
        ibi = any(a < e and b > s for a, b in spans_ibiruba)
        meta = dict(meta_base)
        meta.update({
            "secao": secao[0], "secao_titulo": secao[1], "secoes": ",".join(cobertas),
            "escopo_campus": "ibiruba" if ibi else escopo_padrao,
            "split_id": int(ch.meta.get("split_id", 0)), "split_idx_start": s,
        })
        docs.append(Document(content=ch.content, meta=meta))
    return docs


# ── Chunker do corpo por item numerado (IDX_CHUNKER = item | item_titulo) ────────────────────────────
# Item = linha que começa com "N.N." (nível 2) cujo N é o número da seção de nível 1 em curso. Subitens "N.N.N"
# ficam na unidade do item pai e só viram corte quando o item passa do teto de palavras. Nunca cruza seção.
_RE_ITEM_NUM = re.compile(r"^(\d{1,2})((?:\.\d{1,3})+)\.?\s+\S")


def _n_palavras(texto: str) -> int:
    return len(texto.split())


def _janelas_palavras(texto: str, teto: int, sobreposicao: int) -> list:
    pal = texto.split()
    if len(pal) <= teto:
        return [texto]
    janelas = []
    for ini in range(0, len(pal), teto - sobreposicao):
        janelas.append(" ".join(pal[ini:ini + teto]))
        if ini + teto >= len(pal):
            break
    return janelas


def _unidades_da_secao(linhas: list, num_secao: int) -> list:
    """Unidades = itens "N.N." de UMA seção de nível 1. `linhas` = [{texto, pagina, ibi}]. Texto anterior ao
    primeiro item forma a unidade de introdução (intro=True); `filhos` = índices das linhas "N.N.N" da unidade."""
    unidades, atual = [], None
    for ln in linhas:
        m = _RE_ITEM_NUM.match(ln["texto"])
        mesma_secao = bool(m) and int(m.group(1)) == num_secao
        profundidade = m.group(2).count(".") + 1 if mesma_secao else 0
        if mesma_secao and (profundidade == 2 or atual is None or atual["intro"]):
            atual = {"linhas": [ln], "filhos": [], "intro": False}
            unidades.append(atual)
        elif mesma_secao:
            atual["filhos"].append(len(atual["linhas"]))
            atual["linhas"].append(ln)
        else:
            if atual is None:
                atual = {"linhas": [], "filhos": [], "intro": True}
                unidades.append(atual)
            atual["linhas"].append(ln)
    if len(unidades) > 1 and unidades[0]["intro"] and sum(_n_palavras(l["texto"]) for l in unidades[0]["linhas"]) < 30:
        prox = unidades[1]                       # introdução curta (ex.: só o título): cola no 1º item
        prox["filhos"] = [len(unidades[0]["linhas"]) + f for f in prox["filhos"]]
        prox["linhas"] = unidades[0]["linhas"] + prox["linhas"]
        unidades.pop(0)
    return unidades


def _dividir_unidade(u: dict, teto: int, sobreposicao: int) -> list:
    """[(texto, linhas cobertas)] de uma unidade: inteira se couber no teto; senão corta nos subitens "N.N.N"
    (repetindo a 1ª linha do item, se curta) e, se um subitem sozinho passar do teto, em janelas de palavras."""
    linhas = u["linhas"]
    if sum(_n_palavras(l["texto"]) for l in linhas) <= teto:
        return [("\n".join(l["texto"] for l in linhas), linhas)]
    cortes = [0] + u["filhos"] + [len(linhas)]
    segs = [linhas[a:b] for a, b in zip(cortes, cortes[1:]) if b > a]
    cabeca = linhas[0]["texto"] if _n_palavras(linhas[0]["texto"]) <= 30 else ""
    partes, atual, w = [], [], 0

    def fechar():
        nonlocal atual, w
        if atual:
            txt = "\n".join(l["texto"] for l in atual)
            if partes and cabeca and atual[0]["texto"] != cabeca:
                txt = cabeca + "\n" + txt
            partes.append((txt, atual))
        atual, w = [], 0

    for seg in segs:
        ws = sum(_n_palavras(l["texto"]) for l in seg)
        if ws > teto:
            fechar()
            for jan in _janelas_palavras("\n".join(l["texto"] for l in seg), teto, sobreposicao):
                partes.append((jan, seg))
            continue
        if atual and w + ws > teto:
            fechar()
        atual, w = atual + seg, w + ws
    fechar()
    return partes


def _chunkar_corpo_por_item(linhas: list, meta_fixa: dict, com_titulo: bool) -> list:
    """Chunks tipo="corpo" por item numerado. `linhas` = linhas do corpo em ordem de documento:
    [{texto, pagina, ibi, titulo: (rótulo, título) | None}] (titulo != None marca a linha de título de nível 1)."""
    secoes, atual = [], {"rot": "0", "tit": "PREÂMBULO DO EDITAL", "linhas": [], "titulo_ln": None}
    for ln in linhas:
        if ln["titulo"]:
            secoes.append(atual)
            atual = {"rot": ln["titulo"][0], "tit": ln["titulo"][1], "linhas": [], "titulo_ln": ln}
            if not com_titulo:
                atual["linhas"].append(ln)       # sem prefixo, o título é parte do texto do 1º item
        else:
            atual["linhas"].append(ln)
    secoes.append(atual)

    docs = []
    for sec in secoes:
        ls = sec["linhas"] or ([sec["titulo_ln"]] if sec["titulo_ln"] else [])
        if not ls:
            continue
        digitos = re.match(r"\d+", sec["rot"])
        num = int(digitos.group()) if digitos else 0
        prefixo = ""
        if com_titulo:
            prefixo = (f"[Seção {num} — {sec['tit']}]\n" if num else f"[{sec['tit']}]\n")
        k = 0
        for u in _unidades_da_secao(ls, num):
            for texto, cobertas in _dividir_unidade(u, _CORPO_TETO_PALAVRAS, _CORPO_SOBREPOSICAO):
                pgs = [l["pagina"] for l in cobertas]
                meta = dict(meta_fixa, tipo="corpo", page_number=min(pgs), pagina_ini=min(pgs), pagina_fim=max(pgs),
                            secao=sec["rot"], secao_titulo=sec["tit"], secoes=sec["rot"],
                            escopo_campus="ibiruba" if any(l["ibi"] for l in cobertas) else "todos",
                            split_id=k, split_idx_start=0)
                docs.append(Document(content=prefixo + texto, meta=meta))
                k += 1
    return docs


def _atribuir_ids_estaveis(docs: list, nome: str) -> list:
    """Cópias dos chunks com id = sha256(nome do arquivo + tipo + página + posição entre os chunks do mesmo tipo
    e página). Não usa caminho, conteúdo nem metadados do Haystack: o mesmo PDF gera os mesmos ids em qualquer máquina."""
    posicao = collections.Counter()
    novos = []
    for d in docs:
        chave = (d.meta["tipo"], d.meta["pagina_ini"])
        novos.append(dataclasses.replace(
            d, id=hashlib.sha256(f"{nome}|{chave[0]}|p{chave[1]}|{posicao[chave]}".encode("utf-8")).hexdigest()))
        posicao[chave] += 1
    return novos


def _processar_pdf(caminho_pdf: str) -> dict:
    """Extrai e chunka um PDF. Nada é gravado em disco aqui.

    Retorna {"docs", "secoes", "info", "n_linhas_tabela", "avisos", "estat"}."""
    nome = Path(caminho_pdf).name
    docs: list = []
    avisos: list = []
    secoes = collections.OrderedDict()   # rótulo -> {titulo, tipo, tem_ibiruba}
    estat = collections.Counter()

    with pdfplumber.open(caminho_pdf) as pdf:
        textos = []
        for i, page in enumerate(pdf.pages):
            try:
                textos.append(page.extract_text() or "")
            except Exception as e:
                print(f"⚠️  Aviso: erro ao ler texto da página {i + 1} de {nome}: {e}")
                textos.append("")
        info = _classificar_paginas(textos)
        for d in info:
            estat[d["categoria"]] += 1
            if d["mista"]:
                avisos.append(f"pág. {d['pagina']}: página com trechos de mais de um campus ({d['campi']}); "
                              "texto de legenda não indexado, tabelas mantidas")
        print(f"   📄 {nome}: {len(pdf.pages)} pág(s) — corpo={estat['corpo']}, "
              f"anexo tabela Ibirubá={estat['anexo_tabela_ibiruba']}, "
              f"anexo tabela outros campi={estat['anexo_tabela_outro']} (descartadas), "
              f"anexos gerais={estat['anexo_geral']}")

        # Anexos que são de tabela e se têm alguma página do Ibirubá
        anexos_tab = collections.OrderedDict()
        for d in info:
            if d["anexo"] and d["categoria"].startswith("anexo_tabela"):
                anexos_tab[d["anexo"]] = anexos_tab.get(d["anexo"], False) or d["categoria"] == "anexo_tabela_ibiruba"

        # ── Parte B: cronograma ──────────────────────────────────────────────
        crono = _extrair_cronograma(pdf, textos, info)
        avisos += crono["avisos"]
        if crono["linhas"]:
            for i, bbs in crono["bboxes"].items():
                pg = pdf.pages[i]
                for bb in bbs:
                    pg = pg.outside_bbox(bb)     # remove o texto fragmentado da tabela
                textos[i] = pg.extract_text() or ""
        else:
            avisos.append("cronograma: tabela não detectada; mantido o texto bruto das páginas")

        # ── Percurso das páginas ─────────────────────────────────────────────
        secao_atual = ("0", "PREÂMBULO DO EDITAL")
        estado_lista = "todos"
        n_linhas_tabela = 0
        n_itens_descartados = 0
        # file_path: com ids estáveis guarda só o nome do arquivo (o caminho muda entre máquinas e SO)
        meta_fixa = {"source": nome, "file_path": caminho_pdf if IDX_IDS == "legado" else nome}
        corpo_linhas: list = []

        def registrar(rot, tit, tipo="corpo", tem_ibiruba=True):
            if rot not in secoes:
                secoes[rot] = {"titulo": tit, "tipo": tipo, "tem_ibiruba": tem_ibiruba}
            elif tit and not secoes[rot]["titulo"]:
                secoes[rot]["titulo"] = tit

        registrar("0", "PREÂMBULO DO EDITAL", "preambulo")

        for i, page in enumerate(pdf.pages):
            d = info[i]
            pagina = i + 1
            cat = d["categoria"]
            if d["anexo"]:
                registrar(d["anexo"], d["anexo_titulo"], "anexo_tabela" if d["anexo"] in anexos_tab else "anexo_geral",
                          anexos_tab.get(d["anexo"], True))
                if d["anexo"] in anexos_tab:
                    secoes[d["anexo"]]["tem_ibiruba"] = anexos_tab[d["anexo"]]

            if cat == "anexo_tabela_outro":
                continue

            if cat == "anexo_tabela_ibiruba":
                secao = (d["anexo"], d["anexo_titulo"])
                secao_atual = secao
                base = dict(meta_fixa, page_number=pagina, pagina_ini=pagina, pagina_fim=pagina)
                try:
                    brutas = _linhas_tabela_pagina_cab(page)
                except Exception as e:
                    print(f"⚠️  Aviso: erro ao processar tabelas da página {pagina} de {nome}: {e}")
                    brutas = []
                linhas_ok = [(l, rot) for l, rot in brutas if _validar_cursos([l], nome)[0]]
                for linha, rotulo in linhas_ok:
                    conteudo = linha
                    if IDX_TABELA_CABECALHO:
                        conteudo = _cabecalho_tabela(d["anexo"], d["anexo_titulo"], rotulo, linha) + "\n" + linha
                    docs.append(Document(content=conteudo, meta=dict(
                        base, tipo="tabela", secao=secao[0], secao_titulo=secao[1], secoes=secao[0],
                        escopo_campus="ibiruba", split_id=0, split_idx_start=0)))
                n_linhas_tabela += len(linhas_ok)
                if not d["mista"]:
                    leg = _texto_legenda(page)
                    docs += _chunkar(leg, dict(base, tipo="legenda"), secao, [], [], "ibiruba")
                continue

            # corpo e anexo_geral
            linhas = [l for l in _limpar(textos[i]).split("\n") if l.strip()]
            if cat == "corpo":
                linhas, flags, estado_lista, nd = _filtrar_itens_campus(linhas, estado_lista)
                n_itens_descartados += nd
            elif d["anexo"] in _ANEXOS_LINHAS_POR_CAMPUS:
                linhas, flags, nd = _filtrar_linhas_por_campus(linhas)
                n_itens_descartados += nd
            else:
                flags = [False] * len(linhas)
            texto, offs, spans = _montar_pagina(linhas, flags)
            titulos = _detectar_titulos(linhas, offs, corpo=(cat == "corpo"))
            titulos_reg = []
            for off, rot, tit in titulos:
                if cat == "corpo" and rot in secoes and secoes[rot]["tipo"] == "corpo":
                    rot = rot + "b"                      # o edital tem dois itens "20."
                registrar(rot, tit, "corpo" if cat == "corpo" else "anexo_geral")
                titulos_reg.append((off, rot, tit))
            if cat == "anexo_geral" and d["anexo"] and not titulos_reg and secao_atual[0] != d["anexo"]:
                secao_atual = (d["anexo"], d["anexo_titulo"])   # página de continuação do anexo
            base = dict(meta_fixa, page_number=pagina, pagina_ini=pagina, pagina_fim=pagina,
                        tipo="corpo" if cat == "corpo" else "anexo_geral")
            if cat == "corpo" and IDX_CHUNKER != "atual":
                por_offset = {off: (rot, tit) for off, rot, tit in titulos_reg}
                for lin, off, fl in zip(linhas, offs, flags):
                    corpo_linhas.append({"texto": lin.strip(), "pagina": pagina, "ibi": bool(fl),
                                         "titulo": por_offset.get(off)})
            else:
                docs += _chunkar(texto, base, secao_atual, titulos_reg, spans, "todos")
            if titulos_reg:
                secao_atual = (titulos_reg[-1][1], titulos_reg[-1][2])

        if corpo_linhas:
            docs += _chunkar_corpo_por_item(corpo_linhas, meta_fixa, com_titulo=(IDX_CHUNKER == "item_titulo"))

        # ── Chunks de cronograma ─────────────────────────────────────────────
        if crono["linhas"]:
            registrar("2", "CRONOGRAMA DO PROCESSO SELETIVO")
            for grupo in _agrupar_cronograma(crono["linhas"]):
                conteudo = "\n".join(f"Data: {dt} | Evento: {ev}" for dt, ev, _ in grupo)
                pgs = [p for _, _, p in grupo]
                docs.append(Document(content=conteudo, meta=dict(
                    meta_fixa, tipo="cronograma", secao="2", secao_titulo=secoes["2"]["titulo"], secoes="2",
                    escopo_campus="todos", page_number=min(pgs), pagina_ini=min(pgs), pagina_fim=max(pgs),
                    split_id=0, split_idx_start=0)))

    # Ordem do documento (página; dentro da página, o cronograma depois do corpo). A ordem de gravação
    # é a ordem em que nucleo.py injeta os chunks tipo="cronograma" no contexto do híbrido (modo "todos").
    docs.sort(key=lambda d: (d.meta["pagina_ini"], d.meta["tipo"] == "cronograma"))
    if IDX_IDS == "estavel":
        docs = _atribuir_ids_estaveis(docs, nome)

    return {"nome": nome, "docs": docs, "secoes": secoes, "info": info, "n_linhas_tabela": n_linhas_tabela,
            "avisos": avisos, "n_itens_descartados": n_itens_descartados, "cronograma": crono}


# ═════════════════════════════════════════════════════════════════════════════
# Portões
# ═════════════════════════════════════════════════════════════════════════════

def _portao_legibilidade(docs: list) -> tuple:
    """Rejeita chunks ilegíveis. Retorna (aceitos, rejeitados[(doc, motivo)], estatística_literal)."""
    df = collections.Counter()
    for d in docs:
        df.update({t.lower() for t in _TOKEN.findall(d.content) if len(t) >= 4})

    aceitos, rejeitados, so_literal = [], [], []
    for d in docs:
        if d.meta.get("tipo") not in _TIPOS_COM_PORTAO:
            aceitos.append(d)
            continue
        if len(_TOKEN.findall(d.content)) < _MIN_TOKENS_PORTAO:
            aceitos.append(d)
            continue
        curtos = _frac_curtos(d.content, contar_digitos=False, compostos=True)
        toks4 = [t.lower() for t in _TOKEN.findall(d.content) if len(t) >= 4]
        freq = (sum(1 for t in toks4 if df[t] >= 3) / len(toks4)) if toks4 else 0.0
        motivo = None
        if curtos >= _CURTOS_MAX:
            motivo = f"{100 * curtos:.0f}% dos tokens com 1-2 caracteres"
        elif len(toks4) >= 5 and freq < _FREQ_MIN:
            motivo = f"só {100 * freq:.0f}% dos tokens de 4+ caracteres aparecem em >=3 chunks"
        if motivo:
            rejeitados.append((d, motivo))
        else:
            aceitos.append(d)
            if _frac_curtos(d.content, contar_digitos=True, compostos=True) >= _CURTOS_MAX:
                so_literal.append(d)       # a regra literal (contando dígitos) também o rejeitaria
    return aceitos, rejeitados, so_literal


def _cobertura_secoes(resultado: dict, aceitos: list) -> list:
    """Linhas [(rótulo, título, n_chunks, páginas, status)] para as seções de nível 1 do PDF."""
    nome = resultado["nome"]
    linhas = []
    for rot, s in resultado["secoes"].items():
        if s["tipo"] == "preambulo":
            continue
        meus = [d for d in aceitos if d.meta.get("source") == nome and rot in d.meta["secoes"].split(",")]
        paginas = [p for d in meus for p in range(d.meta["pagina_ini"], d.meta["pagina_fim"] + 1)]
        if meus:
            status = "ok"
        elif s["tipo"] == "anexo_tabela" and not s["tem_ibiruba"]:
            status = "sem oferta do campus (esperado)"
        else:
            status = "FALHA"
        linhas.append((rot, s["titulo"], len(meus), _intervalos(paginas), status))
    return linhas


def _imprimir_tabela_secoes(linhas: list) -> None:
    print("\n📑 Cobertura por seção de nível 1:")
    print(f"   {'seção':<10} {'nº chunks':>9}  {'páginas':<14} {'status':<32} título")
    for rot, tit, n, pgs, status in linhas:
        print(f"   {rot:<10} {n:>9}  {pgs:<14} {status:<32} {tit[:55]}")


# ═════════════════════════════════════════════════════════════════════════════
# Compatibilidade (debug_linhas.py e similares)
# ═════════════════════════════════════════════════════════════════════════════

def extrair_tabelas_como_texto(caminho_pdf: str) -> list:
    """Linhas de tabela do Campus Ibirubá de um PDF (usa a classificação por página)."""
    linhas: list = []
    with pdfplumber.open(caminho_pdf) as pdf:
        textos = [(p.extract_text() or "") for p in pdf.pages]
        info = _classificar_paginas(textos)
        for i, page in enumerate(pdf.pages):
            if info[i]["categoria"] == "anexo_tabela_ibiruba":
                linhas += _linhas_tabela_pagina(page)
    return _validar_cursos(linhas)[0]


# ═════════════════════════════════════════════════════════════════════════════
# Indexação
# ═════════════════════════════════════════════════════════════════════════════

def indexar_documentos(
    pasta_docs: str = "./documentos",
    pasta_chroma: str = "./chroma_data",
) -> dict:
    """
    Processa todos os PDFs de pasta_docs e persiste o índice em pasta_chroma
    via ChromaDocumentStore (embedded, sem servidor).
    Retorna {"n_arquivos", "n_chunks", "n_linhas_tabela", "por_tipo", "secoes", "rejeitados"}.
    Levanta exceção se não houver PDFs, se algo falhar ou (IndexacaoIncompleta) se alguma seção de
    nível 1 ficar sem chunk. O índice anterior só é removido DEPOIS de os portões passarem.
    """
    pasta_docs = str(pasta_docs)
    pasta_chroma = str(pasta_chroma)

    arquivos = sorted(Path(pasta_docs).glob("*.pdf"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {pasta_docs}\n"
            "Coloque os PDFs na pasta e rode novamente."
        )

    if IDX_CHUNKER not in _CHUNKERS:
        raise ValueError(f"IDX_CHUNKER inválido: {IDX_CHUNKER!r} (use {', '.join(_CHUNKERS)})")
    if IDX_IDS not in ("estavel", "legado"):
        raise ValueError(f"IDX_IDS inválido: {IDX_IDS!r} (use estavel ou legado)")
    print(f"⚙️  Variantes: chunker={IDX_CHUNKER} · cabeçalho de tabela={'sim' if IDX_TABELA_CABECALHO else 'não'} · "
          f"ids={IDX_IDS} · prefixos E5={'sim' if E5_PREFIXOS else 'não'}")
    print(f"📄 {len(arquivos)} PDF(s) encontrado(s):")
    for a in arquivos:
        print(f"   • {a.name}")
    print("\n⏳ Extraindo e segmentando...")

    resultados = []
    for arq in arquivos:
        resultados.append(_processar_pdf(str(arq)))
    todos = [d for r in resultados for d in r["docs"]]
    if len({d.id for d in todos}) != len(todos):
        raise RuntimeError("ids de chunk duplicados (o Chroma sobrescreveria chunks)")
    n_linhas_tabela = sum(r["n_linhas_tabela"] for r in resultados)
    for r in resultados:
        for av in r["avisos"]:
            print(f"   ⚠️  {av}")

    # Portão de legibilidade
    aceitos, rejeitados, so_literal = _portao_legibilidade(todos)
    print(f"\n🧹 Portão de legibilidade: {len(rejeitados)} chunk(s) rejeitado(s) de {len(todos)}")
    for d, motivo in rejeitados:
        txt = " ".join(d.content.split())[:80]
        print(f"   ✗ id={d.id[:8]} pág.{d.meta['pagina_ini']} {d.meta['tipo']:<11} ({motivo}) | {txt!r}")
    if so_literal:
        print(f"   ℹ️  {len(so_literal)} chunk(s) aceito(s) teriam sido rejeitados se os tokens numéricos "
              f"contassem como curtos (ex.: tabelas de números): " + ", ".join(d.id[:8] for d in so_literal))

    # Portão de cobertura
    tabela = []
    for r in resultados:
        tabela += _cobertura_secoes(r, aceitos)
    _imprimir_tabela_secoes(tabela)
    faltando = [t for t in tabela if t[4] == "FALHA"]
    if faltando:
        msg = ("Indexação ABORTADA: seção(ões) de nível 1 sem nenhum chunk: "
               + ", ".join(f"{t[0]} ({t[1][:40]})" for t in faltando)
               + ". O índice anterior foi preservado.")
        raise IndexacaoIncompleta(msg)

    # ── A partir daqui grava em disco ────────────────────────────────────────
    chroma_path = Path(pasta_chroma)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f"\n   🗑️  Índice anterior em {pasta_chroma} removido")
    chroma_path.mkdir(parents=True, exist_ok=True)

    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=pasta_chroma)
    embedder = SentenceTransformersDocumentEmbedder(model="intfloat/multilingual-e5-base",
                                                    prefix="passage: " if E5_PREFIXOS else "")
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    embedder.warm_up()

    # Auditoria pós-indexação: cabeçalho de outro campus dentro de um chunk (contaminação real).
    print("\n🔍 Auditoria pós-indexação...")
    n_audit_falhas = 0
    for chunk in aceitos:
        m_hdr = _RE_CAMPUS_OUTRO_TEXTO.search(chunk.content or "")
        if m_hdr:
            print(f"   [AUDITORIA] cabeçalho id={chunk.id[:8]} campus={m_hdr.group().strip()!r} | "
                  f"{chunk.content[:100]!r}")
            n_audit_falhas += 1

    embedded = embedder.run(documents=aceitos)
    writer.run(documents=embedded["documents"])
    n_chunks = document_store.count_documents()

    por_tipo = collections.Counter(d.meta["tipo"] for d in aceitos)
    n_cota = sum(1 for d in aceitos if d.meta["tipo"] != "tabela" and _PADRAO_COTA.search(d.content))
    print(f"\n✅ Indexação concluída! {n_chunks} chunks persistidos em {pasta_chroma}/")
    for t in ("corpo", "tabela", "cronograma", "legenda", "anexo_geral"):
        print(f"   • {t:<12}: {por_tipo.get(t, 0)}")
    print(f"   • chunks não-tabela com padrão de cota 'C<n>:' : {n_cota}  "
          f"(o pós-filtro de responder() em nucleo.py os descarta na consulta)")
    print(f"   • Rejeitados pelo portão de legibilidade: {len(rejeitados)}")
    print(f"   • Linhas descartadas por escopo de campus (itens de lista/linhas de tabela): "
          f"{sum(r['n_itens_descartados'] for r in resultados)}")
    if n_audit_falhas:
        print(f"   ⚠️  Alertas de auditoria: {n_audit_falhas} (revisar acima)")
    else:
        print("   ✅ Auditoria: nenhum chunk com cabeçalho de campus externo")
    _imprimir_tabela_secoes(tabela)

    return {
        "n_arquivos": len(arquivos), "n_chunks": n_chunks, "n_linhas_tabela": n_linhas_tabela,
        "por_tipo": dict(por_tipo), "secoes": tabela,
        "rejeitados": [(d.id, d.meta["pagina_ini"], motivo, " ".join(d.content.split())[:80])
                       for d, motivo in rejeitados],
    }


def main() -> int:
    """CLI: retorna 0 em sucesso e 1 se o portão de cobertura abortar a indexação."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pasta_docs = os.getenv("PASTA_DOCS", "./documentos")
    pasta_chroma = os.getenv("PASTA_CHROMA", "./chroma_data")
    try:
        indexar_documentos(pasta_docs, pasta_chroma)
    except IndexacaoIncompleta as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        return 1
    print("   Agora execute: streamlit run main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
