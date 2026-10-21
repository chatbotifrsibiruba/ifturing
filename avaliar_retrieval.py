"""
avaliar_retrieval.py — Mede só o RETRIEVAL (sem LLM) contra o gabarito de PIPELINE.md.

Cada pergunta do gabarito tem uma coluna EVIDENCIA: trechos literais, separados por " ; ",
que precisam aparecer no contexto recuperado. Para cada pergunta calcula-se o menor k em
que TODAS as evidências estão na união dos textos dos k primeiros chunks, em três modos:

  semantico    busca vetorial pura (top-k do Chroma, sem nenhum pós-processamento)
  hib-top3     mesmo fluxo de nucleo.responder(): semântico → pós-filtro de cotas → injeção de tabelas →
               injeção de cronograma = os 3 chunks tipo="cronograma" mais similares à pergunta
  hib-todos    idem, mas injeta o cronograma inteiro (todos os chunks tipo="cronograma")

O cronograma é sempre selecionado pelo metadado tipo="cronograma" (nucleo.selecionar_cronograma). O critério
antigo (palavra "cronograma" ou >= 3 datas dd/mm/202X) foi removido de nucleo.py; sobrevive aqui só em
_cronograma_like_legado(), para (a) auditar o que ele ainda injetaria no índice novo e (b) reproduzir o fluxo
da coleção antiga da fase 1, que não tem chunks tipo="cronograma" (nela os dois modos híbridos são idênticos).

As perguntas ficam em dois GRUPOS, com métricas sempre separadas:
  legado   as 30 perguntas da fase 1 (A01–A08, B01–B12, C01–C07, D02, E01, E02)
  novas    perguntas criadas depois (F10a…F18c, E03, E04, E05)

Convenções do gabarito (ver PIPELINE.md, seção 4):
  • EVIDENCIA = NEGATIVA   pergunta negativa (D01, D03): fora das métricas, avaliada só pela resposta.
  • EVIDENCIA_LEGADO       evidência da fase 1 (formato antigo do índice). Só vale para coleções
                           marcadas com --legado (comparação com o índice ANTIGO); a referência para o
                           índice novo é sempre a EVIDENCIA atual, sem --legado.

Métricas (por coleção, modo e grupo):
  hit@k   fração de perguntas com k_min <= k, para k = 3, 5, 10
  MRR     média de 1/k_min (0 quando as evidências nunca aparecem completas)
  hit@ctx fração de perguntas cobertas pelo contexto inteiro (no híbrido pode passar de 10)
  chars_ctx  caracteres do contexto enviado ao LLM (média), montado como em nucleo.responder()
  no índice  evidências que não existem em NENHUM chunk da coleção (teto de recall: nenhum
             retriever consegue trazê-las; indica problema de indexação, não de busca)

Para cada pergunta que falha em k=10 o relatório traz a posição (rank) do chunk com a evidência no ranking
semântico completo da coleção ("a 1–2 posições do top-10" = ajuste de ranking; rank alto = problema de
chunking/indexação) e a posição dele no contexto híbrido.

Somente leitura: nunca escreve em coleções Chroma. Aponte --run para coleções temporárias
(./chroma_exp_*) ou para a de produção.

Uso:
    python avaliar_retrieval.py                          # coleção PASTA_CHROMA
    python avaliar_retrieval.py --run base=./chroma_exp_prod --legado base \\
                                --run fase2=./chroma_exp_fase2
    python avaliar_retrieval.py --run producao=./chroma_exp_prod \\
                                --run prefixos=./chroma_exp_prefix::"query: "
    python avaliar_retrieval.py --ids C01,C03,A04
    python avaliar_retrieval.py --tabelas B08,E02 --foco E03,F11a,F10c

A sintaxe NOME=CAMINHO[::PREFIXO_DA_CONSULTA] permite comparar coleções (A/B); com 2 ou mais --run
o relatório inclui a diferença em relação à primeira. Saídas: retrieval_resultado.md e .csv.
"""
import argparse
import contextlib
import csv
import io
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

import nucleo
from nucleo import (
    EMBEDDER_DEVICE,
    MODELO_EMB,
    PASTA_CHROMA,
    TOP_K,
    _PADRAO_COTA,
    _PALAVRAS_CRONOGRAMA,
    _buscar_cronograma_por_keyword,
    _buscar_tabelas_por_keyword,
    cronograma_mais_similares,
    selecionar_cronograma,
)

_PIPELINE_MD = Path(__file__).parent / "PIPELINE.md"
_KS = (3, 5, 10)
_MODOS = ("semantico", "hib-top3", "hib-todos")
_MODO_CRONO = {"hib-top3": "topk", "hib-todos": "todos"}   # modo híbrido → CRONOGRAMA_MODO de nucleo.py
_GRUPOS = ("legado", "novas")
_FOCO_PADRAO = "E03,F11a,F10c,F17b,F10a,F15a"
_TABELAS_PADRAO = "B08,E02"

# As 30 perguntas da fase 1 (linha de base). Não alterar: define o grupo "legado".
_IDS_LEGADO = frozenset(
    [f"A{i:02d}" for i in range(1, 9)]
    + [f"B{i:02d}" for i in range(1, 13)]
    + [f"C{i:02d}" for i in range(1, 8)]
    + ["D02", "E01", "E02"]
)

# Critério ANTIGO de "cronograma-like" (nucleo.py até a fase 2B; REMOVIDO de lá). Ver docstring do módulo.
_DATA_RECENTE_LEGADO = re.compile(r"\d{1,2}/\d{2}/202[4-9]")


def _cronograma_like_legado(d) -> bool:
    if (d.meta or {}).get("tipo") == "tabela":
        return False
    texto = d.content or ""
    return "cronograma" in texto.lower() or len(_DATA_RECENTE_LEGADO.findall(texto)) >= 3


# ── Gabarito com EVIDENCIA ────────────────────────────────────────────────

def _celulas(linha: str) -> list:
    """Divide uma linha de tabela markdown em células, respeitando '\\|' escapado."""
    s = linha.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    partes = re.split(r"(?<!\\)\|", s)
    return [re.sub(r"\\([|*])", r"\1", p).strip() for p in partes]


def _evidencias(bruto: str) -> list:
    return [e.strip() for e in re.split(r"\s;\s", bruto) if e.strip()]


def carregar_gabarito(caminho: Path = _PIPELINE_MD) -> dict:
    """Retorna {id: {pergunta, evidencias, evidencias_legado, negativa, grupo}} para as tabelas do
    PIPELINE.md com coluna EVIDENCIA.

    `evidencias` é [] quando a célula está vazia (gabarito sem evidência) ou é NEGATIVA."""
    gabarito: dict = {}
    cabecalho = None
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip().startswith("|"):
            cabecalho = None
            continue
        celulas = _celulas(linha)
        if celulas and celulas[0] == "#":
            cabecalho = [c.upper() for c in celulas]
            continue
        if cabecalho is None or "EVIDENCIA" not in cabecalho:
            continue
        if not re.fullmatch(r"[A-F]\d{2}[a-z]?", celulas[0]):
            continue
        i_ev = cabecalho.index("EVIDENCIA")
        bruto = celulas[i_ev] if i_ev < len(celulas) else ""
        i_leg = cabecalho.index("EVIDENCIA_LEGADO") if "EVIDENCIA_LEGADO" in cabecalho else None
        bruto_leg = celulas[i_leg] if i_leg is not None and i_leg < len(celulas) else ""
        negativa = bruto.strip().upper() == "NEGATIVA"
        qid = celulas[0]
        gabarito[qid] = {
            "pergunta": celulas[1],
            "negativa": negativa,
            "evidencias": [] if negativa else _evidencias(bruto),
            "evidencias_legado": [] if negativa else _evidencias(bruto_leg),
            "grupo": "legado" if qid in _IDS_LEGADO else "novas",
        }
    return gabarito


def evidencias_para(gab: dict, usar_legado: bool) -> list:
    """Evidências de uma pergunta para uma coleção: a legada só no grupo legado e em coleção --legado."""
    if usar_legado and gab["grupo"] == "legado" and gab["evidencias_legado"]:
        return gab["evidencias_legado"]
    return gab["evidencias"]


def dispara_cronograma(pergunta: str) -> bool:
    return any(kw in pergunta.lower() for kw in _PALAVRAS_CRONOGRAMA)


# ── Execução do retrieval ─────────────────────────────────────────────────

class Colecao:
    """Uma coleção Chroma + embedder de consulta (com prefixo opcional) + docs pré-carregados."""

    def __init__(self, nome: str, caminho: str, prefixo_consulta: str, nome_colecao: str, top_k: int,
                 usar_legado: bool):
        if not Path(caminho).exists():
            sys.exit(f"ERRO: coleção não encontrada em {caminho}")
        self.nome = nome
        self.caminho = caminho
        self.prefixo_consulta = prefixo_consulta
        self.usar_legado = usar_legado
        store = ChromaDocumentStore(collection_name=nome_colecao, persist_path=str(caminho))
        self.n_chunks = store.count_documents()
        if self.n_chunks == 0:
            sys.exit(f"ERRO: coleção vazia em {caminho}")
        # Mesma pré-carga de nucleo.carregar_pipeline()
        todos = store.filter_documents()
        self.todos = todos
        self.textos_indice = [_norm(d.content or "") for d in todos]
        self.por_tipo = Counter((d.meta or {}).get("tipo", "texto") for d in todos)
        self.tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
        self.cronograma_docs = selecionar_cronograma(todos)
        # Índice da fase 1: sem metadado tipo="cronograma" → só o critério antigo consegue achar o cronograma.
        self.crono_legado = not self.cronograma_docs
        if self.crono_legado:
            self.cronograma_docs = [d for d in todos if _cronograma_like_legado(d)]
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "embedder",
            SentenceTransformersTextEmbedder(
                model=MODELO_EMB,
                device=ComponentDevice.from_str(EMBEDDER_DEVICE),
                prefix=prefixo_consulta,
            ),
        )
        self.pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=store, top_k=top_k))
        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        # Ranking semântico de TODOS os chunks (só para diagnóstico de posição; não entra nas métricas)
        self.retriever_completo = ChromaEmbeddingRetriever(document_store=store, top_k=self.n_chunks)

    def semantico(self, pergunta: str) -> tuple:
        """(top-k semântico, embedding da consulta, ranking completo da coleção)."""
        r = self.pipeline.run({"embedder": {"text": pergunta}}, include_outputs_from={"embedder"})
        emb = r["embedder"]["embedding"]
        ranking = self.retriever_completo.run(query_embedding=emb)["documents"]
        return r["retriever"]["documents"], emb, ranking

    def hibrido(self, pergunta: str, docs_semanticos: list, emb: list, modo: str) -> list:
        """Reproduz nucleo.responder() (pós-filtro de cotas + híbrido de tabela + de cronograma)."""
        docs = [
            d for d in docs_semanticos
            if (d.meta or {}).get("tipo") == "tabela" or not _PADRAO_COTA.search(d.content or "")
        ]
        docs_tabela_extra = _buscar_tabelas_por_keyword(pergunta, self.tabela_docs, docs)
        if docs_tabela_extra:
            docs_tabela_ja = [d for d in docs if (d.meta or {}).get("tipo") == "tabela"]
            docs_texto_final = [d for d in docs if (d.meta or {}).get("tipo") != "tabela"][:6]
            docs = docs_tabela_ja + docs_tabela_extra + docs_texto_final
        docs_crono_extra = _buscar_cronograma_por_keyword(
            pergunta, self.cronograma_docs, docs,
            ranquear=lambda k: cronograma_mais_similares(self.pipeline, emb, k),
            modo="todos" if self.crono_legado else _MODO_CRONO[modo],
        )
        if docs_crono_extra:
            docs = docs_crono_extra + docs
        return docs


# ── Métricas ──────────────────────────────────────────────────────────────

def _norm(t: str) -> str:
    return " ".join(t.split()).casefold()


def avaliar_lista(docs: list, evidencias: list) -> dict:
    """k_min e cobertura por k para uma lista ordenada de chunks.

    Uma evidência conta se aparece inteira no texto de algum dos k primeiros chunks
    (a união dos textos não cria trechos que atravessem a fronteira de dois chunks).
    """
    ev_norm = [_norm(e) for e in evidencias]
    textos = [_norm(d.content or "") for d in docs]
    achadas: set = set()
    k_min = None
    cobertura = {}
    for k, texto in enumerate(textos, 1):
        achadas |= {i for i, e in enumerate(ev_norm) if e in texto}
        if k in _KS:
            cobertura[k] = len(achadas)
        if k_min is None and len(achadas) == len(ev_norm):
            k_min = k
    for k in _KS:  # lista menor que k: cobertura final
        cobertura.setdefault(k, len(achadas))
    faltando_10 = [e for i, e in enumerate(evidencias)
                   if not any(ev_norm[i] in t for t in textos[:10])]
    return {
        "k_min": k_min,
        "n_contexto": len(docs),
        "cobertura": cobertura,
        "faltando_10": faltando_10,
    }


def agregar(resultados: list) -> dict:
    n = len(resultados)
    if n == 0:
        return {"n": 0, "hit@3": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0, "hit@ctx": 0.0, "chars": 0.0}
    agg = {"n": n}
    for k in _KS:
        agg[f"hit@{k}"] = sum(1 for r in resultados if r["k_min"] is not None and r["k_min"] <= k) / n
    agg["mrr"] = sum(1 / r["k_min"] for r in resultados if r["k_min"]) / n
    agg["hit@ctx"] = sum(1 for r in resultados if r["k_min"] is not None) / n
    agg["chars"] = sum(r["chars_ctx"] for r in resultados) / n
    return agg


def _id8(d) -> str:
    return (d.id or "?")[:8]


def _chunk_resumo(d, n: int = 80) -> str:
    meta = d.meta or {}
    return (f"{_id8(d)}  tipo={meta.get('tipo', 'texto')}  seção={meta.get('secao', '—')}  "
            f"{' '.join((d.content or '').split())[:n]!r}")


def _ev_curta(e: str, n: int = 70) -> str:
    e = e.replace("|", "\\|")
    return e if len(e) <= n else e[: n - 1] + "…"


def _ranks_evidencia(c: Colecao, ranking: list, evidencias: list) -> dict:
    """{evidência: [(rank no ranking completo, chunk), ...]} ordenado do melhor para o pior rank."""
    rank = {d.id: i for i, d in enumerate(ranking, 1)}
    out = {}
    for e in evidencias:
        en = _norm(e)
        achados = [(rank.get(d.id), d) for d, t in zip(c.todos, c.textos_indice) if en in t]
        out[e] = sorted(achados, key=lambda x: (x[0] is None, x[0] or 0))
    return out


# ── Relatório ─────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _fmt_k(k) -> str:
    return "—" if k is None else str(k)


def _falhas(c, modo, ids, res) -> list:
    return [q for q in ids if res[(c.nome, modo, q)]["k_min"] is None or res[(c.nome, modo, q)]["k_min"] > 10]


def _nota_evidencia(c: Colecao) -> str:
    if c.usar_legado:
        return ("**EVIDENCIA_LEGADO (fase 1) — só para comparar com o índice ANTIGO; NÃO é a referência do índice novo** "
                "(no grupo novas não existe evidência legada: vale a atual)")
    return "EVIDENCIA atual (**referência para o índice novo**)"


def _nota_cronograma(c: Colecao) -> str:
    if c.crono_legado:
        return (f"{len(c.cronograma_docs)} chunks pelo critério antigo (índice sem metadado tipo=cronograma; "
                "hib-top3 e hib-todos são idênticos)")
    return f"{len(c.cronograma_docs)} chunks tipo=cronograma (metadado)"


def _secao_cronograma(L: list, c: Colecao, gabarito: dict, ids_grupo: dict, ctx_ids: dict) -> None:
    docs = c.cronograma_docs
    total = len("\n\n---\n\n".join(d.content or "" for d in docs))
    L += ["", f"## Cronograma injetado — {c.nome}", "",
          f"{_nota_cronograma(c)}. Cronograma inteiro: **{total:,} caracteres ≈ {total / 3.5:,.0f} tokens** "
          f"(caracteres / 3,5; teto `num_ctx` = 8192 tokens, prompt ≈ 430 tokens + resposta até 1024).", "",
          "| chunk | seção | caracteres | tokens est. | início |", "|---|---|---|---|---|"]
    for d in docs:
        n = len(d.content or "")
        L.append(f"| {_id8(d)} | {(d.meta or {}).get('secao', '—')} | {n:,} | {n / 3.5:,.0f} | "
                 f"{' '.join((d.content or '').split())[:60]!r} |")
    perguntas = [q for g in _GRUPOS for q in ids_grupo[g] if dispara_cronograma(gabarito[q]["pergunta"])]
    L += ["", f"Perguntas que disparam o cronograma (palavra de `_PALAVRAS_CRONOGRAMA`): {len(perguntas)} — "
          f"{', '.join(perguntas)}", "",
          "Chunks de cronograma injetados, na ordem do contexto (os já presentes no top-10 semântico não são repetidos):", "",
          "| id | hib-top3 | hib-todos |", "|---|---|---|"]
    for q in perguntas:
        def ids(modo):
            return " ".join(i for i in ctx_ids[(c.nome, modo, q)]["crono"]) or "—"
        L.append(f"| {q} | {ids('hib-top3')} | {ids('hib-todos')} |")


def _secao_criterio_antigo(L: list, c: Colecao, gabarito: dict, ids_grupo: dict) -> None:
    """B: o que o critério antigo ainda injetaria (além dos tipo=cronograma) e se alguma evidência depende disso."""
    if c.crono_legado:
        return
    antigo = [d for d in c.todos if _cronograma_like_legado(d)]
    ids_meta = {d.id for d in c.cronograma_docs}
    so_antigo = [d for d in antigo if d.id not in ids_meta]
    perdidos = [d for d in c.cronograma_docs if d.id not in {x.id for x in antigo}]
    L += ["", f"## Critério antigo de cronograma (removido de nucleo.py) — {c.nome}", "",
          f"O critério antigo (palavra \"cronograma\" ou ≥ 3 datas `dd/mm/202X`, em chunk que não é tabela) selecionaria "
          f"**{len(antigo)}** chunks neste índice: **{len(antigo) - len(so_antigo)}** são tipo=cronograma e "
          f"**{len(so_antigo)}** não são (só ele os injetaria). Chunks tipo=cronograma que ele deixaria de fora: "
          f"{len(perdidos)}.", "",
          "| chunk | tipo | seção | caracteres | início | evidências do gabarito (EVIDENCIA atual) que contêm |",
          "|---|---|---|---|---|---|"]
    todos_ids = [q for g in _GRUPOS for q in ids_grupo[g]]
    for d in so_antigo:
        t = _norm(d.content or "")
        achadas = []
        for q in todos_ids:
            for e in gabarito[q]["evidencias"]:
                if _norm(e) in t:
                    outros = sum(1 for dd, tt in zip(c.todos, c.textos_indice) if dd.id != d.id and _norm(e) in tt)
                    achadas.append(f"{q}: {_ev_curta(e, 50)} (também em {outros} outro(s) chunk(s))")
        meta = d.meta or {}
        L.append(f"| {_id8(d)} | {meta.get('tipo', 'texto')} | {meta.get('secao', '—')} | {len(d.content or ''):,} | "
                 f"{' '.join((d.content or '').split())[:60]!r} | {'; '.join(achadas) or '**nenhuma**'} |")
    if not so_antigo:
        L.append("| (nenhum) | | | | | |")


_RE_TRACE = re.compile(r"id=(?P<id>'[^']*'|\"[^\"]*\")\s+tipo=(?P<tipo>\S+)\s+semantico=(?P<sem>True|False)\s+"
                       r"bate_tipo=(?P<bt>True|False)\s+bate_nome=(?P<bn>True|False)")
_RE_CAB = re.compile(r"\[debug_hibrido\] (termos_pergunta|tipos_solicitados|termos_match)\s*= (.*)")


def _diagnostico_tabelas(c: Colecao, q: str, pergunta: str, docs_sem: list, ctx_todos: list) -> list:
    """E: as tabelas do campus chegam ao contexto híbrido? Se não, qual critério de
    _buscar_tabelas_por_keyword as deixa de fora (trace do próprio nucleo.py via _DEBUG_HIBRIDO)."""
    docs = [d for d in docs_sem if (d.meta or {}).get("tipo") == "tabela" or not _PADRAO_COTA.search(d.content or "")]
    buf = io.StringIO()
    anterior, nucleo._DEBUG_HIBRIDO = nucleo._DEBUG_HIBRIDO, True
    try:
        with contextlib.redirect_stdout(buf):
            injetadas = _buscar_tabelas_por_keyword(pergunta, c.tabela_docs, docs)
    finally:
        nucleo._DEBUG_HIBRIDO = anterior
    trace = buf.getvalue()
    cab = {m.group(1): m.group(2) for m in _RE_CAB.finditer(trace)}
    linhas = {m.group("id").strip("'\""): m for m in _RE_TRACE.finditer(trace)}
    pos_sem = {d.id: i for i, d in enumerate(docs_sem, 1)}
    pos_ctx = {d.id: i for i, d in enumerate(ctx_todos, 1)}
    ids_inj = {d.id for d in injetadas}
    L = ["", f"**{q}** — {pergunta}", ""]
    if "skip" in trace and not cab:
        L.append(f"- porta de entrada: `{trace.strip()}` → nenhuma tabela é injetada")
    else:
        L += [f"- `termos_pergunta` = {cab.get('termos_pergunta', '?')}",
              f"- `tipos_solicitados` (critério 1, tipo de curso) = {cab.get('tipos_solicitados', '?')}",
              f"- `termos_match` (critério 2, nome específico) = {cab.get('termos_match', '?')}"]
    no_ctx = sum(1 for d in c.tabela_docs if d.id in pos_ctx)
    L += [f"- tabelas no contexto híbrido: **{no_ctx}/{len(c.tabela_docs)}** "
          f"({len(ids_inj)} injetadas por palavra-chave + {sum(1 for d in c.tabela_docs if d.id in pos_sem)} já no top-10 semântico)", "",
          "| chunk | tipo inferido | posição no top-10 sem. | ja_semantico | bate_tipo | bate_nome | injetado | posição no contexto híb. | início do chunk |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in c.tabela_docs:
        m = linhas.get(d.id)
        tipo, sem, bt, bn = (m.group("tipo"), m.group("sem"), m.group("bt"), m.group("bn")) if m else ("—",) * 4
        L.append(f"| {_id8(d)} | {tipo} | {pos_sem.get(d.id, '—')} | {sem} | {bt} | {bn} | "
                 f"{'sim' if d.id in ids_inj else 'não'} | {pos_ctx.get(d.id, '**fora**')} | "
                 f"{' '.join((d.content or '').split())[:50]!r} |")
    return L


def gerar_relatorio(colecoes: list, gabarito: dict, ids_grupo: dict, negativas: list, sem_evidencia: list,
                    res: dict, docs_top: dict, ausentes: dict, rank_ev: dict, ctx_ids: dict,
                    diag_tabelas: dict, foco: list) -> str:
    todos_ids = [q for g in _GRUPOS for q in ids_grupo[g]]
    L = ["# Avaliação de retrieval", ""]
    L.append(f"- Perguntas avaliadas: **legado {len(ids_grupo['legado'])}** (fase 1) · **novas {len(ids_grupo['novas'])}**. "
             "As métricas dos dois grupos nunca são misturadas.")
    L.append(f"- Perguntas negativas (avaliadas só pela resposta \"não consta\"): {', '.join(negativas) or '—'} · "
             f"sem evidência (ignoradas): {', '.join(sem_evidencia) or '—'}")
    L.append(f"- Modelo de embedding: `{MODELO_EMB}` (device `{EMBEDDER_DEVICE}`) · top_k semântico: {TOP_K}")
    L.append("- **Referência para o índice novo (fase 2): avaliação com a EVIDENCIA atual, sem `--legado`.** "
             "`--legado` só existe para comparar com o índice antigo da fase 1 (EVIDENCIA_LEGADO).")
    for c in colecoes:
        tipos = ", ".join(f"{t}={n}" for t, n in sorted(c.por_tipo.items()))
        L.append(f"- Coleção **{c.nome}**: `{c.caminho}` — {c.n_chunks} chunks ({tipos}) · cronograma: {_nota_cronograma(c)} · "
                 f"prefixo da consulta: `{c.prefixo_consulta!r}` · evidência: {_nota_evidencia(c)}")
    L += ["", "Modos: **semantico** = top-10 do Chroma; **hib-top3** = fluxo de `nucleo.responder()` injetando os "
          f"{nucleo.CRONOGRAMA_TOP_K} chunks tipo=cronograma mais similares à pergunta (ranking semântico dentro do tipo); "
          "**hib-todos** = idem, injetando o cronograma inteiro.", "",
          "Definições: *k_min* = menor k cujos k primeiros chunks contêm TODAS as evidências; "
          "*hit@k* = k_min ≤ k; *MRR* = média de 1/k_min; *hit@ctx* = coberto pelo contexto inteiro "
          "(no modo híbrido o contexto pode ter mais de 10 chunks); *chars_ctx* = média de caracteres do contexto "
          "(chunks unidos por `\\n\\n---\\n\\n`, como em `nucleo.responder()`).", ""]

    L += ["## Resumo", "",
          "| coleção | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in _GRUPOS:
        for c in colecoes:
            for modo in _MODOS:
                a = agregar([res[(c.nome, modo, q)] for q in ids_grupo[g]])
                if a["n"]:
                    L.append(f"| {c.nome} | {g} | {modo} | {a['n']} | {_pct(a['hit@3'])} | {_pct(a['hit@5'])} | "
                             f"{_pct(a['hit@10'])} | {a['mrr']:.3f} | {_pct(a['hit@ctx'])} | {a['chars']:,.0f} |")

    L += ["", "### Só as perguntas que disparam o cronograma (palavras de `_PALAVRAS_CRONOGRAMA`)", "",
          "| coleção | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in _GRUPOS:
        for c in colecoes:
            qs = [q for q in ids_grupo[g] if dispara_cronograma(gabarito[q]["pergunta"])]
            for modo in _MODOS:
                a = agregar([res[(c.nome, modo, q)] for q in qs])
                if a["n"]:
                    L.append(f"| {c.nome} | {g} | {modo} | {a['n']} | {_pct(a['hit@3'])} | {_pct(a['hit@5'])} | "
                             f"{_pct(a['hit@10'])} | {a['mrr']:.3f} | {_pct(a['hit@ctx'])} | {a['chars']:,.0f} |")

    L += ["", "## Tamanho do contexto no modo híbrido", "",
          "O contexto inteiro vai para o LLM (`nucleo.responder()`), e `chamar_ollama` usa `num_ctx=8192` tokens "
          "(prompt de ~1,5 mil caracteres + resposta de até 1024 tokens). Tokens estimados = caracteres / 3,5 "
          "(aproximação para português).", "",
          "| coleção | grupo | modo | chunks (média / máx) | caracteres (média / máx) | tokens estimados (média / máx) |",
          "|---|---|---|---|---|---|"]
    for c in colecoes:
        for g in _GRUPOS:
            for modo in _MODOS[1:]:
                rs = [res[(c.nome, modo, q)] for q in ids_grupo[g]]
                if rs:
                    nch = [r["n_contexto"] for r in rs]
                    chs = [r["chars_ctx"] for r in rs]
                    L.append(f"| {c.nome} | {g} | {modo} | {sum(nch) / len(nch):.1f} / {max(nch)} | "
                             f"{sum(chs) / len(chs):,.0f} / {max(chs):,} | "
                             f"{sum(chs) / len(chs) / 3.5:,.0f} / {max(chs) / 3.5:,.0f} |")
    for c in colecoes:
        _secao_cronograma(L, c, gabarito, ids_grupo, ctx_ids)
    for c in colecoes:
        _secao_criterio_antigo(L, c, gabarito, ids_grupo)

    L += ["", "## Evidências ausentes do índice (teto de recall)", "",
          "Evidências que não aparecem em nenhum chunk da coleção: nenhuma estratégia de busca resolve; "
          "o problema está na extração/indexação (ou o formato da evidência não é o do índice)."]
    for c in colecoes:
        for g in _GRUPOS:
            aus = [(q, ausentes[(c.nome, q)]) for q in ids_grupo[g] if ausentes[(c.nome, q)]]
            L += ["", f"**{c.nome} · {g}** — {len(aus)} de {len(ids_grupo[g])} pergunta(s) com evidência inexistente no índice"]
            L += [f"- {q}: {e}" for q, e in aus]

    for c in colecoes:
        L += ["", f"## Por pergunta — {c.nome}", "",
              "| id | grupo | pergunta | n ev. | ev. no índice | k_min sem. | k_min top3 | k_min todos | ctx top3 | ctx todos "
              "| ev.@10 sem. | ev.@10 top3 | ev.@10 todos |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in _GRUPOS:
            for q in ids_grupo[g]:
                s, t3, td = (res[(c.nome, m, q)] for m in _MODOS)
                n_ev = s["n_ev"]
                n_idx = n_ev - len(ausentes[(c.nome, q)])
                L.append(f"| {q} | {g} | {gabarito[q]['pergunta'][:60]} | {n_ev} | {n_idx}/{n_ev} | {_fmt_k(s['k_min'])} | "
                         f"{_fmt_k(t3['k_min'])} | {_fmt_k(td['k_min'])} | {t3['n_contexto']} | {td['n_contexto']} | "
                         f"{s['cobertura'][10]}/{n_ev} | {t3['cobertura'][10]}/{n_ev} | {td['cobertura'][10]}/{n_ev} |")

    def linhas_posicao(c, modo, q, so_faltando: bool) -> list:
        r = res[(c.nome, modo, q)]
        ctx = ctx_ids[(c.nome, modo, q)]["todos"]
        pos_ctx = {i: p for p, i in enumerate(ctx, 1)}
        out = []
        for e, achados in rank_ev[(c.nome, q)].items():
            if so_faltando and e not in r["faltando_10"]:
                continue
            if not achados:
                out.append(f"| {q} | {_ev_curta(e)} | **nenhum chunk contém a evidência** | | | | |")
                continue
            for rank, d in achados[:3]:
                meta = d.meta or {}
                dist = "" if rank is None else (f"+{rank - 10}" if rank > 10 else "dentro")
                out.append(f"| {q} | {_ev_curta(e)} | {_id8(d)} | {meta.get('tipo', 'texto')} | {meta.get('secao', '—')} | "
                           f"{rank if rank else '—'} de {c.n_chunks} ({dist}) | {pos_ctx.get(d.id, '—')} |")
        return out

    cab_pos = ["| id | evidência | chunk | tipo | seção | rank semântico | posição no contexto |",
               "|---|---|---|---|---|---|---|"]
    L += ["", "## Falhas em k=10 — posição do chunk com a evidência", "",
          "Para cada evidência que falta nos 10 primeiros chunks: o(s) chunk(s) que a contêm (até 3, do melhor rank), "
          "o **rank dele no ranking semântico completo da coleção** (1 = mais similar; \"+N\" = N posições além do top-10) e a "
          "posição dele no contexto do modo (— = fora do contexto). Rank 11–12 = ajuste de ranking; rank alto = a pergunta "
          "não se parece com o chunk (problema de chunking/indexação)."]
    for c in colecoes:
        for modo in _MODOS:
            falhas = _falhas(c, modo, todos_ids, res)
            L += ["", f"### {c.nome} · {modo} — {len(falhas)} falha(s)", ""]
            if not falhas:
                L.append("(nenhuma)")
                continue
            L += cab_pos
            for q in falhas:
                L += linhas_posicao(c, modo, q, so_faltando=True)
            L += ["", "Pergunta · menor k em que a evidência completa apareceu (— = nunca) · chunks no contexto:", ""]
            for q in falhas:
                r = res[(c.nome, modo, q)]
                L.append(f"- **{q}** ({gabarito[q]['grupo']}) {gabarito[q]['pergunta']} → k_min={_fmt_k(r['k_min'])}, "
                         f"{r['n_contexto']} chunks")

    L += ["", "## Foco: " + ", ".join(foco), ""]
    for c in colecoes:
        for q in foco:
            if q not in gabarito or (c.nome, q) not in rank_ev:
                continue
            status = " · ".join(f"{m}: k_min={_fmt_k(res[(c.nome, m, q)]['k_min'])}" for m in _MODOS)
            L += [f"**{c.nome} · {q}** — {gabarito[q]['pergunta']} ({status})", ""] + cab_pos
            L += linhas_posicao(c, "hib-todos", q, so_faltando=False)
            L.append("")

    L += ["", "## Tabelas do campus no contexto híbrido (B08, E02…)", "",
          "Trace de `_buscar_tabelas_por_keyword` (o próprio `nucleo.py`, com `_DEBUG_HIBRIDO` ligado). "
          "`bate_tipo` = critério 1 (tipo de curso via `_SINONIMOS_TIPO`); `bate_nome` = critério 2 (termo literal "
          "no conteúdo do chunk); `ja_semantico` = já estava no top-10 semântico (não é reinjetado)."]
    for c in colecoes:
        for q, linhas in diag_tabelas.get(c.nome, {}).items():
            L += [f"", f"### {c.nome} · {q}"] + linhas

    if len(colecoes) > 1:
        base = colecoes[0]
        L += ["", f"## Diferença (cada coleção vs. `{base.nome}`)"]
        for c in colecoes[1:]:
            L += ["", f"### {c.nome} − {base.nome}", "",
                  "| grupo | modo | n | Δhit@3 | Δhit@5 | Δhit@10 | ΔMRR | Δhit@ctx |", "|---|---|---|---|---|---|---|---|"]
            for g in _GRUPOS:
                for modo in _MODOS:
                    a0 = agregar([res[(base.nome, modo, q)] for q in ids_grupo[g]])
                    a1 = agregar([res[(c.nome, modo, q)] for q in ids_grupo[g]])
                    if a1["n"]:
                        L.append(f"| {g} | {modo} | {a1['n']} | {100 * (a1['hit@3'] - a0['hit@3']):+.0f} pp | "
                                 f"{100 * (a1['hit@5'] - a0['hit@5']):+.0f} pp | {100 * (a1['hit@10'] - a0['hit@10']):+.0f} pp | "
                                 f"{a1['mrr'] - a0['mrr']:+.3f} | {100 * (a1['hit@ctx'] - a0['hit@ctx']):+.0f} pp |")
            for g in _GRUPOS:
                for modo in _MODOS:
                    L += ["", f"{g} · {modo} — perguntas em que algo mudou:", "",
                          f"| id | k_min {base.nome} | k_min {c.nome} | Δrr | ev.@10 {base.nome} | ev.@10 {c.nome} |",
                          "|---|---|---|---|---|---|"]
                    mudou = False
                    for q in ids_grupo[g]:
                        r0, r1 = res[(base.nome, modo, q)], res[(c.nome, modo, q)]
                        if r0["k_min"] != r1["k_min"] or r0["cobertura"][10] != r1["cobertura"][10]:
                            mudou = True
                            rr0 = 1 / r0["k_min"] if r0["k_min"] else 0.0
                            rr1 = 1 / r1["k_min"] if r1["k_min"] else 0.0
                            L.append(f"| {q} | {_fmt_k(r0['k_min'])} | {_fmt_k(r1['k_min'])} | {rr1 - rr0:+.3f} | "
                                     f"{r0['cobertura'][10]}/{r0['n_ev']} | {r1['cobertura'][10]}/{r1['n_ev']} |")
                    if not mudou:
                        L.append("| (nenhuma) | | | | | |")
    return "\n".join(L) + "\n"


def salvar_csv(caminho: Path, colecoes: list, gabarito: dict, ids_grupo: dict, res: dict, ausentes: dict) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["colecao", "modo", "grupo", "id", "pergunta", "n_evidencias", "k_min",
                    "hit@3", "hit@5", "hit@10", "rr", "ev@3", "ev@5", "ev@10", "n_contexto", "chars_ctx",
                    "faltando@10", "ausente_no_indice"])
        for c in colecoes:
            for modo in _MODOS:
                for g in _GRUPOS:
                    for q in ids_grupo[g]:
                        r = res[(c.nome, modo, q)]
                        k = r["k_min"]
                        w.writerow([
                            c.nome, modo, g, q, gabarito[q]["pergunta"], r["n_ev"],
                            "" if k is None else k,
                            *(int(k is not None and k <= kk) for kk in _KS),
                            f"{(1 / k) if k else 0:.4f}",
                            *(r["cobertura"][kk] for kk in _KS),
                            r["n_contexto"], r["chars_ctx"], " ; ".join(r["faltando_10"]),
                            " ; ".join(ausentes[(c.nome, q)]),
                        ])


# ── Main ──────────────────────────────────────────────────────────────────

def _parse_run(txt: str) -> tuple:
    """'NOME=CAMINHO[::PREFIXO]' → (nome, caminho, prefixo)."""
    if "=" not in txt:
        sys.exit(f"ERRO: --run espera NOME=CAMINHO[::PREFIXO], recebido {txt!r}")
    nome, resto = txt.split("=", 1)
    caminho, _, prefixo = resto.partition("::")
    return nome.strip(), caminho.strip(), prefixo


def _lista_ids(txt: str, gabarito: dict, opcao: str) -> list:
    ids = [i.strip() for i in txt.split(",") if i.strip()]
    invalidos = [i for i in ids if i not in gabarito]
    if invalidos:
        sys.exit(f"ERRO: IDs inexistentes em {opcao}: {invalidos}")
    return ids


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Avalia o retrieval contra a coluna EVIDENCIA do PIPELINE.md")
    ap.add_argument("--run", action="append", default=[], metavar="NOME=CAMINHO[::PREFIXO]",
                    help="coleção a avaliar (repetível). Padrão: producao=$PASTA_CHROMA")
    ap.add_argument("--legado", action="append", default=[], metavar="NOME",
                    help="usa EVIDENCIA_LEGADO (fase 1) nas perguntas do grupo legado para esta coleção; só para "
                         "comparar com o índice ANTIGO (repetível)")
    ap.add_argument("--colecao", default="ifturing", help="nome da coleção Chroma (padrão: ifturing)")
    ap.add_argument("--top-k", type=int, default=TOP_K, help=f"top_k do retriever semântico (padrão: {TOP_K})")
    ap.add_argument("--ids", default=None, help='IDs do gabarito, ex.: "C01,C03,A04" (padrão: todos com EVIDENCIA)')
    ap.add_argument("--foco", default=_FOCO_PADRAO, help=f"perguntas detalhadas na seção Foco (padrão: {_FOCO_PADRAO})")
    ap.add_argument("--tabelas", default=_TABELAS_PADRAO,
                    help=f"perguntas para o diagnóstico de tabelas no contexto (padrão: {_TABELAS_PADRAO})")
    ap.add_argument("--saida", default="retrieval_resultado", help="prefixo dos arquivos de saída")
    args = ap.parse_args()

    gabarito = carregar_gabarito()
    if not gabarito:
        sys.exit("ERRO: nenhuma tabela com coluna EVIDENCIA em PIPELINE.md")
    negativas = sorted(q for q, v in gabarito.items() if v["negativa"])
    sem_ev = sorted(q for q, v in gabarito.items()
                    if not v["negativa"] and not v["evidencias"] and not v["evidencias_legado"])
    ids = sorted(q for q, v in gabarito.items() if v["evidencias"] or v["evidencias_legado"])
    if args.ids:
        pedidos = _lista_ids(args.ids, gabarito, "--ids")
        ids = [i for i in pedidos if i in ids]
    foco = _lista_ids(args.foco, gabarito, "--foco")
    q_tabelas = _lista_ids(args.tabelas, gabarito, "--tabelas")
    ids_grupo = {g: [q for q in ids if gabarito[q]["grupo"] == g] for g in _GRUPOS}

    runs = [_parse_run(r) for r in args.run] or [("producao", PASTA_CHROMA, "")]
    nomes = [n for n, _, _ in runs]
    for n in args.legado:
        if n not in nomes:
            sys.exit(f"ERRO: --legado {n!r} não corresponde a nenhum --run ({nomes})")
    colecoes = [Colecao(n, p, pref, args.colecao, args.top_k, usar_legado=(n in args.legado)) for n, p, pref in runs]

    res: dict = {}
    docs_top: dict = {}
    ausentes: dict = {}
    rank_ev: dict = {}
    ctx_ids: dict = {}
    diag_tabelas: dict = {}
    for c in colecoes:
        print(f"[{c.nome}] {c.n_chunks} chunks ({dict(c.por_tipo)}) — avaliando {len(ids)} perguntas...")
        diag_tabelas[c.nome] = {}
        for q in ids:
            pergunta = gabarito[q]["pergunta"]
            ev = evidencias_para(gabarito[q], c.usar_legado)
            ausentes[(c.nome, q)] = [e for e in ev if not any(_norm(e) in t for t in c.textos_indice)]
            sem, emb, ranking = c.semantico(pergunta)
            if [d.id for d in ranking[:len(sem)]] != [d.id for d in sem]:
                print(f"  [aviso] {q}: top-{len(sem)} do ranking completo difere do retriever de produção")
            rank_ev[(c.nome, q)] = _ranks_evidencia(c, ranking, ev)
            listas = {"semantico": sem}
            for modo in _MODOS[1:]:
                listas[modo] = c.hibrido(pergunta, sem, emb, modo)
            ids_crono = {d.id for d in c.cronograma_docs}
            for modo, docs in listas.items():
                r = avaliar_lista(docs, ev)
                r["n_ev"] = len(ev)
                r["chars_ctx"] = len("\n\n---\n\n".join(d.content or "" for d in docs))  # como em nucleo.responder()
                res[(c.nome, modo, q)] = r
                docs_top[(c.nome, modo, q)] = docs[:3]
                ctx_ids[(c.nome, modo, q)] = {
                    "todos": [d.id for d in docs],
                    "crono": [_id8(d) for d in docs if d.id in ids_crono and d.id not in {x.id for x in sem}],
                }
            if q in q_tabelas:
                diag_tabelas[c.nome][q] = _diagnostico_tabelas(c, q, pergunta, sem, listas["hib-todos"])

    relatorio = gerar_relatorio(colecoes, gabarito, ids_grupo, negativas, sem_ev, res, docs_top, ausentes,
                                rank_ev, ctx_ids, diag_tabelas, foco)
    base = Path(__file__).parent / args.saida
    base.with_suffix(".md").write_text(relatorio, encoding="utf-8")
    salvar_csv(base.with_suffix(".csv"), colecoes, gabarito, ids_grupo, res, ausentes)

    # Console: resumo + falhas em k=10
    print("\n" + relatorio.split("## Por pergunta")[0])
    todos_ids = [q for g in _GRUPOS for q in ids_grupo[g]]
    for c in colecoes:
        for modo in _MODOS:
            falhas = _falhas(c, modo, todos_ids, res)
            print(f"── {c.nome} · {modo}: {len(falhas)} pergunta(s) falham em k=10 ──")
            for q in falhas:
                r = res[(c.nome, modo, q)]
                onde = "não recuperada" if r["k_min"] is None else f"k={r['k_min']}"
                print(f"  [{q}] ({gabarito[q]['grupo']}) {gabarito[q]['pergunta']}  → {onde}")
                print(f"       faltando: {r['faltando_10']}")
                for d in docs_top[(c.nome, modo, q)]:
                    print(f"       · {_chunk_resumo(d)}")
    print(f"\n[saída] {base.with_suffix('.md')}\n[saída] {base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
