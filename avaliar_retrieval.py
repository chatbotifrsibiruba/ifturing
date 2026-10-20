"""
avaliar_retrieval.py — Mede só o RETRIEVAL (sem LLM) contra o gabarito de PIPELINE.md.

Cada pergunta do gabarito tem uma coluna EVIDENCIA: trechos literais, separados por " ; ",
que precisam aparecer no contexto recuperado. Para cada pergunta calcula-se o menor k em
que TODAS as evidências estão na união dos textos dos k primeiros chunks, em dois modos:

  semantico  busca vetorial pura (top-k do Chroma, sem nenhum pós-processamento)
  hibrido    mesmo fluxo de nucleo.responder(): semântico → pós-filtro de cotas →
             injeção de tabelas → injeção de cronograma (na mesma ordem do contexto)

As perguntas ficam em dois GRUPOS, com métricas sempre separadas:
  legado   as 30 perguntas da fase 1 (A01–A08, B01–B12, C01–C07, D02, E01, E02)
  novas    perguntas criadas depois (F10a…F18c, E03, E04, E05)

Convenções do gabarito (ver PIPELINE.md, seção 4):
  • EVIDENCIA = NEGATIVA   pergunta negativa (D01, D03): fora das métricas, avaliada só pela resposta.
  • EVIDENCIA_LEGADO       evidência da fase 1 (formato antigo do índice). Só vale para coleções
                           marcadas com --legado; as demais usam EVIDENCIA.

Métricas (por coleção, modo e grupo):
  hit@k   fração de perguntas com k_min <= k, para k = 3, 5, 10
  MRR     média de 1/k_min (0 quando as evidências nunca aparecem completas)
  hit@ctx fração de perguntas cobertas pelo contexto inteiro (no híbrido pode passar de 10)
  no índice  evidências que não existem em NENHUM chunk da coleção (teto de recall: nenhum
             retriever consegue trazê-las; indica problema de indexação, não de busca)

Somente leitura: nunca escreve em coleções Chroma. Aponte --run para coleções temporárias
(./chroma_exp_*) ou para a de produção.

Uso:
    python avaliar_retrieval.py                          # coleção PASTA_CHROMA
    python avaliar_retrieval.py --run base=./chroma_exp_prod --legado base \\
                                --run fase2=./chroma_exp_fase2
    python avaliar_retrieval.py --run producao=./chroma_exp_prod \\
                                --run prefixos=./chroma_exp_prefix::"query: "
    python avaliar_retrieval.py --ids C01,C03,A04
    python avaliar_retrieval.py --run fase2=./chroma_exp_fase2 --crono-por-meta --saida retrieval_resultado_meta

--crono-por-meta é uma HIPÓTESE (nucleo.py não faz isso): no híbrido, injeta como cronograma só os chunks com
meta tipo="cronograma" em vez da heurística de nucleo.py (palavra "cronograma" ou >= 3 datas). Serve para medir
o efeito de uma futura mudança no nucleo.py; o resultado oficial é sempre o do fluxo atual (sem a flag).

A sintaxe NOME=CAMINHO[::PREFIXO_DA_CONSULTA] permite comparar coleções (A/B); com 2 ou mais --run
o relatório inclui a diferença em relação à primeira. Saídas: retrieval_resultado.md e .csv.
"""
import argparse
import csv
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

from nucleo import (
    EMBEDDER_DEVICE,
    MODELO_EMB,
    PASTA_CHROMA,
    TOP_K,
    _PADRAO_COTA,
    _PADRAO_DATA_RECENTE,
    _buscar_cronograma_por_keyword,
    _buscar_tabelas_por_keyword,
)

_PIPELINE_MD = Path(__file__).parent / "PIPELINE.md"
_KS = (3, 5, 10)
_MODOS = ("semantico", "hibrido")
_GRUPOS = ("legado", "novas")

# As 30 perguntas da fase 1 (linha de base). Não alterar: define o grupo "legado".
_IDS_LEGADO = frozenset(
    [f"A{i:02d}" for i in range(1, 9)]
    + [f"B{i:02d}" for i in range(1, 13)]
    + [f"C{i:02d}" for i in range(1, 8)]
    + ["D02", "E01", "E02"]
)


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


# ── Execução do retrieval ─────────────────────────────────────────────────

class Colecao:
    """Uma coleção Chroma + embedder de consulta (com prefixo opcional) + docs pré-carregados."""

    def __init__(self, nome: str, caminho: str, prefixo_consulta: str, nome_colecao: str, top_k: int,
                 usar_legado: bool, crono_por_meta: bool = False):
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
        self.textos_indice = [_norm(d.content or "") for d in todos]
        self.por_tipo = Counter((d.meta or {}).get("tipo", "texto") for d in todos)
        self.tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
        self.cronograma_docs = [
            d for d in todos
            if (d.meta or {}).get("tipo") != "tabela"
            and (
                "cronograma" in (d.content or "").lower()
                or len(_PADRAO_DATA_RECENTE.findall(d.content or "")) >= 3
            )
        ]
        self.crono_por_meta = False
        if crono_por_meta:
            por_meta = [d for d in todos if (d.meta or {}).get("tipo") == "cronograma"]
            if por_meta:
                self.cronograma_docs, self.crono_por_meta = por_meta, True
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

    def semantico(self, pergunta: str) -> list:
        return self.pipeline.run({"embedder": {"text": pergunta}})["retriever"]["documents"]

    def hibrido(self, pergunta: str, docs_semanticos: list) -> list:
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
        docs_crono_extra = _buscar_cronograma_por_keyword(pergunta, self.cronograma_docs, docs)
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
        return {"n": 0, "hit@3": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0, "hit@ctx": 0.0}
    agg = {"n": n}
    for k in _KS:
        agg[f"hit@{k}"] = sum(1 for r in resultados if r["k_min"] is not None and r["k_min"] <= k) / n
    agg["mrr"] = sum(1 / r["k_min"] for r in resultados if r["k_min"]) / n
    agg["hit@ctx"] = sum(1 for r in resultados if r["k_min"] is not None) / n
    return agg


def _chunk_resumo(d, n: int = 80) -> str:
    meta = d.meta or {}
    return (f"{(d.id or '?')[:8]}  tipo={meta.get('tipo', 'texto')}  seção={meta.get('secao', '—')}  "
            f"{' '.join((d.content or '').split())[:n]!r}")


# ── Relatório ─────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _fmt_k(k) -> str:
    return "—" if k is None else str(k)


def _falhas(c, modo, ids, res) -> list:
    return [q for q in ids if res[(c.nome, modo, q)]["k_min"] is None or res[(c.nome, modo, q)]["k_min"] > 10]


def gerar_relatorio(colecoes: list, gabarito: dict, ids_grupo: dict, negativas: list, sem_evidencia: list,
                    res: dict, docs_top: dict, ausentes: dict) -> str:
    todos_ids = [q for g in _GRUPOS for q in ids_grupo[g]]
    L = ["# Avaliação de retrieval", ""]
    L.append(f"- Perguntas avaliadas: **legado {len(ids_grupo['legado'])}** (fase 1) · **novas {len(ids_grupo['novas'])}**. "
             "As métricas dos dois grupos nunca são misturadas.")
    L.append(f"- Perguntas negativas (avaliadas só pela resposta \"não consta\"): {', '.join(negativas) or '—'} · "
             f"sem evidência (ignoradas): {', '.join(sem_evidencia) or '—'}")
    L.append(f"- Modelo de embedding: `{MODELO_EMB}` (device `{EMBEDDER_DEVICE}`) · top_k semântico: {TOP_K}")
    for c in colecoes:
        tipos = ", ".join(f"{t}={n}" for t, n in sorted(c.por_tipo.items()))
        L.append(f"- Coleção **{c.nome}**: `{c.caminho}` — {c.n_chunks} chunks ({tipos}; "
                 f"{len(c.cronograma_docs)} cronograma-like) · prefixo da consulta: `{c.prefixo_consulta!r}` · "
                 f"evidência: {'LEGADO (fase 1) quando existir' if c.usar_legado else 'EVIDENCIA (atual)'}"
                 + (" · **cronograma injetado por metadado (hipótese; nucleo.py não faz isso)**" if c.crono_por_meta else ""))
    L += ["", "Definições: *k_min* = menor k cujos k primeiros chunks contêm TODAS as evidências; "
          "*hit@k* = k_min ≤ k; *MRR* = média de 1/k_min; *hit@ctx* = coberto pelo contexto inteiro "
          "(no modo híbrido o contexto pode ter mais de 10 chunks).", ""]

    L += ["## Resumo", "",
          "| coleção | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |", "|---|---|---|---|---|---|---|---|---|"]
    for g in _GRUPOS:
        for c in colecoes:
            for modo in _MODOS:
                a = agregar([res[(c.nome, modo, q)] for q in ids_grupo[g]])
                if a["n"]:
                    L.append(f"| {c.nome} | {g} | {modo} | {a['n']} | {_pct(a['hit@3'])} | {_pct(a['hit@5'])} | "
                             f"{_pct(a['hit@10'])} | {a['mrr']:.3f} | {_pct(a['hit@ctx'])} |")

    L += ["", "## Tamanho do contexto no modo híbrido", "",
          "O contexto inteiro vai para o LLM (`nucleo.responder()`), e `chamar_ollama` usa `num_ctx=8192` tokens "
          "(prompt de ~1,5 mil caracteres + resposta de até 1024 tokens). Tokens estimados = caracteres / 3,5 "
          "(aproximação para português).", "",
          "| coleção | grupo | chunks (média / máx) | caracteres (média / máx) | tokens estimados (média / máx) |",
          "|---|---|---|---|---|"]
    for c in colecoes:
        for g in _GRUPOS:
            rs = [res[(c.nome, "hibrido", q)] for q in ids_grupo[g]]
            if rs:
                nch = [r["n_contexto"] for r in rs]
                chs = [r["chars_ctx"] for r in rs]
                L.append(f"| {c.nome} | {g} | {sum(nch) / len(nch):.1f} / {max(nch)} | {sum(chs) / len(chs):,.0f} / {max(chs):,} | "
                         f"{sum(chs) / len(chs) / 3.5:,.0f} / {max(chs) / 3.5:,.0f} |")
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
              "| id | grupo | pergunta | n ev. | ev. no índice | k_min sem. | k_min híb. | ctx híb. | ev.@10 sem. | ev.@10 híb. |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g in _GRUPOS:
            for q in ids_grupo[g]:
                s, h = res[(c.nome, "semantico", q)], res[(c.nome, "hibrido", q)]
                n_ev = s["n_ev"]
                n_idx = n_ev - len(ausentes[(c.nome, q)])
                L.append(f"| {q} | {g} | {gabarito[q]['pergunta'][:60]} | {n_ev} | {n_idx}/{n_ev} | {_fmt_k(s['k_min'])} | "
                         f"{_fmt_k(h['k_min'])} | {h['n_contexto']} | {s['cobertura'][10]}/{n_ev} | {h['cobertura'][10]}/{n_ev} |")

    L += ["", "## Falhas em k=10 (evidências incompletas nos 10 primeiros chunks)"]
    for c in colecoes:
        for modo in _MODOS:
            falhas = _falhas(c, modo, todos_ids, res)
            L += ["", f"### {c.nome} · {modo} — {len(falhas)} falha(s)"]
            for q in falhas:
                r = res[(c.nome, modo, q)]
                onde = ("não recuperada" if r["k_min"] is None else f"apareceu em k={r['k_min']}")
                L += ["", f"**{q}** ({gabarito[q]['grupo']}) — {gabarito[q]['pergunta']}",
                      f"- menor k em que a evidência apareceu: {onde} (contexto com {r['n_contexto']} chunks)",
                      f"- faltando em k=10: {r['faltando_10']}",
                      "- 3 primeiros chunks (id, tipo, seção, 80 caracteres):"]
                L += [f"  - {_chunk_resumo(d)}" for d in docs_top[(c.nome, modo, q)]]

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
                    "hit@3", "hit@5", "hit@10", "rr", "ev@3", "ev@5", "ev@10", "n_contexto", "faltando@10",
                    "ausente_no_indice"])
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
                            r["n_contexto"], " ; ".join(r["faltando_10"]),
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Avalia o retrieval contra a coluna EVIDENCIA do PIPELINE.md")
    ap.add_argument("--run", action="append", default=[], metavar="NOME=CAMINHO[::PREFIXO]",
                    help="coleção a avaliar (repetível). Padrão: producao=$PASTA_CHROMA")
    ap.add_argument("--legado", action="append", default=[], metavar="NOME",
                    help="usa EVIDENCIA_LEGADO (fase 1) nas perguntas do grupo legado para esta coleção (repetível)")
    ap.add_argument("--crono-por-meta", action="store_true",
                    help="HIPÓTESE: injeta no híbrido só os chunks com meta tipo=cronograma (ver docstring)")
    ap.add_argument("--colecao", default="ifturing", help="nome da coleção Chroma (padrão: ifturing)")
    ap.add_argument("--top-k", type=int, default=TOP_K, help=f"top_k do retriever semântico (padrão: {TOP_K})")
    ap.add_argument("--ids", default=None, help='IDs do gabarito, ex.: "C01,C03,A04" (padrão: todos com EVIDENCIA)')
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
        pedidos = [i.strip() for i in args.ids.split(",") if i.strip()]
        invalidos = [i for i in pedidos if i not in gabarito]
        if invalidos:
            sys.exit(f"ERRO: IDs inexistentes: {invalidos}")
        ids = [i for i in pedidos if i in ids]
    ids_grupo = {g: [q for q in ids if gabarito[q]["grupo"] == g] for g in _GRUPOS}

    runs = [_parse_run(r) for r in args.run] or [("producao", PASTA_CHROMA, "")]
    nomes = [n for n, _, _ in runs]
    for n in args.legado:
        if n not in nomes:
            sys.exit(f"ERRO: --legado {n!r} não corresponde a nenhum --run ({nomes})")
    colecoes = [Colecao(n, p, pref, args.colecao, args.top_k, usar_legado=(n in args.legado),
                         crono_por_meta=args.crono_por_meta) for n, p, pref in runs]

    res: dict = {}
    docs_top: dict = {}
    ausentes: dict = {}
    for c in colecoes:
        print(f"[{c.nome}] {c.n_chunks} chunks ({dict(c.por_tipo)}) — avaliando {len(ids)} perguntas...")
        for q in ids:
            pergunta = gabarito[q]["pergunta"]
            ev = evidencias_para(gabarito[q], c.usar_legado)
            ausentes[(c.nome, q)] = [e for e in ev if not any(_norm(e) in t for t in c.textos_indice)]
            sem = c.semantico(pergunta)
            hib = c.hibrido(pergunta, sem)
            for modo, docs in (("semantico", sem), ("hibrido", hib)):
                r = avaliar_lista(docs, ev)
                r["n_ev"] = len(ev)
                r["chars_ctx"] = len("\n\n---\n\n".join(d.content or "" for d in docs))  # como em nucleo.responder()
                res[(c.nome, modo, q)] = r
                docs_top[(c.nome, modo, q)] = docs[:3]

    relatorio = gerar_relatorio(colecoes, gabarito, ids_grupo, negativas, sem_ev, res, docs_top, ausentes)
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
