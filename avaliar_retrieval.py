"""
avaliar_retrieval.py — Mede só o RETRIEVAL (sem LLM) contra o gabarito de PIPELINE.md.

Cada pergunta do gabarito tem uma coluna EVIDENCIA: trechos literais, separados por " ; ",
que precisam aparecer no contexto recuperado. Para cada pergunta calcula-se o menor k em
que TODAS as evidências estão na união dos textos dos k primeiros chunks, em dois modos:

  semantico  busca vetorial pura (top-k do Chroma, sem nenhum pós-processamento)
  hibrido    mesmo fluxo de nucleo.responder(): semântico → pós-filtro de cotas →
             injeção de tabelas → injeção de cronograma (na mesma ordem do contexto)

Métricas (por modo, sobre as perguntas que têm EVIDENCIA):
  hit@k   fração de perguntas com k_min <= k, para k = 3, 5, 10
  MRR     média de 1/k_min (0 quando as evidências nunca aparecem completas)
  hit@ctx fração de perguntas cobertas pelo contexto inteiro (no híbrido pode passar de 10)
  no índice  evidências que não existem em NENHUM chunk da coleção (teto de recall: nenhum
             retriever consegue trazê-las; indica problema de indexação, não de busca)

Somente leitura: nunca escreve em coleções Chroma. Aponte --run para coleções temporárias
(./chroma_exp_*) ou para a de produção.

Uso:
    python avaliar_retrieval.py                          # coleção PASTA_CHROMA
    python avaliar_retrieval.py --run producao=./chroma_exp_prod
    python avaliar_retrieval.py --run producao=./chroma_exp_prod \\
                                --run prefixos=./chroma_exp_prefix::"query: "
    python avaliar_retrieval.py --ids C01,C03,A04

A sintaxe NOME=CAMINHO[::PREFIXO_DA_CONSULTA] permite comparar duas coleções (A/B); com 2 ou
mais --run o relatório inclui a diferença por pergunta em relação à primeira.
Saídas: retrieval_resultado.md e retrieval_resultado.csv.
"""
import argparse
import csv
import re
import sys
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


# ── Gabarito com EVIDENCIA ────────────────────────────────────────────────

def _celulas(linha: str) -> list:
    """Divide uma linha de tabela markdown em células, respeitando '\\|' escapado."""
    s = linha.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    partes = re.split(r"(?<!\\)\|", s)
    return [re.sub(r"\\([|*])", r"\1", p).strip() for p in partes]


def carregar_gabarito(caminho: Path = _PIPELINE_MD) -> dict:
    """Retorna {id: {pergunta, evidencias}} para as tabelas do PIPELINE.md com coluna EVIDENCIA.

    `evidencias` é [] quando a célula está vazia (pergunta negativa ou gabarito em dúvida).
    """
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
        if not re.fullmatch(r"[A-E]\d{2}", celulas[0]):
            continue
        i_ev = cabecalho.index("EVIDENCIA")
        bruto = celulas[i_ev] if i_ev < len(celulas) else ""
        gabarito[celulas[0]] = {
            "pergunta": celulas[1],
            "evidencias": [e.strip() for e in re.split(r"\s;\s", bruto) if e.strip()],
        }
    return gabarito


# ── Execução do retrieval ─────────────────────────────────────────────────

class Colecao:
    """Uma coleção Chroma + embedder de consulta (com prefixo opcional) + docs pré-carregados."""

    def __init__(self, nome: str, caminho: str, prefixo_consulta: str, nome_colecao: str, top_k: int):
        if not Path(caminho).exists():
            sys.exit(f"ERRO: coleção não encontrada em {caminho}")
        self.nome = nome
        self.caminho = caminho
        self.prefixo_consulta = prefixo_consulta
        store = ChromaDocumentStore(collection_name=nome_colecao, persist_path=str(caminho))
        self.n_chunks = store.count_documents()
        if self.n_chunks == 0:
            sys.exit(f"ERRO: coleção vazia em {caminho}")
        # Mesma pré-carga de nucleo.carregar_pipeline()
        todos = store.filter_documents()
        self.textos_indice = [_norm(d.content or "") for d in todos]
        self.tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
        self.cronograma_docs = [
            d for d in todos
            if (d.meta or {}).get("tipo") != "tabela"
            and (
                "cronograma" in (d.content or "").lower()
                or len(_PADRAO_DATA_RECENTE.findall(d.content or "")) >= 3
            )
        ]
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
        return {"n": 0}
    agg = {"n": n}
    for k in _KS:
        agg[f"hit@{k}"] = sum(1 for r in resultados if r["k_min"] is not None and r["k_min"] <= k) / n
    agg["mrr"] = sum(1 / r["k_min"] for r in resultados if r["k_min"]) / n
    agg["hit@ctx"] = sum(1 for r in resultados if r["k_min"] is not None) / n
    return agg


def _chunk_resumo(d, n: int = 80) -> str:
    return f"{(d.id or '?')[:8]}  {' '.join((d.content or '').split())[:n]!r}"


# ── Relatório ─────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _fmt_k(k) -> str:
    return "—" if k is None else str(k)


def gerar_relatorio(colecoes: list, gabarito: dict, ids: list, ids_sem_evidencia: list,
                    res: dict, docs_top: dict) -> str:
    L = ["# Avaliação de retrieval", ""]
    L.append(f"- Perguntas com EVIDENCIA avaliadas: **{len(ids)}** · sem evidência (ignoradas): "
             f"{', '.join(ids_sem_evidencia) or '—'}")
    L.append(f"- Modelo de embedding: `{MODELO_EMB}` (device `{EMBEDDER_DEVICE}`) · top_k semântico: {TOP_K}")
    for c in colecoes:
        L.append(f"- Coleção **{c.nome}**: `{c.caminho}` — {c.n_chunks} chunks "
                 f"({len(c.tabela_docs)} tabela, {len(c.cronograma_docs)} cronograma-like) · "
                 f"prefixo da consulta: `{c.prefixo_consulta!r}`")
    L += ["", "Definições: *k_min* = menor k cujos k primeiros chunks contêm TODAS as evidências; "
          "*hit@k* = k_min ≤ k; *MRR* = média de 1/k_min; *hit@ctx* = coberto pelo contexto inteiro "
          "(no modo híbrido o contexto pode ter mais de 10 chunks).", ""]

    L += ["## Resumo", "", "| coleção | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |", "|---|---|---|---|---|---|---|---|"]
    for c in colecoes:
        for modo in _MODOS:
            a = agregar([res[(c.nome, modo, q)] for q in ids])
            L.append(f"| {c.nome} | {modo} | {a['n']} | {_pct(a['hit@3'])} | {_pct(a['hit@5'])} | "
                     f"{_pct(a['hit@10'])} | {a['mrr']:.3f} | {_pct(a['hit@ctx'])} |")

    L += ["", "## Evidências ausentes do índice (teto de recall)", "",
          "Evidências que não aparecem em nenhum chunk da coleção: nenhuma estratégia de busca resolve; "
          "o problema está na extração/indexação."]
    for c in colecoes:
        aus = [(q, res[(c.nome, "indice", q)]) for q in ids if res[(c.nome, "indice", q)]]
        L += ["", f"**{c.nome}** — {len(aus)} de {len(ids)} pergunta(s) com ao menos uma evidência inexistente no índice"]
        L += [f"- {q}: {e}" for q, e in aus]

    for c in colecoes:
        L += ["", f"## Por pergunta — {c.nome}", "",
              "| id | pergunta | n ev. | ev. no índice | k_min sem. | k_min híb. | ctx híb. | ev.@10 sem. | ev.@10 híb. |",
              "|---|---|---|---|---|---|---|---|---|"]
        for q in ids:
            s, h = res[(c.nome, "semantico", q)], res[(c.nome, "hibrido", q)]
            n_ev = len(gabarito[q]["evidencias"])
            n_idx = n_ev - len(res[(c.nome, "indice", q)])
            L.append(f"| {q} | {gabarito[q]['pergunta'][:60]} | {n_ev} | {n_idx}/{n_ev} | {_fmt_k(s['k_min'])} | {_fmt_k(h['k_min'])} | "
                     f"{h['n_contexto']} | {s['cobertura'][10]}/{n_ev} | {h['cobertura'][10]}/{n_ev} |")

    L += ["", "## Falhas em k=10 (evidências incompletas nos 10 primeiros chunks)"]
    for c in colecoes:
        for modo in _MODOS:
            falhas = [q for q in ids if res[(c.nome, modo, q)]["k_min"] is None
                      or res[(c.nome, modo, q)]["k_min"] > 10]
            L += ["", f"### {c.nome} · {modo} — {len(falhas)} falha(s)"]
            for q in falhas:
                r = res[(c.nome, modo, q)]
                L += ["", f"**{q}** — {gabarito[q]['pergunta']}",
                      f"- faltando em k=10: {r['faltando_10']}",
                      "- 3 primeiros chunks:"]
                L += [f"  - {_chunk_resumo(d)}" for d in docs_top[(c.nome, modo, q)]]

    if len(colecoes) > 1:
        base = colecoes[0]
        L += ["", f"## Diferença por pergunta (cada coleção vs. `{base.nome}`)"]
        for c in colecoes[1:]:
            L += ["", f"### {c.nome} − {base.nome}", "",
                  "| modo | n | Δhit@3 | Δhit@5 | Δhit@10 | ΔMRR |", "|---|---|---|---|---|---|"]
            for modo in _MODOS:
                a0 = agregar([res[(base.nome, modo, q)] for q in ids])
                a1 = agregar([res[(c.nome, modo, q)] for q in ids])
                L.append(f"| {modo} | {a1['n']} | {100 * (a1['hit@3'] - a0['hit@3']):+.0f} pp | "
                         f"{100 * (a1['hit@5'] - a0['hit@5']):+.0f} pp | {100 * (a1['hit@10'] - a0['hit@10']):+.0f} pp | "
                         f"{a1['mrr'] - a0['mrr']:+.3f} |")
            for modo in _MODOS:
                L += ["", f"{modo} — perguntas em que algo mudou:", "",
                      f"| id | k_min {base.nome} | k_min {c.nome} | Δrr | ev.@10 {base.nome} | ev.@10 {c.nome} |",
                      "|---|---|---|---|---|---|"]
                mudou = False
                for q in ids:
                    r0, r1 = res[(base.nome, modo, q)], res[(c.nome, modo, q)]
                    if r0["k_min"] != r1["k_min"] or r0["cobertura"][10] != r1["cobertura"][10]:
                        mudou = True
                        n_ev = len(gabarito[q]["evidencias"])
                        rr0 = 1 / r0["k_min"] if r0["k_min"] else 0.0
                        rr1 = 1 / r1["k_min"] if r1["k_min"] else 0.0
                        L.append(f"| {q} | {_fmt_k(r0['k_min'])} | {_fmt_k(r1['k_min'])} | {rr1 - rr0:+.3f} | "
                                 f"{r0['cobertura'][10]}/{n_ev} | {r1['cobertura'][10]}/{n_ev} |")
                if not mudou:
                    L.append("| (nenhuma) | | | | | |")
    return "\n".join(L) + "\n"


def salvar_csv(caminho: Path, colecoes: list, gabarito: dict, ids: list, res: dict) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["colecao", "modo", "id", "pergunta", "n_evidencias", "k_min",
                    "hit@3", "hit@5", "hit@10", "rr", "ev@3", "ev@5", "ev@10", "n_contexto", "faltando@10",
                    "ausente_no_indice"])
        for c in colecoes:
            for modo in _MODOS:
                for q in ids:
                    r = res[(c.nome, modo, q)]
                    k = r["k_min"]
                    w.writerow([
                        c.nome, modo, q, gabarito[q]["pergunta"], len(gabarito[q]["evidencias"]),
                        "" if k is None else k,
                        *(int(k is not None and k <= kk) for kk in _KS),
                        f"{(1 / k) if k else 0:.4f}",
                        *(r["cobertura"][kk] for kk in _KS),
                        r["n_contexto"], " ; ".join(r["faltando_10"]),
                        " ; ".join(res[(c.nome, "indice", q)]),
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
    ap.add_argument("--colecao", default="ifturing", help="nome da coleção Chroma (padrão: ifturing)")
    ap.add_argument("--top-k", type=int, default=TOP_K, help=f"top_k do retriever semântico (padrão: {TOP_K})")
    ap.add_argument("--ids", default=None, help='IDs do gabarito, ex.: "C01,C03,A04" (padrão: todos com EVIDENCIA)')
    ap.add_argument("--saida", default="retrieval_resultado", help="prefixo dos arquivos de saída")
    args = ap.parse_args()

    gabarito = carregar_gabarito()
    if not gabarito:
        sys.exit("ERRO: nenhuma tabela com coluna EVIDENCIA em PIPELINE.md")
    ids_sem = sorted(q for q, v in gabarito.items() if not v["evidencias"])
    ids = sorted(q for q, v in gabarito.items() if v["evidencias"])
    if args.ids:
        pedidos = [i.strip() for i in args.ids.split(",") if i.strip()]
        invalidos = [i for i in pedidos if i not in gabarito]
        if invalidos:
            sys.exit(f"ERRO: IDs inexistentes: {invalidos}")
        ids = [i for i in pedidos if gabarito[i]["evidencias"]]

    runs = [_parse_run(r) for r in args.run] or [("producao", PASTA_CHROMA, "")]
    colecoes = [Colecao(n, p, pref, args.colecao, args.top_k) for n, p, pref in runs]

    res: dict = {}
    docs_top: dict = {}
    for c in colecoes:
        print(f"[{c.nome}] {c.n_chunks} chunks ({len(c.tabela_docs)} tabela, "
              f"{len(c.cronograma_docs)} cronograma-like) — avaliando {len(ids)} perguntas...")
        for q in ids:
            pergunta, ev = gabarito[q]["pergunta"], gabarito[q]["evidencias"]
            res[(c.nome, "indice", q)] = [e for e in ev if not any(_norm(e) in t for t in c.textos_indice)]
            sem = c.semantico(pergunta)
            hib = c.hibrido(pergunta, sem)
            for modo, docs in (("semantico", sem), ("hibrido", hib)):
                res[(c.nome, modo, q)] = avaliar_lista(docs, ev)
                docs_top[(c.nome, modo, q)] = docs[:3]

    relatorio = gerar_relatorio(colecoes, gabarito, ids, ids_sem, res, docs_top)
    base = Path(__file__).parent / args.saida
    base.with_suffix(".md").write_text(relatorio, encoding="utf-8")
    salvar_csv(base.with_suffix(".csv"), colecoes, gabarito, ids, res)

    # Console: resumo + falhas em k=10
    print("\n" + relatorio.split("## Por pergunta")[0])
    for c in colecoes:
        for modo in _MODOS:
            falhas = [q for q in ids if res[(c.nome, modo, q)]["k_min"] is None
                      or res[(c.nome, modo, q)]["k_min"] > 10]
            print(f"── {c.nome} · {modo}: {len(falhas)} pergunta(s) falham em k=10 ──")
            for q in falhas:
                r = res[(c.nome, modo, q)]
                print(f"  [{q}] {gabarito[q]['pergunta']}")
                print(f"       faltando: {r['faltando_10']}")
                for d in docs_top[(c.nome, modo, q)]:
                    print(f"       · {_chunk_resumo(d)}")
    print(f"\n[saída] {base.with_suffix('.md')}\n[saída] {base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
