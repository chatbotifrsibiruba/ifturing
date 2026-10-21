"""
teste_concorrencia.py — N requisições simultâneas pelo fluxo COMPLETO de nucleo.responder()
(embedding + busca + reranker opcional + híbrido + Ollama), com perguntas do gabarito (PIPELINE.md).

Uso:
    python teste_concorrencia.py --modelo gemma3:4b --concorrencia 1,2,4,8 --rodadas 3
    python teste_concorrencia.py --modelo gemma3:4b --ollama-urls http://localhost:11434,http://localhost:11435
    RERANKER_ATIVO=1 RERANKER_DEVICE=cuda python teste_concorrencia.py --modelo gemma3:4b

Em cada rodada de um nível de concorrência c, c threads disparam ao mesmo tempo (barreira), cada uma com uma
pergunta diferente do gabarito (as perguntas percorrem a lista em sequência, sem repetir dentro da rodada).
Latência = tempo de parede de responder() na thread; vazão = respostas / tempo de parede das rodadas (resp/min).
Antes de medir roda preaquecer() e uma pergunta sequencial (carrega o modelo no Ollama); --sem-aquecimento pula.

Saída (fora do git): resultado_concorrencia.csv (uma linha por requisição) e resultado_concorrencia.md (resumo).
"""
import argparse
import contextlib
import csv
import io
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).parent
SAIDA_CSV = AQUI / "resultado_concorrencia.csv"
SAIDA_MD = AQUI / "resultado_concorrencia.md"


def percentil(valores: list, p: float) -> float:
    """Percentil p (0-100) por interpolação linear; lista vazia → 0."""
    if not valores:
        return 0.0
    v = sorted(valores)
    k = (len(v) - 1) * p / 100
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def media(valores: list) -> float:
    return statistics.fmean(valores) if valores else 0.0


def disparar(nucleo, cfg: dict, perguntas: list) -> tuple:
    """Dispara len(perguntas) requisições ao mesmo tempo. Devolve (linhas, segundos de parede da rodada)."""
    n = len(perguntas)
    barreira = threading.Barrier(n)

    def uma(qid_pergunta):
        qid, pergunta = qid_pergunta
        barreira.wait()
        t0 = time.perf_counter()
        try:
            r = nucleo.responder(pergunta, cfg)
            exc = ""
        except Exception as e:                       # responder() já captura; isto é só rede de segurança
            r, exc = {"erro": True}, f"{type(e).__name__}: {e}"
        return {"id": qid, "latencia_s": time.perf_counter() - t0, "resp": r, "excecao": exc}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        linhas = list(ex.map(uma, perguntas))
    return linhas, time.perf_counter() - t0


def resumir(nivel: int, rodadas: list) -> dict:
    """Agrega as linhas de um nível. rodadas = [(linhas, parede_s), ...]."""
    linhas = [l for lin, _ in rodadas for l in lin]
    parede = sum(p for _, p in rodadas)
    ok = [l for l in linhas if not l["resp"].get("erro") and not l["excecao"]]
    lat = [l["latencia_s"] for l in ok]
    pegar = lambda campo: [l["resp"].get(campo, 0.0) or 0.0 for l in ok]
    urls = {}
    for l in ok:
        u = l["resp"].get("ollama_url") or "-"
        urls[u] = urls.get(u, 0) + 1
    return {
        "concorrencia": nivel, "requisicoes": len(linhas), "erros": len(linhas) - len(ok),
        "p50_s": percentil(lat, 50), "p95_s": percentil(lat, 95), "max_s": max(lat, default=0.0),
        "vazao_resp_min": 60 * len(ok) / parede if parede > 0 else 0.0,
        "retrieval_s": media(pegar("tempo_retrieval_s")), "rerank_s": media(pegar("tempo_rerank_s")),
        "llm_s": media(pegar("tempo_llm_s")), "geracao_s": media(pegar("tempo_geracao_s")),
        "parede_s": parede, "urls": urls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=None, help="id do modelo no Ollama (padrão: modelo padrão do app)")
    ap.add_argument("--concorrencia", default="1,2,4,8", help="níveis separados por vírgula (padrão 1,2,4,8)")
    ap.add_argument("--rodadas", type=int, default=3, help="rodadas por nível (padrão 3)")
    ap.add_argument("--ollama-urls", default=None, help="OLLAMA_URLS (lista separada por vírgula)")
    ap.add_argument("--sem-aquecimento", action="store_true", help="não roda preaquecer() nem a pergunta de aquecimento")
    args = ap.parse_args()
    niveis = [int(x) for x in args.concorrencia.split(",") if x.strip()]
    if args.ollama_urls:
        os.environ["OLLAMA_URLS"] = args.ollama_urls            # antes de importar nucleo (lido na importação)

    sys.stdout.reconfigure(encoding="utf-8")
    import nucleo
    import avaliar_retrieval as ar
    gab = ar.carregar_gabarito()
    ids = sorted(gab)
    modelo = args.modelo or nucleo._MODELO_PADRAO_ID
    cfg = {"tipo": "ollama", "modelo": modelo}
    cab = (f"modelo={modelo} | OLLAMA_URLS={nucleo.OLLAMA_URLS} | RERANKER_ATIVO={int(nucleo.RERANKER_ATIVO)}"
           + (f" ({nucleo.RERANKER_MODELO}, {nucleo.RERANKER_CANDIDATOS} cand., ml={nucleo.RERANKER_MAX_LENGTH}, "
              f"{nucleo.RERANKER_DEVICE})" if nucleo.RERANKER_ATIVO else "")
           + f" | EMBEDDER_DEVICE={nucleo.EMBEDDER_DEVICE}")
    print(cab)

    silencio = io.StringIO()          # responder() imprime dezenas de linhas [timing]; some durante a medição
    if not args.sem_aquecimento:
        with contextlib.redirect_stdout(silencio):
            aq = nucleo.preaquecer()
            r0 = nucleo.responder(gab[ids[0]]["pergunta"], cfg)
        print(f"aquecimento: {aq} | pergunta de aquecimento {'COM ERRO' if r0.get('erro') else 'ok'} "
              f"({r0.get('tempo_total_s')}s)")

    resumos, todas = [], []
    cursor = 0
    for nivel in niveis:
        rodadas = []
        for k in range(args.rodadas):
            perguntas = [(ids[(cursor + i) % len(ids)], gab[ids[(cursor + i) % len(ids)]]["pergunta"])
                         for i in range(nivel)]
            cursor += nivel
            with contextlib.redirect_stdout(silencio):
                linhas, parede = disparar(nucleo, cfg, perguntas)
            rodadas.append((linhas, parede))
            for l in linhas:
                todas.append({"concorrencia": nivel, "rodada": k + 1, **l})
        s = resumir(nivel, rodadas)
        resumos.append(s)
        print(f"c={nivel}: p50 {s['p50_s']:.2f}s | p95 {s['p95_s']:.2f}s | máx {s['max_s']:.2f}s | "
              f"{s['vazao_resp_min']:.1f} resp/min | retrieval {s['retrieval_s']:.2f}s | rerank {s['rerank_s']:.2f}s | "
              f"llm {s['llm_s']:.2f}s | erros {s['erros']}/{s['requisicoes']}")

    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concorrencia", "rodada", "id", "latencia_s", "erro", "excecao", "tempo_retrieval_s",
                    "tempo_rerank_s", "tempo_llm_s", "tempo_geracao_s", "tokens_saida", "ollama_url"])
        for l in todas:
            r = l["resp"]
            w.writerow([l["concorrencia"], l["rodada"], l["id"], f"{l['latencia_s']:.3f}", int(bool(r.get("erro"))),
                        l["excecao"], r.get("tempo_retrieval_s", ""), r.get("tempo_rerank_s", ""),
                        r.get("tempo_llm_s", ""), r.get("tempo_geracao_s", ""), r.get("tokens_saida", ""),
                        r.get("ollama_url") or ""])

    L = [f"# Teste de concorrência — {datetime.now():%Y-%m-%d %H:%M}", "", f"`{cab}`", "",
         f"{args.rodadas} rodada(s) por nível; cada rodada dispara c requisições simultâneas (perguntas distintas do "
         f"gabarito). Tempos em segundos, medidos; latência/percentis só das requisições sem erro.", "",
         "| c | req | erros | p50 | p95 | máx | vazão (resp/min) | retrieval | rerank | llm | geração | instâncias |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in resumos:
        urls = ", ".join(f"{u}: {n}" for u, n in sorted(s["urls"].items())) or "—"
        L.append(f"| {s['concorrencia']} | {s['requisicoes']} | {s['erros']} | {s['p50_s']:.2f} | {s['p95_s']:.2f} | "
                 f"{s['max_s']:.2f} | {s['vazao_resp_min']:.1f} | {s['retrieval_s']:.2f} | {s['rerank_s']:.2f} | "
                 f"{s['llm_s']:.2f} | {s['geracao_s']:.2f} | {urls} |")
    L += ["", "retrieval = embedding+busca; rerank = cross-encoder (0 se desligado); llm = chamada inteira ao Ollama "
          "(inclui fila, carga e prefill); geração = só a fase de geração de tokens informada pelo Ollama."]
    SAIDA_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nsalvo: {SAIDA_CSV.name}, {SAIDA_MD.name}")


if __name__ == "__main__":
    main()
