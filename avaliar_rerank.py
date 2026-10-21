"""
avaliar_rerank.py — Experimento OFFLINE de reranking (cross-encoder) sobre o retrieval do IF Turing.
Não altera nucleo.py nem nenhum índice: só lê uma coleção Chroma temporária.

Fluxo medido (por pergunta do gabarito com EVIDENCIA):
  1. busca semântica exata no índice → 30 candidatos (ranking completo[:30]);
  2. o reranker pontua (pergunta, texto do chunk) para os 30 UMA vez; os scores vão para rerank_cache/ (JSON);
  3. cenários RERANK_CANDIDATOS = 15, 20 e 30 usam os N primeiros desses 30 (sem repontuar), reordenam por
     score e ficam com os 10 primeiros (TOP_K);
  4. a lista reordenada é avaliada pura ("semantico") e dentro do fluxo híbrido de produção ("hib-todos":
     pós-filtro de cotas + tabelas + cronograma inteiro), com as mesmas métricas de avaliar_retrieval.py.
O baseline é o ranking semântico sem reranker (10 primeiros).

Uso (a ordem das configurações é fixa: bge-reranker-v2-m3 a 256 e 384, depois o menor):
    python avaliar_rerank.py pontuar      # pesado: pontua e grava rerank_cache/*.json (retomável)
    python avaliar_rerank.py pontuar --completo   # ignora o corte de 20 min e completa as 57 perguntas
    python avaliar_rerank.py relatar      # só lê o cache e escreve retrieval_rerank_resumo.md

Regras de execução (modo sem supervisão):
  • se uma configuração estimar > 20 min, usa só o subconjunto foco (E03, F16a, F18a, A07, E01 + 10 perguntas
    aleatórias fixas, semente 20261021) e isso é marcado no relatório;
  • se um modelo falhar ao carregar, registra o erro e tenta UM substituto; trust_remote_code nunca é usado;
  • tempos de 15/20 candidatos são DERIVADOS proporcionalmente do medido com 30.
"""
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).parent
PASTA_CACHE = AQUI / "rerank_cache"
SAIDA_MD = AQUI / "retrieval_rerank_resumo.md"

INDICE = "./chroma_exp_f3_final"                                  # índice final da Fase 3 (item_titulo + cabeçalho)
OPCOES = {"gatilho": "palavra", "listar": "1"}                    # mesmas opções finais da Fase 3
N_CANDIDATOS = 30
CENARIOS = (15, 20, 30)
LIMITE_S = 20 * 60
MAX_LENGTHS = (256, 384)
FOCO = ["E03", "F16a", "F18a", "A07", "E01"]
SEMENTE = 20261021

# (modelo, substituto se falhar). O substituto nunca é um modelo que já esteja na lista.
MODELOS = [
    ("BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"),
    ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "BAAI/bge-reranker-base"),
]

# T1000 (Turing TU117, 4 GB): 896 núcleos CUDA × 2 FLOP × ~1,39 GHz ≈ 2,5 TFLOPS fp32 de pico.
T1000_PICO_FLOPS = 2.5e12
T1000_UTIL = (0.25, 0.40)           # fração do pico realista em inferência fp32 de transformer (ESTIMATIVA)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _slug(modelo: str) -> str:
    return modelo.replace("/", "__")


def _caminho_cache(modelo: str, ml: int) -> Path:
    return PASTA_CACHE / f"{_slug(modelo)}__ml{ml}.json"


def _gravar_json(caminho: Path, dados: dict) -> None:
    caminho.parent.mkdir(exist_ok=True)
    tmp = caminho.with_suffix(".tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, caminho)


def _ler_json(caminho: Path, padrao):
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except Exception:
        return padrao


# ── Perguntas e coleção ───────────────────────────────────────────────────

def carregar_base():
    import avaliar_retrieval as ar
    gab = ar.carregar_gabarito()
    ids = sorted(q for q, v in gab.items() if v["evidencias"])
    colecao = ar.Colecao("final", INDICE, "", "ifturing", 10, usar_legado=False, opcoes=OPCOES)
    return ar, gab, ids, colecao


def ordem_perguntas(ids: list) -> tuple:
    """(todas, foco): foco primeiro em qualquer execução, para que um corte por tempo preserve o essencial."""
    resto = [q for q in ids if q not in FOCO]
    aleatorias = random.Random(SEMENTE).sample(resto, 10)
    foco = [q for q in FOCO if q in ids] + sorted(aleatorias)
    return foco + [q for q in ids if q not in foco], foco


def candidatos(ar, gab, ids, c) -> dict:
    """Top-30 semântico exato por pergunta (ids dos chunks), em cache para retomar."""
    caminho = PASTA_CACHE / "candidatos.json"
    cache = _ler_json(caminho, {})
    if cache.get("_indice") != [INDICE, c.n_chunks]:
        cache = {"_indice": [INDICE, c.n_chunks]}
    mudou = False
    for q in ids:
        if q in cache:
            continue
        sem, emb, ranking = c.semantico(gab[q]["pergunta"])
        cache[q] = [d.id for d in ranking[:N_CANDIDATOS]]
        mudou = True
    if mudou:
        _gravar_json(caminho, cache)
    return cache


# ── Pontuação (pesado) ────────────────────────────────────────────────────

def _carregar_reranker(modelo: str, ml: int):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(modelo, max_length=ml, device="cpu", trust_remote_code=False)


def _contar_tokens(tok, pares: list, ml: int) -> int:
    return int(sum(len(tok(q, t, truncation=True, max_length=ml)["input_ids"]) for q, t in pares))


def pontuar(forcar_completo: bool = False) -> None:
    """forcar_completo (--completo): pontua TODAS as perguntas mesmo que a estimativa passe de 20 min, completando
    o que o corte por tempo deixou de fora (o cache já pontuado é reaproveitado)."""
    PASTA_CACHE.mkdir(exist_ok=True)
    ar, gab, ids, c = carregar_base()
    todas, foco = ordem_perguntas(ids)
    docs_por_id = {d.id: d for d in c.todos}
    cand = candidatos(ar, gab, ids, c)
    log(f"{len(ids)} perguntas; candidatos prontos; foco={foco}")
    erros_caminho = PASTA_CACHE / "erros.json"
    erros = _ler_json(erros_caminho, [])
    usados = set()

    for modelo_pedido, substituto in MODELOS:
        modelo, usando_substituto = modelo_pedido, False
        for ml in MAX_LENGTHS:
            cam = _caminho_cache(modelo, ml)
            while True:
                try:
                    cache = _ler_json(cam, None) or {"modelo": modelo, "pedido": modelo_pedido, "max_length": ml,
                                                    "substituto": usando_substituto, "perguntas": {}}
                    if forcar_completo and cache.get("subconjunto"):
                        cache["subconjunto"], cache["completo"] = False, False
                    if cache.get("completo"):
                        log(f"[{modelo} ml={ml}] já completo no cache ({len(cache['perguntas'])} perguntas)")
                        break
                    log(f"[{modelo} ml={ml}] carregando…")
                    t0 = time.perf_counter()
                    rr = _carregar_reranker(modelo, ml)
                    from transformers import AutoTokenizer
                    tok = rr.tokenizer if getattr(rr, "tokenizer", None) is not None else AutoTokenizer.from_pretrained(modelo)
                    hf = rr.model
                    params_total = int(sum(p.numel() for p in hf.parameters()))
                    params_corpo = int(sum(p.numel() for n, p in hf.named_parameters() if "embeddings" not in n))
                    log(f"[{modelo} ml={ml}] carregado em {time.perf_counter() - t0:.1f}s; "
                        f"{params_total / 1e6:.0f}M parâmetros ({params_corpo / 1e6:.0f}M fora dos embeddings)")
                    rr.predict([("aquecimento", "texto de aquecimento")], batch_size=1)
                    cache.update({"params_total": params_total, "params_corpo": params_corpo, "indice": INDICE,
                                  "cpu": platform.processor(), "threads_torch": __import__("torch").get_num_threads()})
                    break
                except Exception as e:
                    erros.append({"quando": datetime.now().isoformat(timespec="seconds"), "modelo": modelo,
                                  "max_length": ml, "erro": f"{type(e).__name__}: {e}"[:600]})
                    _gravar_json(erros_caminho, erros)
                    log(f"[{modelo} ml={ml}] FALHOU: {type(e).__name__}: {str(e)[:300]}")
                    if usando_substituto or substituto in usados or substituto == modelo_pedido:
                        log(f"[{modelo_pedido}] sem substituto disponível; seguindo em frente")
                        cache = None
                        break
                    modelo, usando_substituto = substituto, True
                    cam = _caminho_cache(modelo, ml)
                    log(f"[{modelo_pedido}] tentando UM substituto: {modelo}")
            if not cache or cache.get("completo"):
                continue
            usados.add(modelo)

            # sonda de tempo (1ª pergunta do foco) → decide o subconjunto
            def pontuar_pergunta(q: str) -> None:
                ids_c = cand[q]
                pares = [(gab[q]["pergunta"], docs_por_id[i].content or "") for i in ids_c]
                t = time.perf_counter()
                scores = rr.predict(pares, batch_size=16, show_progress_bar=False)
                dt = time.perf_counter() - t
                cache["perguntas"][q] = {"ids": ids_c, "scores": [float(s) for s in scores],
                                         "tempo_s": round(dt, 3), "n_pares": len(pares),
                                         "tokens": _contar_tokens(tok, pares, ml)}
                _gravar_json(cam, cache)

            if "subconjunto" not in cache:
                q0 = foco[0]
                if q0 not in cache["perguntas"] or cache["perguntas"][q0]["ids"] != cand[q0]:
                    pontuar_pergunta(q0)
                est = cache["perguntas"][q0]["tempo_s"] * len(todas)
                cache["subconjunto"] = est > LIMITE_S
                cache["estimativa_s"] = round(est)
                _gravar_json(cam, cache)
                log(f"[{modelo} ml={ml}] sonda: {cache['perguntas'][q0]['tempo_s']:.1f}s para 30 candidatos → "
                    f"estimativa {est / 60:.1f} min para {len(todas)} perguntas"
                    + (" > 20 min: SÓ SUBCONJUNTO FOCO" if cache["subconjunto"] else ""))
            alvo = foco if cache["subconjunto"] else todas
            for i, q in enumerate(alvo, 1):
                if q in cache["perguntas"] and cache["perguntas"][q]["ids"] == cand[q]:
                    continue
                pontuar_pergunta(q)
                if i % 5 == 0 or i == len(alvo):
                    log(f"[{modelo} ml={ml}] {i}/{len(alvo)} perguntas "
                        f"(última: {q}, {cache['perguntas'][q]['tempo_s']:.1f}s)")
            cache["completo"] = True
            cache["subconjunto"] = bool(cache["subconjunto"])
            _gravar_json(cam, cache)
            log(f"[{modelo} ml={ml}] CONCLUÍDO ({len(cache['perguntas'])} perguntas"
                f"{', subconjunto foco' if cache['subconjunto'] else ''})")
            del rr
    log("pontuação terminada")


# ── Relatório (só lê o cache) ─────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def avaliar_config(ar, gab, c, cache_cfg: dict, qs: list, n: int | None) -> dict:
    """Métricas de um cenário. n=None → baseline (semântico sem reranker, 10 primeiros)."""
    docs_por_id = {d.id: d for d in c.todos}
    res = {"semantico": {}, "hib-todos": {}}
    ordens = {}
    for q in qs:
        info = cache_cfg["perguntas"][q]
        ids_c = info["ids"]
        if n is None:
            ordem = ids_c[:10]
        else:
            idx = sorted(range(n), key=lambda i: (-info["scores"][i], i))      # empate → ordem semântica
            ordem = [ids_c[i] for i in idx][:10]
        ordens[q] = ordem
        docs = [docs_por_id[i] for i in ordem]
        ev = ar.evidencias_para(gab[q], False)
        for modo, lista in (("semantico", docs), ("hib-todos", c.hibrido(gab[q]["pergunta"], docs, None, "hib-todos"))):
            r = ar.avaliar_lista(lista, ev)
            r["n_ev"] = len(ev)
            r["chars_ctx"] = len("\n\n---\n\n".join(d.content or "" for d in lista))
            r["ordem_ids"] = [d.id for d in lista]
            res[modo][q] = r
    return {"res": res, "ordens": ordens}


def relatar() -> None:
    ar, gab, ids, c = carregar_base()
    todas, foco = ordem_perguntas(ids)
    erros = _ler_json(PASTA_CACHE / "erros.json", [])
    cand = _ler_json(PASTA_CACHE / "candidatos.json", {})
    configs = []                                        # (rótulo, cache) na ordem de execução
    for modelo_pedido, substituto in MODELOS:
        for ml in MAX_LENGTHS:
            for m in (modelo_pedido, substituto):
                cache = _ler_json(_caminho_cache(m, ml), None)
                if cache and cache.get("perguntas"):
                    configs.append((m, ml, cache))
                    break
    L = ["# Reranking sobre o retrieval do IF Turing — resumo", "",
         f"Gerado em {datetime.now():%Y-%m-%d %H:%M} por `avaliar_rerank.py relatar` a partir de `rerank_cache/`. "
         f"Índice: `{INDICE}` ({c.n_chunks} chunks; item+título, cabeçalho de tabela, ids estáveis); opções de consulta "
         f"{OPCOES}; híbrido = `hib-todos` (cronograma inteiro). **Este experimento só lê índices temporários; o reranker "
         "existe em `nucleo.py` como opção desligada por padrão (`RERANKER_ATIVO`, ver GPUS.md).**", ""]
    # ── decisões e ressalvas
    L += ["## Decisões, falhas e ressalvas (modo sem supervisão)", "",
          f"- Modelos: grande `{MODELOS[0][0]}` (max_length 256 e 384); menor `{MODELOS[1][0]}`; substituto de reserva "
          f"`{MODELOS[0][1]}` (um único substituto por modelo). `trust_remote_code` **nunca** foi usado; nada foi "
          "instalado no venv (usei só `sentence-transformers` 6.1 / `transformers` 5.20 / `torch` 2.14 CPU já existentes).",
          f"- Cada pergunta tem os {N_CANDIDATOS} primeiros candidatos do ranking semântico exato; pontuados UMA vez por "
          "(modelo, max_length). Os cenários 15 e 20 usam os 15/20 primeiros desses 30, sem repontuar; o reranker "
          "reordena e ficam os 10 primeiros.",
          "- Baseline = ranking semântico exato (10 primeiros) sem reranker; ele pode diferir em ±1 chunk do retriever "
          "aproximado do Chroma usado em relatórios anteriores.",
          "- **Tempos de 15 e 20 candidatos são DERIVADOS** (tempo medido com 30 × N/30), não medidos.",
          f"- Hardware desta medição: {platform.processor()}, torch com {configs[0][2].get('threads_torch', '?') if configs else '?'} "
          "threads, **CPU, fp32**. Não há GPU nesta máquina.",
          "- Perguntas do subconjunto foco: " + ", ".join(foco) + f" (as 10 últimas são aleatórias com semente fixa {SEMENTE})."]
    if erros:
        L += ["- **Erros registrados:**"] + [f"  - {e['quando']} · `{e['modelo']}` ml={e['max_length']}: {e['erro']}" for e in erros]
    else:
        L += ["- Nenhum erro de carregamento ou pontuação foi registrado."]
    for m, ml, cache in configs:
        sub = " **(SUBCONJUNTO FOCO: estimativa > 20 min)**" if cache.get("subconjunto") else ""
        subst = f" (substituto de `{cache['pedido']}`)" if cache.get("substituto") else ""
        compl = "" if cache.get("completo") else " **(INCOMPLETO: execução interrompida)**"
        L.append(f"- `{m}`{subst} ml={ml}: {len(cache['perguntas'])} perguntas pontuadas{sub}{compl}; "
                 f"estimativa da sonda {cache.get('estimativa_s', '?')} s.")
    L.append("")

    # ── métricas
    L += ["## Métricas por configuração", "",
          "`semantico` = lista reordenada pelo reranker (10 primeiros); `híbrido` = o mesmo fluxo de `nucleo.responder()` "
          "alimentado por essa lista. Grupos legado e novas separados.", "",
          "| configuração | N cand. | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    resultados = {}
    baselines_vistos = set()
    grupos = ("legado", "novas")

    def linhas(rotulo, ncand, qs, av):
        out = []
        for g in grupos:
            qg = [q for q in qs if gab[q]["grupo"] == g]
            for modo in ("semantico", "hib-todos"):
                a = ar.agregar([av["res"][modo][q] for q in qg])
                if a["n"]:
                    out.append(f"| {rotulo} | {ncand} | {g} | {'híbrido' if modo == 'hib-todos' else 'semântico'} | {a['n']} | "
                               f"{_pct(a['hit@3'])} | {_pct(a['hit@5'])} | {_pct(a['hit@10'])} | {a['mrr']:.3f} | "
                               f"{_pct(a['hit@ctx'])} | {a['chars']:,.0f} |")
        return out

    if configs:
        ref = configs[0][2]
        qs_completo = [q for q in todas if q in ref["perguntas"]]
    for m, ml, cache in configs:
        qs = [q for q in todas if q in cache["perguntas"]]
        completo = len(qs) == len(ids)
        rot = f"{m.split('/')[-1]} ml={ml}"
        resultados[(m, ml, None)] = base = avaliar_config(ar, gab, c, cache, qs, None)
        if tuple(qs) not in baselines_vistos:      # o baseline só muda quando o conjunto de perguntas muda
            baselines_vistos.add(tuple(qs))
            L += linhas(f"baseline (sem reranker){'' if completo else ' — mesmo subconjunto de ' + rot}", "—", qs, base)
        for n in CENARIOS:
            av = avaliar_config(ar, gab, c, cache, qs, n)
            resultados[(m, ml, n)] = av
            L += linhas(rot + ("" if completo else " (SUBCONJUNTO)"), n, qs, av)
    L.append("")

    # ── todas as configurações no MESMO subconjunto (o bge só roda no foco; o menor roda em tudo)
    comuns = [q for q in foco if all(q in cache["perguntas"] for _, _, cache in configs)]
    if configs and len(comuns) < len(ids):
        L += [f"## Comparação no MESMO subconjunto foco ({len(comuns)} perguntas: "
              f"{sum(1 for q in comuns if gab[q]['grupo'] == 'legado')} do legado, "
              f"{sum(1 for q in comuns if gab[q]['grupo'] == 'novas')} das novas)", "",
              "O reranker grande só foi pontuado nestas perguntas (estimativa > 20 min para as 57); aqui o menor é "
              "restrito ao mesmo conjunto para a comparação ser justa. n pequeno: leia como indício, não como medida.", "",
              "| configuração | N cand. | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        L += linhas("baseline (sem reranker)", "—", comuns,
                    avaliar_config(ar, gab, c, configs[0][2], comuns, None))
        for m, ml, cache in configs:
            for n in CENARIOS:
                L += linhas(f"{m.split('/')[-1]} ml={ml}", n, comuns, avaliar_config(ar, gab, c, cache, comuns, n))
        L.append("")

    # ── variação por pergunta
    L += ["## Perguntas que mudaram em relação ao baseline (menor k que cobre a evidência, híbrido)", "",
          "| configuração | N | melhoraram | pioraram | iguais |", "|---|---|---|---|---|"]
    for m, ml, cache in configs:
        qs = [q for q in todas if q in cache["perguntas"]]
        b = resultados[(m, ml, None)]["res"]["hib-todos"]
        for n in CENARIOS:
            r = resultados[(m, ml, n)]["res"]["hib-todos"]
            melh = pior = 0
            for q in qs:
                kb = b[q]["k_min"] or 10 ** 6
                kr = r[q]["k_min"] or 10 ** 6
                melh += kr < kb
                pior += kr > kb
            L.append(f"| {m.split('/')[-1]} ml={ml} | {n} | {melh} | {pior} | {len(qs) - melh - pior} |")
    L.append("")

    # ── tempo
    L += ["## Tempo do reranker por pergunta (CPU, fp32)", "",
          "Medido: `predict` de 30 pares (pergunta, chunk), lote 16, média e mediana por pergunta. "
          "**15 e 20 são DERIVADOS** proporcionalmente (tempo(30) × N/30).", "",
          "| configuração | perguntas medidas | tokens/pergunta (30 pares) | 30 cand. (medido) média / mediana | 20 cand. (derivado) | 15 cand. (derivado) |",
          "|---|---|---|---|---|---|"]
    for m, ml, cache in configs:
        ts = [v["tempo_s"] for v in cache["perguntas"].values()]
        tk = [v["tokens"] for v in cache["perguntas"].values()]
        L.append(f"| {m.split('/')[-1]} ml={ml} | {len(ts)} | {np.mean(tk):,.0f} | {np.mean(ts):.2f} s / {np.median(ts):.2f} s | "
                 f"{np.mean(ts) * 20 / 30:.2f} s | {np.mean(ts) * 15 / 30:.2f} s |")
    L.append("")

    # ── T1000
    L += ["## Estimativa para uma NVIDIA T1000 (fp32) — **ESTIMATIVA, não medida**", "",
          "Não há GPU nesta máquina. Método: FLOPs ≈ 2 × parâmetros do corpo do transformador (sem embeddings) × tokens "
          "processados (contados com o tokenizer, truncados em max_length; ignora atenção e o custo de lotes com padding); "
          f"tempo = FLOPs / (pico fp32 da T1000 ≈ {T1000_PICO_FLOPS / 1e12:.1f} TFLOPS × utilização "
          f"{int(T1000_UTIL[0] * 100)}–{int(T1000_UTIL[1] * 100)}%). A utilização é uma suposição típica de inferência fp32 "
          "de transformers; transferência de dados e carga do modelo não entram. O 'efeito CPU' é o mesmo cálculo com o "
          "tempo medido, para calibrar a conta.", "",
          "| configuração | pesos fp32 (GB) | GFLOPs por pergunta (30 pares) | CPU medida (GFLOPS efetivos) | T1000 30 cand. (estimativa) | T1000 20 cand. | T1000 15 cand. |",
          "|---|---|---|---|---|---|---|"]
    for m, ml, cache in configs:
        if "params_corpo" not in cache:
            continue
        tk = np.mean([v["tokens"] for v in cache["perguntas"].values()])
        ts = np.mean([v["tempo_s"] for v in cache["perguntas"].values()])
        fl = 2 * cache["params_corpo"] * tk
        lo, hi = fl / (T1000_PICO_FLOPS * T1000_UTIL[1]), fl / (T1000_PICO_FLOPS * T1000_UTIL[0])
        L.append(f"| {m.split('/')[-1]} ml={ml} | {cache['params_total'] * 4 / 1e9:.2f} | {fl / 1e9:,.0f} | {fl / ts / 1e9:,.0f} | "
                 f"≈ {lo:.2f}–{hi:.2f} s | ≈ {lo * 20 / 30:.2f}–{hi * 20 / 30:.2f} s | ≈ {lo * 15 / 30:.2f}–{hi * 15 / 30:.2f} s |")
    L += ["", "Memória: a T1000 tem 4 GB ou 8 GB (conferir); os pesos fp32 da tabela acima mais ativações de lote 16 precisam "
          "caber nisso, e o e5-base (~1,1 GB fp32) soma se `EMBEDDER_DEVICE=cuda`. Ver GPUS.md.", ""]

    # ── evidência por pergunta foco
    L += ["## Evidência por pergunta foco (E03, F16a, F18a, A07, E01)", "",
          "`sem.` = menor k no semântico reordenado; `híb.` = menor k no contexto híbrido (— = nunca cobre); "
          "`posição das evidências` = posição, no ranking reordenado dos N candidatos, do melhor chunk de cada "
          "evidência (✗ = fora dos N candidatos).", "",
          "| pergunta | configuração | N | k sem. | k híb. | posição das evidências |", "|---|---|---|---|---|---|"]
    docs_texto = {d.id: ar._norm(d.content or "") for d in c.todos}
    for q in ["E03", "F16a", "F18a", "A07", "E01"]:
        if q not in ids:
            continue
        ev = ar.evidencias_para(gab[q], False)
        ids_c = cand.get(q, [])

        def posicoes(ordem_ids, n_cand):
            partes = []
            for e in ev:
                en = ar._norm(e)
                pos = [i + 1 for i, d in enumerate(ordem_ids) if en in docs_texto[d]]
                partes.append(str(min(pos)) if pos else "✗")
            return ", ".join(partes)

        rb = None
        for m, ml, cache in configs[:1]:
            if q in cache["perguntas"]:
                bb = resultados[(m, ml, None)]
                L.append(f"| {q} | baseline | 10 | {bb['res']['semantico'][q]['k_min'] or '—'} | "
                         f"{bb['res']['hib-todos'][q]['k_min'] or '—'} | {posicoes(ids_c[:30], 30)} (ranking semântico) |")
        for m, ml, cache in configs:
            if q not in cache["perguntas"]:
                continue
            for n in CENARIOS:
                av = resultados[(m, ml, n)]
                info = cache["perguntas"][q]
                idx = sorted(range(n), key=lambda i: (-info["scores"][i], i))
                ordem_n = [info["ids"][i] for i in idx]
                L.append(f"| {q} | {m.split('/')[-1]} ml={ml} | {n} | {av['res']['semantico'][q]['k_min'] or '—'} | "
                         f"{av['res']['hib-todos'][q]['k_min'] or '—'} | {posicoes(ordem_n, n)} |")
    L.append("")
    SAIDA_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"escrito {SAIDA_MD}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pontuar":
        pontuar(forcar_completo="--completo" in sys.argv)
    elif cmd == "relatar":
        relatar()
    else:
        sys.exit("uso: python avaliar_rerank.py pontuar|relatar")
