"""
teste_modelos.py — Compara modelos usando o mesmo fluxo de produção RAG.

Reutiliza carregar_pipeline() de teste_timing.py e os blocos de retrieval
de nucleo.py. Não altera nenhum dos dois.

Uso:
    python teste_modelos.py
    python teste_modelos.py --modelos "qwen3:0.6b,qwen2.5:3b,gemma3:4b"
    python teste_modelos.py --ids "A01,B01,C01,C02,C03,C04"
"""
import os
import sys
import csv
import re
import time
import argparse
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()

from teste_timing import carregar_pipeline
from nucleo import (
    OLLAMA_URL,
    _PADRAO_COTA,
    _buscar_tabelas_por_keyword,
    _buscar_cronograma_por_keyword,
    perguntar_llm,
    _OllamaOffline,
)

# IDs padrão: 4 datas (incluindo data de prova), 2 cursos, 1 duração,
# 1 vagas, 1 documentos, 1 caso de borda/critério
_IDS_PADRAO = ["C01", "C02", "C03", "C04", "A01", "B01", "A05", "A04", "E03", "E01"]

_PIPELINE_MD = Path(__file__).parent / "PIPELINE.md"
_SAIDA_CSV   = Path(__file__).parent / "resultado_modelos.csv"
_SAIDA_MD    = Path(__file__).parent / "resultado_modelos.md"


# ── Gabarito ──────────────────────────────────────────────────────────────

def carregar_gabarito() -> dict:
    """Parse todos os blocos A-E de PIPELINE.md. Retorna {id: {pergunta, resposta_esperada}}."""
    texto = _PIPELINE_MD.read_text(encoding="utf-8")
    gabarito: dict = {}
    for m in re.finditer(r"\|\s*([A-E]\d{2})\s*\|([^|]+)\|([^|]+)\|", texto):
        qid = m.group(1).strip()
        gabarito[qid] = {
            "pergunta": m.group(2).strip(),
            "resposta_esperada": m.group(3).strip(),
        }
    return gabarito


# ── Auto-avaliação ────────────────────────────────────────────────────────

_STOP_EVAL = frozenset({
    "e", "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "que", "para", "com", "por", "se", "um",
    "uma", "ao", "aos", "ou", "mas", "sim", "nao", "campus", "ibiruba",
    "curso", "cursos", "deve",
})


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _keywords(esperada: str) -> list:
    tokens = re.split(r"[,;()/|\s]+", _norm(esperada))
    return [t for t in tokens if len(t) >= 3 and t not in _STOP_EVAL]


def avaliar(resposta: str, esperada: str) -> str:
    kws = _keywords(esperada)
    if not kws:
        return "REVISAR"
    resp = _norm(resposta)
    n = sum(1 for k in kws if k in resp)
    ratio = n / len(kws)
    if ratio >= 0.8:
        return "OK"
    elif ratio >= 0.4:
        return "REVISAR"
    return "FALHOU"


# ── Disponibilidade Ollama ────────────────────────────────────────────────

def _ollama_disponiveis() -> set:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


# ── Execução de uma pergunta ──────────────────────────────────────────────

def _executar(pipeline, tabela_docs: list, cronograma_docs: list,
              pergunta: str, modelo_cfg: dict) -> dict:
    """Reproduz nucleo.responder() sem @st.cache_resource."""
    t0 = time.perf_counter()

    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]

    docs = [
        d for d in docs
        if (d.meta or {}).get("tipo") == "tabela"
        or not _PADRAO_COTA.search(d.content or "")
    ]

    docs_tabela_extra = _buscar_tabelas_por_keyword(pergunta, tabela_docs, docs)
    if docs_tabela_extra:
        docs_tabela_ja   = [d for d in docs if (d.meta or {}).get("tipo") == "tabela"]
        docs_texto_final = [d for d in docs if (d.meta or {}).get("tipo") != "tabela"][:6]
        docs = docs_tabela_ja + docs_tabela_extra + docs_texto_final

    docs_crono_extra = _buscar_cronograma_por_keyword(pergunta, cronograma_docs, docs)
    if docs_crono_extra:
        docs = docs_crono_extra + docs

    if not docs:
        return {
            "resposta": "⚠️ Nenhum trecho relevante encontrado.",
            "tempo_total_s": round(time.perf_counter() - t0, 3),
            "tokens_saida": 0,
        }

    contexto = "\n\n---\n\n".join(d.content for d in docs)

    try:
        r = perguntar_llm(pergunta, contexto, modelo_cfg)
    except _OllamaOffline:
        return {
            "resposta": "ERRO: Ollama offline",
            "tempo_total_s": round(time.perf_counter() - t0, 3),
            "tokens_saida": 0,
        }

    return {
        "resposta": r["texto"],
        "tempo_total_s": round(time.perf_counter() - t0, 3),
        "tokens_saida": r["tokens_saida"],
    }


# ── Impressão de tabela terminal ──────────────────────────────────────────

def _imprimir_tabela(resultados: dict, modelos: list, ids: list, gabarito: dict) -> None:
    col_id = 4
    col_q  = 36
    col_m  = 18

    sep = "-" * (col_id + 1 + col_q + 1 + col_m * len(modelos))
    cabecalho_m = "".join(m[:col_m - 1].ljust(col_m) for m in modelos)

    print()
    print(sep)
    print(f"{'ID'.ljust(col_id)} {'Pergunta'.ljust(col_q)} {cabecalho_m}")
    print(sep)

    for qid in ids:
        pergunta = gabarito[qid]["pergunta"]
        p_curta  = (pergunta[:col_q - 3] + "...") if len(pergunta) > col_q else pergunta
        celulas  = ""
        for m in modelos:
            cell = resultados.get(m, {}).get(qid)
            if cell is None:
                s = "—"
            else:
                s = f"{cell['resultado']} {cell['tempo']:.1f}s"
            celulas += s[:col_m - 1].ljust(col_m)
        print(f"{qid.ljust(col_id)} {p_curta.ljust(col_q)} {celulas}")

    print(sep)
    print()
    print("Resumo por modelo:")
    for m in modelos:
        res    = resultados.get(m, {})
        vals   = [v for v in res.values() if v]
        n_ok   = sum(1 for v in vals if v["resultado"] == "OK")
        n_rev  = sum(1 for v in vals if v["resultado"] == "REVISAR")
        n_fail = sum(1 for v in vals if v["resultado"] == "FALHOU")
        t_med  = (sum(v["tempo"] for v in vals) / len(vals)) if vals else 0.0
        print(f"  {m:<28} OK={n_ok}  REVISAR={n_rev}  FALHOU={n_fail}  "
              f"tempo_médio={t_med:.1f}s")
    print()


# ── Persistência ──────────────────────────────────────────────────────────

def _salvar_csv(resultados: dict, modelos: list, ids: list, gabarito: dict) -> None:
    with _SAIDA_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pergunta", "modelo", "resposta", "resultado",
                    "tempo_total_s", "tokens_saida"])
        for qid in ids:
            for m in modelos:
                cell = resultados.get(m, {}).get(qid)
                if cell is None:
                    continue
                w.writerow([
                    qid, gabarito[qid]["pergunta"], m,
                    cell["resposta"], cell["resultado"],
                    cell["tempo"], cell["tokens"],
                ])
    print(f"[saída] CSV salvo em {_SAIDA_CSV}")


def _salvar_md(resultados: dict, modelos: list, ids: list, gabarito: dict) -> None:
    linhas = ["# Resultado: Comparação de Modelos", ""]
    linhas.append("| ID | Pergunta | " + " | ".join(modelos) + " |")
    linhas.append("|---|---| " + " | ".join(["---"] * len(modelos)) + " |")
    for qid in ids:
        cells = []
        for m in modelos:
            cell = resultados.get(m, {}).get(qid)
            cells.append(f"{cell['resultado']} ({cell['tempo']:.1f}s)" if cell else "—")
        linhas.append(f"| {qid} | {gabarito[qid]['pergunta']} | " + " | ".join(cells) + " |")
    linhas += ["", "## Resumo por modelo", ""]
    for m in modelos:
        vals  = [v for v in resultados.get(m, {}).values() if v]
        n_ok  = sum(1 for v in vals if v["resultado"] == "OK")
        n_rev = sum(1 for v in vals if v["resultado"] == "REVISAR")
        n_fal = sum(1 for v in vals if v["resultado"] == "FALHOU")
        linhas.append(f"- **{m}**: OK={n_ok}, REVISAR={n_rev}, FALHOU={n_fal}")
    linhas.append("")
    _SAIDA_MD.write_text("\n".join(linhas), encoding="utf-8")
    print(f"[saída] Markdown salvo em {_SAIDA_MD}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara modelos RAG usando o gabarito de PIPELINE.md"
    )
    parser.add_argument("--modelos", default=None,
                        help='Modelos separados por vírgula, ex: "qwen3:0.6b,qwen2.5:3b"')
    parser.add_argument("--ids", default=None,
                        help=f'IDs do gabarito, ex: "A01,B01,C01". Padrão: {",".join(_IDS_PADRAO)}')
    args = parser.parse_args()

    # Gabarito
    gabarito = carregar_gabarito()
    if not gabarito:
        print("ERRO: Não foi possível parsear o gabarito de PIPELINE.md")
        sys.exit(1)

    ids = [i.strip() for i in args.ids.split(",")] if args.ids else _IDS_PADRAO
    invalidos = [i for i in ids if i not in gabarito]
    if invalidos:
        print(f"ERRO: IDs não encontrados no gabarito: {invalidos}")
        print(f"IDs disponíveis: {sorted(gabarito)}")
        sys.exit(1)

    # Resolve modelos
    if args.modelos:
        solicitados = [m.strip() for m in args.modelos.split(",")]
    else:
        solicitados = ["qwen3:0.6b"]

    maritaca_ids = {m for m in solicitados if "sabia" in m.lower() or "maritaca" in m.lower()}
    ollama_ids   = [m for m in solicitados if m not in maritaca_ids]

    disponiveis_ollama = _ollama_disponiveis()
    tem_maritaca = os.getenv("MARITACA_API_KEY", "").strip() != ""

    modelos_ok: list = []
    modelos_tipo: dict = {}
    for m in ollama_ids:
        if m in disponiveis_ollama:
            modelos_ok.append(m)
            modelos_tipo[m] = "ollama"
        else:
            print(f"[aviso] Modelo {m!r} não disponível no Ollama — pulando")
    for m in maritaca_ids:
        if tem_maritaca:
            modelos_ok.append(m)
            modelos_tipo[m] = "maritaca"
        else:
            print(f"[aviso] {m!r} requer MARITACA_API_KEY — pulando (chave não configurada)")

    if not modelos_ok:
        print("ERRO: Nenhum modelo disponível. Verifique se o Ollama está rodando.")
        sys.exit(1)

    # Confirmação
    print("\n" + "=" * 60)
    print("Modelos a testar:")
    for m in modelos_ok:
        print(f"  • {m} ({modelos_tipo[m]})")
    print(f"\nPerguntas ({len(ids)} IDs):")
    for qid in ids:
        print(f"  {qid}: {gabarito[qid]['pergunta'][:70]}")
    print("=" * 60)
    print()

    # Pipeline (sem @st.cache_resource)
    pipeline, tabela_docs, cronograma_docs = carregar_pipeline()

    # Execução
    resultados: dict = {m: {} for m in modelos_ok}

    for modelo_id in modelos_ok:
        modelo_cfg = {"tipo": modelos_tipo[modelo_id], "modelo": modelo_id}

        print(f"\n{'=' * 60}")
        print(f"MODELO: {modelo_id}")
        print(f"{'=' * 60}")

        # Warmup — excluído do timing
        print(f"[warmup] Aquecendo {modelo_id!r} ...", end=" ", flush=True)
        t_w = time.perf_counter()
        try:
            _executar(pipeline, tabela_docs, cronograma_docs,
                      "Quais são os cursos?", modelo_cfg)
            print(f"OK ({time.perf_counter() - t_w:.1f}s)")
        except Exception as exc:
            print(f"aviso ({exc})")

        # Perguntas cronometradas
        for qid in ids:
            pergunta = gabarito[qid]["pergunta"]
            esperada = gabarito[qid]["resposta_esperada"]
            print(f"\n  [{qid}] {pergunta}")
            cell = _executar(pipeline, tabela_docs, cronograma_docs, pergunta, modelo_cfg)
            resultado = avaliar(cell["resposta"], esperada)
            print(f"  Resposta : {cell['resposta'][:120]}"
                  f"{'...' if len(cell['resposta']) > 120 else ''}")
            print(f"  Esperado : {esperada[:80]}")
            print(f"  → {resultado}  ({cell['tempo_total_s']:.1f}s, "
                  f"{cell['tokens_saida']} tokens)")
            resultados[modelo_id][qid] = {
                "resposta": cell["resposta"],
                "resultado": resultado,
                "tempo": cell["tempo_total_s"],
                "tokens": cell["tokens_saida"],
            }

    _imprimir_tabela(resultados, modelos_ok, ids, gabarito)
    _salvar_csv(resultados, modelos_ok, ids, gabarito)
    _salvar_md(resultados, modelos_ok, ids, gabarito)


if __name__ == "__main__":
    main()
