"""
teste_modelos.py — Compara modelos usando o mesmo fluxo de produção RAG.

Reutiliza carregar_pipeline() de teste_timing.py e os blocos de retrieval
de nucleo.py. Não altera nenhum dos dois.

Uso:
    python teste_modelos.py
    python teste_modelos.py --modelos "qwen3:0.6b,qwen2.5:3b,gemma3:4b"
    python teste_modelos.py --ids "A01,B01,C01,C02,C03,C04"
    python teste_modelos.py --todas --modelos "gemma3:4b"      # todas as perguntas com resposta esperada

Modos offline (não chamam nenhum modelo nem carregam o índice):
    python teste_modelos.py --reavaliar resultado_modelos.csv  # reavalia respostas gravadas (antigo × novo)
    python teste_modelos.py --autoteste                        # reavalia respostas reais embutidas no código

Avaliador (regra v2; a regra antiga fica congelada em _avaliar_v1 só para comparação):
  • datas são comparadas pelo valor (dia, mês, ano opcional), não pelo texto: "07 a 08 de dezembro de 2026",
    "07 e 08/12/2026" e "07 a 08/12/2026" são equivalentes; "30 de setembro de 2026" = "30/09/2026";
  • perguntas de contagem ("Quantos…?" com número esperado) só valem OK se a resposta AFIRMAR o número
    (dígito ou por extenso) fora de linhas cruas do contexto e sem ser outra grandeza (ex.: "4 semestres");
  • a coluna "observação" avisa quando a resposta repete linhas cruas do contexto ("Curso:", "Data:", "Evento:").
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
from avaliar_retrieval import _IDS_LEGADO, _celulas
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

# D01 e D03 (negativas) são da fase 1, mas não estão em _IDS_LEGADO (que só lista as perguntas com evidência
# de retrieval); aqui entram no grupo legado. O grupo "novas" é o complemento: F10a…F18c, E03, E04, E05.
_NEGATIVAS_LEGADO = frozenset({"D01", "D03"})
_GRUPOS = ("legado", "novas")

_PIPELINE_MD = Path(__file__).parent / "PIPELINE.md"
_SAIDA_CSV   = Path(__file__).parent / "resultado_modelos.csv"
_SAIDA_MD    = Path(__file__).parent / "resultado_modelos.md"


def grupo_de(qid: str) -> str:
    return "legado" if qid in _IDS_LEGADO or qid in _NEGATIVAS_LEGADO else "novas"


# ── Gabarito ──────────────────────────────────────────────────────────────

def carregar_gabarito() -> dict:
    """Lê as tabelas do gabarito (blocos A–F) de PIPELINE.md.

    Retorna {id: {pergunta, resposta_esperada, nao_deve_conter, negativa, grupo}}.
    A 3ª coluna é a resposta esperada (blocos A–D, F) ou "o que verificar" (bloco E); 'NAO DEVE CONTER' é
    opcional (lista vazia quando ausente). `resposta_esperada` vazia = pergunta sem resposta esperada.
    """
    gabarito: dict = {}
    cabecalho = None
    for linha in _PIPELINE_MD.read_text(encoding="utf-8").splitlines():
        if not linha.strip().startswith("|"):
            cabecalho = None
            continue
        celulas = _celulas(linha)
        if celulas and celulas[0] == "#":
            cabecalho = [c.upper() for c in celulas]
            continue
        if cabecalho is None or not re.fullmatch(r"[A-F]\d{2}[a-z]?", celulas[0]) or len(celulas) < 3:
            continue
        i_nao = next((i for i, c in enumerate(cabecalho) if c.startswith("NAO DEVE")), None)
        i_ev = cabecalho.index("EVIDENCIA") if "EVIDENCIA" in cabecalho else None
        raw_nao = celulas[i_nao] if i_nao is not None and i_nao < len(celulas) else ""
        ev = celulas[i_ev].strip().upper() if i_ev is not None and i_ev < len(celulas) else ""
        esperada = celulas[2].strip()
        qid = celulas[0]
        gabarito[qid] = {
            "pergunta": celulas[1].strip(),
            "resposta_esperada": esperada,
            "nao_deve_conter": [p.strip() for p in re.split(r"[,;]", raw_nao) if p.strip()],
            "negativa": ev == "NEGATIVA" or "pergunta negativa" in esperada.lower(),
            "grupo": grupo_de(qid),
        }
    return gabarito


# ── Auto-avaliação ────────────────────────────────────────────────────────

_STOP_EVAL = frozenset({
    "e", "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "que", "para", "com", "por", "se", "um",
    "uma", "ao", "aos", "ou", "mas", "sim", "nao", "campus", "ibiruba",
    "curso", "cursos", "deve", "ate",
    # Palavras de unidade/quantidade — o número isolado já é suficiente
    "vagas", "vaga", "nomes", "distintos", "total", "semestres", "semestre",
})

# Mapeamento dígito → forma por extenso (português) para correspondência numérica.
_NUM_EXT = {
    "1": "um", "2": "dois", "3": "tres", "4": "quatro", "5": "cinco",
    "6": "seis", "7": "sete", "8": "oito", "9": "nove", "10": "dez",
    "30": "trinta", "32": "trinta e dois",
}
_EXT_NUM = {v: k for k, v in _NUM_EXT.items()}

# Nomes de curso normalizados para contagem de repetições
_CURSOS_NORM = [
    "agropecuaria", "informatica", "mecanica", "eletrotecnica",
    "agronomia", "ciencia da computacao", "engenharia mecanica", "matematica",
]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _keywords(esperada: str) -> list:
    tokens = re.split(r"[,;()/|\s]+", _norm(esperada))
    # Mantém token se: tem ≥ 3 chars OU é puramente numérico (ex: "32", "4")
    return [t for t in tokens if t not in _STOP_EVAL and (len(t) >= 3 or t.isdigit())]


def _kw_presente(kw: str, resp_norm: str) -> bool:
    """Verifica se kw está em resp_norm, aceitando dígito ↔ forma por extenso."""
    if kw in resp_norm:
        return True
    # dígito → extenso: "4" aceita "quatro"
    if kw in _NUM_EXT and _norm(_NUM_EXT[kw]) in resp_norm:
        return True
    # extenso → dígito: "quatro" aceita "4"
    if kw in _EXT_NUM and _EXT_NUM[kw] in resp_norm:
        return True
    return False


def _contar_repeticoes(resposta: str) -> int:
    """Conta quantas vezes nomes de curso aparecem mais de uma vez na resposta."""
    resp = _norm(resposta)
    total = 0
    for nome in _CURSOS_NORM:
        count = 0
        pos = 0
        while True:
            idx = resp.find(nome, pos)
            if idx == -1:
                break
            count += 1
            pos = idx + 1
        if count > 1:
            total += count - 1
    return total


def _avaliar_v1(resposta: str, esperada: str,
                nao_deve_conter: list | None = None) -> tuple[str, int]:
    """REGRA ANTIGA (congelada; só para comparar com a v2). Não usar para avaliar de verdade.

    Palavras-chave da resposta esperada por substring, dígito ↔ extenso; limiares 0,8 (OK) e 0,4 (REVISAR).
    Defeitos que motivaram a v2: datas por extenso viravam REVISAR ("12" não aparece em "de dezembro") e
    qualquer "4" no texto (ex.: "Duração: 4 semestres") valia como a contagem 4.
    """
    resp = _norm(resposta)
    n_rep = _contar_repeticoes(resposta)

    if nao_deve_conter:
        for proibida in nao_deve_conter:
            if _norm(proibida) in resp:
                return "FALHOU", n_rep

    kws = _keywords(esperada)
    if not kws:
        return "REVISAR", n_rep

    n = sum(1 for k in kws if _kw_presente(k, resp))
    ratio = n / len(kws)
    if ratio >= 0.8:
        return "OK", n_rep
    elif ratio >= 0.4:
        return "REVISAR", n_rep
    return "FALHOU", n_rep


# ── Datas: comparação por valor ───────────────────────────────────────────

_MESES = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7,
          "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}
_MES_ALT = "|".join(_MESES)
_RE_DATA_NUM = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{4}))?(?![\d/])")
_RE_DATA_EXT = re.compile(rf"(?<!\d)(\d{{1,2}})\s*(?:o\s*)?(?:de\s+)?({_MES_ALT})\b(?:\s*(?:de\s+)?(\d{{4}}))?")
# "07 a 08/12/2026", "07 e 08 de dezembro": o dia solto antes de uma data completa herda mês e ano dela
_RE_DIA_SOLTO = re.compile(r"(?<![\d/])(\d{1,2})\s*(?:a|e|ate|ao|-|–)\s*$")


def _extrair_datas(texto: str) -> tuple[list, str]:
    """Datas (dia, mês, ano|None) de um texto JÁ normalizado (_norm) e o texto sem elas.

    Reconhece dd/mm[/aaaa], "dd [de] <mês> [de aaaa]" e dias soltos de intervalo ("07 a 08 de dezembro de 2026").
    """
    achados = []
    for m in _RE_DATA_NUM.finditer(texto):
        achados.append((m.start(), m.end(), int(m.group(1)), int(m.group(2)), m.group(3)))
    for m in _RE_DATA_EXT.finditer(texto):
        achados.append((m.start(), m.end(), int(m.group(1)), _MESES[m.group(2)], m.group(3)))
    achados = [a for a in achados if 1 <= a[2] <= 31 and 1 <= a[3] <= 12]
    achados.sort()
    datas, spans, fim = [], [], -1
    for ini, f, dia, mes, ano in achados:
        if ini < fim:  # sobreposição com uma data já aceita
            continue
        ano = int(ano) if ano else None
        pre = _RE_DIA_SOLTO.search(texto[:ini])
        if pre and pre.start() >= fim:
            d0 = int(pre.group(1))
            if 1 <= d0 <= 31:
                datas.append({"dia": d0, "mes": mes, "ano": ano})
                ini = pre.start()
        datas.append({"dia": dia, "mes": mes, "ano": ano})
        spans.append((ini, f))
        fim = f
    resto, pos = [], 0
    for ini, f in spans:
        resto.append(texto[pos:ini])
        pos = f
    resto.append(texto[pos:])
    return datas, " ".join(resto)


def _data_presente(d: dict, datas_resp: list) -> bool:
    return any(d["dia"] == r["dia"] and d["mes"] == r["mes"]
               and (d["ano"] is None or r["ano"] is None or d["ano"] == r["ano"])
               for r in datas_resp)


# ── Contagem ──────────────────────────────────────────────────────────────

_UNIDADES_NUM = ["zero", "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "onze",
                 "doze", "treze", "catorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"]
_DEZENAS_NUM = {20: "vinte", 30: "trinta", 40: "quarenta", 50: "cinquenta", 60: "sessenta", 70: "setenta",
                80: "oitenta", 90: "noventa"}
_FEMININO = {"um": "uma", "dois": "duas"}
# Grandezas que um número pode quantificar sem ser a contagem pedida
_UNIDADES_GRANDEZA = ("semestres", "semestre", "vagas", "vaga", "horas", "hora", "minutos", "minuto",
                      "questoes", "questao", "turnos", "turno", "anos", "ano")


def _formas_extenso(n: int) -> list:
    """Formas por extenso (sem acento) de 0–100, no masculino e no feminino."""
    if n < 20:
        base = [_UNIDADES_NUM[n]]
        if n == 14:
            base.append("quatorze")
    elif n < 100:
        dez, uni = (n // 10) * 10, n % 10
        base = [_DEZENAS_NUM[dez]] if uni == 0 else [f"{_DEZENAS_NUM[dez]} e {_UNIDADES_NUM[uni]}"]
    elif n == 100:
        base = ["cem"]
    else:
        return []
    return base + [re.sub(r"\b(um|dois)\b", lambda m: _FEMININO[m.group(1)], f) for f in base
                   if re.search(r"\b(um|dois)\b", f)]


_PALAVRA_NUM = {f: n for n in range(0, 101) for f in _formas_extenso(n)}
_RE_NUM_INICIAL = re.compile(
    r"^\W*(\d+|(?:" + "|".join(sorted((re.escape(f) for f in _PALAVRA_NUM), key=len, reverse=True)) + r"))\b")


def _numero_esperado(esperada_norm: str) -> int | None:
    """Número com que a resposta esperada COMEÇA ("4 quatro", "4 cursos superiores", "Dois turnos, …"), se houver."""
    m = _RE_NUM_INICIAL.match(esperada_norm)
    if not m:
        return None
    t = m.group(1)
    return int(t) if t.isdigit() else _PALAVRA_NUM[t]


def _eh_contagem(pergunta_norm: str, esperada_norm: str) -> int | None:
    """Número esperado quando é pergunta de contagem ("Quantos/Quantas…?" cuja resposta esperada começa por número)."""
    if re.search(r"\bquant[oa]s?\b", pergunta_norm):
        return _numero_esperado(esperada_norm)
    return None


_RE_CRU_PREFIXO = re.compile(r"^\s*(?:[-*•]\s*)?(?:curso|data|evento)\s*:", re.IGNORECASE)
_RE_CRU_CAMPO = re.compile(r"(?:^|\s)(?:curso|data|evento)\s*:\s", re.IGNORECASE)


def _linha_crua(linha: str) -> bool:
    """Linha copiada do contexto: começa com "Curso:"/"Data:"/"Evento:" ou tem 2+ separadores " | " de célula."""
    return bool(_RE_CRU_PREFIXO.match(linha)) or linha.count(" | ") >= 2


def _linhas_cruas(resposta: str) -> list:
    return [l for l in resposta.splitlines() if _linha_crua(l)]


def observacao_linhas_cruas(resposta: str) -> str:
    """Texto para a coluna "observação"; vazio se a resposta não repete linhas cruas do contexto."""
    comeca = bool(_RE_CRU_PREFIXO.match(resposta))
    linhas = _linhas_cruas(resposta)
    n = len(linhas)
    if n < 2 and len(resposta.splitlines()) <= 1:  # tudo numa linha só: conta os campos "Curso:"/"Data:"/"Evento:"
        n = max(n, len(_RE_CRU_CAMPO.findall(resposta)) if _linha_crua(resposta) or comeca else 0)
    if comeca or n >= 2:
        partes = [f"{n} linha(s) crua(s)" if n else "linha crua"]
        if comeca:
            partes.append("começa com rótulo do contexto (Curso:/Data:/Evento:)")
        return "repete linhas cruas do contexto: " + "; ".join(partes)
    return ""


def _afirma_numero(resposta: str, n: int, pergunta_norm: str) -> bool:
    """A resposta afirma o número n (dígito ou extenso) fora de linhas cruas e sem ser outra grandeza?"""
    texto = _norm("\n".join(l for l in resposta.splitlines() if not _linha_crua(l)))
    padroes = [rf"(?<![\d.,/:\-])({n})(?![\d/:]|[.,]\d)"] + [rf"\b({re.escape(f)})\b" for f in _formas_extenso(n)]
    for p in padroes:
        for m in re.finditer(p, texto):
            resto = texto[m.end():m.end() + 40]
            resto = re.sub(r"^\s*\*?\s*(?:\([^)]*\)\s*)?", "", resto)
            prox = re.match(r"[a-z]+", resto)
            if prox and prox.group(0) in _UNIDADES_GRANDEZA and prox.group(0).rstrip("s") not in pergunta_norm:
                continue  # ex.: "4 semestres" numa pergunta sobre quantidade de cursos
            return True
    return False


# ── Avaliador v2 ──────────────────────────────────────────────────────────

def avaliar(resposta: str, esperada: str, nao_deve_conter: list | None = None,
            pergunta: str = "", negativa: bool = False) -> dict:
    """Avalia a resposta contra o gabarito. Retorna {resultado, repeticoes, observacao, motivo}.

    resultado: "OK", "REVISAR" ou "FALHOU".
    Regras:
    1. Se qualquer palavra de nao_deve_conter aparecer → FALHOU imediatamente.
    2. Pergunta negativa (D01, D03): OK se a resposta diz que não encontrou/não consta; senão REVISAR.
    3. Datas valem pelo valor (dia, mês, ano opcional), em qualquer formato.
    4. Números curtos são aceitos sem a unidade (e.g. "32" sem "vagas"); dígito ↔ extenso.
    5. Contagem ("Quantos…?"): só OK se a resposta afirmar o número esperado fora de linhas cruas do contexto.
    """
    resp = _norm(resposta)
    n_rep = _contar_repeticoes(resposta)
    obs = observacao_linhas_cruas(resposta)

    def saida(res: str, motivo: str = "") -> dict:
        return {"resultado": res, "repeticoes": n_rep, "observacao": obs, "motivo": motivo}

    for proibida in nao_deve_conter or []:
        if _norm(proibida) in resp:
            return saida("FALHOU", f"contém proibida: {proibida!r}")

    if negativa:
        if re.search(r"nao (?:encontrei|consta|ha |existe|oferece|foi encontrad)", resp):
            return saida("OK", "pergunta negativa respondida com \"não consta/não encontrei\"")
        return saida("REVISAR", "pergunta negativa sem \"não encontrei\": conferir se inventou")

    esp = _norm(esperada)
    datas_esp, resto = _extrair_datas(esp)
    kws = _keywords(resto)
    if not kws and not datas_esp:
        return saida("REVISAR", "sem palavras-chave na resposta esperada")

    datas_resp, _ = _extrair_datas(resp)
    n = sum(1 for k in kws if _kw_presente(k, resp)) + sum(1 for d in datas_esp if _data_presente(d, datas_resp))
    total = len(kws) + len(datas_esp)
    ratio = n / total
    motivo = f"{n}/{total} itens esperados"
    if ratio < 0.4:
        return saida("FALHOU", motivo)
    if ratio < 0.8:
        return saida("REVISAR", motivo)

    num = _eh_contagem(_norm(pergunta), esp)
    if num is not None and not _afirma_numero(resposta, num, _norm(pergunta)):
        return saida("REVISAR", f"contagem: não afirma o total {num} fora de linhas cruas ({motivo})")
    return saida("OK", motivo)


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


# ── Seleção das perguntas ─────────────────────────────────────────────────

def _selecionar_ids(gabarito: dict, args) -> list:
    if args.todas:
        if args.ids:
            print("ERRO: use --todas OU --ids, não os dois")
            sys.exit(1)
        ids = sorted(q for q, v in gabarito.items() if v["resposta_esperada"])
        fora = sorted(q for q, v in gabarito.items() if not v["resposta_esperada"])
        n_g = {g: sum(1 for q in ids if gabarito[q]["grupo"] == g) for g in _GRUPOS}
        print(f"[--todas] {len(ids)} perguntas entram (legado={n_g['legado']}, novas={n_g['novas']}); "
              f"{len(fora)} ficam de fora por falta de resposta esperada: {', '.join(fora) or 'nenhuma'}")
        negativas = [q for q in ids if gabarito[q]["negativa"]]
        if negativas:
            print(f"[--todas] negativas (OK = \"não encontrei\"): {', '.join(negativas)}")
        return ids
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else _IDS_PADRAO
    invalidos = [i for i in ids if i not in gabarito]
    if invalidos:
        print(f"ERRO: IDs não encontrados no gabarito: {invalidos}")
        print(f"IDs disponíveis: {sorted(gabarito)}")
        sys.exit(1)
    return ids


# ── Resumo (sempre separado por grupo) ────────────────────────────────────

def _contagens(vals: list, campo: str = "resultado") -> dict:
    return {r: sum(1 for v in vals if v[campo] == r) for r in ("OK", "REVISAR", "FALHOU")}


def _linhas_resumo(resultados: dict, modelos: list, gabarito: dict, campo: str = "resultado") -> list:
    out = []
    for m in modelos:
        for g in _GRUPOS:
            vals = [v for q, v in resultados.get(m, {}).items() if v and gabarito[q]["grupo"] == g]
            if not vals:
                continue
            c = _contagens(vals, campo)
            out.append((m, g, len(vals), c))
    return out


# ── Impressão de tabela terminal ──────────────────────────────────────────

def _imprimir_tabela(resultados: dict, modelos: list, ids: list, gabarito: dict) -> None:
    col_id = 5
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
                s = f"{cell['resultado']}{'*' if cell.get('observacao') else ''} {cell['tempo']:.1f}s"
            celulas += s[:col_m - 1].ljust(col_m)
        print(f"{qid.ljust(col_id)} {p_curta.ljust(col_q)} {celulas}")

    print(sep)
    print("(* = a resposta repete linhas cruas do contexto; ver coluna observação)")
    print()
    print("Resumo por modelo e grupo:")
    for m, g, n, c in _linhas_resumo(resultados, modelos, gabarito):
        vals = [v for q, v in resultados[m].items() if gabarito[q]["grupo"] == g]
        t_med = sum(v["tempo"] for v in vals) / len(vals)
        print(f"  {m:<28} {g:<7} n={n:<3} OK={c['OK']}  REVISAR={c['REVISAR']}  FALHOU={c['FALHOU']}  "
              f"tempo_médio={t_med:.1f}s")
    print()


# ── Persistência ──────────────────────────────────────────────────────────

def _salvar_csv(resultados: dict, modelos: list, ids: list, gabarito: dict) -> None:
    with _SAIDA_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "grupo", "pergunta", "modelo", "resposta", "resultado",
                    "tempo_total_s", "tokens_saida", "repeticoes", "observacao"])
        for qid in ids:
            for m in modelos:
                cell = resultados.get(m, {}).get(qid)
                if cell is None:
                    continue
                w.writerow([
                    qid, gabarito[qid]["grupo"], gabarito[qid]["pergunta"], m,
                    cell["resposta"], cell["resultado"],
                    cell["tempo"], cell["tokens"],
                    cell.get("repeticoes", 0), cell.get("observacao", ""),
                ])
    print(f"[saída] CSV salvo em {_SAIDA_CSV}")


def _md_escape(t: str) -> str:
    return " ".join(str(t).split()).replace("|", "\\|")


def _salvar_md(resultados: dict, modelos: list, ids: list, gabarito: dict) -> None:
    linhas = ["# Resultado: Comparação de Modelos", ""]
    linhas.append("| ID | Grupo | Pergunta | " + " | ".join(modelos) + " |")
    linhas.append("|---|---|---| " + " | ".join(["---"] * len(modelos)) + " |")
    for qid in ids:
        cells = []
        for m in modelos:
            cell = resultados.get(m, {}).get(qid)
            cells.append(f"{cell['resultado']}{'*' if cell.get('observacao') else ''} ({cell['tempo']:.1f}s)"
                         if cell else "—")
        linhas.append(f"| {qid} | {gabarito[qid]['grupo']} | {gabarito[qid]['pergunta']} | " + " | ".join(cells) + " |")
    linhas += ["", "`*` = a resposta repete linhas cruas do contexto (ver observações).", "",
               "## Resumo por modelo e grupo", ""]
    for m, g, n, c in _linhas_resumo(resultados, modelos, gabarito):
        linhas.append(f"- **{m}** · {g} (n={n}): OK={c['OK']}, REVISAR={c['REVISAR']}, FALHOU={c['FALHOU']}")
    obs = [(qid, m, resultados[m][qid]["observacao"]) for qid in ids for m in modelos
           if resultados.get(m, {}).get(qid, {}).get("observacao")]
    linhas += ["", "## Observações", ""]
    linhas += [f"- {qid} · {m}: {o}" for qid, m, o in obs] or ["(nenhuma)"]
    linhas.append("")
    _SAIDA_MD.write_text("\n".join(linhas), encoding="utf-8")
    print(f"[saída] Markdown salvo em {_SAIDA_MD}")


# ── Modos offline: --reavaliar e --autoteste ──────────────────────────────

# Respostas reais (gemma3:4b) com o resultado esperado da regra v2 e o que a regra v1 dava.
_AUTOTESTE = [
    ("C01", "As inscrições para o processo seletivo ocorrem no período das 14h do dia 30 de setembro de 2026 às "
            "23h59min do dia 19 de outubro de 2026, exclusivamente via INTERNET no endereço ingresso.ifrs.edu.br.",
     "OK", "REVISAR"),
    ("C03", "O prazo para recurso após as provas é de 07 a 08 de dezembro de 2026.", "OK", "REVISAR"),
    ("C02", "Os cursos aplicam a prova nos seguintes horários: Curso Integrado, Concomitante e "
            "Concomitantes/Subsequente: 8h30min Curso Subsequente: 14h Curso Graduação: 14h, horário de Brasília",
     "FALHOU", "FALHOU"),
    ("E01", "4", "OK", "OK"),
    ("E01", "Curso: Técnico em Mecânica | Turnos: Noite | Duração (semestres): 4 semestres* | Total de Vagas: 30\n"
            "Curso: Técnico em Eletrotécnica** | Turnos: Noite | Duração (semestres): 4 | Total de Vagas: 30\n"
            "Curso: Técnico em Agropecuária | Turnos: Manhã e Tarde | Duração (semestres): 3* | Total de Vagas: 32\n"
            "Curso: Técnico em Informática | Turnos: Manhã e Tarde | Duração (semestres): 3* | Total de Vagas: 32\n"
            "Curso: Técnico em Mecânica | Turnos: Manhã e Tarde | Duração (semestres): 3* | Total de Vagas: 32",
     "REVISAR", "OK"),
]

# Casos extras de cobertura das regras novas (não vêm de execuções reais; o esperado da v2 foi escrito à mão e a v1
# só é mostrada, sem conferir).
_AUTOTESTE_EXTRA = [
    ("C03", "O recurso pode ser feito nos dias 07 e 08/12/2026.", "OK", None),
    ("C03", "Prazo: 07 a 08/12/2026.", "OK", None),
    ("C03", "O prazo é 08 de dezembro de 2026.", "REVISAR", None),
    ("C01", "As inscrições vão de 30 de setembro a 19 de outubro.", "OK", None),
    ("C01", "As inscrições vão de 30/09 até 19/10.", "OK", None),
    ("C02", "A prova será aplicada em 6 de dezembro de 2026.", "OK", None),
    ("C07", "O resultado final sai em 3 de fevereiro de 2027.", "OK", None),
    ("E01", "O campus oferece quatro cursos técnicos.", "OK", None),
    ("E01", "Existem 4 cursos técnicos: Agropecuária, Informática, Mecânica e Eletrotécnica.", "OK", None),
    ("E01", "Os técnicos têm duração de 4 semestres.", "REVISAR", None),
    ("B12", "Há 4 cursos superiores.", "OK", None),
    ("A04", "O curso tem 32 vagas.", "OK", None),
    ("D01", "Não encontrei essa informação nos documentos.", "OK", None),
    ("D01", "Sim, o campus oferece Direito.", "REVISAR", None),
]


def _grade(casos: list, gabarito: dict) -> list:
    """Avalia casos (qid, resposta, esperado_v2, esperado_v1) e devolve dicts com v1, v2 e se bateu."""
    linhas = []
    for qid, resposta, esp_v2, esp_v1 in casos:
        g = gabarito[qid]
        v1, _ = _avaliar_v1(resposta, g["resposta_esperada"], g["nao_deve_conter"])
        v2 = avaliar(resposta, g["resposta_esperada"], g["nao_deve_conter"], g["pergunta"], g["negativa"])
        linhas.append({"id": qid, "esperada": g["resposta_esperada"], "resposta": resposta, "v1": v1,
                       "v2": v2["resultado"], "obs": v2["observacao"], "motivo": v2["motivo"],
                       "esp_v2": esp_v2, "esp_v1": esp_v1,
                       "ok": v2["resultado"] == esp_v2 and (esp_v1 is None or v1 == esp_v1)})
    return linhas


def _imprimir_autoteste(titulo: str, linhas: list) -> None:
    print(f"\n── {titulo} ──")
    for l in linhas:
        resp = " ".join(l["resposta"].split())
        print(f"[{'PASSOU' if l['ok'] else 'DIVERGE'}] {l['id']}  esperado v2={l['esp_v2']}  "
              f"antiga(v1)={l['v1']}  nova(v2)={l['v2']}")
        print(f"    resposta : {resp[:110]}{'…' if len(resp) > 110 else ''}")
        print(f"    gabarito : {l['esperada'][:80]}")
        print(f"    motivo   : {l['motivo']}" + (f"  | observação: {l['obs']}" if l["obs"] else ""))


def autoteste(gabarito: dict) -> int:
    ausentes = [q for q, *_ in _AUTOTESTE + _AUTOTESTE_EXTRA if q not in gabarito]
    if ausentes:
        print(f"ERRO: ids do autoteste fora do gabarito: {ausentes}")
        return 1
    reais, extras = _grade(_AUTOTESTE, gabarito), _grade(_AUTOTESTE_EXTRA, gabarito)
    _imprimir_autoteste("Respostas reais do gemma3:4b (pedido)", reais)
    _imprimir_autoteste("Casos extras de cobertura das regras (escritos à mão)", extras)
    todos = reais + extras
    print("\n── Casos em que a regra nova mudou o resultado em relação à antiga ──")
    mudou = [l for l in todos if l["v1"] != l["v2"]]
    for l in mudou:
        print(f"  {l['id']}: {l['v1']} → {l['v2']}  ({' '.join(l['resposta'].split())[:70]}…)")
    print(f"  total: {len(mudou)} de {len(todos)} (reais: {sum(1 for l in reais if l['v1'] != l['v2'])} de {len(reais)})")
    falhas = [l for l in todos if not l["ok"]]
    print(f"\n{'TODOS OS CASOS PASSARAM' if not falhas else f'{len(falhas)} CASO(S) DIVERGEM DO ESPERADO'} "
          f"({len(todos) - len(falhas)}/{len(todos)})")
    return 0 if not falhas else 1


def reavaliar(caminho: Path, gabarito: dict) -> int:
    if not caminho.exists():
        print(f"ERRO: arquivo não encontrado: {caminho}")
        return 1
    with caminho.open(newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if not linhas or not {"id", "modelo", "resposta", "resultado"} <= set(linhas[0]):
        print("ERRO: o CSV precisa das colunas id, modelo, resposta e resultado (formato de resultado_modelos.csv)")
        return 1
    saida, ignoradas = [], []
    for l in linhas:
        qid = l["id"]
        g = gabarito.get(qid)
        if g is None or not g["resposta_esperada"]:
            ignoradas.append(qid)
            continue
        v2 = avaliar(l["resposta"], g["resposta_esperada"], g["nao_deve_conter"], g["pergunta"], g["negativa"])
        saida.append({"id": qid, "grupo": g["grupo"], "pergunta": g["pergunta"], "modelo": l["modelo"],
                      "resposta": l["resposta"], "antigo": l["resultado"], "novo": v2["resultado"],
                      "observacao": v2["observacao"], "motivo": v2["motivo"]})
    base = caminho.with_name(caminho.stem + "_reavaliado")
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "grupo", "pergunta", "modelo", "resposta", "resultado_antigo", "resultado_novo",
                    "mudou", "observacao", "motivo"])
        for s in saida:
            w.writerow([s["id"], s["grupo"], s["pergunta"], s["modelo"], s["resposta"], s["antigo"], s["novo"],
                        int(s["antigo"] != s["novo"]), s["observacao"], s["motivo"]])
    modelos = sorted({s["modelo"] for s in saida})
    md = [f"# Reavaliação de {caminho.name}", "",
          "Respostas já gravadas, sem chamar modelo. *antigo* = resultado gravado no CSV; *novo* = regra v2.", "",
          "## Resumo por modelo e grupo", "", "| modelo | grupo | n | antigo (OK/REV/FALHOU) | novo (OK/REV/FALHOU) |",
          "|---|---|---|---|---|"]
    for m in modelos:
        for g in _GRUPOS:
            vs = [s for s in saida if s["modelo"] == m and s["grupo"] == g]
            if vs:
                a, n = _contagens(vs, "antigo"), _contagens(vs, "novo")
                md.append(f"| {m} | {g} | {len(vs)} | {a['OK']}/{a['REVISAR']}/{a['FALHOU']} | "
                          f"{n['OK']}/{n['REVISAR']}/{n['FALHOU']} |")
    md += ["", "## Lado a lado", "", "| id | grupo | modelo | antigo | novo | mudou | observação |", "|---|---|---|---|---|---|---|"]
    for s in saida:
        md.append(f"| {s['id']} | {s['grupo']} | {s['modelo']} | {s['antigo']} | {s['novo']} | "
                  f"{'**sim**' if s['antigo'] != s['novo'] else ''} | {_md_escape(s['observacao'])} |")
    if ignoradas:
        md += ["", f"Ignoradas (id fora do gabarito ou sem resposta esperada): {', '.join(sorted(set(ignoradas)))}"]
    base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    mudou = [s for s in saida if s["antigo"] != s["novo"]]
    print(f"[reavaliar] {len(saida)} respostas reavaliadas; {len(mudou)} mudaram de resultado")
    for s in mudou:
        print(f"  {s['id']} · {s['modelo']}: {s['antigo']} → {s['novo']}  ({s['motivo']})")
    print(f"[saída] {base.with_suffix('.csv')}\n[saída] {base.with_suffix('.md')}")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Compara modelos RAG usando o gabarito de PIPELINE.md"
    )
    parser.add_argument("--modelos", default=None,
                        help='Modelos separados por vírgula, ex: "qwen3:0.6b,qwen2.5:3b"')
    parser.add_argument("--ids", default=None,
                        help=f'IDs do gabarito, ex: "A01,B01,C01". Padrão: {",".join(_IDS_PADRAO)}')
    parser.add_argument("--todas", action="store_true",
                        help="roda todas as perguntas do gabarito que tenham resposta esperada (legado e novas)")
    parser.add_argument("--reavaliar", metavar="CAMINHO.csv", default=None,
                        help="OFFLINE: reavalia as respostas gravadas num resultado_modelos.csv (antigo × novo)")
    parser.add_argument("--autoteste", action="store_true",
                        help="OFFLINE: reavalia respostas reais embutidas no código e confere o resultado esperado")
    args = parser.parse_args()

    # Gabarito
    gabarito = carregar_gabarito()
    if not gabarito:
        print("ERRO: Não foi possível parsear o gabarito de PIPELINE.md")
        sys.exit(1)

    if args.autoteste:
        sys.exit(autoteste(gabarito))
    if args.reavaliar:
        sys.exit(reavaliar(Path(args.reavaliar), gabarito))

    ids = _selecionar_ids(gabarito, args)

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
        print(f"  {qid} [{gabarito[qid]['grupo']}]: {gabarito[qid]['pergunta'][:70]}")
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
            g = gabarito[qid]
            pergunta, esperada = g["pergunta"], g["resposta_esperada"]
            print(f"\n  [{qid}] {pergunta}")
            cell = _executar(pipeline, tabela_docs, cronograma_docs, pergunta, modelo_cfg)
            av = avaliar(cell["resposta"], esperada, g["nao_deve_conter"], pergunta, g["negativa"])
            print(f"  Resposta : {cell['resposta'][:120]}"
                  f"{'...' if len(cell['resposta']) > 120 else ''}")
            print(f"  Esperado : {esperada[:80]}")
            rep_info = f", rep={av['repeticoes']}" if av["repeticoes"] else ""
            print(f"  → {av['resultado']}  ({cell['tempo_total_s']:.1f}s, "
                  f"{cell['tokens_saida']} tokens{rep_info})")
            if av["observacao"]:
                print(f"  ⚠ {av['observacao']}")
            resultados[modelo_id][qid] = {
                "resposta": cell["resposta"],
                "resultado": av["resultado"],
                "tempo": cell["tempo_total_s"],
                "tokens": cell["tokens_saida"],
                "repeticoes": av["repeticoes"],
                "observacao": av["observacao"],
            }

    _imprimir_tabela(resultados, modelos_ok, ids, gabarito)
    _salvar_csv(resultados, modelos_ok, ids, gabarito)
    _salvar_md(resultados, modelos_ok, ids, gabarito)


if __name__ == "__main__":
    main()
