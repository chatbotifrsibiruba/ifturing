"""
fix_agg_csv.py — Recalcula um relatorio_ragas*_agg.csv a partir do
relatorio_ragas*.csv correspondente, ignorando valores NaN nas médias
(em vez de deixar a média inteira virar NaN por causa de 1-2 falhas
pontuais do juiz RAGAS/Maritaca em perguntas específicas).

Uso:
    # Rodada 1 (padrão)
    python fix_agg_csv.py

    # Rodada 2
    python fix_agg_csv.py --dataset relatorio_ragas_2.csv --saida relatorio_ragas_2_agg.csv

    # Rodada 3
    python fix_agg_csv.py --dataset relatorio_ragas_3.csv --saida relatorio_ragas_3_agg.csv
"""

import csv
import math
import argparse
from pathlib import Path

CAMPOS_METRICA = [
    "tempo_retrieval_s",
    "tempo_llm_s",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def to_float_or_none(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except ValueError:
        return None


def media_segura(valores):
    limpos = [v for v in valores if v is not None]
    if not limpos:
        return None
    return round(sum(limpos) / len(limpos), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="relatorio_ragas.csv",
                         help="Arquivo bruto de entrada (uma linha por pergunta)")
    parser.add_argument("--saida", default="relatorio_ragas_agg.csv",
                         help="Arquivo agregado de saída (uma linha por modelo)")
    args = parser.parse_args()

    bruto = Path(args.dataset)
    agg = Path(args.saida)

    if not bruto.exists():
        print(f"❌ Arquivo não encontrado: {bruto}")
        return

    with open(bruto, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    por_branch = {}
    for linha in linhas:
        chave = (linha["branch"], linha["modelo"])
        por_branch.setdefault(chave, []).append(linha)

    novas_linhas = []
    for (branch, modelo), entradas in por_branch.items():
        n = len(entradas)
        linha_agg = {
            "branch": branch,
            "modelo": modelo,
            "n_perguntas": n,
        }

        for campo in CAMPOS_METRICA:
            valores = [to_float_or_none(e.get(campo)) for e in entradas]
            n_validos = sum(1 for v in valores if v is not None)
            n_nan = n - n_validos
            if n_nan > 0:
                print(f"  ⚠️  {branch}/{modelo}: {n_nan} valor(es) NaN/ausente em '{campo}' — ignorado(s) na média")
            chave_campo = f"{campo.replace('_s', '')}_medio" if campo in ("tempo_retrieval_s", "tempo_llm_s") else f"{campo}_medio"
            linha_agg[chave_campo] = media_segura(valores)

        novas_linhas.append(linha_agg)

    campos_finais = [
        "branch", "modelo", "n_perguntas",
        "tempo_retrieval_medio_s", "tempo_llm_medio_s",
        "faithfulness_medio", "answer_relevancy_medio",
        "context_precision_medio", "context_recall_medio",
        "answer_correctness_medio",
    ]

    linhas_saida = []
    for linha in novas_linhas:
        linhas_saida.append({
            "branch": linha["branch"],
            "modelo": linha["modelo"],
            "n_perguntas": linha["n_perguntas"],
            "tempo_retrieval_medio_s": linha.get("tempo_retrieval_medio"),
            "tempo_llm_medio_s": linha.get("tempo_llm_medio"),
            "faithfulness_medio": linha.get("faithfulness_medio"),
            "answer_relevancy_medio": linha.get("answer_relevancy_medio"),
            "context_precision_medio": linha.get("context_precision_medio"),
            "context_recall_medio": linha.get("context_recall_medio"),
            "answer_correctness_medio": linha.get("answer_correctness_medio"),
        })

    with open(agg, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos_finais)
        writer.writeheader()
        writer.writerows(linhas_saida)

    print(f"\n✅ {agg} recalculado com {len(linhas_saida)} linha(s), ignorando NaN nas médias.")
    for linha in linhas_saida:
        print(f"   {linha['branch']}/{linha['modelo']}: answer_correctness_medio = {linha['answer_correctness_medio']}")


if __name__ == "__main__":
    main()