"""
migrar_csv_para_db.py — Importa os relatorio_ragas*.csv já existentes
(rodadas 1, 2, 3...) para o banco SQLite (ifturing.db), preservando o
histórico já coletado em vez de perdê-lo na migração pro banco.

Uso:
    python migrar_csv_para_db.py
    python migrar_csv_para_db.py --arquivos relatorio_ragas.csv relatorio_ragas_2.csv relatorio_ragas_3.csv
"""

import csv
import math
import argparse
from pathlib import Path

from db import init_db, upsert_modelo, get_or_create_sessao, inserir_consulta

NOME_BONITO = {
    "llama3":         "LLaMA 3",
    "llama3:latest":  "LLaMA 3",
    "mistral":        "Mistral",
    "mistral:latest": "Mistral",
    "gemma2:27b":     "Gemma 2 27B",
}


def to_float(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except ValueError:
        return None


def to_int(v):
    f = to_float(v)
    return int(f) if f is not None else None


def migrar_arquivo(caminho: str, numero_rodada: int):
    p = Path(caminho)
    if not p.exists():
        print(f"  (arquivo não encontrado, pulando: {caminho})")
        return 0

    with open(p, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    if not linhas:
        print(f"  (arquivo vazio: {caminho})")
        return 0

    sessao_uid = f"avaliar_rodada_{numero_rodada}"
    branch = linhas[0].get("branch", f"rodada_{numero_rodada}")
    sessao_id = get_or_create_sessao(sessao_uid, origem="avaliar", branch=branch)

    modelos_vistos = set()
    n = 0
    for linha in linhas:
        modelo_id = linha["modelo"]
        if modelo_id not in modelos_vistos:
            upsert_modelo(modelo_id, nome_bonito=NOME_BONITO.get(modelo_id, modelo_id))
            modelos_vistos.add(modelo_id)

        ragas = {
            "faithfulness":       to_float(linha.get("faithfulness")),
            "answer_relevancy":   to_float(linha.get("answer_relevancy")),
            "context_precision":  to_float(linha.get("context_precision")),
            "context_recall":     to_float(linha.get("context_recall")),
            "answer_correctness": to_float(linha.get("answer_correctness")),
        }
        # só grava avaliação RAGAS se tiver pelo menos um valor
        ragas_valido = ragas if any(v is not None for v in ragas.values()) else None

        inserir_consulta(
            sessao_id=sessao_id,
            modelo_id=modelo_id,
            pergunta=linha.get("pergunta", ""),
            resposta_gerada=linha.get("resposta_gerada", ""),
            resposta_esperada=linha.get("resposta_esperada"),
            tempo_retrieval_s=to_float(linha.get("tempo_retrieval_s")),
            tempo_llm_s=to_float(linha.get("tempo_llm_s")),
            tokens_entrada=to_int(linha.get("tokens_entrada")),
            tokens_saida=to_int(linha.get("tokens_saida")),
            ragas=ragas_valido,
        )
        n += 1

    print(f"  ✅ {caminho} → {n} consultas migradas (sessão: {sessao_uid})")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arquivos", nargs="+",
        default=["relatorio_ragas.csv", "relatorio_ragas_2.csv", "relatorio_ragas_3.csv"],
        help="Lista de CSVs a migrar, em ordem de rodada",
    )
    args = parser.parse_args()

    print("🗄️  Inicializando banco...")
    init_db()

    print("\n📥 Migrando CSVs...")
    total = 0
    for i, arq in enumerate(args.arquivos, 1):
        total += migrar_arquivo(arq, i)

    print(f"\n✅ Migração concluída: {total} consultas no total, em {len(args.arquivos)} rodada(s).")
    print("   Banco salvo em: ifturing.db")


if __name__ == "__main__":
    main()
