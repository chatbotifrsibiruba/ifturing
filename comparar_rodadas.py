"""
comparar_rodadas.py — Consolida múltiplas rodadas de avaliação (variance
analysis) e gera uma tabela Markdown + gráfico com barras de erro, prontos
para enviar direto pra equipe (ex: Laura) ou colar no artigo.

Lê os arquivos relatorio_ragas_agg.csv, relatorio_ragas_2_agg.csv,
relatorio_ragas_3_agg.csv (ou quantos você tiver) e calcula média e
desvio padrão de cada métrica entre as rodadas, por modelo.

INSTALAÇÃO (uma vez):
    pip install pandas matplotlib

USO:
    # Usa os arquivos padrão: relatorio_ragas_agg.csv, _2_agg.csv, _3_agg.csv
    python comparar_rodadas.py

    # Especificando arquivos manualmente (qualquer quantidade)
    python comparar_rodadas.py --arquivos relatorio_ragas_agg.csv relatorio_ragas_2_agg.csv relatorio_ragas_3_agg.csv

SAÍDA (pasta ./resultados/):
    tabela_rodadas.md          → tabela Markdown com média ± desvio padrão por modelo/métrica
    grafico_rodadas_tempo.png  → barras: tempo médio de LLM por modelo, com barra de erro
    grafico_rodadas_ragas.png  → barras agrupadas: métricas RAGAS por modelo, com barra de erro
"""

import argparse
import statistics
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("./resultados")
OUT_DIR.mkdir(exist_ok=True)

NOME_BONITO = {
    "llama3":         "LLaMA 3",
    "llama3:latest":  "LLaMA 3",
    "mistral":        "Mistral",
    "mistral:latest": "Mistral",
    "gemma2:27b":     "Gemma 2 27B",
}


def nome_bonito(modelo_id: str) -> str:
    return NOME_BONITO.get(modelo_id, modelo_id)


METRICAS = {
    "tempo_llm_medio_s":         "LLM (s)",
    "faithfulness_medio":        "faithfulness",
    "answer_relevancy_medio":    "answer_relevancy",
    "context_precision_medio":   "context_precision",
    "context_recall_medio":      "context_recall",
    "answer_correctness_medio":  "answer_correctness",
}


def carregar_rodadas(arquivos: list[str]) -> pd.DataFrame:
    """Lê cada CSV agregado e empilha tudo, marcando de qual rodada veio."""
    dfs = []
    for i, arq in enumerate(arquivos, 1):
        p = Path(arq)
        if not p.exists():
            print(f"  (arquivo não encontrado, pulando: {arq})")
            continue
        df = pd.read_csv(p)
        df["rodada"] = i
        dfs.append(df)
        print(f"  rodada {i}: {arq} — {len(df)} modelo(s)")

    if not dfs:
        raise SystemExit("❌ Nenhum arquivo de rodada encontrado. Confira os caminhos com --arquivos.")

    return pd.concat(dfs, ignore_index=True)


def consolidar(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula média e desvio padrão de cada métrica, por modelo, entre rodadas."""
    linhas = []
    for modelo, grupo in df.groupby("modelo"):
        linha = {"modelo": modelo, "modelo_nome": nome_bonito(modelo), "n_rodadas": len(grupo)}
        for col, label in METRICAS.items():
            valores = grupo[col].dropna().tolist()
            if valores:
                media = round(statistics.mean(valores), 4)
                desvio = round(statistics.stdev(valores), 4) if len(valores) > 1 else 0.0
            else:
                media, desvio = None, None
            linha[f"{col}_media"] = media
            linha[f"{col}_desvio"] = desvio
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("modelo_nome")


def gerar_tabela_md(consolidado: pd.DataFrame, n_rodadas_total: int) -> str:
    md = f"# Comparação entre {n_rodadas_total} rodadas de avaliação\n\n"
    md += "Cada valor é **média ± desvio padrão** entre as rodadas (100 perguntas do Golden Dataset por rodada).\n\n"

    colunas = ["Modelo"] + list(METRICAS.values())
    md += "| " + " | ".join(colunas) + " |\n"
    md += "|" + "---|" * len(colunas) + "\n"

    for _, r in consolidado.iterrows():
        celulas = [r["modelo_nome"]]
        for col in METRICAS:
            media = r[f"{col}_media"]
            desvio = r[f"{col}_desvio"]
            if media is None:
                celulas.append("—")
            else:
                celulas.append(f"{media} ± {desvio}")
        md += "| " + " | ".join(celulas) + " |\n"

    return md


def grafico_tempo(consolidado: pd.DataFrame):
    modelos = consolidado["modelo_nome"].tolist()
    medias = consolidado["tempo_llm_medio_s_media"].tolist()
    desvios = consolidado["tempo_llm_medio_s_desvio"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(modelos, medias, yerr=desvios, capsize=6, color="#1B5E3F")
    ax.set_ylabel("Tempo médio de LLM (s)")
    ax.set_title("Latência média entre rodadas (barra de erro = desvio padrão)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grafico_rodadas_tempo.png", dpi=150)
    plt.close()
    print("  ✅ grafico_rodadas_tempo.png")


def grafico_ragas(consolidado: pd.DataFrame):
    cols = ["faithfulness_medio", "answer_relevancy_medio", "context_precision_medio",
            "context_recall_medio", "answer_correctness_medio"]
    labels = [METRICAS[c] for c in cols]

    modelos = consolidado["modelo_nome"].tolist()
    x = range(len(modelos))
    largura = 0.15

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, col in enumerate(cols):
        medias = consolidado[f"{col}_media"].tolist()
        desvios = consolidado[f"{col}_desvio"].tolist()
        posicoes = [xi + i * largura for xi in x]
        ax.bar(posicoes, medias, largura, yerr=desvios, capsize=3, label=labels[i])

    ax.set_xticks([xi + largura * 2 for xi in x])
    ax.set_xticklabels(modelos)
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1)
    ax.set_title("Métricas RAGAS entre rodadas (barra de erro = desvio padrão)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grafico_rodadas_ragas.png", dpi=150)
    plt.close()
    print("  ✅ grafico_rodadas_ragas.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arquivos", nargs="+",
        default=["relatorio_ragas_agg.csv", "relatorio_ragas_2_agg.csv", "relatorio_ragas_3_agg.csv"],
        help="Lista de arquivos *_agg.csv, um por rodada",
    )
    args = parser.parse_args()

    print("📥 Carregando rodadas...")
    df = carregar_rodadas(args.arquivos)
    n_rodadas = df["rodada"].nunique()

    print("\n📊 Consolidando (média ± desvio padrão)...")
    consolidado = consolidar(df)

    tabela_md = gerar_tabela_md(consolidado, n_rodadas)
    (OUT_DIR / "tabela_rodadas.md").write_text(tabela_md, encoding="utf-8")
    print("  ✅ tabela_rodadas.md")

    print("\n📈 Gerando gráficos...")
    grafico_tempo(consolidado)
    grafico_ragas(consolidado)

    print(f"\n{'='*55}")
    print(f"✅ Comparação de {n_rodadas} rodadas concluída! Resultados em: {OUT_DIR}/")
    print(f"{'='*55}\n")
    print(tabela_md)


if __name__ == "__main__":
    main()