"""
analisar_resultados.py — Consolida os dados coletados para o artigo do ERBD

Lê os arquivos gerados por app.py (logs/) e avaliar.py (relatorio_ragas*.csv)
e produz:
  - Tabelas resumo em Markdown (prontas pra colar no artigo)
  - Gráficos comparativos (latência, throughput, RAGAS) em PNG
  - Um CSV único consolidado, caso queira levar pro Excel/SPSS

INSTALAÇÃO (uma vez):
    pip install pandas matplotlib

USO:
    # Analisa tudo que encontrar na pasta atual
    python analisar_resultados.py

    # Especificando arquivos
    python analisar_resultados.py --ragas relatorio_ragas.csv --ragas-agg relatorio_ragas_agg.csv --logs-dir logs/

SAÍDA (pasta ./analise/):
    tabela_desempenho.md       → latência/tokens por modelo (Markdown)
    tabela_qualidade.md        → métricas RAGAS por modelo (Markdown)
    grafico_latencia.png       → barras: tempo retrieval vs LLM por modelo
    grafico_throughput.png     → barras: tokens/s por modelo
    grafico_ragas.png          → radar/barras: métricas RAGAS por modelo
    dados_consolidados.csv     → tudo junto, uma linha por consulta
"""

import argparse
import json
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("./analise")
OUT_DIR.mkdir(exist_ok=True)

# Nomes de exibição, usados de forma consistente em todas as tabelas e
# gráficos do relatório/artigo. Os IDs brutos (llama3, gemma2:27b, etc.)
# só existem nos CSVs internos — o material final sempre usa estes nomes.
NOME_BONITO = {
    "llama3":                    "LLaMA 3",
    "llama3:latest":             "LLaMA 3",
    "mistral":                   "Mistral",
    "mistral:latest":            "Mistral",
    "gemma2:27b":                "Gemma 2 27B",
}


def nome_bonito(modelo_id: str) -> str:
    return NOME_BONITO.get(modelo_id, modelo_id)


# ── Carregamento ──────────────────────────────────────────────────────
def carregar_logs_streamlit(logs_dir: str) -> pd.DataFrame:
    """Lê todos os logs/log_*.jsonl gerados pelo app.py em uso real."""
    arquivos = sorted(glob.glob(f"{logs_dir}/log_*.jsonl"))
    if not arquivos:
        print(f"  (nenhum log encontrado em {logs_dir}/)")
        return pd.DataFrame()

    linhas = []
    for arq in arquivos:
        with open(arq, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if linha:
                    linhas.append(json.loads(linha))

    df = pd.DataFrame(linhas)
    print(f"  {len(df)} consultas carregadas de {len(arquivos)} arquivo(s) de log")
    return df


def carregar_ragas(caminho: str) -> pd.DataFrame:
    if not Path(caminho).exists():
        print(f"  (arquivo não encontrado: {caminho})")
        return pd.DataFrame()
    df = pd.read_csv(caminho)
    print(f"  {len(df)} avaliações carregadas de {caminho}")

    # Restringe à rodada oficial: só os 3 modelos locais do experimento.
    # Evita que resquícios de testes antigos (ex: llama-3.3-70b-versatile
    # via Groq, ou execuções de teste com --limite) contaminem as
    # tabelas e gráficos finais do artigo.
    modelos_oficiais = set(NOME_BONITO.keys())
    antes = len(df)
    df = df[df["modelo"].isin(modelos_oficiais)]
    if len(df) < antes:
        print(f"  ({antes - len(df)} linha(s) de modelos fora do experimento oficial foram descartadas)")

    return df


# ── Tabelas em Markdown ──────────────────────────────────────────────
def tabela_desempenho(df_logs: pd.DataFrame, df_ragas: pd.DataFrame) -> str:
    """
    Tabela de latência/throughput por modelo — prioriza o relatorio_ragas.csv
    (rodada oficial do avaliar.py com o Golden Dataset), já que é a fonte
    controlada e completa (mesmo número de perguntas por modelo). Cai para
    os logs do app.py apenas se a avaliação formal ainda não tiver sido
    rodada.
    """
    if not df_ragas.empty:
        g = df_ragas.groupby("modelo").agg(
            n_consultas=("modelo", "count"),
            tempo_retrieval_medio_s=("tempo_retrieval_s", "mean"),
            tempo_llm_medio_s=("tempo_llm_s", "mean"),
            tokens_saida_medio=("tokens_saida", "mean"),
        ).round(3).reset_index()
        g["tokens_por_s_medio"] = (g["tokens_saida_medio"] / g["tempo_llm_medio_s"]).round(2)
        g["modelo_nome"] = g["modelo"].apply(nome_bonito)

        md = "| Modelo | N | Retrieval (s) | LLM (s) | Tokens/s (fim a fim) |\n"
        md += "|---|---|---|---|---|\n"
        for _, r in g.sort_values("modelo_nome").iterrows():
            md += (
                f"| {r['modelo_nome']} | {int(r['n_consultas'])} | "
                f"{r['tempo_retrieval_medio_s']} | {r['tempo_llm_medio_s']} | "
                f"{r['tokens_por_s_medio']} |\n"
            )
        return md

    elif not df_logs.empty:
        g = df_logs.groupby("modelo_id").agg(
            n_consultas=("modelo_id", "count"),
            tempo_total_medio_s=("tempo_total_s", "mean"),
            tempo_retrieval_medio_s=("tempo_retrieval_s", "mean"),
            tempo_llm_medio_s=("tempo_llm_s", "mean"),
            tokens_saida_medio=("tokens_saida", "mean"),
            tokens_por_s_medio=("tokens_por_s", "mean"),
            taxa_nao_encontrado_pct=("nao_encontrado", lambda s: round(s.mean() * 100, 1)),
        ).round(3).reset_index()
        g["modelo_nome"] = g["modelo_id"].apply(nome_bonito)

        md = "| Modelo | N | Tempo total (s) | Retrieval (s) | LLM (s) | Tokens/s | % não encontrado |\n"
        md += "|---|---|---|---|---|---|---|\n"
        for _, r in g.sort_values("modelo_nome").iterrows():
            md += (
                f"| {r['modelo_nome']} | {int(r['n_consultas'])} | "
                f"{r.get('tempo_total_medio_s', '—')} | "
                f"{r['tempo_retrieval_medio_s']} | {r['tempo_llm_medio_s']} | "
                f"{r.get('tokens_por_s_medio', '—')} | "
                f"{r.get('taxa_nao_encontrado_pct', '—')} |\n"
            )
        return md

    return "*(sem dados de desempenho — rode o app.py ou avaliar.py primeiro)*\n"


def tabela_qualidade(df_ragas: pd.DataFrame) -> str:
    if df_ragas.empty:
        return "*(sem dados RAGAS — rode o avaliar.py com o Golden Dataset primeiro)*\n"

    cols_ragas = ["faithfulness", "answer_relevancy", "context_precision",
                  "context_recall", "answer_correctness"]
    cols_presentes = [c for c in cols_ragas if c in df_ragas.columns]

    g = df_ragas.groupby("modelo")[cols_presentes].mean().round(4).reset_index()
    g["modelo_nome"] = g["modelo"].apply(nome_bonito)

    md = "| Modelo | " + " | ".join(cols_presentes) + " |\n"
    md += "|---|" + "---|" * len(cols_presentes) + "\n"
    for _, r in g.sort_values("modelo_nome").iterrows():
        vals = " | ".join(
            "—" if pd.isna(r[c]) else str(r[c]) for c in cols_presentes
        )
        md += f"| {r['modelo_nome']} | {vals} |\n"
    return md


# ── Gráficos ─────────────────────────────────────────────────────────
def grafico_latencia(df_logs: pd.DataFrame, df_ragas: pd.DataFrame):
    if not df_ragas.empty:
        g = df_ragas.groupby("modelo")[["tempo_retrieval_s", "tempo_llm_s"]].mean()
    elif not df_logs.empty:
        g = df_logs.groupby("modelo_id")[["tempo_retrieval_s", "tempo_llm_s"]].mean()
    else:
        return

    g.index = [nome_bonito(m) for m in g.index]
    g = g.sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    g.plot(kind="bar", stacked=True, ax=ax, color=["#2C7A4D", "#E8A33D"])
    ax.set_ylabel("Tempo médio (s)")
    ax.set_xlabel("Modelo")
    ax.set_title("Latência média por etapa (retrieval vs. geração LLM)")
    ax.legend(["Retrieval", "LLM"])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grafico_latencia.png", dpi=150)
    plt.close()
    print("  ✅ grafico_latencia.png")


def grafico_throughput(df_logs: pd.DataFrame, df_ragas: pd.DataFrame):
    """
    Throughput (tokens/s) por modelo — prioriza o relatorio_ragas.csv
    (rodada oficial), calculando tokens_saida / tempo_llm_s por pergunta
    e tirando a média. Cai para os logs do app.py (que já trazem
    tokens_por_s pronto) apenas se a avaliação formal não existir.
    """
    if not df_ragas.empty:
        df = df_ragas.copy()
        df = df[df["tempo_llm_s"] > 0]
        df["tokens_por_s"] = df["tokens_saida"] / df["tempo_llm_s"]
        g = df.groupby("modelo")["tokens_por_s"].mean()
    elif not df_logs.empty and "tokens_por_s" in df_logs.columns:
        g = df_logs.groupby("modelo_id")["tokens_por_s"].mean()
    else:
        print("  (grafico_throughput.png pulado — sem dados de tokens/s)")
        return

    g.index = [nome_bonito(m) for m in g.index]
    g = g.sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    g.plot(kind="barh", ax=ax, color="#1B5E3F")
    ax.set_xlabel("Tokens gerados por segundo (fim a fim: prefill + geração)")
    ax.set_ylabel("Modelo")
    ax.set_title("Throughput médio por modelo (tempo total da chamada)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grafico_throughput.png", dpi=150)
    plt.close()
    print("  ✅ grafico_throughput.png")


def grafico_ragas(df_ragas: pd.DataFrame):
    cols_ragas = ["faithfulness", "answer_relevancy", "context_precision",
                  "context_recall", "answer_correctness"]
    cols_presentes = [c for c in cols_ragas if c in df_ragas.columns and df_ragas[c].notna().any()]

    if df_ragas.empty or not cols_presentes:
        return

    g = df_ragas.groupby("modelo")[cols_presentes].mean()
    g.index = [nome_bonito(m) for m in g.index]
    g = g.sort_index()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    g.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score (0–1)")
    ax.set_xlabel("Modelo")
    ax.set_title("Métricas RAGAS por modelo")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grafico_ragas.png", dpi=150)
    plt.close()
    print("  ✅ grafico_ragas.png")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas",     default="relatorio_ragas.csv")
    parser.add_argument("--ragas-agg", default="relatorio_ragas_agg.csv")
    parser.add_argument("--logs-dir",  default="logs")
    args = parser.parse_args()

    print("📥 Carregando dados...")
    df_logs  = carregar_logs_streamlit(args.logs_dir)
    df_ragas = carregar_ragas(args.ragas)

    print("\n📊 Gerando tabelas...")
    md_desempenho = tabela_desempenho(df_logs, df_ragas)
    md_qualidade  = tabela_qualidade(df_ragas)

    (OUT_DIR / "tabela_desempenho.md").write_text(
        "# Tabela de Desempenho por Modelo\n\n" + md_desempenho, encoding="utf-8"
    )
    (OUT_DIR / "tabela_qualidade.md").write_text(
        "# Tabela de Qualidade (RAGAS) por Modelo\n\n" + md_qualidade, encoding="utf-8"
    )
    print("  ✅ tabela_desempenho.md")
    print("  ✅ tabela_qualidade.md")

    print("\n📈 Gerando gráficos...")
    grafico_latencia(df_logs, df_ragas)
    grafico_throughput(df_logs, df_ragas)
    grafico_ragas(df_ragas)

    print("\n💾 Consolidando CSV único...")
    if not df_ragas.empty:
        df_ragas.to_csv(OUT_DIR / "dados_consolidados.csv", index=False)
        print(f"  ✅ dados_consolidados.csv ({len(df_ragas)} linhas, de avaliar.py — rodada oficial)")
    elif not df_logs.empty:
        df_logs.to_csv(OUT_DIR / "dados_consolidados.csv", index=False)
        print(f"  ✅ dados_consolidados.csv ({len(df_logs)} linhas, de logs de uso real)")

    print(f"\n{'='*55}")
    print(f"✅ Análise concluída! Resultados em: {OUT_DIR}/")
    print(f"{'='*55}")
    print("\nPrévia da tabela de desempenho:\n")
    print(md_desempenho)
    if not df_ragas.empty:
        print("\nPrévia da tabela de qualidade:\n")
        print(md_qualidade)


if __name__ == "__main__":
    main()
