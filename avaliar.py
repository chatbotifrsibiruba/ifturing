"""
avaliar.py — Avaliação automática com Golden Dataset + RAGAS
Para o artigo do ERBD.

Lê uma planilha CSV com perguntas e respostas esperadas (Golden Dataset,
elaborado pelo Vitório e Lorenzo), roda cada pergunta contra o RAG do
IF Turing, e calcula métricas RAGAS comparando a resposta gerada com a
resposta esperada.

O juiz RAGAS (LLM que avalia faithfulness, relevancy, etc.) usa a
Maritaca AI (API compatível com OpenAI), evitando dependência da OpenAI.
Os modelos avaliados continuam rodando 100% localmente via Ollama —
a Maritaca é usada apenas na etapa de avaliação/julgamento, não no
chatbot em produção.

────────────────────────────────────────────────────────────────────
FORMATO ESPERADO DO GOLDEN DATASET (CSV ou XLSX, exportado do Drive):

    pergunta,resposta_esperada
    "Quais são as datas do processo seletivo?","As inscrições vão de..."
    "Existe taxa de inscrição?","Sim, o valor é..."
────────────────────────────────────────────────────────────────────

INSTALAÇÃO (uma vez):
    pip install ragas datasets pandas openpyxl langchain-openai

CONFIGURAÇÃO (.env):
    MARITACA_API_KEY=sua_chave_aqui
    MARITACA_BASE_URL=https://chat.maritaca.ai/api
    MARITACA_MODEL=sabia-3

USO:
    python avaliar.py --dataset golden_dataset.csv --branch llama3 --modelo llama3
    python avaliar.py --dataset golden_dataset.csv --branch mistral --modelo mistral
    python avaliar.py --dataset golden_dataset.csv --branch gemma --modelo gemma2:27b

    # Limitar a N perguntas para um teste rápido
    python avaliar.py --dataset golden_dataset.csv --branch llama3 --modelo llama3 --limite 10

Gera:
    relatorio_ragas.csv       → uma linha por pergunta, todas as branches juntas
    relatorio_ragas_agg.csv   → uma linha por branch/modelo, pronto pra tabela do artigo
"""

import os
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from db import init_db, upsert_modelo, get_or_create_sessao, inserir_consulta

# Importa a lógica de retrieval de produção para garantir paridade com o chatbot.
# top_k, filtro de cotas, hybrid retrieval de tabelas e cronograma, e o prompt
# são os mesmos usados em nucleo.responder(). Resultados RAGAS assim refletem
# o pipeline real, não uma configuração de avaliação separada.
from nucleo import (
    PASTA_CHROMA,
    MODELO_EMB,
    TOP_K,
    _PADRAO_COTA,
    _PADRAO_DATA_RECENTE,
    _buscar_tabelas_por_keyword,
    _buscar_cronograma_por_keyword,
    perguntar_llm,
    _OllamaOffline,
    _MaritacaError,
)

load_dotenv()

MARITACA_API_KEY   = os.getenv("MARITACA_API_KEY", "")
MARITACA_BASE_URL  = os.getenv("MARITACA_BASE_URL", "https://chat.maritaca.ai/api")
MARITACA_MODEL     = os.getenv("MARITACA_MODEL", "sabia-3")


# ── Carrega o Golden Dataset ─────────────────────────────────────────
def carregar_dataset(caminho: str, limite: int = None) -> list:
    caminho = Path(caminho)
    linhas = []

    if caminho.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(caminho)
        for _, row in df.iterrows():
            linhas.append({
                "pergunta": str(row["pergunta"]).strip(),
                "resposta_esperada": str(row["resposta_esperada"]).strip(),
            })
    else:
        with open(caminho, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                linhas.append({
                    "pergunta": row["pergunta"].strip(),
                    "resposta_esperada": row["resposta_esperada"].strip(),
                })

    if limite:
        linhas = linhas[:limite]

    return linhas


# ── Pipeline de retrieval (sem Streamlit — não usa st.cache_resource) ─
def carregar_pipeline():
    """Carrega o ChromaDocumentStore e pré-carrega chunks para hybrid retrieval.

    Espelha o setup de nucleo.carregar_pipeline() sem a camada de cache do
    Streamlit, para que avaliar.py possa rodar como script standalone.
    """
    from haystack import Pipeline
    from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
    from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=PASTA_CHROMA)
    n_chunks = document_store.count_documents()
    print(f"[setup] {n_chunks} chunks no índice ChromaDB")

    todos = document_store.filter_documents()
    tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
    cronograma_docs = [
        d for d in todos
        if (d.meta or {}).get("tipo") != "tabela"
        and (
            "cronograma" in (d.content or "").lower()
            or len(_PADRAO_DATA_RECENTE.findall(d.content or "")) >= 3
        )
    ]
    print(f"[setup] {len(tabela_docs)} chunks de tabela pré-carregados")
    print(f"[setup] {len(cronograma_docs)} chunks de cronograma pré-carregados")

    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component(
        "retriever",
        ChromaEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
    )
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    return pipeline, tabela_docs, cronograma_docs


# ── Retrieval idêntico ao pipeline de produção ───────────────────────
def recuperar_contexto(pipeline, tabela_docs: list, cronograma_docs: list, pergunta: str) -> tuple[str, list]:
    """Executa as mesmas etapas de retrieval de nucleo.responder().

    Retorna (contexto, docs) onde contexto é a string passada ao LLM e
    docs é a lista de Document para métricas de retrieval.
    """
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]

    # Filtro de cotas (igual a nucleo.responder)
    docs = [
        d for d in docs
        if (d.meta or {}).get("tipo") == "tabela" or not _PADRAO_COTA.search(d.content or "")
    ]

    # Hybrid retrieval: tabelas
    docs_tabela_extra = _buscar_tabelas_por_keyword(pergunta, tabela_docs, docs)
    if docs_tabela_extra:
        docs_tabela_ja   = [d for d in docs if (d.meta or {}).get("tipo") == "tabela"]
        docs_texto_final = [d for d in docs if (d.meta or {}).get("tipo") != "tabela"][:6]
        docs = docs_tabela_ja + docs_tabela_extra + docs_texto_final

    # Hybrid retrieval: cronograma
    docs_crono_extra = _buscar_cronograma_por_keyword(pergunta, cronograma_docs, docs)
    if docs_crono_extra:
        docs = docs_crono_extra + docs

    if not docs:
        return "", []

    contexto = "\n\n---\n\n".join(d.content for d in docs)
    return contexto, docs


# ── Responder usando o pipeline de produção ──────────────────────────
def responder(pipeline, tabela_docs: list, cronograma_docs: list, pergunta: str, modelo_cfg: dict) -> dict:
    t0 = time.perf_counter()
    contexto, docs = recuperar_contexto(pipeline, tabela_docs, cronograma_docs, pergunta)
    t_retrieval = time.perf_counter() - t0

    if not docs:
        return {
            "resposta": "Não encontrei essa informação nos documentos.",
            "contexto": "",
            "tempo_retrieval_s": round(t_retrieval, 3),
            "tempo_llm_s": 0.0,
            "tokens_entrada": 0,
            "tokens_saida": 0,
        }

    t1 = time.perf_counter()
    try:
        r = perguntar_llm(pergunta, contexto, modelo_cfg)
    except (_OllamaOffline, _MaritacaError) as exc:
        return {
            "resposta": f"[ERRO: {exc}]",
            "contexto": contexto,
            "tempo_retrieval_s": round(t_retrieval, 3),
            "tempo_llm_s": round(time.perf_counter() - t1, 3),
            "tokens_entrada": 0,
            "tokens_saida": 0,
        }

    return {
        "resposta": r["texto"],
        "contexto": contexto,
        "tempo_retrieval_s": round(t_retrieval, 3),
        "tempo_llm_s": r.get("tempo_geracao_s", round(time.perf_counter() - t1, 3)),
        "tokens_entrada": r["tokens_entrada"],
        "tokens_saida": r["tokens_saida"],
    }


# ── Juiz RAGAS via Maritaca ──────────────────────────────────────────
def montar_juiz_maritaca():
    """
    Monta o LLM-juiz do RAGAS usando a Maritaca AI (API compatível com
    OpenAI). Levanta um erro claro se a chave não estiver configurada,
    em vez de deixar o RAGAS falhar tentando usar a OpenAI por padrão.
    """
    if not MARITACA_API_KEY:
        raise RuntimeError(
            "MARITACA_API_KEY não configurada no .env. "
            "Defina MARITACA_API_KEY, MARITACA_BASE_URL e MARITACA_MODEL "
            "antes de rodar a avaliação com RAGAS."
        )

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=MARITACA_MODEL,
        api_key=MARITACA_API_KEY,
        base_url=MARITACA_BASE_URL,
        temperature=0.0,
        n=1,  # a API da Maritaca não aceita n>1 (múltiplas gerações por chamada)
    )
    return LangchainLLMWrapper(chat)


def montar_embeddings_juiz():
    """
    Embeddings usados pelo RAGAS para métricas de similaridade semântica.
    Usa o mesmo modelo de embeddings do pipeline de retrieval
    (intfloat/multilingual-e5-base) via HuggingFace, evitando também
    aqui a dependência da OpenAI.

    NOTA METODOLÓGICA (mencionar no artigo, seção de limitações):
    usar o mesmo modelo de embeddings do retrieval como "juiz" de
    avaliação pode introduzir viés circular. Se o tempo permitir, vale
    testar com um segundo modelo de embeddings independente (ex.:
    nomic-embed-text, já disponível no Ollama do servidor) para
    validar a robustez dos resultados.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf_embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMB)
    return LangchainEmbeddingsWrapper(hf_embeddings)


# ── RAGAS ────────────────────────────────────────────────────────────
def avaliar_com_ragas(perguntas, respostas, contextos, resp_esperadas) -> dict:
    """
    Calcula métricas RAGAS: faithfulness, answer_relevancy, context_precision,
    context_recall, answer_correctness — usando a Maritaca como juiz.

    Se o pacote ragas ou langchain-openai não estiverem instalados, ou se
    a MARITACA_API_KEY não estiver configurada, cai no fallback de
    similaridade lexical (Jaccard), avisando claramente no console.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )

        juiz_llm = montar_juiz_maritaca()
        juiz_embeddings = montar_embeddings_juiz()

        # A API da Maritaca não aceita n>1 (múltiplas gerações numa única
        # chamada): "Sorry, we currently don't support making more than
        # one request at a time. Please change the value of 'n' to 1."
        # Métricas como faithfulness e answer_relevancy usam múltiplas
        # gerações por padrão (via `strictness`) para checagem de
        # consistência interna. Forçamos strictness=1 em todas as
        # métricas que tiverem esse atributo, para ficar compatível.
        metricas = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
        for m in metricas:
            if hasattr(m, "strictness"):
                m.strictness = 1

        dataset = Dataset.from_dict({
            "question":       perguntas,
            "answer":         respostas,
            "contexts":       [[c] for c in contextos],
            "ground_truth":   resp_esperadas,
        })

        print(f"   🧑‍⚖️  Juiz RAGAS: Maritaca ({MARITACA_MODEL})")
        print(f"   🧑‍⚖️  Embeddings do juiz: {MODELO_EMB} (mesmo do retrieval — ver nota no código)")

        # Timeout maior e menos chamadas em paralelo — a API da Maritaca
        # é mais lenta que a OpenAI e derruba jobs por TimeoutError com
        # a configuração padrão do RAGAS, principalmente na métrica
        # answer_correctness (que faz mais chamadas por pergunta).
        run_config = RunConfig(
            timeout=180,       # segundos por chamada individual (padrão é 60s)
            max_retries=3,     # tenta de novo antes de desistir do job
            max_wait=60,       # espera máxima entre retries
            max_workers=2,     # menos chamadas simultâneas para a API da Maritaca
        )

        resultado = evaluate(
            dataset,
            metrics=metricas,
            llm=juiz_llm,
            embeddings=juiz_embeddings,
            run_config=run_config,
        )

        df = resultado.to_pandas()
        return {
            "faithfulness":       df["faithfulness"].tolist(),
            "answer_relevancy":   df["answer_relevancy"].tolist(),
            "context_precision":  df["context_precision"].tolist(),
            "context_recall":     df["context_recall"].tolist(),
            "answer_correctness": df["answer_correctness"].tolist(),
        }

    except (ImportError, RuntimeError) as e:
        print(f"\n⚠️  RAGAS com juiz Maritaca indisponível ({e}).")
        print("   Usando fallback de similaridade lexical (Jaccard) apenas para answer_correctness.\n")

        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        scores = [round(jaccard(r, g), 4) for r, g in zip(respostas, resp_esperadas)]
        return {
            "faithfulness":       [None] * len(respostas),
            "answer_relevancy":   [None] * len(respostas),
            "context_precision":  [None] * len(respostas),
            "context_recall":     [None] * len(respostas),
            "answer_correctness": scores,  # aproximação via Jaccard
        }


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV ou XLSX com pergunta,resposta_esperada")
    parser.add_argument("--branch",  required=True, help="Nome da branch/config (ex: llama3, mistral, gemma)")
    parser.add_argument("--modelo",  required=True, help="ID do modelo Ollama (ex: llama3, mistral, gemma2:27b)")
    parser.add_argument("--limite",  type=int, default=None, help="Limitar número de perguntas (teste rápido)")
    parser.add_argument("--saida",   default="relatorio_ragas.csv")
    args = parser.parse_args()

    modelo_cfg = {"tipo": "ollama", "modelo": args.modelo}

    print(f"\n🌿 Branch/config: {args.branch}")
    print(f"🤖 Modelo avaliado: {args.modelo} (ollama)")

    init_db()
    upsert_modelo(modelo_id=args.modelo)
    sessao_uid = f"avaliar_{args.branch}_{datetime.now().isoformat()}"
    sessao_id = get_or_create_sessao(
        sessao_uid=sessao_uid,
        origem="avaliar",
        branch=args.branch,
    )

    dados = carregar_dataset(args.dataset, args.limite)
    print(f"📋 {len(dados)} perguntas carregadas do Golden Dataset\n")

    pipeline, tabela_docs, cronograma_docs = carregar_pipeline()

    perguntas, respostas, contextos, resp_esperadas = [], [], [], []
    linhas_brutas = []

    for i, item in enumerate(dados, 1):
        pergunta = item["pergunta"]
        esperada = item["resposta_esperada"]
        print(f"  ⏳ [{i}/{len(dados)}] {pergunta[:55]}...")

        try:
            r = responder(pipeline, tabela_docs, cronograma_docs, pergunta, modelo_cfg)
        except Exception as e:
            print(f"     ❌ erro: {e}")
            r = {"resposta": "", "contexto": "", "tempo_retrieval_s": 0, "tempo_llm_s": 0,
                 "tokens_entrada": 0, "tokens_saida": 0}

        perguntas.append(pergunta)
        respostas.append(r["resposta"])
        contextos.append(r["contexto"])
        resp_esperadas.append(esperada)

        linhas_brutas.append({
            "branch": args.branch,
            "modelo": args.modelo,
            "pergunta": pergunta,
            "resposta_esperada": esperada,
            "resposta_gerada": r["resposta"][:300].replace("\n", " "),
            "tempo_retrieval_s": r["tempo_retrieval_s"],
            "tempo_llm_s": r["tempo_llm_s"],
            "tokens_entrada": r["tokens_entrada"],
            "tokens_saida": r["tokens_saida"],
        })

        time.sleep(1)  # evitar sobrecarregar o Ollama local

    print("\n🧮 Calculando métricas RAGAS...")
    ragas_scores = avaliar_com_ragas(perguntas, respostas, contextos, resp_esperadas)

    for i, linha in enumerate(linhas_brutas):
        linha["faithfulness"]       = ragas_scores["faithfulness"][i]
        linha["answer_relevancy"]   = ragas_scores["answer_relevancy"][i]
        linha["context_precision"]  = ragas_scores["context_precision"][i]
        linha["context_recall"]     = ragas_scores["context_recall"][i]
        linha["answer_correctness"] = ragas_scores["answer_correctness"][i]

    print("\n💾 Gravando resultados no banco...")
    for i, linha in enumerate(linhas_brutas):
        ragas_dict = {
            "faithfulness":       linha.get("faithfulness"),
            "answer_relevancy":   linha.get("answer_relevancy"),
            "context_precision":  linha.get("context_precision"),
            "context_recall":     linha.get("context_recall"),
            "answer_correctness": linha.get("answer_correctness"),
        }
        try:
            inserir_consulta(
                sessao_id=sessao_id,
                modelo_id=args.modelo,
                pergunta=linha["pergunta"],
                resposta_gerada=respostas[i],
                resposta_esperada=linha["resposta_esperada"],
                tempo_retrieval_s=linha["tempo_retrieval_s"],
                tempo_llm_s=linha["tempo_llm_s"],
                tokens_entrada=linha["tokens_entrada"],
                tokens_saida=linha["tokens_saida"],
                ragas=ragas_dict,
            )
        except Exception as e:
            print(f"[db] erro ao gravar consulta {i + 1}: {e}")

    # ── Salva CSV bruto (append entre execuções de branches diferentes) ──
    saida = Path(args.saida)
    ja_existe = saida.exists()
    with open(saida, "a", newline="", encoding="utf-8") as f:
        campos = list(linhas_brutas[0].keys())
        writer = csv.DictWriter(f, fieldnames=campos)
        if not ja_existe:
            writer.writeheader()
        writer.writerows(linhas_brutas)

    # ── Resumo agregado desta execução ──
    def media(campo):
        vals = [l[campo] for l in linhas_brutas if l[campo] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    print(f"\n{'='*55}")
    print(f"📊 RESULTADO — {args.branch} ({args.modelo})")
    print(f"   Tempo retrieval médio: {media('tempo_retrieval_s')}s")
    print(f"   Tempo LLM médio:       {media('tempo_llm_s')}s")
    print(f"   answer_correctness:    {media('answer_correctness')}")
    if media("faithfulness") is not None:
        print(f"   faithfulness:          {media('faithfulness')}")
        print(f"   answer_relevancy:      {media('answer_relevancy')}")
        print(f"   context_precision:     {media('context_precision')}")
        print(f"   context_recall:        {media('context_recall')}")
    print(f"{'='*55}\n")

    saida_agg = Path(args.saida.replace(".csv", "_agg.csv"))
    ja_existe_agg = saida_agg.exists()
    linha_agg = {
        "branch": args.branch,
        "modelo": args.modelo,
        "n_perguntas": len(dados),
        "tempo_retrieval_medio_s": media("tempo_retrieval_s"),
        "tempo_llm_medio_s": media("tempo_llm_s"),
        "faithfulness_medio": media("faithfulness"),
        "answer_relevancy_medio": media("answer_relevancy"),
        "context_precision_medio": media("context_precision"),
        "context_recall_medio": media("context_recall"),
        "answer_correctness_medio": media("answer_correctness"),
    }
    with open(saida_agg, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linha_agg.keys()))
        if not ja_existe_agg:
            writer.writeheader()
        writer.writerow(linha_agg)

    print(f"✅ Resultados salvos em: {saida}")
    print(f"✅ Resumo agregado em:   {saida_agg}")
    print("   (rode para os outros modelos e compare no mesmo CSV)")


if __name__ == "__main__":
    main()
