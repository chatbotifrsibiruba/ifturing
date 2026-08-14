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
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL         = os.getenv("OLLAMA_URL", "http://localhost:11434")
PASTA_FAISS        = os.getenv("PASTA_FAISS", "./faiss_index")
MODELO_EMB         = "intfloat/multilingual-e5-base"

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


# ── Pipeline de retrieval ────────────────────────────────────────────
def carregar_pipeline():
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    store_path = Path(PASTA_FAISS) / "store.pkl"
    with open(store_path, "rb") as f:
        documentos = pickle.load(f)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documentos)

    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    return pipeline


# ── Chamada ao modelo local (Ollama) ─────────────────────────────────
def chamar_ollama(prompt: str, modelo: str) -> dict:
    import requests
    t0 = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        },
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    tempo = time.perf_counter() - t0
    return {
        "texto": d.get("response", ""),
        "tokens_entrada": d.get("prompt_eval_count", 0),
        "tokens_saida": d.get("eval_count", 0),
        "tempo_llm_s": round(tempo, 3),
    }


PROMPT_TEMPLATE = """Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS.
Responda à pergunta abaixo usando APENAS o contexto fornecido.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""


def responder(pipeline, pergunta: str, modelo: str) -> dict:
    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
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

    contexto = "\n\n---\n\n".join([d.content for d in docs])
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)

    r = chamar_ollama(prompt, modelo)

    return {
        "resposta": r["texto"],
        "contexto": contexto,
        "tempo_retrieval_s": round(t_retrieval, 3),
        "tempo_llm_s": r["tempo_llm_s"],
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
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
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

    print(f"\n🌿 Branch/config: {args.branch}")
    print(f"🤖 Modelo avaliado: {args.modelo} (ollama)")

    dados = carregar_dataset(args.dataset, args.limite)
    print(f"📋 {len(dados)} perguntas carregadas do Golden Dataset\n")

    pipeline = carregar_pipeline()

    perguntas, respostas, contextos, resp_esperadas = [], [], [], []
    linhas_brutas = []

    for i, item in enumerate(dados, 1):
        pergunta = item["pergunta"]
        esperada = item["resposta_esperada"]
        print(f"  ⏳ [{i}/{len(dados)}] {pergunta[:55]}...")

        try:
            r = responder(pipeline, pergunta, args.modelo)
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