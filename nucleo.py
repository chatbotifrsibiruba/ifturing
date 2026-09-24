"""
nucleo.py — Lógica compartilhada entre as páginas do IF Turing.
Backend puro: constantes, MODELOS, pipeline RAG, Ollama, persistência.
Sem layout ou CSS do Streamlit.
"""
import os
import time
import pickle
import psutil
import requests
import streamlit as st
from pathlib import Path
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from dotenv import load_dotenv
from db import upsert_modelo, inserir_consulta

load_dotenv()

PASTA_FAISS   = os.getenv("PASTA_FAISS",   "./faiss_index")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3:latest")
MODELO_EMB    = "intfloat/multilingual-e5-base"
TOP_K         = 5

MODELOS = {
    "🖥️ LLaMA 3 (Ollama local)": {
        "tipo": "ollama", "modelo": MODELO_OLLAMA,
        "descricao": "Local no servidor", "params_b": 8.0, "quantizacao": "Q4_0",
    },
    "🖥️ Mistral 7B (Ollama local)": {
        "tipo": "ollama", "modelo": "mistral",
        "descricao": "Local no servidor", "params_b": 7.2, "quantizacao": "Q4_0",
    },
    "🖥️ Gemma 2 27B (Ollama local)": {
        "tipo": "ollama", "modelo": "gemma2:27b",
        "descricao": "Local no servidor", "params_b": 27.2, "quantizacao": "Q4_0",
    },
    "🖥️ Qwen3 0.6B (Ollama local)": {
        "tipo": "ollama", "modelo": "qwen3:0.6b",
        "descricao": "Local no servidor", "params_b": 0.6, "quantizacao": "Q4_0",
    },
}

MODELO_PADRAO_SIMPLES = "🖥️ Qwen3 0.6B (Ollama local)"


def registrar_modelos() -> None:
    for nome, cfg in MODELOS.items():
        upsert_modelo(
            modelo_id=cfg["modelo"],
            nome_bonito=nome,
            tipo=cfg.get("tipo", "ollama"),
            params_b=cfg.get("params_b"),
            quantizacao=cfg.get("quantizacao"),
        )


@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def carregar_pipeline():
    store_path = Path(PASTA_FAISS) / "store.pkl"
    if not store_path.exists():
        return None, 0
    with open(store_path, "rb") as f:
        documentos = pickle.load(f)
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documentos)
    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=TOP_K))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    return pipeline, len(documentos)


class _OllamaOffline(Exception):
    pass


_MSG_OLLAMA_OFFLINE = (
    "⚠️ Estou com instabilidade no momento. Por favor, tente novamente "
    "em alguns instantes ou entre em contato pelo site oficial do processo seletivo."
)


def chamar_ollama(prompt: str, modelo: str) -> dict:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": modelo, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 1024}},
            timeout=300,
        )
        r.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
            requests.exceptions.RequestException) as _exc:
        raise _OllamaOffline() from _exc
    d = r.json()
    ns = 1_000_000_000
    return {
        "texto": d.get("response", ""),
        "tokens_entrada": d.get("prompt_eval_count", 0),
        "tokens_saida": d.get("eval_count", 0),
        "tokens_total": d.get("prompt_eval_count", 0) + d.get("eval_count", 0),
        "tempo_carga_s": round(d.get("load_duration", 0) / ns, 3),
        "tempo_prefill_s": round(d.get("prompt_eval_duration", 0) / ns, 3),
        "tempo_geracao_s": round(d.get("eval_duration", 0) / ns, 3),
    }


def perguntar_llm(pergunta: str, contexto: str, modelo_cfg: dict) -> dict:
    prompt = f"""Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS.
Sua função é guiar as pessoas interessadas em entrar na instituição de forma inclusiva e acessível.
Responda à pergunta abaixo usando APENAS o contexto fornecido.
Se a resposta puder ser deduzida com base nas informações disponíveis, responda deixando claro que é uma inferência.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""
    return chamar_ollama(prompt, modelo_cfg["modelo"])


def responder(pergunta: str, modelo_cfg: dict) -> dict:
    pipeline, _ = carregar_pipeline()
    if pipeline is None:
        raise RuntimeError("Pipeline não carregado — execute indexar.py primeiro.")

    proc = psutil.Process()
    ram_antes = proc.memory_info().rss / 1024**2
    t_inicio = time.perf_counter()

    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
    t_retrieval = time.perf_counter() - t0

    if not docs:
        return {
            "resposta": "⚠️ Nenhum trecho relevante encontrado nos documentos.",
            "tempo_total_s": round(time.perf_counter() - t_inicio, 3),
            "tempo_retrieval_s": round(t_retrieval, 3), "tempo_llm_s": 0.0,
            "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
            "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
            "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
            "ram_delta_mb": 0.0, "nao_encontrado": True, "erro": False,
            "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
        }

    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join([d.content for d in docs])

    t1 = time.perf_counter()
    try:
        r = perguntar_llm(pergunta, contexto, modelo_cfg)
    except _OllamaOffline:
        print(f"[ollama] falha de conexão ao consultar modelo {modelo_cfg['modelo']}")
        return {
            "resposta": _MSG_OLLAMA_OFFLINE,
            "tempo_total_s": round(time.perf_counter() - t_inicio, 3),
            "tempo_retrieval_s": round(t_retrieval, 3), "tempo_llm_s": 0.0,
            "docs_recuperados": len(docs),
            "score_max": round(max(scores), 4) if scores else 0.0,
            "score_medio": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "chars_contexto": len(contexto), "tokens_entrada": 0, "tokens_saida": 0,
            "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
            "ram_delta_mb": 0.0, "nao_encontrado": False, "erro": True,
            "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
        }
    t_llm = time.perf_counter() - t1

    t_total = time.perf_counter() - t_inicio
    ram_depois = proc.memory_info().rss / 1024**2
    texto = r["texto"]
    tok_saida = r["tokens_saida"]

    return {
        "resposta": texto,
        "tempo_total_s": round(t_total, 3),
        "tempo_retrieval_s": round(t_retrieval, 3),
        "tempo_llm_s": round(t_llm, 3),
        "tempo_carga_s": r.get("tempo_carga_s", 0.0),
        "tempo_prefill_s": r.get("tempo_prefill_s", 0.0),
        "tempo_geracao_s": r.get("tempo_geracao_s", 0.0),
        "docs_recuperados": len(docs),
        "score_max": round(max(scores), 4) if scores else 0.0,
        "score_medio": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "chars_contexto": len(contexto),
        "tokens_entrada": r["tokens_entrada"],
        "tokens_saida": tok_saida,
        "tokens_total": r["tokens_total"],
        "tokens_por_s": round(tok_saida / t_llm, 2) if t_llm > 0 else 0.0,
        "chars_resposta": len(texto),
        "ram_delta_mb": round(ram_depois - ram_antes, 1),
        "nao_encontrado": "não encontrei essa informação" in texto.lower(),
        "erro": False,
    }


def salvar_consulta_no_banco(sessao_id: int, modelo_cfg: dict, pergunta: str, m: dict) -> int | None:
    try:
        return inserir_consulta(
            sessao_id=sessao_id,
            modelo_id=modelo_cfg["modelo"],
            pergunta=pergunta,
            resposta_gerada=m["resposta"],
            tempo_retrieval_s=m["tempo_retrieval_s"],
            tempo_llm_s=m["tempo_llm_s"],
            tempo_total_s=m["tempo_total_s"],
            tokens_entrada=m["tokens_entrada"],
            tokens_saida=m["tokens_saida"],
            docs_recuperados=m["docs_recuperados"],
            score_max=m["score_max"],
            score_medio=m["score_medio"],
            nao_encontrado=m["nao_encontrado"],
            erro=m["erro"],
        )
    except Exception as e:
        print(f"[db] erro ao gravar consulta: {e}")
        return None


def formatar_metricas(m: dict, modelo_id: str) -> str:
    return (
        f"⏱ **{m['tempo_total_s']}s** total "
        f"(retrieval {m['tempo_retrieval_s']}s · LLM {m['tempo_llm_s']}s) · "
        f"🔢 {m['tokens_entrada']}→{m['tokens_saida']} tokens · "
        f"⚡ {m['tokens_por_s']} tok/s · "
        f"📄 {m['docs_recuperados']} trechos (score máx {m['score_max']}) · "
        f"🤖 {modelo_id}"
    )
