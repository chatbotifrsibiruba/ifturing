import os
import json
import time
import psutil
import datetime
import platform
import pickle
import requests
import streamlit as st
from pathlib import Path
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from dotenv import load_dotenv
from db import init_db, upsert_modelo, get_or_create_sessao, inserir_consulta, inserir_feedback

load_dotenv()
init_db()

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

for _nome, _cfg in MODELOS.items():
    upsert_modelo(
        modelo_id=_cfg["modelo"],
        nome_bonito=_nome,
        tipo=_cfg.get("tipo", "ollama"),
        params_b=_cfg.get("params_b"),
        quantizacao=_cfg.get("quantizacao"),
    )

st.set_page_config(
    page_title="Admin — IF Turing",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Painel Admin — IF Turing")
st.caption("Modo análise: todas as métricas visíveis. Use para avaliar modelos e qualidade das respostas.")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuração")
    st.markdown("---")
    modelo_selecionado = st.selectbox(
        "Modelo de IA",
        options=list(MODELOS.keys()),
        help="Troque o modelo para comparar respostas",
    )
    cfg = MODELOS[modelo_selecionado]
    st.markdown(f"**Tipo:** `{cfg['tipo']}`")
    st.markdown(f"**Modelo:** `{cfg['modelo']}`")
    st.markdown(f"**Parâmetros:** ~{cfg['params_b']}B")
    st.markdown(f"**Quantização:** {cfg['quantizacao']}")

    st.markdown("---")
    st.markdown("## 📊 Métricas da sessão")
    if "admin_logs" in st.session_state and st.session_state.admin_logs:
        logs = st.session_state.admin_logs
        n = len(logs)
        med = lambda k: round(sum(l[k] for l in logs) / n, 3)
        st.metric("Consultas", n)
        st.metric("Tempo total médio", f"{med('tempo_total_s')} s")
        st.metric("→ Retrieval médio", f"{med('tempo_retrieval_s')} s")
        st.metric("→ LLM médio", f"{med('tempo_llm_s')} s")
        st.metric("Throughput médio", f"{med('tokens_por_s')} tok/s")
    else:
        st.info("Nenhuma consulta ainda.")

    st.markdown("---")
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.admin_historico = [{
            "role": "assistant",
            "content": "Painel admin iniciado. Faça uma pergunta para testar o modelo selecionado.",
        }]
        st.rerun()

# ── Carrega pipeline ──────────────────────────────────────────────────
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

pipeline, n_chunks = carregar_pipeline()

if pipeline is None:
    st.error("⚠️ Índice não encontrado. Execute `python indexar.py` primeiro.", icon="🚨")
    st.stop()

# ── Funções RAG ───────────────────────────────────────────────────────
def chamar_ollama(prompt: str, modelo: str) -> dict:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.3, "num_predict": 1024}},
        timeout=300,
    )
    r.raise_for_status()
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

def perguntar_llm(pergunta: str, contexto: str) -> dict:
    prompt = f"""Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS.
Sua função é guiar as pessoas interessadas em entrar na instituição de forma inclusiva e acessível.
Responda à pergunta abaixo usando APENAS o contexto fornecido.
Se a resposta puder ser deduzida com base nas informações disponíveis, responda deixando claro que é uma inferência.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""
    return chamar_ollama(prompt, cfg["modelo"])

def responder(pergunta: str) -> dict:
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
    r = perguntar_llm(pergunta, contexto)
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

# ── Estado da sessão ──────────────────────────────────────────────────
if "admin_sessao_id" not in st.session_state:
    st.session_state.admin_sessao_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_admin"

if "admin_db_sessao_id" not in st.session_state:
    try:
        st.session_state.admin_db_sessao_id = get_or_create_sessao(
            sessao_uid=st.session_state.admin_sessao_id,
            origem="admin",
            modo="detalhado",
        )
    except Exception as e:
        print(f"[db] erro ao criar sessão admin: {e}")
        st.session_state.admin_db_sessao_id = None

if "admin_historico" not in st.session_state:
    st.session_state.admin_historico = [{
        "role": "assistant",
        "content": "Painel admin iniciado. Faça uma pergunta para testar o modelo selecionado.",
    }]

if "admin_logs" not in st.session_state:
    st.session_state.admin_logs = []

# ── Renderiza histórico ───────────────────────────────────────────────
for msg in st.session_state.admin_historico:
    with st.chat_message(msg["role"], avatar="🔬" if msg["role"] == "assistant" else "🧑‍💻"):
        st.markdown(msg["content"])
        if "metricas" in msg:
            st.caption(msg["metricas"])
        if msg["role"] == "assistant" and msg.get("consulta_id"):
            col_util, col_nao_util, col_resto = st.columns([1, 1, 8])
            chave_base = f"feedback_{msg['consulta_id']}"
            if chave_base not in st.session_state:
                with col_util:
                    if st.button("👍", key=f"{chave_base}_up", help="Resposta útil"):
                        try:
                            inserir_feedback(msg["consulta_id"], util=True)
                            st.session_state[chave_base] = "up"
                            st.rerun()
                        except Exception as e:
                            print(f"Erro ao salvar feedback: {e}")
                with col_nao_util:
                    if st.button("👎", key=f"{chave_base}_down", help="Resposta não útil"):
                        try:
                            inserir_feedback(msg["consulta_id"], util=False)
                            st.session_state[chave_base] = "down"
                            st.rerun()
                        except Exception as e:
                            print(f"Erro ao salvar feedback: {e}")
            else:
                emoji = "👍" if st.session_state[chave_base] == "up" else "👎"
                st.caption(f"{emoji} Obrigado pelo feedback!")

# ── Input ─────────────────────────────────────────────────────────────
pergunta = st.chat_input("Pergunta de teste...")

if pergunta:
    st.session_state.admin_historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner(f"Consultando {cfg['modelo']}..."):
            try:
                m = responder(pergunta)
                consulta_id = None
                try:
                    consulta_id = inserir_consulta(
                        sessao_id=st.session_state.admin_db_sessao_id,
                        modelo_id=cfg["modelo"],
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
                    print(f"[db] erro ao gravar consulta admin: {e}")
                st.session_state.admin_logs.append({
                    "modelo_id": cfg["modelo"],
                    "tipo": cfg["tipo"],
                    "params_b": cfg["params_b"],
                    "quantizacao": cfg["quantizacao"],
                    "tempo_total_s": m["tempo_total_s"],
                    "tempo_retrieval_s": m["tempo_retrieval_s"],
                    "tempo_llm_s": m["tempo_llm_s"],
                    "tokens_por_s": m["tokens_por_s"],
                })
            except Exception as e:
                m = {
                    "resposta": f"❌ Erro ao processar: {e}",
                    "tempo_total_s": 0.0, "tempo_retrieval_s": 0.0, "tempo_llm_s": 0.0,
                    "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
                    "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
                    "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
                    "ram_delta_mb": 0.0, "nao_encontrado": False, "erro": True,
                    "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
                }
                consulta_id = None

        st.markdown(m["resposta"])
        linha_metricas = (
            f"⏱ **{m['tempo_total_s']}s** total "
            f"(retrieval {m['tempo_retrieval_s']}s · LLM {m['tempo_llm_s']}s) · "
            f"🔢 {m['tokens_entrada']}→{m['tokens_saida']} tokens · "
            f"⚡ {m['tokens_por_s']} tok/s · "
            f"📄 {m['docs_recuperados']} trechos (score máx {m['score_max']}) · "
            f"🤖 {cfg['modelo']}"
        )
        st.caption(linha_metricas)

    st.session_state.admin_historico.append({
        "role": "assistant",
        "content": m["resposta"],
        "metricas": linha_metricas,
        "consulta_id": consulta_id,
    })
