import os
import json
import time
import datetime
import pickle
import requests
import streamlit as st
from pathlib import Path
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────
PASTA_FAISS   = os.getenv("PASTA_FAISS",   "./faiss_index")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
LOG_PATH      = Path("./logs")
LOG_PATH.mkdir(exist_ok=True)

MODELOS = {
    "🌐 LLaMA 3.3 70B (Groq)": {
        "tipo": "groq",
        "modelo": "llama-3.3-70b-versatile",
        "descricao": "Nuvem via Groq API",
    },
    "🖥️ LLaMA 3 (Ollama local)": {
        "tipo": "ollama",
        "modelo": "llama3:latest",
        "descricao": "Local no servidor",
    },
    "🖥️ Mistral 7B (Ollama local)": {
        "tipo": "ollama",
        "modelo": "mistral",
        "descricao": "Local no servidor",
    },
    "🖥️ Gemma 2 27B (Ollama local)": {
        "tipo": "ollama",
        "modelo": "gemma2:27b",
        "descricao": "Local no servidor",
    },
}

# ── Página ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IF Turing — IFRS Ibirubá",
    page_icon="🎓",
    layout="centered",
)

st.markdown("""
    <style>
        .block-container { max-width: 780px; padding-top: 2rem; }
        .stChatMessage { border-radius: 12px; }
        h1 { color: #1a5276; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 IF Turing")
st.caption("Assistente do Processo Seletivo — IFRS Campus Ibirubá")

with st.sidebar:
    st.markdown("## ⚙️ Configuração")
    st.markdown("---")
    modelo_selecionado = st.selectbox(
        "Modelo de IA",
        options=list(MODELOS.keys()),
        help="Troque o modelo para comparar respostas no artigo"
    )
    cfg = MODELOS[modelo_selecionado]
    st.markdown(f"**Tipo:** `{cfg['tipo']}`")
    st.markdown(f"**Modelo:** `{cfg['modelo']}`")
    st.markdown(f"**Onde roda:** {cfg['descricao']}")
    st.markdown("---")
    st.markdown("## 📊 Logs desta sessão")
    if "logs" in st.session_state and st.session_state.logs:
        total     = len(st.session_state.logs)
        tempo_med = round(sum(l["tempo_s"] for l in st.session_state.logs) / total, 2)
        st.metric("Perguntas", total)
        st.metric("Tempo médio", f"{tempo_med}s")
        if st.button("💾 Exportar logs CSV"):
            import csv, io
            output = io.StringIO()
            campos = list(st.session_state.logs[0].keys())
            writer = csv.DictWriter(output, fieldnames=campos)
            writer.writeheader()
            writer.writerows(st.session_state.logs)
            st.download_button(
                "⬇️ Baixar CSV",
                data=output.getvalue(),
                file_name=f"logs_ifturing_{datetime.date.today()}.csv",
                mime="text/csv"
            )
    else:
        st.info("Nenhuma pergunta feita ainda.")
    st.markdown("---")
    if st.button("🗑️ Limpar conversa"):
        st.session_state.historico = [{
            "role": "assistant",
            "content": (
                "Olá! 👋 Sou o **IF Turing**, assistente do processo seletivo do "
                "IFRS Campus Ibirubá. Pode me perguntar sobre cursos, documentos, "
                "datas, cotas, inscrições e muito mais!"
            ),
        }]
        st.rerun()

# ── Carrega índice ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def carregar_pipeline():
    store_path = Path(PASTA_FAISS) / "store.pkl"

    if not store_path.exists():
        return None

    with open(store_path, "rb") as f:
        documentos = pickle.load(f)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documentos)

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(model="intfloat/multilingual-e5-base"),
    )
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(document_store=document_store, top_k=5),
    )
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    return pipeline


pipeline = carregar_pipeline()

if pipeline is None:
    st.error(
        "⚠️ Índice não encontrado. Execute `python indexar.py` para indexar os PDFs primeiro.",
        icon="🚨",
    )
    st.stop()

# ── Funções RAG ──────────────────────────────────────────────────────
def chamar_groq(prompt: str, modelo: str) -> tuple[str, int]:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content, response.usage.total_tokens


def chamar_ollama(prompt: str, modelo: str) -> tuple[str, int]:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data["response"], data.get("eval_count", 0)


def perguntar_llm(pergunta: str, contexto: str) -> tuple[str, int]:
    prompt = f"""Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS.
Sua função é guiar as pessoas interessadas em entrar na instituição de forma inclusiva e acessível.
Responda à pergunta abaixo usando APENAS o contexto fornecido.
Se a resposta puder ser deduzida com base nas informações disponíveis, responda deixando claro que é uma inferência.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

    if cfg["tipo"] == "groq":
        return chamar_groq(prompt, cfg["modelo"])
    else:
        return chamar_ollama(prompt, cfg["modelo"])


def responder(pergunta: str) -> tuple[str, float, int, int]:
    t0 = time.time()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]

    if not docs:
        return "⚠️ Nenhum trecho relevante encontrado nos documentos.", 0.0, 0, 0

    contexto = "\n\n---\n\n".join([d.content for d in docs])
    resposta, tokens = perguntar_llm(pergunta, contexto)
    tempo = round(time.time() - t0, 2)
    return resposta, tempo, tokens, len(docs)


def salvar_log(pergunta: str, resposta: str, tempo: float, tokens: int, docs: int):
    entrada = {
        "timestamp":        datetime.datetime.now().isoformat(),
        "modelo_nome":      modelo_selecionado,
        "modelo_id":        cfg["modelo"],
        "tipo":             cfg["tipo"],
        "pergunta":         pergunta,
        "resposta":         resposta[:300].replace("\n", " "),
        "tempo_s":          tempo,
        "tokens":           tokens,
        "docs_recuperados": docs,
    }
    st.session_state.logs.append(entrada)
    log_file = LOG_PATH / f"log_{datetime.date.today()}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


# ── Histórico de chat ────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []
    st.session_state.historico.append({
        "role": "assistant",
        "content": (
            "Olá! 👋 Sou o **IF Turing**, assistente do processo seletivo do "
            "IFRS Campus Ibirubá. Pode me perguntar sobre cursos, documentos, "
            "datas, cotas, inscrições e muito mais!"
        ),
    })

if "logs" not in st.session_state:
    st.session_state.logs = []

# ── Renderiza histórico ──────────────────────────────────────────────
for msg in st.session_state.historico:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "🧑‍🎓"):
        st.markdown(msg["content"])

# ── Input do usuário ─────────────────────────────────────────────────
pergunta = st.chat_input("Digite sua dúvida sobre o processo seletivo...")

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner(f"Buscando com {cfg['modelo']}..."):
            try:
                resposta, tempo, tokens, n_docs = responder(pergunta)
                salvar_log(pergunta, resposta, tempo, tokens, n_docs)
            except Exception as e:
                resposta, tempo, tokens, n_docs = f"❌ Erro ao processar: {e}", 0, 0, 0
        st.markdown(resposta)
        st.caption(f"⏱ {tempo}s · 🔢 {tokens} tokens · 📄 {n_docs} trechos · 🤖 {cfg['modelo']}")

    st.session_state.historico.append({"role": "assistant", "content": resposta})
