import os
import requests
import streamlit as st
from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from dotenv import load_dotenv

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────
PASTA_FAISS   = os.getenv("PASTA_FAISS",   "./faiss_index")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3:latest")

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

# ── Carrega índice ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def carregar_pipeline():
    document_store = ChromaDocumentStore(persist_path=PASTA_FAISS)

    if document_store.count_documents() == 0:
        return None

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(model="intfloat/multilingual-e5-base"),
    )
    pipeline.add_component(
        "retriever",
        ChromaEmbeddingRetriever(document_store=document_store, top_k=5),
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
def perguntar_ollama(pergunta: str, contexto: str) -> str:
    prompt = f"""Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS.
Sua função é guiar as pessoas interessadas em entrar na instituição de forma inclusiva e acessível.
Responda à pergunta abaixo usando APENAS o contexto fornecido.
Se a resposta puder ser deduzida com base nas informações disponíveis, responda deixando claro que é uma inferência.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODELO_OLLAMA,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            }
        },
        timeout=120
    )
    r.raise_for_status()
    return r.json()["response"]


def responder(pergunta: str) -> str:
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]

    if not docs:
        return "⚠️ Nenhum trecho relevante encontrado nos documentos."

    contexto = "\n\n---\n\n".join([d.content for d in docs])
    return perguntar_ollama(pergunta, contexto)


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
        with st.spinner("Buscando nos documentos..."):
            try:
                resposta = responder(pergunta)
            except Exception as e:
                resposta = f"❌ Erro ao processar: {e}"
        st.markdown(resposta)

    st.session_state.historico.append({"role": "assistant", "content": resposta})