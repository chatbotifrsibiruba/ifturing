import os
import pickle
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
PASTA_FAISS = os.getenv("PASTA_FAISS", "./faiss_index")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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
    store_path = Path(PASTA_FAISS) / "store.pkl"

    if not store_path.exists():
        return None, None

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

    groq_client = Groq(api_key=GROQ_API_KEY)

    return pipeline, groq_client


pipeline, groq_client = carregar_pipeline()

if pipeline is None:
    st.error(
        "⚠️ Índice não encontrado. Execute `python indexar.py` para indexar os PDFs primeiro.",
        icon="🚨",
    )
    st.stop()

# ── Funções RAG ──────────────────────────────────────────────────────
def perguntar_groq(pergunta: str, contexto: str) -> str:
    prompt = f"""Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS.
Sua função é guiar as pessoas interessadas em entrar na instituição de forma inclusiva e acessível.
Responda à pergunta abaixo usando APENAS o contexto fornecido.
Se a resposta puder ser deduzida com base nas informações disponíveis, responda deixando claro que é uma inferência.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content


def responder(pergunta: str) -> str:
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]

    if not docs:
        return "⚠️ Nenhum trecho relevante encontrado nos documentos."

    contexto = "\n\n---\n\n".join([d.content for d in docs])
    return perguntar_groq(pergunta, contexto)


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
