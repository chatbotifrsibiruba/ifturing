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

# ── Configurações ────────────────────────────────────────────────────
PASTA_FAISS   = os.getenv("PASTA_FAISS",   "./faiss_index")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3:latest")
MODELO_EMB    = "intfloat/multilingual-e5-base"
TOP_K         = 5
LOG_PATH      = Path("./logs")
LOG_PATH.mkdir(exist_ok=True)

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

for _nome, _cfg in MODELOS.items():
    upsert_modelo(
        modelo_id=_cfg["modelo"],
        nome_bonito=_nome,
        tipo=_cfg.get("tipo", "ollama"),
        params_b=_cfg.get("params_b"),
        quantizacao=_cfg.get("quantizacao"),
    )

# ── Metadados do ambiente ────────────────────────────────────────────
@st.cache_data
def coletar_ambiente() -> dict:
    return {
        "so":           f"{platform.system()} {platform.release()}",
        "python":       platform.python_version(),
        "cpu_modelo":   platform.processor() or "desconhecido",
        "cpu_nucleos":  psutil.cpu_count(logical=False),
        "cpu_threads":  psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "modelo_emb":   MODELO_EMB,
        "top_k":        TOP_K,
    }

AMBIENTE = coletar_ambiente()

# ── Página ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IF Turing — IFRS Ibirubá",
    page_icon="🎓",
    layout="wide",
)

# ── Estado do modo ───────────────────────────────────────────────────
if "modo" not in st.session_state:
    st.session_state.modo = "simples"

MODO_SIMPLES = st.session_state.modo == "simples"

# ── CSS Global (bloco único consolidado) ─────────────────────────────
st.markdown("""
<style>
/*
 * IF TURING — CSS CONSOLIDADO
 * Regra geral: tudo é claro por padrão (fundo branco/cinza, texto escuro).
 * Exceções escuras: sidebar (#2B2B2B) e balão do usuário (#1A1A1A).
 * Todas as regras usam !important para sobrepor o tema padrão do Streamlit.
 * Ordem das regras: da mais geral para a mais específica, garantindo que
 * regras mais específicas (balão do usuário) sobrescrevam as gerais.
 */

/* ── 0. Chrome do Streamlit ─────────────────────────────────────────── */
#MainMenu, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* NÃO esconder o header — o botão "»" de reabrir a sidebar vive nele.
   Tornamos transparente/sem altura visual, mas overflow:visible preserva
   o botão de toggle posicionado absolutamente dentro dele. */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    border-bottom: none !important;
    height: 0 !important;
    overflow: visible !important;
}

/* Botão de reabrir sidebar quando colapsada: nunca esconder */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

/* ── 1. Base da página: fundo cinza claro, texto escuro ─────────────── */
html, body {
    background-color: #EDEDED !important;
    color: #1A1A1A !important;
}
.stApp {
    background-color: #EDEDED !important;
    color: #1A1A1A !important;
}
section.main,
section.main > div {
    background-color: #EDEDED !important;
    color: #1A1A1A !important;
}

/* ── 2. Card central (block-container): branco, texto escuro em tudo ── */
.main .block-container {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.09) !important;
    margin: 16px 20px 80px 20px !important;
    padding: 0 !important;
    max-width: calc(100% - 40px) !important;
    overflow: hidden !important;
}
/* Força cor escura em todos os descendentes do card.
   Regras mais específicas abaixo sobrescrevem esta para os casos excepcionais. */
.main .block-container * {
    color: #1A1A1A !important;
}

/* ── 3. Sidebar: fundo escuro, TODO texto branco ────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #2B2B2B !important;
    padding-top: 0 !important;
}
/* Seletor universal cobre p, span, label, a, h*, .stMarkdown, metric, etc. */
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr {
    border-color: #3A3A3A !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #3A3A3A !important;
    border-color: #555555 !important;
}
[data-testid="stSidebar"] .stExpander {
    background-color: #333333 !important;
    border-color: #444444 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #3A3A3A !important;
    border-color: #555555 !important;
}

/* ── 4. Botões na área principal: fundo branco, hover verde ─────────── */
/* Cobre as variações de DOM do Streamlit (stButton container, kind attr) */
div[data-testid="stButton"] button,
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
button[kind="secondary"],
button[kind="secondary"] p,
button[kind="secondary"] span,
.main .stButton button {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #D5D5D5 !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] button:hover,
button[kind="secondary"]:hover,
.main .stButton button:hover {
    background-color: #F0FFF0 !important;
    border-color: #00A300 !important;
    color: #1A1A1A !important;
}
/* Botões DA SIDEBAR sobrescrevem as regras acima (seletor mais específico) */
[data-testid="stSidebar"] div[data-testid="stButton"] button,
[data-testid="stSidebar"] div[data-testid="stButton"] button p,
[data-testid="stSidebar"] div[data-testid="stButton"] button span {
    background-color: #3A3A3A !important;
    color: #FFFFFF !important;
    border-color: #555555 !important;
}

/* ── 5. Input do chat e barra sticky do rodapé ──────────────────────── */
/* Todos os níveis do container fixo ao fundo */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
}
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-top: 1px solid #EBEBEB !important;
    padding: 10px 16px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #F0F0F0 !important;
    color: #1A1A1A !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 24px !important;
    padding: 10px 20px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background-color: #00A300 !important;
    color: #FFFFFF !important;
    border: none !important;
    outline: none !important;
    border-radius: 50% !important;
}
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── 6. Balões do chat ──────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    padding: 6px 0 !important;
}

/* Bot: esquerda, cinza claro, texto escuro */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background-color: #E8E8E8 !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent * {
    color: #1A1A1A !important;
}

/* Usuário: direita, escuro — ÚNICA exceção de fundo escuro fora da sidebar.
   Este seletor é mais específico que ".main .block-container *",
   portanto vence na cascata mesmo ambos tendo !important. */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background-color: #1A1A1A !important;
    border-radius: 16px 4px 16px 16px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent * {
    color: #FFFFFF !important;
}

/* ── 7. Classes personalizadas IF Turing ────────────────────────────── */
.if-topbar {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #EBEBEB !important;
    padding: 14px 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}
.if-topbar-left {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    color: #1A1A1A !important;
}
.if-topbar-status {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    background-color: #F0F0F0 !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 13px !important;
    color: #555555 !important;
}
.if-dot {
    width: 8px !important;
    height: 8px !important;
    background-color: #00A300 !important;
    border-radius: 50% !important;
    display: inline-block !important;
    flex-shrink: 0 !important;
}
.if-card-header {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #F0F0F0 !important;
    padding: 14px 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}
.if-card-header-left {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.if-card-title  { font-weight: 700 !important; font-size: 15px !important; color: #1A1A1A !important; }
.if-card-subtitle { font-size: 11px !important; color: #888888 !important; margin-top: 2px !important; }
.if-card-online {
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
    font-size: 13px !important;
    color: #555555 !important;
}

/* ── 8. Força bruta: garante tema claro em toda a área da aplicação ─── */
/* Estas regras ficam no FINAL para vencer qualquer regra anterior,
   incluindo as injetadas pelo tema padrão do Streamlit. */
:root {
    color-scheme: light !important;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    color: #1A1A1A !important;
}

/* Sidebar: sobrescreve a regra genérica acima */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Balão do usuário: sobrescreve a regra genérica acima */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent * {
    color: #FFFFFF !important;
}

/* Botão de envio do chat: ícone branco no fundo verde */
[data-testid="stChatInput"] button,
[data-testid="stChatInput"] button * {
    color: #FFFFFF !important;
    background-color: #00A300 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:24px 20px 8px;">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
<div style="width:42px;height:42px;background:#FFF;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0;">🎓</div>
<div>
<div style="font-weight:700;font-size:17px;color:#FFF;line-height:1.2;">IF Turing</div>
<div style="font-size:12px;color:#A0A0A0;">IFRS Ibirubá</div>
</div>
</div>
<div style="display:flex;align-items:center;gap:7px;margin-bottom:24px;">
<div style="width:8px;height:8px;background:#00A300;border-radius:50%;flex-shrink:0;"></div>
<span style="color:#A0A0A0;font-size:13px;">Bot online</span>
</div>
<div style="font-size:10px;color:#A0A0A0;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">Painel</div>
<div style="background:#00A300;border-radius:10px;padding:11px 16px;margin-bottom:24px;display:flex;align-items:center;gap:10px;">
<span style="font-size:17px;">💬</span>
<span style="color:#FFF;font-weight:700;font-size:15px;">Conversas</span>
</div>
<div style="border-top:1px solid #3A3A3A;margin-top:8px;"></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="padding:14px 20px 0;">
<div style="font-size:10px;color:#A0A0A0;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">Atalhos</div>
<div style="display:flex;flex-direction:column;gap:14px;margin-bottom:20px;">
<a href="https://ifrs.edu.br/ibiruba/" target="_blank" style="color:#FFF;text-decoration:none;font-size:14px;">Site IFRS Ibirubá</a>
<a href="#" style="color:#FFF;text-decoration:none;font-size:14px;">Vem pro IF Ibirubá</a>
<a href="#" style="color:#FFF;text-decoration:none;font-size:14px;">Cronograma do processo seletivo</a>
</div>
</div>
""", unsafe_allow_html=True)

    if MODO_SIMPLES:
        modelo_selecionado = MODELO_PADRAO_SIMPLES
        cfg = MODELOS[modelo_selecionado]
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.historico = [{
                "role": "assistant",
                "content": (
                    "Olá! 👋 Sou o **IF Turing**, assistente do processo seletivo do "
                    "IFRS Campus Ibirubá. Pode me perguntar sobre cursos, documentos, "
                    "datas, cotas, inscrições e muito mais!"
                ),
            }]
            st.rerun()
    else:
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
        st.markdown(f"**Parâmetros:** ~{cfg['params_b']}B")
        st.markdown(f"**Quantização:** {cfg['quantizacao']}")
        st.markdown(f"**Onde roda:** {cfg['descricao']}")

        with st.expander("🖥️ Ambiente de execução"):
            st.markdown(f"**SO:** {AMBIENTE['so']}")
            st.markdown(f"**Python:** {AMBIENTE['python']}")
            st.markdown(f"**CPU:** {AMBIENTE['cpu_nucleos']} núcleos / {AMBIENTE['cpu_threads']} threads")
            st.markdown(f"**RAM:** {AMBIENTE['ram_total_gb']} GB")
            st.markdown(f"**Embeddings:** `{AMBIENTE['modelo_emb']}`")
            st.markdown(f"**top_k:** {AMBIENTE['top_k']}")

        st.markdown("---")
        st.markdown("## 📊 Métricas da sessão")

        if "logs" in st.session_state and st.session_state.logs:
            logs = st.session_state.logs
            n = len(logs)
            med = lambda k: round(sum(l[k] for l in logs) / n, 3)

            st.metric("Consultas", n)
            st.metric("Tempo total médio", f"{med('tempo_total_s')} s")
            st.metric("→ Retrieval médio", f"{med('tempo_retrieval_s')} s")
            st.metric("→ LLM médio", f"{med('tempo_llm_s')} s")
            st.metric("Throughput médio", f"{med('tokens_por_s')} tok/s")
            st.metric("Tokens saída (média)", f"{med('tokens_saida')}")

            st.markdown("---")
            st.markdown("### 💾 Exportar dados")

            import csv, io
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(logs[0].keys()))
            w.writeheader()
            w.writerows(logs)
            st.download_button(
                "⬇️ CSV bruto (por consulta)",
                data=buf.getvalue(),
                file_name=f"ifturing_bruto_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            agregado = {}
            for l in logs:
                agregado.setdefault(l["modelo_id"], []).append(l)

            linhas_agg = []
            for mid, entradas in agregado.items():
                m = len(entradas)
                avg = lambda k: round(sum(e[k] for e in entradas) / m, 3)
                tempos = sorted(e["tempo_total_s"] for e in entradas)
                linhas_agg.append({
                    "modelo_id": mid,
                    "tipo": entradas[0]["tipo"],
                    "params_b": entradas[0]["params_b"],
                    "quantizacao": entradas[0]["quantizacao"],
                    "n_consultas": m,
                    "tempo_total_medio_s": avg("tempo_total_s"),
                    "tempo_total_min_s": min(tempos),
                    "tempo_total_max_s": max(tempos),
                    "tempo_total_mediana_s": round(tempos[m // 2], 3),
                    "tempo_retrieval_medio_s": avg("tempo_retrieval_s"),
                    "tempo_llm_medio_s": avg("tempo_llm_s"),
                    "pct_tempo_llm": round(avg("tempo_llm_s") / avg("tempo_total_s") * 100, 1) if avg("tempo_total_s") else 0,
                    "tokens_entrada_medio": avg("tokens_entrada"),
                    "tokens_saida_medio": avg("tokens_saida"),
                    "tokens_por_s_medio": avg("tokens_por_s"),
                    "chars_resposta_medio": avg("chars_resposta"),
                    "taxa_nao_encontrado": round(sum(1 for e in entradas if e["nao_encontrado"]) / m * 100, 1),
                    "taxa_erro": round(sum(1 for e in entradas if e["erro"]) / m * 100, 1),
                })

            buf2 = io.StringIO()
            w2 = csv.DictWriter(buf2, fieldnames=list(linhas_agg[0].keys()))
            w2.writeheader()
            w2.writerows(linhas_agg)
            st.download_button(
                "⬇️ CSV agregado (por modelo)",
                data=buf2.getvalue(),
                file_name=f"ifturing_agregado_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Nenhuma consulta registrada ainda.")

        st.markdown("---")
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.historico = [{
                "role": "assistant",
                "content": (
                    "Olá! 👋 Sou o **IF Turing**, assistente do processo seletivo do "
                    "IFRS Campus Ibirubá. Pode me perguntar sobre cursos, documentos, "
                    "datas, cotas, inscrições e muito mais!"
                ),
            }]
            st.rerun()

    # Toggle de modo + rodapé
    rotulo_botao = "🔬 Ver modo análise" if MODO_SIMPLES else "🎓 Ver modo usuário"
    if st.button(rotulo_botao, use_container_width=True):
        st.session_state.modo = "detalhado" if MODO_SIMPLES else "simples"
        st.rerun()
    st.markdown("""
<div style="text-align:center;color:#555;font-size:11px;padding:10px 0 16px;border-top:1px solid #3A3A3A;margin-top:8px;">
Assistente virtual
</div>
""", unsafe_allow_html=True)

# ── Carrega índice ───────────────────────────────────────────────────
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

# ── Topbar da área de chat ───────────────────────────────────────────
st.markdown("""
<div class="if-topbar">
    <div class="if-topbar-left">
        <span style="font-size:20px;cursor:pointer;color:#555;">☰</span>
        <span style="font-size:17px;">💬</span>
        <span>Conversas</span>
    </div>
    <div class="if-topbar-status">
        <span class="if-dot"></span>
        <span>Bot online</span>
    </div>
</div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Índice não encontrado. Execute `python indexar.py` para indexar os PDFs primeiro.", icon="🚨")
    st.stop()

# ── Funções RAG ──────────────────────────────────────────────────────
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

def salvar_log(pergunta: str, m: dict) -> int | None:
    entrada = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sessao_id": st.session_state.sessao_id,
        "modo": st.session_state.modo,
        "modelo_nome": modelo_selecionado,
        "modelo_id": cfg["modelo"],
        "tipo": cfg["tipo"],
        "params_b": cfg["params_b"],
        "quantizacao": cfg["quantizacao"],
        "modelo_emb": MODELO_EMB,
        "top_k": TOP_K,
        "chunks_indice": n_chunks,
        "pergunta": pergunta,
        "chars_pergunta": len(pergunta),
        "resposta": m["resposta"][:500].replace("\n", " "),
        "chars_resposta": m["chars_resposta"],
        "tempo_total_s": m["tempo_total_s"],
        "tempo_retrieval_s": m["tempo_retrieval_s"],
        "tempo_llm_s": m["tempo_llm_s"],
        "tempo_carga_s": m.get("tempo_carga_s", 0.0),
        "tempo_prefill_s": m.get("tempo_prefill_s", 0.0),
        "tempo_geracao_s": m.get("tempo_geracao_s", 0.0),
        "tokens_entrada": m["tokens_entrada"],
        "tokens_saida": m["tokens_saida"],
        "tokens_total": m["tokens_total"],
        "tokens_por_s": m["tokens_por_s"],
        "docs_recuperados": m["docs_recuperados"],
        "score_max": m["score_max"],
        "score_medio": m["score_medio"],
        "chars_contexto": m["chars_contexto"],
        "ram_delta_mb": m["ram_delta_mb"],
        "cpu_nucleos": AMBIENTE["cpu_nucleos"],
        "ram_total_gb": AMBIENTE["ram_total_gb"],
        "nao_encontrado": m["nao_encontrado"],
        "erro": m["erro"],
    }
    st.session_state.logs.append(entrada)
    log_file = LOG_PATH / f"log_{datetime.date.today()}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")

    consulta_id = None
    try:
        consulta_id = inserir_consulta(
            sessao_id=st.session_state.db_sessao_id,
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
        print(f"[db] erro ao gravar consulta: {e}")
    return consulta_id

# ── Estado da sessão ─────────────────────────────────────────────────
if "sessao_id" not in st.session_state:
    st.session_state.sessao_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if "db_sessao_id" not in st.session_state:
    try:
        st.session_state.db_sessao_id = get_or_create_sessao(
            sessao_uid=st.session_state.sessao_id,
            origem="app",
            modo=st.session_state.modo,
        )
    except Exception as e:
        print(f"[db] erro ao criar sessão: {e}")
        st.session_state.db_sessao_id = None

if "historico" not in st.session_state:
    st.session_state.historico = [{
        "role": "assistant",
        "content": (
            "Olá! 👋 Sou o **IF Turing**, assistente do processo seletivo do "
            "IFRS Campus Ibirubá. Pode me perguntar sobre cursos, documentos, "
            "datas, cotas, inscrições e muito mais!"
        ),
    }]

if "logs" not in st.session_state:
    st.session_state.logs = []

# ── Cabeçalho do card de chat ────────────────────────────────────────
n_msgs = len(st.session_state.historico)
st.markdown(f"""
<div class="if-card-header">
    <div class="if-card-header-left">
        <span style="font-size:24px;">🎓</span>
        <div>
            <div class="if-card-title">IF Turing</div>
            <div class="if-card-subtitle">
                Assistente do Processo Seletivo &middot; {n_msgs} msgs
            </div>
        </div>
    </div>
    <div class="if-card-online">
        <span class="if-dot"></span>
        <span>online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Renderiza histórico ──────────────────────────────────────────────
for msg in st.session_state.historico:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "🧑‍🎓"):
        st.markdown(msg["content"])
        if not MODO_SIMPLES and "metricas" in msg:
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

# ── Sugestões rápidas (só no modo simples, só no início) ─────────────
if MODO_SIMPLES and len(st.session_state.historico) == 1:
    st.markdown(
        '<p style="font-size:13px;color:#888;font-weight:600;'
        'margin:14px 0 8px;text-transform:uppercase;letter-spacing:.04em;">'
        'Perguntas frequentes</p>',
        unsafe_allow_html=True
    )
    sugestoes = [
        "📅 Quais são as datas do processo seletivo?",
        "📄 Quais documentos preciso para me inscrever?",
        "🎯 Como funcionam as cotas?",
        "💰 Existe taxa de inscrição?",
    ]
    cols = st.columns(2)
    pergunta_sugerida = None
    for i, s in enumerate(sugestoes):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                pergunta_sugerida = s.split(" ", 1)[1]
else:
    pergunta_sugerida = None

# ── Input do usuário ─────────────────────────────────────────────────
pergunta_digitada = st.chat_input("Digite sua dúvida sobre o processo seletivo...")
pergunta = pergunta_sugerida or pergunta_digitada

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="🎓"):
        spinner_msg = "Buscando nos documentos..." if MODO_SIMPLES else f"Consultando {cfg['modelo']}..."
        with st.spinner(spinner_msg):
            try:
                m = responder(pergunta)
                consulta_id = salvar_log(pergunta, m)
            except Exception as e:
                m = {
                    "resposta": f"❌ Erro ao processar: {e}",
                    "tempo_total_s": 0.0, "tempo_retrieval_s": 0.0, "tempo_llm_s": 0.0,
                    "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
                    "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
                    "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
                    "ram_delta_mb": 0.0, "nao_encontrado": False, "erro": True,
                }
                consulta_id = salvar_log(pergunta, m)

        st.markdown(m["resposta"])

        linha_metricas = (
            f"⏱ **{m['tempo_total_s']}s** total "
            f"(retrieval {m['tempo_retrieval_s']}s · LLM {m['tempo_llm_s']}s) · "
            f"🔢 {m['tokens_entrada']}→{m['tokens_saida']} tokens · "
            f"⚡ {m['tokens_por_s']} tok/s · "
            f"📄 {m['docs_recuperados']} trechos (score máx {m['score_max']}) · "
            f"🤖 {cfg['modelo']}"
        )
        if not MODO_SIMPLES:
            st.caption(linha_metricas)

    st.session_state.historico.append({
        "role": "assistant",
        "content": m["resposta"],
        "metricas": linha_metricas,
        "consulta_id": consulta_id,
    })
    if pergunta_sugerida:
        st.rerun()

