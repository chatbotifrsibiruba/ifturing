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
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────
PASTA_FAISS   = os.getenv("PASTA_FAISS",   "./faiss_index")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3:latest")
MODELO_EMB    = "intfloat/multilingual-e5-base"
TOP_K         = 5
LOG_PATH      = Path("./logs")
LOG_PATH.mkdir(exist_ok=True)

MODELOS = {
    "🌐 LLaMA 3.3 70B (Groq)": {
        "tipo": "groq",
        "modelo": "llama-3.3-70b-versatile",
        "descricao": "Nuvem via Groq API",
        "params_b": 70.0,
        "quantizacao": "FP16 (servidor Groq)",
    },
    "🖥️ LLaMA 3 (Ollama local)": {
        "tipo": "ollama",
        "modelo": MODELO_OLLAMA,
        "descricao": "Local no servidor",
        "params_b": 8.0,
        "quantizacao": "Q4_0",
    },
    "🖥️ Mistral 7B (Ollama local)": {
        "tipo": "ollama",
        "modelo": "mistral",
        "descricao": "Local no servidor",
        "params_b": 7.2,
        "quantizacao": "Q4_0",
    },
    "🖥️ Gemma 2 27B (Ollama local)": {
        "tipo": "ollama",
        "modelo": "gemma2:27b",
        "descricao": "Local no servidor",
        "params_b": 27.2,
        "quantizacao": "Q4_0",
    },
}

# ── Metadados do ambiente (constantes por execução) ──────────────────
@st.cache_data
def coletar_ambiente() -> dict:
    return {
        "so":             f"{platform.system()} {platform.release()}",
        "python":         platform.python_version(),
        "cpu_modelo":     platform.processor() or "desconhecido",
        "cpu_nucleos":    psutil.cpu_count(logical=False),
        "cpu_threads":    psutil.cpu_count(logical=True),
        "ram_total_gb":   round(psutil.virtual_memory().total / 1024**3, 1),
        "modelo_emb":     MODELO_EMB,
        "top_k":          TOP_K,
    }

AMBIENTE = coletar_ambiente()

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
        st.metric("Tempo total médio",     f"{med('tempo_total_s')} s")
        st.metric("→ Retrieval médio",     f"{med('tempo_retrieval_s')} s")
        st.metric("→ LLM médio",           f"{med('tempo_llm_s')} s")
        st.metric("Throughput médio",      f"{med('tokens_por_s')} tok/s")
        st.metric("Tokens saída (média)",  f"{med('tokens_saida')}")

        st.markdown("---")
        st.markdown("### 💾 Exportar dados")

        # CSV completo — uma linha por consulta
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

        # CSV agregado — uma linha por modelo (pronto para tabela do artigo)
        agregado = {}
        for l in logs:
            mid = l["modelo_id"]
            agregado.setdefault(mid, []).append(l)

        linhas_agg = []
        for mid, entradas in agregado.items():
            m = len(entradas)
            avg = lambda k: round(sum(e[k] for e in entradas) / m, 3)
            tempos = sorted(e["tempo_total_s"] for e in entradas)
            linhas_agg.append({
                "modelo_id":            mid,
                "tipo":                 entradas[0]["tipo"],
                "params_b":             entradas[0]["params_b"],
                "quantizacao":          entradas[0]["quantizacao"],
                "n_consultas":          m,
                "tempo_total_medio_s":  avg("tempo_total_s"),
                "tempo_total_min_s":    min(tempos),
                "tempo_total_max_s":    max(tempos),
                "tempo_total_mediana_s": round(tempos[m // 2], 3),
                "tempo_retrieval_medio_s": avg("tempo_retrieval_s"),
                "tempo_llm_medio_s":    avg("tempo_llm_s"),
                "pct_tempo_llm":        round(avg("tempo_llm_s") / avg("tempo_total_s") * 100, 1) if avg("tempo_total_s") else 0,
                "tokens_entrada_medio": avg("tokens_entrada"),
                "tokens_saida_medio":   avg("tokens_saida"),
                "tokens_por_s_medio":   avg("tokens_por_s"),
                "chars_resposta_medio": avg("chars_resposta"),
                "taxa_nao_encontrado":  round(sum(1 for e in entradas if e["nao_encontrado"]) / m * 100, 1),
                "taxa_erro":            round(sum(1 for e in entradas if e["erro"]) / m * 100, 1),
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
    pipeline.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(model=MODELO_EMB),
    )
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
    )
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    return pipeline, len(documentos)


pipeline, n_chunks = carregar_pipeline()

if pipeline is None:
    st.error(
        "⚠️ Índice não encontrado. Execute `python indexar.py` para indexar os PDFs primeiro.",
        icon="🚨",
    )
    st.stop()

# ── Funções RAG ──────────────────────────────────────────────────────
def chamar_groq(prompt: str, modelo: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    u = response.usage
    return {
        "texto":           response.choices[0].message.content,
        "tokens_entrada":  u.prompt_tokens,
        "tokens_saida":    u.completion_tokens,
        "tokens_total":    u.total_tokens,
        "tempo_carga_s":   0.0,   # Groq não expõe tempo de carga do modelo
    }


def chamar_ollama(prompt: str, modelo: str) -> dict:
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

    ns = 1_000_000_000  # nanosegundos → segundos
    return {
        "texto":            d.get("response", ""),
        "tokens_entrada":   d.get("prompt_eval_count", 0),
        "tokens_saida":     d.get("eval_count", 0),
        "tokens_total":     d.get("prompt_eval_count", 0) + d.get("eval_count", 0),
        "tempo_carga_s":    round(d.get("load_duration", 0) / ns, 3),
        "tempo_prefill_s":  round(d.get("prompt_eval_duration", 0) / ns, 3),
        "tempo_geracao_s":  round(d.get("eval_duration", 0) / ns, 3),
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

    if cfg["tipo"] == "groq":
        return chamar_groq(prompt, cfg["modelo"])
    return chamar_ollama(prompt, cfg["modelo"])


def responder(pergunta: str) -> dict:
    proc = psutil.Process()
    ram_antes = proc.memory_info().rss / 1024**2

    t_inicio = time.perf_counter()

    # ── Etapa 1: retrieval (embedding + busca) ──
    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
    t_retrieval = time.perf_counter() - t0

    if not docs:
        return {
            "resposta":          "⚠️ Nenhum trecho relevante encontrado nos documentos.",
            "tempo_total_s":     round(time.perf_counter() - t_inicio, 3),
            "tempo_retrieval_s": round(t_retrieval, 3),
            "tempo_llm_s":       0.0,
            "docs_recuperados":  0,
            "score_max":         0.0,
            "score_medio":       0.0,
            "chars_contexto":    0,
            "tokens_entrada":    0,
            "tokens_saida":      0,
            "tokens_total":      0,
            "tokens_por_s":      0.0,
            "chars_resposta":    0,
            "ram_delta_mb":      0.0,
            "nao_encontrado":    True,
            "erro":              False,
        }

    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join([d.content for d in docs])

    # ── Etapa 2: geração pela LLM ──
    t1 = time.perf_counter()
    r = perguntar_llm(pergunta, contexto)
    t_llm = time.perf_counter() - t1

    t_total = time.perf_counter() - t_inicio
    ram_depois = proc.memory_info().rss / 1024**2

    texto = r["texto"]
    tok_saida = r["tokens_saida"]

    return {
        "resposta":          texto,
        "tempo_total_s":     round(t_total, 3),
        "tempo_retrieval_s": round(t_retrieval, 3),
        "tempo_llm_s":       round(t_llm, 3),
        "tempo_carga_s":     r.get("tempo_carga_s", 0.0),
        "tempo_prefill_s":   r.get("tempo_prefill_s", 0.0),
        "tempo_geracao_s":   r.get("tempo_geracao_s", 0.0),
        "docs_recuperados":  len(docs),
        "score_max":         round(max(scores), 4) if scores else 0.0,
        "score_medio":       round(sum(scores) / len(scores), 4) if scores else 0.0,
        "chars_contexto":    len(contexto),
        "tokens_entrada":    r["tokens_entrada"],
        "tokens_saida":      tok_saida,
        "tokens_total":      r["tokens_total"],
        "tokens_por_s":      round(tok_saida / t_llm, 2) if t_llm > 0 else 0.0,
        "chars_resposta":    len(texto),
        "ram_delta_mb":      round(ram_depois - ram_antes, 1),
        "nao_encontrado":    "não encontrei essa informação" in texto.lower(),
        "erro":              False,
    }


def salvar_log(pergunta: str, m: dict):
    entrada = {
        # Identificação
        "timestamp":         datetime.datetime.now().isoformat(),
        "sessao_id":         st.session_state.sessao_id,
        "modelo_nome":       modelo_selecionado,
        "modelo_id":         cfg["modelo"],
        "tipo":              cfg["tipo"],
        "params_b":          cfg["params_b"],
        "quantizacao":       cfg["quantizacao"],
        # Configuração do RAG
        "modelo_emb":        MODELO_EMB,
        "top_k":             TOP_K,
        "chunks_indice":     n_chunks,
        # Consulta
        "pergunta":          pergunta,
        "chars_pergunta":    len(pergunta),
        "resposta":          m["resposta"][:500].replace("\n", " "),
        "chars_resposta":    m["chars_resposta"],
        # Latência
        "tempo_total_s":     m["tempo_total_s"],
        "tempo_retrieval_s": m["tempo_retrieval_s"],
        "tempo_llm_s":       m["tempo_llm_s"],
        "tempo_carga_s":     m.get("tempo_carga_s", 0.0),
        "tempo_prefill_s":   m.get("tempo_prefill_s", 0.0),
        "tempo_geracao_s":   m.get("tempo_geracao_s", 0.0),
        # Tokens
        "tokens_entrada":    m["tokens_entrada"],
        "tokens_saida":      m["tokens_saida"],
        "tokens_total":      m["tokens_total"],
        "tokens_por_s":      m["tokens_por_s"],
        # Qualidade do retrieval
        "docs_recuperados":  m["docs_recuperados"],
        "score_max":         m["score_max"],
        "score_medio":       m["score_medio"],
        "chars_contexto":    m["chars_contexto"],
        # Recursos
        "ram_delta_mb":      m["ram_delta_mb"],
        "cpu_nucleos":       AMBIENTE["cpu_nucleos"],
        "ram_total_gb":      AMBIENTE["ram_total_gb"],
        # Flags
        "nao_encontrado":    m["nao_encontrado"],
        "erro":              m["erro"],
    }

    st.session_state.logs.append(entrada)

    log_file = LOG_PATH / f"log_{datetime.date.today()}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


# ── Estado da sessão ─────────────────────────────────────────────────
if "sessao_id" not in st.session_state:
    st.session_state.sessao_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

# ── Renderiza histórico ──────────────────────────────────────────────
for msg in st.session_state.historico:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "🧑‍🎓"):
        st.markdown(msg["content"])
        if "metricas" in msg:
            st.caption(msg["metricas"])

# ── Input do usuário ─────────────────────────────────────────────────
pergunta = st.chat_input("Digite sua dúvida sobre o processo seletivo...")

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner(f"Consultando {cfg['modelo']}..."):
            try:
                m = responder(pergunta)
                salvar_log(pergunta, m)
            except Exception as e:
                m = {
                    "resposta": f"❌ Erro ao processar: {e}",
                    "tempo_total_s": 0.0, "tempo_retrieval_s": 0.0, "tempo_llm_s": 0.0,
                    "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
                    "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
                    "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
                    "ram_delta_mb": 0.0, "nao_encontrado": False, "erro": True,
                }
                salvar_log(pergunta, m)

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

    st.session_state.historico.append({
        "role": "assistant",
        "content": m["resposta"],
        "metricas": linha_metricas,
    })