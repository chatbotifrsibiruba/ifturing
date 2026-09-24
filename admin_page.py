import os
import datetime
import streamlit as st
from dotenv import load_dotenv
from db import init_db, get_or_create_sessao, inserir_feedback
from nucleo import MODELOS, registrar_modelos, carregar_pipeline, responder, salvar_consulta_no_banco, formatar_metricas

load_dotenv()
init_db()
registrar_modelos()

_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
if not _ADMIN_PASSWORD:
    _ADMIN_PASSWORD = "ifturing2026"
    print("[AVISO] ADMIN_PASSWORD não definida no .env — usando senha padrão de desenvolvimento.")

if not st.session_state.get("admin_autenticado"):
    st.title("🔐 Acesso ao Painel Admin")
    _senha = st.text_input("Senha de acesso", type="password")
    if st.button("Entrar"):
        if _senha == _ADMIN_PASSWORD:
            st.session_state.admin_autenticado = True
            st.rerun()
        else:
            st.error("Senha incorreta")
    st.stop()

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
pipeline, n_chunks = carregar_pipeline()

if pipeline is None:
    st.error("⚠️ Índice não encontrado. Execute `python indexar.py` primeiro.", icon="🚨")
    st.stop()

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
                m = responder(pergunta, cfg)
                consulta_id = salvar_consulta_no_banco(st.session_state.admin_db_sessao_id, cfg, pergunta, m)
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
        linha_metricas = formatar_metricas(m, cfg["modelo"])
        st.caption(linha_metricas)

    st.session_state.admin_historico.append({
        "role": "assistant",
        "content": m["resposta"],
        "metricas": linha_metricas,
        "consulta_id": consulta_id,
    })
