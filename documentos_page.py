import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from indexar import indexar_documentos

load_dotenv()

PASTA_DOCS  = Path(os.getenv("PASTA_DOCS",  "./documentos"))
PASTA_FAISS =      os.getenv("PASTA_FAISS", "./faiss_index")
PASTA_DOCS.mkdir(exist_ok=True)

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

st.title("📁 Gestão de Documentos")
st.caption("Os documentos abaixo alimentam as respostas do chatbot. Adicione editais novos ou remova os desatualizados.")

uploaded = st.file_uploader("Adicionar novo PDF", type=["pdf"], accept_multiple_files=True)
if uploaded:
    for arquivo in uploaded:
        destino = PASTA_DOCS / arquivo.name
        with open(destino, "wb") as f:
            f.write(arquivo.getbuffer())
    st.success(f"{len(uploaded)} arquivo(s) salvo(s). Clique em 'Reindexar' para atualizar o chatbot.")
    st.rerun()

col1, col2 = st.columns([1, 3])
with col1:
    reindexar = st.button("🔄 Reindexar documentos", type="primary", use_container_width=True)

if reindexar:
    with st.spinner("Reindexando... isso pode levar alguns minutos."):
        try:
            resultado = indexar_documentos(pasta_docs=str(PASTA_DOCS), pasta_faiss=PASTA_FAISS)
            st.success(f"✅ Reindexado! {resultado['n_arquivos']} arquivo(s), {resultado['n_chunks']} chunks.")
            st.cache_resource.clear()
        except Exception as e:
            st.error(f"❌ Erro ao reindexar: {e}")

st.divider()
st.subheader("Documentos atuais")

arquivos = sorted(PASTA_DOCS.glob("*.pdf"))

if not arquivos:
    st.info("Nenhum PDF encontrado. Adicione documentos acima.")
else:
    for arquivo in arquivos:
        col_nome, col_tam, col_del = st.columns([5, 2, 1])
        with col_nome:
            st.markdown(f"📄 **{arquivo.name}**")
        with col_tam:
            tamanho_mb = arquivo.stat().st_size / (1024 * 1024)
            st.caption(f"{tamanho_mb:.1f} MB")
        with col_del:
            chave_confirmacao = f"confirmar_del_{arquivo.name}"
            if st.session_state.get(chave_confirmacao):
                if st.button("Confirmar?", key=f"confirma_{arquivo.name}"):
                    arquivo.unlink()
                    st.session_state.pop(chave_confirmacao)
                    st.rerun()
            else:
                if st.button("🗑️", key=f"del_{arquivo.name}", help="Remover"):
                    st.session_state[chave_confirmacao] = True
                    st.rerun()

    st.caption(f"Total: {len(arquivos)} documento(s). Após adicionar ou remover, clique em 'Reindexar documentos'.")
