# Ponto de entrada da aplicação IF Turing.
#
# Execute com:
#   streamlit run main.py --server.port 8504
#
# Rotas disponíveis:
#   /app   → interface pública (padrão, redireciona / para cá)
#   /admin → painel administrativo

import streamlit as st

st.set_page_config(
    page_title="IF Turing — IFRS Ibirubá",
    page_icon="🎓",
    layout="wide",
)

pagina_app        = st.Page("app_publico.py",    title="IF Turing",   icon="🎓", url_path="app",        default=True)
pagina_admin      = st.Page("admin_page.py",     title="Admin",       icon="🔧", url_path="admin")
pagina_documentos = st.Page("documentos_page.py", title="Documentos", icon="📁", url_path="documentos")

pg = st.navigation([pagina_app, pagina_admin, pagina_documentos], position="hidden")
pg.run()
