import os
from pathlib import Path

# Conteúdo dos arquivos
files = {
    "app.py": '''import streamlit as st
import os
from pathlib import Path
import pickle
import tempfile
from utils.document_processor import processar_documentos
from utils.rag_engine import RAGEngine

# Configuração da página
st.set_page_config(
    page_title="IFTuring - Assistente Virtual",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #006400 0%, #008000 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 5px solid #7b1fa2;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização do estado da sessão
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'documentos_carregados' not in st.session_state:
    st.session_state.documentos_carregados = False

# Sidebar para configurações
with st.sidebar:
    st.image("https://ifrs.edu.br/wp-content/uploads/2022/03/logo-ifrs-verde.png", width=200)
    st.title("⚙️ Configurações")
    
    # Upload de documentos
    st.subheader("📄 Documentos")
    uploaded_files = st.file_uploader(
        "Faça upload dos PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Arraste ou selecione os editais do IFRS"
    )
    
    if uploaded_files:
        if st.button("🔄 Processar Documentos", use_container_width=True):
            with st.spinner("Processando documentos..."):
                temp_dir = tempfile.mkdtemp()
                pdf_paths = []
                
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(temp_path)
                
                try:
                    documentos = processar_documentos(pdf_paths)
                    
                    if documentos:
                        st.session_state.rag_engine = RAGEngine(documentos)
                        st.session_state.documentos_carregados = True
                        
                        os.makedirs("faiss_index", exist_ok=True)
                        with open("faiss_index/store.pkl", "wb") as f:
                            pickle.dump(documentos, f)
                        
                        st.success(f"✅ {len(documentos)} chunks processados!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
    
    if st.button("📂 Carregar Índice Salvo", use_container_width=True):
        try:
            with open("faiss_index/store.pkl", "rb") as f:
                documentos = pickle.load(f)
                st.session_state.rag_engine = RAGEngine(documentos)
                st.session_state.documentos_carregados = True
                st.success("✅ Índice carregado!")
                st.rerun()
        except FileNotFoundError:
            st.error("⚠️ Índice não encontrado")
    
    st.divider()
    
    # Status do sistema
    st.subheader("📊 Status")
    if st.session_state.documentos_carregados:
        st.success("Sistema Online")
    else:
        st.warning("Sistema Offline")
    st.metric("Mensagens", len(st.session_state.chat_history))
    
    st.divider()
    
    # Informações do projeto
    st.markdown("""
    ### 📚 IFTuring
    Assistente Virtual para o Processo Seletivo do IFRS Campus Ibirubá
    
    **Projeto:** PVH4343-2025  
    **Vigência:** 2025-2026
    """)

# Página principal
st.markdown("""
<div class="main-header">
    <h1>🤖 IFTuring - Assistente Virtual</h1>
    <p style="font-size: 1.2rem;">Processo Seletivo IFRS Campus Ibirubá</p>
</div>
""", unsafe_allow_html=True)

# Área do chat
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Exibir histórico
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Você:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 IFTuring:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input do usuário
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Digite sua pergunta:",
            key="user_input",
            height=100,
            placeholder="Ex: Quais são os documentos necessários para isenção da taxa?"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            submit_button = st.form_submit_button("📤 Enviar", use_container_width=True)
        with col_btn2:
            clear_button = st.form_submit_button("🗑️ Limpar", use_container_width=True)
    
    if submit_button and user_input.strip():
        if not st.session_state.documentos_carregados:
            st.warning("⚠️ Carregue os documentos primeiro!")
        else:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.spinner("🤔 Buscando resposta..."):
                try:
                    resposta = st.session_state.rag_engine.consultar(user_input)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resposta
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

with col2:
    st.subheader("❓ Perguntas Frequentes")
    
    faq_examples = [
        "Como funciona o sistema de cotas?",
        "Qual o prazo para solicitar isenção da taxa?",
        "Quais documentos para matrícula?",
        "Como funciona o processo seletivo?",
        "Quem tem direito à isenção?"
    ]
    
    for i, question in enumerate(faq_examples):
        if st.button(f"{i+1}. {question}", use_container_width=True, key=f"faq_{i}"):
            if st.session_state.documentos_carregados:
                with st.spinner("Buscando..."):
                    resposta = st.session_state.rag_engine.consultar(question)
                    st.session_state.chat_history = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": resposta}
                    ]
                    st.rerun()

# Rodapé
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>IFTuring - Projeto de Pesquisa, Ensino e Extensão do IFRS Campus Ibirubá</p>
    <p>Código SIGAA: PVH4343-2025 | Vigência: abril a dezembro de 2026</p>
</div>
""", unsafe_allow_html=True)
''',
    
    "utils/__init__.py": '',
    
    "utils/rag_engine.py": '''from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3:latest")

class RAGEngine:
    def __init__(self, documentos):
        self.document_store = InMemoryDocumentStore()
        self.document_store.write_documents(documentos)

        self.pipeline = Pipeline()
        self.pipeline.add_component('embedder', SentenceTransformersTextEmbedder(
            model='intfloat/multilingual-e5-base'
        ))
        self.pipeline.add_component('retriever', InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        ))
        self.pipeline.connect('embedder.embedding', 'retriever.query_embedding')

    def consultar(self, pergunta):
        resultado = self.pipeline.run({'embedder': {'text': pergunta}})
        docs = resultado['retriever']['documents']

        if not docs:
            return '⚠️ Nenhum trecho relevante encontrado nos documentos do processo seletivo.'

        contexto = '\\n\\n---\\n\\n'.join([d.content for d in docs])
        return self._perguntar_ollama(pergunta, contexto)

    def _perguntar_ollama(self, pergunta, contexto):
        prompt = f"""Você é um assistente especializado nos documentos relacionados ao processo seletivo do IFRS Campus Ibirubá.
Responda à pergunta abaixo usando APENAS o contexto fornecido.

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODELO_OLLAMA, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 1024}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("response", "")
''',
    
    "utils/document_processor.py": '''from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

def processar_documentos(pdf_paths):
    if not pdf_paths:
        return []
    
    document_store = InMemoryDocumentStore()
    
    pipeline = Pipeline()
    pipeline.add_component('converter', PyPDFToDocument())
    pipeline.add_component('cleaner', DocumentCleaner())
    pipeline.add_component('splitter', DocumentSplitter(
        split_by='word',
        split_length=150,
        split_overlap=20
    ))
    pipeline.add_component('embedder', SentenceTransformersDocumentEmbedder(
        model='intfloat/multilingual-e5-base'
    ))
    pipeline.add_component('writer', DocumentWriter(document_store=document_store))
    
    pipeline.connect('converter', 'cleaner')
    pipeline.connect('cleaner', 'splitter')
    pipeline.connect('splitter', 'embedder')
    pipeline.connect('embedder', 'writer')
    
    pipeline.run({'converter': {'sources': pdf_paths}})
    
    return document_store.filter_documents()
''',
    
    "requirements.txt": '''streamlit>=1.28.0
haystack-ai>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
requests>=2.31.0
pypdf>=3.0.0
pydantic>=2.0.0
torch>=2.0.0
numpy>=1.24.0
''',
    
    ".env.example": '''# URL do servidor Ollama
OLLAMA_URL=http://localhost:11434

# Modelo Ollama a utilizar
MODELO_OLLAMA=llama3:latest

# Configurações do modelo de embeddings
MODEL_NAME=intfloat/multilingual-e5-base
''',
    
    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
venv/
env/

# Dados
faiss_index/
data/
*.pkl
*.pdf

# Configurações
.env
.streamlit/secrets.toml

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
'''
}

# Criar estrutura de diretórios
base_dir = Path("chatbotIFRS_streamlit")
directories = [
    "utils",
    "data",
    "faiss_index"
]

for dir_path in directories:
    full_path = base_dir / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Criado diretório: {full_path}")

# Criar arquivos
for file_path, content in files.items():
    full_path = base_dir / file_path
    
    # Criar diretório pai se não existir
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Criado arquivo: {full_path}")
