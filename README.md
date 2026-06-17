# 🎓 IF Turing — Chatbot IFRS Campus Ibirubá

Chatbot RAG para tirar dúvidas sobre o processo seletivo do IFRS Campus Ibirubá.

## Tecnologias
- Python + Streamlit
- Haystack (pipeline RAG)
- SentenceTransformers (`intfloat/multilingual-e5-base`)
- Groq API (LLaMA 3.3 70B)

## Estrutura
```
ifturing/
├── app.py            ← app Streamlit (interface do chat)
├── indexar.py        ← script para indexar os PDFs
├── requirements.txt
├── .env.example      ← modelo do arquivo de configuração
├── documentos/       ← coloque os PDFs aqui
└── faiss_index/      ← gerado automaticamente pelo indexar.py
```

## Como executar

### 1. Instalar dependências
```powershell
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
```powershell
# Copiar o arquivo de exemplo
copy .env.example .env

# Abrir e preencher sua GROQ_API_KEY
notepad .env
```

### 3. Adicionar os PDFs
Coloque os PDFs do processo seletivo dentro da pasta `documentos/`.

### 4. Indexar os documentos
```powershell
python indexar.py
```
> ⚠️ Só precisa rodar quando os PDFs mudarem.

### 5. Executar o chatbot
```powershell
streamlit run app.py
```

O app abrirá automaticamente no navegador em `http://localhost:8501`.

---

> As chaves de API não estão no repositório. Configure o arquivo `.env` localmente.
