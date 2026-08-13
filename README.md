# 🎓 IF Turing — Chatbot RAG para o Processo Seletivo do IFRS Campus Ibirubá

Chatbot baseado em RAG (Retrieval-Augmented Generation) que responde dúvidas sobre o processo seletivo do IFRS Campus Ibirubá a partir dos PDFs oficiais. Desenvolvido como produto de extensão e como objeto de pesquisa para comparação de LLMs, com resultados direcionados a um artigo submetido ao **ERBD (Escola Regional de Banco de Dados)**.

> Este projeto está em desenvolvimento ativo como parte de um projeto de extensão e pesquisa do IFRS Campus Ibirubá.

---

## 📋 Sumário

- [Sobre o projeto](#sobre-o-projeto)
- [Por que apenas modelos locais](#por-que-apenas-modelos-locais)
- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Configuração de modelos locais (Ollama)](#configuração-de-modelos-locais-ollama)
- [Golden Dataset](#golden-dataset)
- [Métricas e avaliação](#métricas-e-avaliação)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Contribuindo](#contribuindo)
- [Equipe](#equipe)
- [Licença](#licença)

---

## Sobre o projeto

O **IF Turing** tem dois objetivos integrados:

**Produto:** um assistente de chat que ajuda candidatos a tirar dúvidas sobre o processo seletivo do IFRS Campus Ibirubá — cursos ofertados, editais, documentos necessários, cotas, prazos e taxas de inscrição. As respostas são geradas exclusivamente a partir dos PDFs oficiais indexados, reduzindo alucinações.

**Pesquisa:** uma plataforma instrumentada para comparar diferentes configurações de LLM e estratégias de retrieval, medindo tanto desempenho (latência decomposta, throughput, consumo de tokens) quanto qualidade das respostas (métricas RAGAS contra um Golden Dataset de 100 perguntas elaborado pela equipe). Os dados coletados alimentam o artigo submetido ao ERBD.

---

## Por que apenas modelos locais

O projeto foi deliberadamente limitado a modelos executados localmente via Ollama. Essa decisão tem quatro razões principais:

**Comparação justa de hardware.** Os três modelos (LLaMA 3 8B, Mistral 7B e Gemma 2 27B) rodam no mesmo servidor do IFRS, sob as mesmas condições de CPU, GPU e RAM. Isso torna a comparação de latência e throughput controlada e replicável. Um modelo em nuvem mediria essencialmente a velocidade do datacenter do provedor externo — não do ambiente que o IFRS efetivamente opera.

**Reprodutibilidade e independência.** Modelos locais não dependem de disponibilidade externa, rate limits, políticas de uso de terceiros ou mudanças de contrato. Qualquer colaborador com acesso ao servidor de pesquisa consegue replicar os experimentos na íntegra.

**Sem custo por token.** A avaliação completa (100 perguntas × 3 modelos = 300 chamadas) e o uso contínuo em produção não geram custos variáveis, aproveitando o servidor de pesquisa já disponível na instituição (64 GB RAM, GPU dedicada).

**Isolamento da variável "modelo".** O artigo compara especificamente o impacto da escolha do LLM mantendo fixos o retriever e os embeddings. Rodar todos os modelos em ambiente equivalente isola essa variável e fortalece as conclusões.

O projeto não depende de nenhuma API externa paga. Toda a inferência é executada na infraestrutura local do IFRS via Ollama.

---

## 🚀 Funcionalidades

- **Chat RAG** com respostas fundamentadas nos PDFs oficiais do processo seletivo
- **Dois modos de interface** selecionáveis em tempo real:
  - Modo usuário final — chat limpo, voltado ao candidato, sem exposição de métricas
  - Modo análise detalhada — sidebar com seleção de modelo, métricas por consulta e exportação de dados
- **Suporte a 3 LLMs locais**, trocáveis em runtime sem reiniciar o app:
  - LLaMA 3 8B, Mistral 7B e Gemma 2 27B via Ollama (execução local no servidor de pesquisa)
- **Instrumentação completa** por consulta: latência total, latência de retrieval, latência LLM, tokens de entrada/saída, throughput (tokens/s), delta de RAM, scores de similaridade dos chunks recuperados
- **Logs automáticos** em formato JSONL diário, em `logs/`
- **Avaliação automática** com RAGAS contra o Golden Dataset (com fallback para similaridade Jaccard)
- **Geração de tabelas e gráficos** prontos para o artigo (Markdown + PNG)

---

## 🏗️ Arquitetura

### Fluxo de dados

```
PDFs oficiais
    │
    ▼ indexar.py (executa uma vez)
┌─────────────────────────────────────────┐
│  PyPDF → Cleaner → Splitter             │
│  (chunks de 150 palavras, overlap 20)   │
│        ↓                                │
│  SentenceTransformers Embedder          │
│  (intfloat/multilingual-e5-base)        │
│        ↓                                │
│  InMemoryDocumentStore → store.pkl      │
└─────────────────────────────────────────┘
                │
                ▼ app.py (em execução)
┌─────────────────────────────────────────┐
│  Pergunta do usuário                    │
│        ↓                                │
│  Embedder (mesma codificação)           │
│        ↓                                │
│  InMemoryEmbeddingRetriever (top_k=5)   │
│        ↓                                │
│  Contexto (chunks relevantes)           │
│        ↓                                │
│  LLM local via Ollama                   │
│        ↓                                │
│  Resposta + métricas                    │
└─────────────────────────────────────────┘
```

### Scripts e quando usar cada um

| Arquivo | Finalidade | Quando executar |
|---|---|---|
| `indexar.py` | Processa os PDFs e gera o índice vetorial | Uma vez, e a cada atualização dos PDFs |
| `app.py` | Interface Streamlit — modo usuário e modo análise | Sempre que o chatbot precisar estar disponível |
| `avaliar.py` | Avaliação automática com RAGAS contra o Golden Dataset | Para cada configuração de modelo comparada no artigo |
| `analisar_resultados.py` | Consolida dados, gera tabelas Markdown e gráficos PNG | Depois de coletar dados de todos os modelos |
| `create_project.py` | Script de scaffolding da versão inicial | Não é necessário — arquivo legado |

---

## 📦 Pré-requisitos

- Python 3.11 ou superior
- pip
- Git
- Ollama — obrigatório, pois todos os modelos rodam localmente (ver seção [Configuração de modelos locais (Ollama)](#configuração-de-modelos-locais-ollama))

---

## ⚙️ Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/chatbotifrsibiruba/ifturing.git
cd ifturing
```

### 2. Criar e ativar o ambiente virtual

**Linux / macOS / servidor:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Em distribuições Debian/Ubuntu recentes (Python 3.11+), o pip pode recusar instalações fora de um venv devido ao PEP 668. Use sempre o ambiente virtual — nunca `--break-system-packages` em máquinas compartilhadas.

### 3. Instalar as dependências principais

```bash
pip install -r requirements.txt
```

### 4. Instalar dependências opcionais para pesquisa

Para rodar a avaliação RAGAS e gerar os gráficos do artigo:

```bash
pip install ragas datasets pandas openpyxl matplotlib
```

> Sem esse passo, `avaliar.py` funciona com fallback de similaridade Jaccard e `analisar_resultados.py` não encontrará o pandas.

### 5. Configurar as variáveis de ambiente

```bash
# Linux / macOS
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

Abra o arquivo `.env` e preencha os valores:

```dotenv
# Pasta com os PDFs do processo seletivo
PASTA_DOCS=./documentos

# Pasta onde o índice serializado será salvo
PASTA_FAISS=./faiss_index

# URL do servidor Ollama
OLLAMA_URL=http://localhost:11434

# Modelo padrão do Ollama (usado no modo usuário final)
MODELO_OLLAMA=llama3:latest
```


### 6. Adicionar os PDFs

Coloque os PDFs oficiais do processo seletivo na pasta `documentos/`. Qualquer arquivo `.pdf` presente nessa pasta será indexado.

### 7. Indexar os documentos

```bash
python indexar.py
```

Esse processo baixa o modelo de embeddings na primeira execução (aproximadamente 1 GB) e pode levar alguns minutos dependendo do número de PDFs. O índice é salvo em `faiss_index/store.pkl`.

> Repita esse passo sempre que os PDFs forem atualizados.

---

## 🤖 Uso

### Iniciar o chatbot

```bash
streamlit run app.py
```

O app abre automaticamente no navegador em `http://localhost:8501`.

Para acessar de outro dispositivo na mesma rede (útil no servidor do laboratório):

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

**Modo usuário final:** interface limpa com sugestões de perguntas frequentes — indicado para candidatos ao processo seletivo.

**Modo análise detalhada:** clique em "Ver modo análise" no canto superior direito para ativar a sidebar com seleção de modelo, métricas em tempo real e exportação de CSVs da sessão.

---

### Avaliar modelos com o Golden Dataset

Execute uma avaliação para cada modelo que deseja comparar no artigo. Os resultados são acumulados no mesmo CSV a cada execução.

```bash
# LLaMA 3 8B (modelo padrão)
python avaliar.py \
  --dataset golden_dataset.csv \
  --branch main \
  --tipo ollama \
  --modelo llama3

# Mistral 7B
python avaliar.py \
  --dataset golden_dataset.csv \
  --branch exp/mistral \
  --tipo ollama \
  --modelo mistral

# Gemma 2 27B
python avaliar.py \
  --dataset golden_dataset.csv \
  --branch exp/gemma27b \
  --tipo ollama \
  --modelo gemma2:27b

# Teste rápido com 10 perguntas
python avaliar.py \
  --dataset golden_dataset.csv \
  --branch main \
  --tipo ollama \
  --modelo llama3 \
  --limite 10
```

Os resultados são salvos em:
- `relatorio_ragas.csv` — uma linha por pergunta avaliada (acumulativo entre execuções)
- `relatorio_ragas_agg.csv` — uma linha por configuração de modelo, pronto para a tabela do artigo

---

### Gerar tabelas e gráficos para o artigo

```bash
python analisar_resultados.py
```

Ou especificando os arquivos manualmente:

```bash
python analisar_resultados.py \
  --ragas relatorio_ragas.csv \
  --ragas-agg relatorio_ragas_agg.csv \
  --logs-dir logs/
```

Os artefatos são gerados em `analise/`:

| Arquivo | Conteúdo |
|---|---|
| `tabela_desempenho.md` | Latência e throughput por modelo (Markdown) |
| `tabela_qualidade.md` | Métricas RAGAS por modelo (Markdown) |
| `grafico_latencia.png` | Barras empilhadas: retrieval vs. LLM por modelo |
| `grafico_throughput.png` | Barras horizontais: tokens/s por modelo |
| `grafico_ragas.png` | Barras agrupadas: métricas RAGAS por modelo |
| `dados_consolidados.csv` | Todos os dados em um CSV único |

---

## 🖥️ Configuração de modelos locais (Ollama)

### Instalar o Ollama

**Linux / servidor:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** baixe o instalador em [ollama.com](https://ollama.com).

### Baixar os modelos

```bash
ollama pull llama3          # LLaMA 3 8B (~4.7 GB)
ollama pull mistral         # Mistral 7B (~4.1 GB)
ollama pull gemma2:27b      # Gemma 2 27B (~16 GB — requer servidor com RAM suficiente)
```

### Verificar se o Ollama está rodando

```bash
curl http://localhost:11434/api/tags
```

Configure `OLLAMA_URL` no `.env` se o Ollama estiver em um servidor diferente:

```dotenv
OLLAMA_URL=http://192.168.1.100:11434
```

---

## 📊 Golden Dataset

O Golden Dataset é um conjunto de 100 perguntas sobre o processo seletivo do IFRS Campus Ibirubá, com respostas esperadas elaboradas manualmente pela equipe a partir dos editais oficiais. É usado como referência para calcular as métricas RAGAS.

**Formato do arquivo CSV (`pergunta,resposta_esperada`):**

```csv
pergunta,resposta_esperada
"Quais são os cursos técnicos integrados ao Ensino Médio?","Os cursos técnicos integrados são Agropecuária, Informática e Mecânica."
"Os cursos do IFRS são pagos?","Não. Todos os cursos do IFRS são 100% gratuitos."
```

O arquivo `golden_dataset.csv` já está incluído no repositório. O script `avaliar.py` também aceita arquivos no formato XLSX (exportados do Google Sheets).

---

## 🧪 Métricas e avaliação

O framework [RAGAS](https://docs.ragas.io) calcula as seguintes métricas, todas no intervalo [0, 1]:

| Métrica | O que mede |
|---|---|
| `faithfulness` | Se a resposta gerada é factualmente consistente com o contexto recuperado (sem alucinações) |
| `answer_relevancy` | Se a resposta é relevante para a pergunta feita |
| `context_precision` | Se os chunks recuperados são relevantes para a pergunta |
| `context_recall` | Se o retrieval conseguiu trazer todos os chunks necessários para responder |
| `answer_correctness` | Similaridade entre a resposta gerada e a resposta esperada do Golden Dataset |

Quando o pacote `ragas` não está instalado, o script `avaliar.py` exibe um aviso e usa similaridade de Jaccard como fallback apenas para `answer_correctness`.

---

## 📁 Estrutura de pastas

```
ifturing/
│
├── app.py                  ← Interface Streamlit (chatbot + modo análise)
├── indexar.py              ← Indexa os PDFs e gera o store.pkl
├── avaliar.py              ← Avaliação automática com RAGAS
├── analisar_resultados.py  ← Gera tabelas e gráficos para o artigo
├── create_project.py       ← Scaffolding da versão inicial (arquivo legado)
│
├── requirements.txt        ← Dependências principais do projeto
├── .env.example            ← Modelo de configuração de variáveis de ambiente
├── .env                    ← Configuração local (não commitado)
├── golden_dataset.csv      ← 100 perguntas do processo seletivo com respostas esperadas
│
├── documentos/             ← Coloque os PDFs do processo seletivo aqui (não commitados)
│
├── faiss_index/            ← Gerado por indexar.py (não commitado)
│   └── store.pkl           ← Índice vetorial serializado (InMemoryDocumentStore)
│
├── logs/                   ← Logs automáticos por sessão, gerados pelo app.py
│   └── log_YYYY-MM-DD.jsonl
│
└── analise/                ← Saída do analisar_resultados.py
    ├── tabela_desempenho.md
    ├── tabela_qualidade.md
    ├── grafico_latencia.png
    ├── grafico_throughput.png
    ├── grafico_ragas.png
    └── dados_consolidados.csv
```

> **Nota sobre o nome `faiss_index/`:** apesar do nome da pasta, o projeto usa `InMemoryDocumentStore` do Haystack serializado com pickle. O pacote `faiss-cpu` está listado nas dependências mas o armazenamento atual não depende de FAISS.

---

## 🤝 Contribuindo

### Convenção de branches

| Branch | Uso |
|---|---|
| `main` | Versão estável — configuração de referência do artigo (LLaMA 3 8B via Ollama) |
| `exp/<nome>` | Experimentos com configurações alternativas (ex: `exp/mistral`, `exp/gemma27b`, `exp/top-k-10`) |

Ao testar uma nova configuração de modelo ou parâmetro de retrieval, crie uma branch `exp/` e rode `avaliar.py` com `--branch exp/<nome>`. O CSV acumulativo permite comparar todas as configurações na mesma análise final.

### Adicionar um novo modelo

1. Adicione o modelo ao dicionário `MODELOS` em `app.py`, seguindo o mesmo schema (tipo, modelo, descricao, params_b, quantizacao)
2. Se for via Ollama, baixe o modelo com `ollama pull <nome>`
3. Execute `avaliar.py` com `--tipo ollama --modelo <nome>` para coletar as métricas
4. Atualize `analisar_resultados.py` se necessário para ajustar os gráficos

---

## 👥 Equipe

Projeto desenvolvido por uma equipe de estudantes e servidores do IFRS Campus Ibirubá, sob orientação de um professor supervisor, com contribuições em desenvolvimento do RAG, integração de modelos, avaliação, conteúdo, site de divulgação e infraestrutura de servidor.

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT — uma licença permissiva e de código aberto, que permite uso, cópia, modificação e distribuição livres, inclusive para fins comerciais, desde que o aviso de copyright original seja mantido. Consulte o arquivo [LICENSE](LICENSE) para os termos completos.