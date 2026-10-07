# IF Turing — Auditoria do Pipeline RAG

> **Status:** Parte 1 — Mapeamento documentado + correções de riscos aplicadas  
> **Última reindexação:** 30/09/2025 (conforme mtime de `chroma_data/`)  
> **Índice ativo:** `./chroma_data/` (ChromaDB embedded, coleção `ifturing`)  
> **⚠️ Índice dessincronizado:** o edital de 2026 foi removido de `documentos/` mas seus 448 chunks ainda constam no índice. Reindexar para limpar.

---

## 1. Documentos indexados

A cada `python indexar.py` o índice é **destruído e recriado do zero** (`shutil.rmtree` antes de reconstituir).

**Estado atual do disco (`documentos/`):** 1 PDF

| Arquivo | Conteúdo | Modificado |
|---|---|---|
| `EDITAL Nº XX-2026 … 2027 (1).pdf` | Edital processo seletivo 1º/2027 (vigente) | 22/09/2026 |

**Estado atual do índice ChromaDB (30/09/2025 — stale):** 2 PDFs indexados

| Arquivo no índice | Chunks texto | Chunks tabela | Total | Em disco? |
|---|---|---|---|---|
| `EDITAL Nº XX-2026 … 2027 (1).pdf` | 914 | 9 | **923** | ✅ |
| `EDITAL-No-78-2025 … 2026-1.pdf` | 439 | 9 | **448** | ❌ removido |
| **Total no índice** | **1 353** | **18** | **1 371** | |

> Após reindexação o total esperado é **~923 chunks** (só o edital 2027, 9 tabelas).

Subconjunto relevante pré-carregado em memória:
- **25 chunks de cronograma** — chunks de texto que contêm a palavra "cronograma" **ou** ≥ 3 datas no formato `dd/mm/202X` (regex `\d{1,2}/\d{2}/202[4-9]`)
- **18 chunks de tabela** — todos os chunks com `meta["tipo"] == "tabela"`

### Parâmetros de chunking (texto normal)

| Parâmetro | Valor |
|---|---|
| Modelo de embedding | `intfloat/multilingual-e5-base` |
| Estratégia de split | por palavras (`split_by="word"`) |
| Tamanho do chunk | 150 palavras |
| Overlap | 20 palavras |
| Top-K retrieval | 10 |

Tabelas **não são divididas**: cada linha da tabela vira um chunk próprio, sem
passar pelo splitter.

### Chunks de tabela indexados (dados reais, 30/09/2025)

**Edital 2027 (9 chunks):**

```
Curso: Técnico em Agropecuária         | Turnos: Manhã e Tarde | Duração: 3*  sem | Vagas: 32
Curso: Técnico em Informática          | Turnos: Manhã e Tarde | Duração: 3*  sem | Vagas: 32
Curso: Técnico em Mecânica             | Turnos: Manhã e Tarde | Duração: 3*  sem | Vagas: 32
Curso: Técnico em Eletrotécnica**      | Turnos: Noite         | Duração: 4   sem | Vagas: 30
Curso: Técnico em Mecânica             | Turnos: Noite         | Duração: 4*  sem | Vagas: 30
Curso: Bacharelado em Agronomia        | Turnos: Manhã e Tarde | Duração: 10* sem | Vagas: 30
Curso: Bacharelado em Ciência da Comp. | Turnos: Noite         | Duração: 9   sem | Vagas: 30
Curso: Bacharelado em Engenharia Mec.  | Turnos: Noite         | Duração: 10* sem | Vagas: 30
Curso: Licenciatura em Matemática**    | Turnos: Noite         | Duração: 8*  sem | Vagas: 30
```

**Edital 2026 (9 chunks — mesmos cursos, anos diferentes):**

```
Curso: Técnico em Agropecuária         | Turnos: Manhã e Tarde | Duração: 3*  sem | Vagas: 32
Curso: Técnico em Informática          | Turnos: Manhã e Tarde | Duração: 3*  sem | Vagas: 32
Curso: Técnico em Mecânica             | Turnos: Manhã e Tarde | Duração: 3*  sem | Vagas: 32
Curso: Técnico em Mecânica             | Turnos: Noite         | Duração: 4*  sem | Vagas: 30
Curso: Técnico em Eletrotécnica        | Turnos: Noite         | Duração: 4*  sem | Vagas: 30
Curso: Bacharelado em Agronomia        | Turnos: Manhã e Tarde | Duração: 10* sem | Vagas: 30
Curso: Bacharelado em Ciência da Comp. | Turnos: Noite         | Duração: 9   sem | Vagas: 30
Curso: Bacharelado em Engenharia Mec.  | Turnos: Noite         | Duração: 10* sem | Vagas: 30
Curso: Licenciatura em Matemática**    | Turnos: Noite         | Duração: 8*  sem | Vagas: 30
```

> **⚠️ Risco de duplicidade:** Os dois editais têm quase os mesmos cursos.
> Quando a pergunta é sobre "cursos disponíveis", o contexto enviado ao LLM
> pode conter 18 linhas (2×9) referentes ao mesmo conjunto de cursos,
> aumentando o risco de listas repetidas ou contagens dobradas.

---

## 2. Prompt enviado ao LLM

Definido em `nucleo.py:448–463`, função `perguntar_llm()`.

### Template completo

```
Você é um assistente especializado nos documentos do processo seletivo do IFRS.
ESCOPO: Todas as informações fornecidas no contexto abaixo referem-se exclusivamente
ao IFRS – Campus Ibirubá (Instituto Federal do Rio Grande do Sul, Campus Ibirubá).
Ao responder, trate o contexto como sendo sempre sobre esse campus específico —
não presuma nem mencione outros campi do IFRS, a menos que o usuário pergunte
explicitamente sobre outro campus.
Responda à pergunta usando SOMENTE as informações do contexto abaixo.
Regras obrigatórias:
- Responda diretamente, em português, sem introduções nem rótulos como "Resposta:",
  "Inferência:" ou similares.
- Nunca comente sobre como a resposta foi construída, deduzida ou inferida.
- COMPLETUDE OBRIGATÓRIA: ao listar cursos, percorra TODO o contexto e inclua TODOS
  os cursos do tipo perguntado. Cada linha que começa com "Curso:" é um curso diferente
  — não pule nem omita nenhum. Só encerre a lista depois de verificar todas as linhas
  "Curso:" presentes.
- Use exatamente os nomes, turnos e dados que aparecem no contexto — não invente nem
  complete informações ausentes.
- Cada linha do contexto é um registro independente: use APENAS os dados que aparecem
  juntos na MESMA linha. Nunca combine o nome de um curso com turno, duração ou vagas
  de uma linha diferente.
- Se o contexto trouxer cursos de categorias diferentes (técnico integrado, técnico
  subsequente, superior), responda apenas com os cursos que pertencem à categoria
  perguntada, sem misturar categorias.
- Se a informação não estiver no contexto, responda apenas:
  "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}
```

### Exemplo preenchido (pergunta sobre cursos técnicos)

**Pergunta:** "Quais são os cursos técnicos disponíveis no campus Ibirubá?"

**Contexto enviado (montado pelo pipeline):**
```
Curso: Técnico em Agropecuária | Turnos: Manhã e Tarde | Duração (semestres): 3* | Total de Vagas: 32

---

Curso: Técnico em Informática | Turnos: Manhã e Tarde | Duração (semestres): 3* | Total de Vagas: 32

---

[… demais chunks …]
```

**Parâmetros Ollama:**
```json
{
  "model": "qwen3:0.6b",
  "stream": false,
  "think": false,
  "options": { "temperature": 0.0, "num_predict": 1024, "num_ctx": 8192 }
}
```

---

## 3. Fluxo de retrieval passo a passo

```
Pergunta do usuário
       │
       ▼
[1] Embedding da query
    modelo: intfloat/multilingual-e5-base
    componente: SentenceTransformersTextEmbedder
    saída: vetor de 768 dimensões
       │
       ▼
[2] Busca semântica (ChromaEmbeddingRetriever)
    top_k = 10
    documento_store: ChromaDocumentStore (embedded, sem servidor)
    path: ./chroma_data/, coleção "ifturing"
    saída: lista de até 10 docs com scores de similaridade cosseno
       │
       ▼
[3] Pós-filtro de cotas
    regex: C\d{1,2}:
    remove chunks de texto que descrevem distribuição de cotas (C1:, C2: …)
    NÃO remove tabelas (meta["tipo"] == "tabela" é excluído do filtro)
       │
       ▼
[4] Retrieval híbrido de tabelas (por palavra-chave)
    dispara se a pergunta contém alguma palavra de _PALAVRAS_TABELA
    lógica dual:
      a) tipo-de-curso: "superiores" → expande para {licenciatura, bacharelado, superior}
      b) nome-específico: "informática" → busca literal em content.lower()
    injeta chunks de tabela não-semânticos (não presentes no top-10)
    ao injetar tabelas: textos são limitados a 6 chunks para preservar contexto
    limite hard: máximo 20 tabelas injetadas ([:20])
       │
       ▼
[5] Retrieval híbrido de cronograma (por palavra-chave)
    dispara se a pergunta contém alguma palavra de _PALAVRAS_CRONOGRAMA
    busca nos 25 chunks de cronograma pré-carregados
    injeta os que não estão já no contexto semântico
    posição: cronograma vai ANTES dos demais (LLM lê datas primeiro)
       │
       ▼
[6] Montagem do contexto final
    ordem: [cronograma] + [tabelas já no top-k] + [tabelas injetadas] + [texto, máx 6]
    separador: "\n\n---\n\n"
    tamanho típico: 2.000–8.000 chars
       │
       ▼
[7] Chamada ao LLM (Ollama local ou Maritaca)
    via nucleo.perguntar_llm() → nucleo.chamar_ollama() / nucleo.chamar_maritaca()
    temperatura: 0.0 (determinístico)
    num_ctx: 8192 tokens
    num_predict: 1024 tokens (limite de resposta)
       │
       ▼
Resposta + métricas → UI + ifturing.db (SQLite)
```

### Falhas conhecidas por etapa (histórico de bugs reais)

| Etapa | Bug corrigido | Commit |
|---|---|---|
| [1] Extração do PDF | Regex `campus\s+ibirub[aá]` não detectava variantes maiúsculas → páginas gerais descartadas | `a15c3b9` |
| [2] Indexação tabelas | Cabeçalho multi-nível gerava valores `C1`, `C2`, `C3` nas colunas erradas | `13a3f95` |
| [2] Indexação tabelas | Tabelas de outros campi (Bento Gonçalves, Caxias etc.) estavam sendo incluídas | `08cf502` |
| [2] Indexação texto | Parágrafos de distribuição de cotas poluíam o índice semântico | `5623270` |
| [4] Híbrido tabelas | Match literal `"técnicos"` não encontrava chunks com `"técnico"` | `960f2b3` |
| [4] Híbrido tabelas | Chunks de texto sem relação eram trazidos em excesso, ocupando contexto | `2b1ba90` |
| [5] Híbrido cronograma | Chunks de continuação (sem a palavra "cronograma") não eram capturados | `79b6f5d` |
| [7] LLM | `temperature` era `0.3` em `avaliar.py` e não estava definido explicitamente no Ollama → respostas não determinísticas | `2065790` |
| [7] LLM | Qwen3 emite tokens `<think>` que ocupam `eval_count` mas ficam em campo separado → `response` chegava vazio | `2065790` / `19859f5` |
| [7] Prompt | Sem escopo explícito de campus → LLM misturava dados de outros campi do IFRS | `e67e09b` |
| [7] Prompt | Sem regra de separação de categorias → LLM misturava cursos técnicos com superiores | `e7fa6ef` |
| DB | FK constraint ao inserir consulta antes de registrar modelo → falha silenciosa | `664f3cc` |

### Riscos atuais não corrigidos

| Área | Problema |
|---|---|
| `avaliar.py` | Usa `top_k=5`, `temperature=0.3`, sem hybrid retrieval e prompt diferente do produção → **resultados RAGAS não refletem o pipeline real** |
| `avaliar.py` | `num_ctx` não configurado explicitamente → pode usar padrão do Ollama (2048 tokens), diferente dos 8192 da produção |
| Índice | Dois editais com mesmos cursos duplicam as tabelas → risco de LLM contar cursos em dobro ou misturar anos |
| `_buscar_tabelas_por_keyword` | Limite hard de 20 tabelas pode silenciosamente truncar o contexto em perguntas amplas |
| `.env.example` | Ainda referencia `PASTA_FAISS` (obsoleto) em vez de `PASTA_CHROMA` |
| `carregar_pipeline()` | Retorna tupla de 4 elementos, mas `app_publico.py:540` e `admin_page.py:70` desempacotam com `pipeline, n_chunks, *_` — correto, mas `n_chunks` deveria refletir o estado **após** a reindexação, não só ao carregar (o valor fica em cache `st.cache_resource` indefinidamente) |
| Embedder de consulta (`nucleo.py`) | CUDA OOM: GPU compartilhada com outros processos; sem `device="cpu"` o `SentenceTransformersTextEmbedder` pode tentar alocar na GPU e falhar → **mitigado** (ver R8) |
| Modelos pequenos (0.6B) | Qwen3 0.6B lista datas sem rótulo ou duplicadas em perguntas de cronograma — **em investigação** |

---

## 4. Gabarito de referência (conjunto QA de regressão)

**Fonte:** extraído diretamente dos chunks de tabela e cronograma indexados em 30/09/2025.  
**Referência temporal:** perguntas sem qualificador de ano referem-se ao edital mais recente
(Edital Nº XX-2026, processo seletivo 1º/2027).

As respostas esperadas devem ser validadas contra os PDFs originais antes de usar
em produção — especialmente as datas, que podem mudar de versão para versão do edital.

---

### Bloco A — Cursos técnicos

| # | Pergunta | Resposta esperada (palavras-chave obrigatórias) |
|---|---|---|
| A01 | Quais são os cursos técnicos disponíveis no campus Ibirubá? | Agropecuária, Informática, Mecânica, Eletrotécnica (4 nomes distintos) |
| A02 | Quais cursos técnicos são oferecidos no turno da manhã? | Agropecuária, Informática, Mecânica (Manhã e Tarde) |
| A03 | Quais cursos técnicos são oferecidos no turno da noite? | Eletrotécnica, Mecânica (Noite) |
| A04 | Quantas vagas tem o curso Técnico em Agropecuária? | 32 vagas |
| A05 | Qual é a duração do curso Técnico em Informática? | 3 semestres |
| A06 | O campus oferece Técnico em Eletrotécnica? Qual é o turno e quantas vagas? | Sim, turno Noite, 30 vagas, 4 semestres |
| A07 | Quantas vagas têm os cursos técnicos noturnos no total? | 60 vagas (Eletrotécnica 30 + Mecânica noite 30) |
| A08 | O Técnico em Mecânica é oferecido em quantos turnos? | Dois turnos: Manhã e Tarde (3 sem, 32 vagas) e Noite (4 sem, 30 vagas) |

---

### Bloco B — Cursos superiores

| # | Pergunta | Resposta esperada (palavras-chave obrigatórias) |
|---|---|---|
| B01 | Quais são os cursos superiores oferecidos no campus Ibirubá? | Agronomia, Ciência da Computação, Engenharia Mecânica, Matemática (4 nomes) |
| B02 | Quais são os bacharelados disponíveis? | Agronomia, Ciência da Computação, Engenharia Mecânica (3 cursos) |
| B03 | Há alguma licenciatura no campus Ibirubá? | Sim, Licenciatura em Matemática |
| B04 | Qual é a duração do Bacharelado em Ciência da Computação? | 9 semestres |
| B05 | Qual é o turno do Bacharelado em Agronomia e quantas vagas? | Manhã e Tarde, 30 vagas |
| B06 | Quantas vagas tem a Licenciatura em Matemática? | 30 vagas |
| B07 | Qual é a duração do Bacharelado em Engenharia Mecânica? | 10 semestres |
| B08 | Quais cursos superiores são oferecidos no período noturno? | Ciência da Computação, Engenharia Mecânica, Matemática (3 cursos) |
| B09 | Quais são os cursos superiores do campus Ibirubá? | Agronomia, Ciência da Computação, Engenharia Mecânica, Matemática (4 cursos) |
| B10 | Qual é a duração do Bacharelado em Ciência da Computação e da Licenciatura em Matemática? | Ciência da Computação: 9 semestres; Matemática: 8 semestres |
| B11 | Agronomia e Engenharia Mecânica são bacharelados? | Sim, ambos são Bacharelados |
| B12 | Quantos cursos superiores existem no campus Ibirubá? | 4 cursos superiores |

---

### Bloco C — Datas e cronograma (Edital 2027, datas de 2026)

| # | Pergunta | Resposta esperada (palavras-chave obrigatórias) |
|---|---|---|
| C01 | Quando são as inscrições para o processo seletivo? | 30/09 a 19/10/2026 |
| C02 | Quando acontece a aplicação das provas? | 06/12/2026 |
| C03 | Qual é o prazo para recurso após as provas? | 07 a 08/12/2026 |
| C04 | Quando são publicados os locais de prova? | 27/11/2026 |
| C05 | Quando é publicada a Classificação Geral e Aprovados? | 05/01/2027 |
| C06 | Quando é a Chamada Pública? | 27/01/2027 |
| C07 | Qual é a data limite para resultado final da matrícula na Chamada Pública? | até 03/02/2027 |

---

### Bloco D — Regras gerais (espera-se "Não encontrei" ou resposta negativa)

| # | Pergunta | Resposta esperada |
|---|---|---|
| D01 | O campus Ibirubá oferece curso de Direito? | "Não encontrei essa informação nos documentos." |
| D02 | Qual é o valor da taxa de inscrição? | Resposta deve mencionar isenção ou ausência de taxa; **não deve** inventar um valor |
| D03 | Há curso de Tecnologia em Análise e Desenvolvimento de Sistemas? | "Não encontrei essa informação nos documentos." |

---

### Casos de borda críticos (verificação de qualidade)

| # | Pergunta | O que verificar |
|---|---|---|
| E01 | Quantos cursos técnicos existem no total? | Deve dizer **4 nomes distintos** (não 5 entradas), ou distinguir claramente que Mecânica aparece em dois turnos |
| E02 | Liste todos os cursos do campus Ibirubá | Deve listar os 4 técnicos + 4 superiores sem repetir nenhum e sem misturar categorias |
| E03 | Quais documentos preciso para me inscrever? | Deve responder com base no edital; **não deve** listar documentos de outros editais misturados |
| E04 | Quais são as datas do processo seletivo? (sem especificar ano) | Deve responder preferencialmente com as datas de 2026/2027 (edital mais recente); **não deve** misturar com datas de 2025 (edital 78-2025) |

---

## 5. Instruções para uso do gabarito em QA manual

1. Acesse o chatbot (modo simples ou admin).
2. Para cada pergunta acima, verifique:
   - A resposta contém **todas** as palavras-chave marcadas como obrigatórias?
   - A resposta **não contém** nenhuma informação inventada?
   - No caso de listagens (bloco A/B), a lista está **completa** (nenhum curso omitido)?
   - Nos casos do bloco D, o bot respondeu "Não encontrei..." sem inventar dados?
3. Registre: PASS / FAIL / PARCIAL com comentário livre.
4. Divergências devem ser reportadas no arquivo `divergencias_qa.md` com: pergunta, resposta obtida, resposta esperada, data, modelo usado.

---

## 6. Riscos corrigidos (pós-auditoria)

| # | Risco | Status | O que foi feito |
|---|---|---|---|
| R1 | `avaliar.py` usava configuração diferente da produção | ✅ Corrigido | Reescrito para importar `_PADRAO_COTA`, `_PADRAO_DATA_RECENTE`, `_buscar_tabelas_por_keyword`, `_buscar_cronograma_por_keyword` e `perguntar_llm` de `nucleo.py`. Pipeline próprio (sem Streamlit) mas com `top_k=TOP_K` (10), `temperature=0.0`, hybrid retrieval de tabelas e cronograma, filtro de cotas e prompt de produção. Expõe `contexto` no retorno para uso nas métricas RAGAS. |
| R2 | `.env.example` referenciava `PASTA_FAISS` (obsoleto) | ✅ Corrigido | Substituído por `PASTA_CHROMA=./chroma_data` |
| R3 | Índice stale com dados do edital de 2026 removido | ✅ Corrigido | Reindexação completa a partir do único edital vigente (`documentos/`). Índice final: 405 chunks de texto + 9 de tabela = 414 total. |
| R4 | Datas ambíguas (2025 e 2026 misturadas) | ✅ Corrigido | Resolvido como consequência da reindexação (R3). |
| R5 | `_RE_CAMPUS_OUTRO` com `[:(]` causava falso positivo em texto inline | ✅ Corrigido | `indexar.py`: `_RE_CAMPUS_OUTRO` (extrator de texto) agora exige só `:`. Criada `_RE_CAMPUS_OUTRO_TABELA` com `{0,2}` palavras e `[:(]` para suportar os dois formatos de cabeçalho de annexo presentes no PDF. |
| R6 | Extrator de tabelas capturava 54 chunks de outros campi | ✅ Corrigido | Consequência direta de R5. Após a correção, apenas as 4 páginas de annexo do Campus Ibirubá são mantidas, produzindo os 9 chunks esperados (Técnicos + Superiores). |
| R7 | Whitelist `_CURSOS_IBIRUBA` incompleta (apenas Técnicos) | ✅ Corrigido | O filtro de auditoria (Opção C) detectou automaticamente 4 cursos superiores legítimos de Ibirubá (Bacharelado em Agronomia, Ciência da Computação, Engenharia Mecânica; Licenciatura em Matemática) que não estavam na whitelist. Adicionados ao frozenset. |
| R8 | CUDA OOM no embedder de consulta | ✅ Corrigido | `SentenceTransformersTextEmbedder` em `nucleo.carregar_pipeline()` agora lê a variável de ambiente `EMBEDDER_DEVICE` (default `"cpu"`), evitando alocação na GPU compartilhada. Indexador (`SentenceTransformersDocumentEmbedder` em `indexar.py`) não foi alterado — usa classe diferente e roda em contexto exclusivo. |

---

## 7. Próximos passos (Partes 2–4)

- **Parte 2:** Avaliar se ChromaDB persiste corretamente entre reinícios; comparar Chroma+SQLite vs alternativas para camada administrativa.
- **Parte 3:** Proposta e mockup/implementação de interface de QA para Lourenço/Vitória.
- **Parte 4:** Script `teste_regressao.py` que executa o gabarito acima automaticamente e gera relatório pass/fail.
