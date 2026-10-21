# Auditoria da indexação e do retrieval — IF Turing (Fase 1)

Escopo: diagnóstico e ferramenta de avaliação. Nenhum arquivo de código foi alterado (`nucleo.py` e `indexar.py` intactos) e a coleção de produção não foi tocada.

**Como foi medido**
- Este checkout **não tem `./chroma_data`**. A "produção" abaixo é `./chroma_exp_prod`, reconstruída chamando `indexar.indexar_documentos()` sem modificá-lo (176 chunks: 167 texto + 9 tabela).
- O Python do sistema não tinha Haystack; foi usado um venv fora do repositório com as versões resolvidas de `requirements.txt`: haystack-ai 3.4.0, chroma-haystack 5.0.0, sentence-transformers-haystack 1.0.0, sentence-transformers 6.1.0, chromadb 1.5.9, pdfplumber 0.11.10. O `requirements.txt` só fixa mínimos; extração e metadados podem variar na sua versão.
- PDF analisado: `documentos/EDITAL-No-15-2026-…-1.pdf` (120 páginas).
- Resultados do avaliador: `retrieval_resultado.md` e `retrieval_resultado.csv` (ambos no `.gitignore`).

---

## 0. Estado do repositório

HEAD `199c8c2` (branch `main`). Árvore de trabalho antes desta fase: `M .gitignore`, `M PIPELINE.md`, `?? avaliar_retrieval.py` (já existiam, sem commit).

| Item pedido | Existe? | Observação |
|---|---|---|
| Extração estruturada do cronograma ("Data \| Evento") em `indexar.py` | **Não** | O cronograma é texto corrido das páginas 3–5; só é identificado na consulta por heurística (`nucleo.py:38,308-314`) e contado no log (`indexar.py:364-370`). |
| Avaliador com normalização de datas em `teste_modelos.py` | **Não** | Só normaliza acentos e dígito↔extenso (`_norm`, `_kw_presente`). `_keywords` tokeniza por `/`, então "12" isolado já conta como acerto. |
| Flags `--top-k` / `--max-sem` | **Não** (parcial) | `teste_modelos.py` só tem `--modelos` e `--ids`. `--top-k` existe apenas em `avaliar_retrieval.py`; `--max-sem` não existe. |
| Filtro de chunks ilegíveis | **Não** | Só há o filtro de cotas `C\d{1,2}:` (`indexar.py:328-338`; `nucleo.py:502-510`). |

---

## 1. Pipeline exatamente como está no código

### 1.1 Extração por tipo de conteúdo

| Conteúdo | Código | Método |
|---|---|---|
| **Corpo** | `indexar.py:92-139` (`extrair_texto_ibiruba`) | `page.extract_text()` (L114); remove cabeçalho institucional `_RE_CABECALHO_PAGINA` (L57-63, aplicado L117); meta `{file_path, page_number}` (L128). |
| **Tabelas** | `indexar.py:142-251` (`extrair_tabelas_como_texto`) | `page.extract_tables()` (L170); 4 primeiras colunas `_LABELS` (L24); herda col. 0 de célula mesclada (L187-204); descarta rodapé "Observa…" ou >15 palavras (L210); whitelist `_CURSOS_IBIRUBA` (L77-89, L231-244). Uma linha = um chunk com meta `{source, tipo:"tabela"}` (L307-310), sem split. |
| **Cronograma** | — | **Sem extração própria.** Tabela de 2 colunas das págs. 3–5 sai do `extract_text` com data e evento intercalados (ex.: `07 e questionamentos) quanto ao gabarito… 08/12/2026 preenchimento…`). Detectado só na consulta: contém "cronograma" ou ≥3 datas `dd/mm/202[4-9]` (`nucleo.py:38,308-314`). |
| **Anexos** | mesmo caminho do corpo/tabelas | Só as págs. 44, 45, 69 e 84 sobrevivem à máquina de estados; seu texto (cabeçalhos de tabela rotacionados) entra como texto comum e sai ilegível. Os demais anexos são descartados (estado `outro`). |

### 1.2 Filtros e máquina de estados de campus

- **Texto** (`indexar.py:108-131`): 3 estados `general` → `ibiruba` → `outro`, iniciando em `general`. O regex roda no **texto bruto** da página; `Campus Ibirubá[:(]` tem precedência sobre outro campus na mesma página (L119-122); o estado é herdado pelas páginas seguintes; só `general`/`ibiruba` são mantidos (L124).
- **Tabelas** (`indexar.py:158,165-168`): booleano `ibiruba_ativo`, inicia `False`; liga no cabeçalho Ibirubá, desliga no de outro campus (mesma precedência).
- Regex de campus: L26-69; nomes de outros campi: L33-37.
- **Cotas**: chunks de texto com `C\d{1,2}:` são removidos após o split (L327-338), e de novo no pós-filtro de `responder()` (`nucleo.py:502-510`).
- `DocumentCleaner()` com padrões (L314): remove linhas vazias e espaços extras.

### 1.3 Chunker

`indexar.py:315-319`: `DocumentSplitter(split_by="word", split_length=150, split_overlap=20)`. `respect_sentence_boundary` não é passado (padrão `False`), `split_threshold=0`. **Não respeita limites de sentença.**

### 1.4 Embedders, retriever e coleção

| Parâmetro | Documentos (`indexar.py`) | Consulta (`nucleo.py`) |
|---|---|---|
| Classe | `SentenceTransformersDocumentEmbedder` (L320-322) | `SentenceTransformersTextEmbedder` (L324-330) |
| Modelo | `intfloat/multilingual-e5-base` | `MODELO_EMB` = mesmo (L28) |
| Device | não definido (automático) | `ComponentDevice.from_str(EMBEDDER_DEVICE)`, padrão `cpu` (L30) |
| `prefix` | `''` | `''` |
| `normalize_embeddings` | `False` (padrão) | `False` (padrão) |
| Prefixo E5 `passage: ` / `query: ` | **Não usa** | **Não usa** |
| Retriever | — | `ChromaEmbeddingRetriever(top_k=10)` (L29,L331) |

- Normas medidas nos vetores gravados: **1,0** (o modelo já normaliza), dimensão 768.
- Coleção: `ChromaDocumentStore(collection_name="ifturing", persist_path=…)` (`indexar.py:312`), sem `distance_function` → padrão **`l2`**. Medido: `{'hnsw:space': 'l2'}`. Com vetores unitários a ordem por L2 é equivalente à do cosseno.

### 1.5 Metadados salvos por chunk

| Tipo | Metadados |
|---|---|
| Texto | `file_path`, `page_number` (**sempre 1**, ver §2.4), `split_id`, `split_idx_start`, `source_id`, `_split_overlap` (serializado em string, L373-377) |
| Tabela | `source`, `tipo="tabela"` |

---

## 2. Qualidade dos chunks (176 no índice)

### 2.1 Tamanho (palavras)

| tipo | n | mín | p25 | mediana | p75 | máx | <30 | >200 |
|---|---|---|---|---|---|---|---|---|
| texto | 161 | 37 | 150 | 150 | 150 | 272 | 0 | 2 |
| cronograma (subconjunto do texto, heurística de `nucleo.py`) | 6 | 146 | 164 | 174 | 175 | 179 | 0 | 0 |
| tabela | 9 | 16 | 17 | 18 | 18 | 19 | 9 (esperado) | 0 |
| **total** | 176 | 16 | 150 | 150 | 150 | 272 | 9 | 2 |

Os 2 chunks >200 palavras são lixo de anexo: `7f546278` (255) e `a0d30ceb` (272).

### 2.2 Origem dos chunks de texto

| Páginas | Chunks | Ilegíveis (fração < 0,5) |
|---|---|---|
| Corpo: 1–10, 12, 29 | 31 | 0 |
| Anexo, pág. 44 | 33 | 31 |
| Anexo, pág. 45 | 2 | 2 |
| Anexo, pág. 69 | 17 | 14 |
| Anexo, pág. 84 | 84 | 81 |
| **Total** | **167** | **128** |

Ou seja, 136 de 167 chunks de texto (81%) vêm de 4 páginas de anexo cujo texto é cabeçalho de tabela rotacionado, e 128 (77%) são ilegíveis. Os dados úteis dessas páginas já estão nos 9 chunks de tabela.

### 2.3 Início no meio de frase, duplicados

| Métrica | Índice todo (167) | Só corpo (31) |
|---|---|---|
| Trecho anterior não termina em `. ! ? : ;` | 150 (90%) | 18 (58%) |
| Primeiro caractere minúsculo | 130 (78%) | 15 |
| Corte no meio de uma linha | 151 (90%) | — |

- Duplicados exatos (conteúdo normalizado): **0**.
- Quase-duplicados (Jaccard de 5-gramas de palavras ≥ 0,5): **3 pares**, todos linhas de tabela (J = 0,56) que só diferem pelo curso (`3772b09c`~`3c469c8f`, `3772b09c`~`a86a1529`, `3c469c8f`~`a86a1529`). Nenhum em texto.
- Chunk removido pelo filtro de cotas: 1 (`6ff4bd8f`, pág. 45, também ilegível) — não representa perda de conteúdo.

### 2.4 Os 10 chunks de menor fração de tokens (4+ caracteres) presentes em ≥3 chunks

Todos com fração 0,00, 150 palavras, texto de anexo.

| id | pág. | 100 primeiros caracteres |
|---|---|---|
| ab308669 | 44 | `et t t d n ro o o e a dd d d n a f e e e d a c c c f m al la l a a a m i fr r rl a i a a a i a m ld ` |
| 80ff5322 | 44 | `i i t u tgt a t a a aa u / / i /l gpap po r rl u r e e eu a ot t t l uo oio n o , , , i f n up pep f` |
| f83a80fe | 69 | `pp l e daúes orbrt a l i rcidn c ea ot a sp e , *i n g ti en s r a ga d e i rl m eg mapue /e eas nn ` |
| 076b98d6 | 69 | `ue pa re r rp sn s ei att a t dea l a o o /i - g i p nE 1 ur t0 an e e l s t vgioo a nr u, g oa ip a` |
| 02b41849 | 69 | `g D da ú e e ) s b p n q ;e lt iu n ceC ea d 4d, e t :e ec n P on t re m e h es a n ds r d c oe ea u` |
| 1a56303d | 69 | `d ;i r c e e a is cCê lc l6 m n a o :c r el a aiP n a d e tp os( eúPs b n c o o l e D a iEc g )c an ` |
| 4e2b26a7 | 69 | `c t d cr aeud epgro pirt s e ) aa a n l o dm di u g o e eu i n nin a n t d tl t e e í o eg ud go e r` |
| 194b0671 | 69 | `a v/ a a 2 C p g p a e va ú n r ab sd g; lc i ia a d c Cs pa a5.,i t t V:o aa Cau q g i a tg u ao n ` |
| 8d640c16 | 69 | `m s/ s an E o ale nd - g s -o 1r i n 2 i ov no a vt ( M pg ae ra ggé es ard t; as a il C ; omC/3 p2 ` |
| 0c1e798e | 69 | `ur pa v ú iet na basf /l c de iCc o erl aa ia Vo ,n ra da p ú g ui a d a b t a 1s o li, t d c5p oe a` |

(Tokens: `\w{4,}` em minúsculas, fração calculada sobre ocorrências; chunks com 0 tokens de 4+ caracteres contam como 0.) Distribuição: mediana 0,71; 18 chunks < 0,5; 22 < 0,6.

### 2.5 Outros achados de qualidade

- **`page_number` é sempre 1** nos 167 chunks de texto: o `DocumentSplitter` sobrescreve o metadado. A página real foi recuperada reexecutando cleaner+splitter página a página (ids coincidem com os do índice; `split_idx_start` confere em 168/168).
- **Contaminação por outros campi** (cabeçalho `Campus X:` dentro de chunk Ibirubá): `3a0cc0df` e `4deb1928` (pág. 12), `4a94b750` e `9b61e9cd` (pág. 29). Causa: o estado é decidido por página e Ibirubá tem precedência. A auditoria pós-indexação do `indexar.py` emitiu 24 alertas; a maioria é falso positivo ("Rua Gen. Osório", nomes de campi em frases do corpo).
- **Páginas mantidas**: 1–10, 12, 29, 44, 45, 69, 84 (16). **Descartadas: 104 de 120.**

---

## 3. Cobertura por seção

Títulos detectados no PDF: 36 de nível 1 (seções 1–20, o segundo "20" do edital e Anexos I–XV) e 156 de nível 2 (`N.M`). Cobertura = fração dos caracteres da seção contida em chunks do índice.

**Causa raiz do buraco**: a seção 9.5 lista endereços de todos os campi; na pág. 11 aparece `9.5.1 Campus Alvorada:` e o estado vira `outro`. Ele só volta a `ibiruba` nas págs. 12, 29, 44, 45, 69 e 84 (cada uma tem "Campus Ibirubá" no texto), e as páginas intermediárias herdam `outro`.

### 3.1 Nível 1

| Seção | Título | Pág. | Chunks | Cobertura |
|---|---|---|---|---|
| 1 | Informações gerais | 1 | 7 | 100% |
| 2 | Cronograma | 3 | 7 | 100% |
| 3 | Quem pode participar | 5 | 4 | 100% |
| 4 | Vagas e cotas | 6 | 4 | 100% |
| 5 | Cursos, vagas e matérias | 7 | 1 | 100% |
| 6 | Isenção da taxa | 8 | 2 | 100% |
| 7 | Inscrições | 8 | 7 | 100% |
| 8 | Confirmação da inscrição | 10 | 2 | 100% |
| 9 | Condição especial | 11 | 2 | 25% |
| **10** | Comissão de heteroidentificação | 14 | **0** | 0% |
| **11** | Realização da prova | 16 | **0** | 0% |
| **12** | Recurso do gabarito | 20 | **0** | 0% |
| **13** | Sorteio público | 20 | **0** | 0% |
| **14** | Classificação | 20 | **0** | 0% |
| **15** | Preenchimento das vagas | 21 | **0** | 0% |
| **16** | Matrícula | 22 | **0** | 0% |
| **17** | Chamada pública | 24 | **0** | 0% |
| **18** | Documentação para matrícula | 25 | **0** | 0% |
| **19** | Comprovação de requisitos (vaga reservada) | 26 | **0** | 0% |
| 20 | Endereço e e-mail dos campi | 27 | 2 | 25% |
| **20 (bis)** | Disposições gerais e finais | 31 | **0** | 0% |
| **Anexo I** | | 32 | **0** | 0% |
| Anexo II | Vagas integrado/concomitante | 33 | 35 (lixo) | 5% |
| **Anexo III** | | 58 | **0** | 0% |
| Anexo IV | Subsequentes | 65 | 17 (lixo) | 4% |
| Anexo V | Graduação | 76 | 84 (lixo) | 7% |
| **Anexos VI–XV** | formulários e declarações | 94–120 | **0** | 0% |

- **Sem nenhum chunk: 23 de 36** — seções 10–19, o "20 bis", Anexos I, III e VI–XV.
- **Partidas em vários chunks (corpo)**: 1 (7), 2 (7), 3 (4), 4 (4), 6 (2), 7 (7), 8 (2), 9 (2), 20 (2). (Anexos II, IV e V também aparecem partidos, mas em chunks de lixo.)
- **Chunks que misturam duas seções de nível 1: 7 de 167**, todos em fronteiras adjacentes:

| Chunk | Seções |
|---|---|
| 65f18489 | 1 / 2 |
| 7195dd06 | 1 / 2 |
| 6467fed5 | 2 / 3 |
| 4b0a5ade | 3 / 4 |
| d045100c | 4 / 5 |
| 4b1f654b | 6 / 7 |
| 624db6cb | 7 / 8 |

### 3.2 Nível 2

103 de 156 subseções sem nenhum chunk; 22 partidas em ≥2 chunks; 19 chunks misturam ≥2 subseções; cobertura média de caracteres por subseção: 33%.

---

## 4. Avaliação de retrieval (`avaliar_retrieval.py`)

30 perguntas com `EVIDENCIA` (ignoradas: D01, D03 negativas; E03, E04 ambíguas). `k_min` = menor k em que **todas** as evidências aparecem na união dos textos dos k primeiros chunks. O modo híbrido reproduz `nucleo.responder()`: semântico → pós-filtro de cotas → injeção de tabelas → injeção de cronograma. Em produção o LLM recebe o **contexto inteiro** (10 a 16 chunks), então `hit@ctx` é o teto real; `hit@k` mede qualidade do ranking.

### 4.1 Resumo (n = 30)

| coleção | modo | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |
|---|---|---|---|---|---|---|
| produção | semântico | 67% | 70% | 80% | 0,539 | 80% |
| produção | híbrido | 57% | 80% | 90% | 0,493 | 97% |
| prefixos E5 | semântico | 63% | 70% | 80% | 0,541 | 80% |
| prefixos E5 | híbrido | 53% | 80% | 97% | 0,472 | 97% |

Nenhuma evidência está ausente do índice (0 de 30): o problema é de ranking e ruído, não de falta de conteúdo.

### 4.2 `k_min` por pergunta (— = não recuperada)

| id | n ev. | sem. prod. | híb. prod. | sem. pref. | híb. pref. |
|---|---|---|---|---|---|
| A01 | 4 | — | 5 | — | 5 |
| A02 | 3 | 3 | 3 | 3 | 3 |
| A03 | 2 | 2 | 2 | 2 | 2 |
| A04 | 1 | 1 | 1 | 1 | 1 |
| A05 | 1 | 1 | 1 | 1 | 1 |
| A06 | 1 | 1 | 1 | 1 | 1 |
| A07 | 2 | 2 | 2 | 2 | 2 |
| A08 | 2 | 2 | 2 | 2 | 2 |
| B01 | 4 | — | 4 | — | 4 |
| B02 | 3 | 3 | 3 | 5 | 5 |
| B03 | 1 | — | 1 | — | 1 |
| B04 | 1 | 1 | 1 | 1 | 1 |
| B05 | 1 | 1 | 1 | 1 | 1 |
| B06 | 1 | 1 | 1 | 1 | 1 |
| B07 | 1 | 1 | 1 | 1 | 1 |
| B08 | 3 | 6 | 12 | 4 | 10 |
| B09 | 4 | — | 4 | — | 4 |
| B10 | 2 | 2 | 2 | 2 | 2 |
| B11 | 2 | 2 | 2 | 2 | 2 |
| B12 | 4 | — | 4 | — | 4 |
| C01 | 2 | 8 | 12 | 7 | 10 |
| C02 | 1 | 3 | 4 | 2 | 4 |
| C03 | 2 | 5 | 7 | 6 | 8 |
| C04 | 2 | 1 | 1 | 1 | 2 |
| C05 | 2 | 1 | 4 | 1 | 3 |
| C06 | 2 | 1 | 5 | 1 | 5 |
| C07 | 2 | 1 | 3 | 1 | 4 |
| D02 | 3 | 1 | 6 | 1 | 6 |
| E01 | 4 | 6 | 6 | 7 | 7 |
| E02 | 8 | — | — | — | — |

### 4.3 Perguntas que falham em k = 10 (coleção de produção)

| Pergunta | k_min semântico | k_min híbrido | Falha em | 3 primeiros chunks (do modo que falha) |
|---|---|---|---|---|
| A01 | não recuperada | 5 | semântico | c92e38e6, 7472db22, a0d30ceb |
| B01 | não recuperada | 4 | semântico | 7472db22, c92e38e6, 4b0a5ade |
| B03 | não recuperada | 1 | semântico | 7472db22, c92e38e6, a0d30ceb |
| B09 | não recuperada | 4 | semântico | 7472db22, c92e38e6, 4b0a5ade |
| B12 | não recuperada | 4 | semântico | 7472db22, c92e38e6, a0d30ceb |
| B08 | 6 | 12 | híbrido | 65f18489, 7195dd06, d1133e6d |
| C01 | 8 | 12 | híbrido | d1133e6d, d2231fb4, c204d1d2 |
| E02 | não recuperada | não recuperada | ambos | c92e38e6, 7472db22, a0d30ceb |

Texto (80 caracteres) dos chunks citados:

| id | início |
|---|---|
| c92e38e6 | `Campus Ibirubá (a descrição das vagas está após a tabela) s Vagas Prova m a g Re` |
| 7472db22 | `Campus Ibirubá: (a descrição das vagas está após a tabela) Cursos Superiores Tur` |
| a0d30ceb | `Campus Ibirubá: A descrição da tabela encontra-se ao final da mesma. Vagas Sorte` |
| 4b0a5ade | `ingressar nos cursos Concomitantes/Subsequentes: 3.5.1. Do campus Sertão, o estu` |
| 65f18489 | `1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h` |
| 7195dd06 | `primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in` |
| d1133e6d | `sistema de Pré-Matrícula a documentação comprovando ser indígena ou quilombola. ` |
| d2231fb4 | `sistema será disponibilizado no site ingresso.ifrs.edu.br. Publicação do resulta` |
| c204d1d2 | `Divulgação do resultado dos recursos e gabarito definitivo da prova. A 16/12/202` |

Leitura:
- As 5 perguntas de curso que falham no semântico são vencidas por chunks de anexo ilegíveis que começam com "Campus Ibirubá…"; o híbrido as resolve injetando as linhas de tabela.
- B08 e C01 falham no híbrido porque a injeção de 6 chunks de cronograma no início empurra para a posição 12 chunks que já estavam no top-10 semântico (rank 6 e 8). O LLM ainda os recebe (hit@ctx).
- **E02** falha em tudo: o gatilho de tabela não casa nenhum termo de "Liste todos os cursos do campus Ibirubá" (`campus` e `ibirubá` são stop words, `cursos` é genérico), então 0 tabelas são injetadas apesar de as 9 linhas existirem no índice.
- Na coleção com prefixos, as falhas semânticas são as mesmas 6; no híbrido sobra só a E02 (B08 e C01 passam a k = 10).

---

## 5. A/B dos prefixos E5

Coleção `./chroma_exp_prefix`: **mesmos 176 chunks** de `chroma_exp_prod` (ids idênticos), re-embedados com `SentenceTransformersDocumentEmbedder(prefix="passage: ")`; consultas com `SentenceTransformersTextEmbedder(prefix="query: ")`. Nomes de parâmetro confirmados na versão instalada: `prefix` e `suffix`.

| modo | Δhit@3 | Δhit@5 | Δhit@10 | ΔMRR |
|---|---|---|---|---|
| semântico | −3 pp | 0 | 0 | +0,003 |
| híbrido | −3 pp | 0 | +7 pp | −0,021 |

Perguntas em que o `k_min` mudou (produção → prefixos):

| Modo | Mudanças |
|---|---|
| semântico | B02 3→5, B08 6→4, C01 8→7, C02 3→2, C03 5→6, E01 6→7 |
| híbrido | B02 3→5, B08 12→10, C01 12→10, C03 7→8, C04 1→2, C05 4→3, C07 3→4, E01 6→7 |

**Conclusão**: sem ganho consistente. Os ranks mudam 1 a 3 posições nos dois sentidos. O +7 pp de hit@10 no híbrido são 2 perguntas (B08 e C01) que cruzam o corte de 12→10. Com n = 30 e um índice 77% ilegível, o teste é pouco informativo; repetir após a Fase 2.

---

## 6. Gabarito: o que não foi confirmado no PDF

Todas as evidências de A/B/E foram conferidas nas tabelas do Campus Ibirubá (págs. 44, 45, 69, 84) e as do bloco C nas págs. 3–5.

| Pergunta | Situação |
|---|---|
| **D02** | A evidência existe (item 7.4: R$ 40,00 / R$ 80,00 / R$ 110,00), mas a **resposta esperada contradiz o PDF** ("isenção ou ausência de taxa"). `EVIDENCIA` preenchida; gabarito a revisar. |
| **E03** | Sem evidência: inscrição é via gov.br/CPF (item 7.2); documentos só aparecem na matrícula (seção 18), hoje fora do índice. Pergunta ambígua. |
| **E04** | Sem evidência: exigiria o cronograma inteiro. |
| D01, D03 | Negativas, sem evidência aplicável (o curso de D03 existe em outros campi, pág. 85). |
| C04, C06 | Evidência necessária mas não suficiente: "Divulgação dos locais de prova" e "27/01/2027" aparecem 2 vezes no PDF (a segunda é a linha "23 a 27/01/2027"). |
| C03 | `07 e` é evidência fraca, embora única no PDF. |

---

## 7. Proposta para a Fase 2 (não implementada)

Achados que a motivam: 104/120 páginas descartadas; seções 10–19 ausentes; 77% dos chunks de texto ilegíveis e dominando o top-k; `page_number` inútil; 58–90% de cortes no meio de frase; 4 chunks contaminados por outros campi; cronograma intercalado.

1. **Chunking por seção.** Segmentar o corpo (págs. 1–34) pelos títulos `N.` e `N.M`, prependendo o título a cada chunk (`"7. DAS INSCRIÇÕES > 7.4"`). Seções grandes se dividem apenas em subitens `N.M.K` (máx. ~250 palavras); subseções pequenas se fundem. Overlap zero ou de um item. Se restar split por palavra, `respect_sentence_boundary=True`.
2. **Escopo de campus por trecho, não por página.** Nas listas de endereços (9.5 e 20.1) manter só o bloco do Ibirubá; fora delas, escopo `geral`. Em páginas de anexo, ignorar o texto e usar só as tabelas já estruturadas, o que elimina os 136 chunks de lixo e a contaminação das págs. 12 e 29. Avaliar incluir os formulários (Anexos VI–XV) como `tipo="anexo"`.
3. **Metadados**: `secao`, `secao_titulo`, `subsecao`, `pagina_ini`/`pagina_fim` calculados por nós (o splitter zera `page_number`), `tipo` (`texto`/`tabela`/`cronograma`/`anexo`), `escopo_campus` (`geral`/`ibiruba`), `fonte`.
4. **Cronograma estruturado**: `extract_tables` nas págs. 3–5 → linhas `Data: 07 a 08/12/2026 | Evento: …` com `tipo="cronograma"`; o retrieval híbrido passa a filtrar por metadado em vez das heurísticas de regex de `nucleo.py`.
5. **Portões de qualidade na indexação**: rejeitar chunks de baixa legibilidade (fração de tokens frequentes), falhar se houver cabeçalho de outro campus e exigir ≥ 1 chunk por seção de nível 1.
6. **Gabarito**: corrigir D02, resolver E03/E04 e acrescentar perguntas sobre as seções 11–18; ampliar a regra de tabela para consultas do tipo "liste todos os cursos" (E02).
7. **Reteste**: repetir o A/B dos prefixos `passage:`/`query:` no índice limpo e fixar `distance_function="cosine"` explicitamente.

---

## Anexo — arquivos gerados nesta fase

| Arquivo | Estado |
|---|---|
| `avaliar_retrieval.py` | Já existia sem commit; acrescentados: checagem "evidência ausente do índice" e coluna Δrr no A/B |
| `PIPELINE.md` | Coluna `EVIDENCIA` nas tabelas do gabarito + nota sobre D02 |
| `.gitignore` | `retrieval_resultado.*` e `/chroma_exp_*` |
| `retrieval_resultado.md` / `.csv` | Saídas do avaliador (ignoradas pelo git) |
| `chroma_exp_prod/`, `chroma_exp_prefix/` | Coleções temporárias (ignoradas pelo git) |
| `auditoria_indexacao.md` | Este relatório |


---

# Fase 2 — correção da cobertura do índice

Escopo: corrigir a cobertura do índice com base nos achados da fase 1 (seções 2 e 3 acima). Alterados: `indexar.py`, `avaliar_retrieval.py` e o gabarito (`PIPELINE.md`). **Não** foram alterados `nucleo.py`, a coleção de produção nem rodada a reindexação de produção: não existe `./chroma_data` neste checkout, e toda a validação usou coleções temporárias `./chroma_exp_*` (`chroma_exp_fase2` é a nova; `chroma_exp_prod` é a da fase 1, mantida como linha de base). Nada foi commitado.

## F2.0 Resumo

| Parte | Resultado |
|---|---|
| **A — escopo de campus e páginas** | **Concluída.** 23 de 36 seções sem chunk → **0** (o Anexo III não tem oferta do Ibirubá e fica como "n/a esperado"). Descarte global de 104 páginas → classificação por página: 31 de corpo e 28 de anexo geral sempre mantidas; só as 57 páginas de tabela de outros campi são descartadas. Chunks ilegíveis 128 → **0**. `page_number` sempre 1 → **59 páginas distintas** (e `pagina_ini`/`pagina_fim`). Contaminação por outros campi 4 → **0**. O indexador agora **falha (exit 1)** se uma seção de nível 1 ficar sem chunk (testado). |
| **B — cronograma estruturado** | **Concluída.** 26 linhas "Data \| Evento" reconstruídas de `find_tables`, cada data associada ao evento certo; **conferidas visualmente, uma a uma, contra o PDF renderizado (págs. 3–5)**. 6 chunks `tipo="cronograma"`; o texto bruto fragmentado da tabela foi removido das páginas. |
| **C — gabarito** | **Concluída.** D02, E03, E04, C03, C04, C06 corrigidos; D01/D03 marcadas `NEGATIVA`; **24 perguntas novas** (seções 10 a 18: 2 a 4 por seção) mais E05; todas com evidência literal de ocorrência única no PDF. As 30 perguntas antigas ficam separadas. |
| **D — teste** | **Cobertura verde; ranking do híbrido NÃO está verde.** Teto de recuperação (`hit@ctx`) do grupo legado igual à linha de base (97%) e o grupo novo passa de 4% para 85%. Mas hit@3/5/10 do grupo legado no fluxo atual do `nucleo.py` caem (híbrido 57/80/90 → 50/67/73). Causa identificada (F2.6): a heurística de `nucleo.py` injeta agora 14 chunks "cronograma-like" na frente do contexto. Com a injeção por metadado (hipótese, **não** aplicada no `nucleo.py`) o híbrido legado volta a 57/77/90. Decisão pendente sua: ver F2.7. |

## F2.1 O que mudou nos arquivos

| Arquivo | Mudança |
|---|---|
| `indexar.py` | Reescrito (mantém `indexar_documentos()`, `extrair_tabelas_como_texto()` e o formato do índice; removidos `extrair_texto_ibiruba()` e o filtro de cotas). Funções novas: `_classificar_paginas`, `_filtrar_itens_campus`, `_filtrar_linhas_por_campus`, `_texto_legenda`, `_extrair_cronograma`, `_chunkar`, `_portao_legibilidade`, `_cobertura_secoes`; CLI em `main()`. O índice anterior só é removido **depois** de os portões passarem. |
| `avaliar_retrieval.py` | IDs com sufixo (`F10a`); `NEGATIVA`; coluna `EVIDENCIA_LEGADO` + `--legado`; grupos `legado`/`novas` separados em todas as saídas; chunks das falhas com tipo e seção; tamanho do contexto do híbrido; `--crono-por-meta` (hipótese, desligada por padrão). |
| `PIPELINE.md` | Gabarito (seção 4): ver F2.4. |
| `.gitignore` | `retrieval_resultado_*` (saída da hipótese). |

Parâmetros do chunker **inalterados** (`split_by=word`, 150, overlap 20), assim como o embedder, a métrica `l2` e a ausência de prefixos E5 (fase 3).

## F2.2 Parte A — escopo de campus e páginas

### Classificação por página

| Categoria | Páginas | Tratamento |
|---|---|---|
| (i) corpo do edital (antes do 1º "ANEXO") | 1–31 (31 págs.) | Sempre mantidas; listas por item (abaixo) |
| (ii) anexo de tabela do Ibirubá | 44, 45, 69, 84 (4 págs.) | Tabelas estruturadas (9 linhas) + legenda; texto bruto não indexado |
| (iii) anexo de tabela de outro campus | 57 págs. (Anexos II, III, IV e V sem Ibirubá) | Descartadas |
| (iv) anexo geral | 32 (Anexo I) e 94–120 (Anexos VI a XV): 28 págs. | Sempre mantidas |

Um anexo é "de tabela" quando alguma de suas páginas tem cabeçalho de campus nos dois formatos do edital (`Campus X (a descrição das vagas…` e `Campus X: A descrição da tabela…`, com variantes `Campus Canoas(a descrição` e `Campus X: (a descrição…`). O campus vale até o próximo cabeçalho e é herdado pelas páginas de continuação sem cabeçalho (ex.: págs. 73 e 78). Nenhuma página do Ibirubá mistura campi (o aviso de página mista não disparou).

### Listas por item (9.5 e 20.1)

Item de lista = linha `[N.N.N] Campus X:`; vale até o próximo item ou o próximo título/item numerado. Mantidos: introdução + item do Ibirubá.

| Lista | Itens | Mantidos | Descartados |
|---|---|---|---|
| 9.5 (atendimento especial) | 18 | 1 (9.5.8) | 17 |
| 20.1 (endereços) | 19 (o edital não tem 20.1.3; o último, Zona Norte, não é numerado) | 1 (20.1.10) | 18 |

208 linhas do corpo descartadas. Os dois itens do Ibirubá viram chunks `tipo=corpo`, `escopo_campus=ibiruba` (págs. 12 e 29). Antes, as págs. 12 e 29 entravam inteiras, com os endereços de Canoas, Feliz, Farroupilha e Porto Alegre.

### Páginas de anexo de tabela do Ibirubá: legendas

As 4 páginas foram extraídas com legibilidade; **nada foi perdido** além dos fragmentos de cabeçalho de coluna rotacionado ("Vagas Prova", "Reserva Ensino Público", "Renda igual ou inferior…"), cujo conteúdo estrutural já vem de `extract_tables`.

- O embaralhamento da fase 1 vinha de `y_tolerance=3` (padrão do `extract_text`), que funde linhas muito próximas. O parágrafo "descrição das vagas" é texto de fonte de **1 pt e cor branca**: com `y_tolerance=1` sai limpo.
- Para a legenda usa-se o texto **fora** dos bounding boxes das tabelas (cabeçalho "Campus Ibirubá…", notas `*mais estágio obrigatório.` e `** curso com parte da carga horária EaD.`, e o parágrafo por curso com C1 a C10) mais os rodapés de tabela lidos das células (`Observação: …`, pág. 69). Linhas de dados (já em `tipo=tabela`) e cabeçalhos rotacionados não entram.
- Resultado: **32 chunks `tipo=legenda`** (págs. 44, 45, 69 e 84).
- ⚠️ **O parágrafo "descrição das vagas" é invisível no PDF renderizado** (branco, 1 pt; é descrição para leitores de tela, como diz o próprio cabeçalho "a descrição das vagas está após a tabela"). O conteúdo é coerente com a tabela visível (conferido na pág. 45: C1 = 13 e C2 = 1 nas duas fontes). Foi indexado por pedido ("descrição das vagas"), mas fica a decisão: se preferir só texto visível, basta ignorar caracteres com `non_stroking_color == (1.0,)` e `size < 2`.

### Anexo IX (tabela de pesos por curso)

Não tem cabeçalho de campus (cai em "anexo geral"), mas cada linha começa pelo campus. Mantidos título e linhas do **Ibirubá**; descartadas as linhas dos outros campi, suas continuações e as linhas do cabeçalho rotacionado (99 linhas). **Limitação:** os nomes das 5 colunas de peso estão em texto rotacionado ilegível; as 4 linhas do Ibirubá (`Ibirubá Bacharelado em Agronomia 2 2 2 2 2`…) foram indexadas **sem** cabeçalho. A ordem provável (Linguagens, Matemática, Ciências da Natureza, Ciências Humanas, Redação) é inferência minha e **não** foi incluída no índice.

### Portão de qualidade e de cobertura

- **Legibilidade** (corpo, legenda e anexo geral): rejeita chunk com ≥ 40% de tokens de 1–2 caracteres ou com fração de tokens de 4+ caracteres presentes em ≥ 3 chunks do corpus < 0,30. **Rejeitados: 0** de 173. Margem observada: menor fração de tokens frequentes entre os aceitos = 0,33 (Anexo VIII, conteúdo programático legítimo); na fase 1 os ilegíveis ficavam em 0,00. Maior fração de tokens curtos entre os aceitos: 0,37.
  - **Desvio da regra literal:** tokens só de dígitos não contam como "curtos" e chunks com menos de 20 tokens não são avaliados. Motivo: a regra literal (contando dígitos) rejeitaria **17 chunks legítimos** (14 de corpo com itens numerados como "9.1.1." e 3 de anexo, entre eles a tabela de pesos), todos com 40–64% de tokens curtos. O indexador imprime quais seriam esses 17 (`ids`) a cada execução.
- **Cobertura:** o indexador levanta `IndexacaoIncompleta` e sai com **código 1** se alguma seção de nível 1 (itens 1 a 20, o segundo "20" do edital e anexos gerais ou com tabela do campus) ficar sem chunk. **Testado:** simulando a perda dos chunks da seção 12 → `exit code: 1`, mensagem "Indexação ABORTADA: seção(ões) de nível 1 sem nenhum chunk: 12 (RECURSO DO GABARITO DA PROVA). O índice anterior foi preservado.", e o diretório de destino não foi tocado.
- A tabela abaixo é impressa ao final de **toda** indexação (também na falha).

### Metadados reais por chunk

`secao` (rótulo), `secao_titulo`, `secoes` (todas as seções que o chunk toca, ex. `"11,12,13"`), `pagina_ini`, `pagina_fim`, `page_number` (= `pagina_ini`; o splitter o sobrescrevia com 1), `tipo` (`corpo` | `tabela` | `cronograma` | `legenda` | `anexo_geral`), `escopo_campus` (`todos` | `ibiruba`), `source`, `file_path`, `split_id`, `split_idx_start`. A coleção Chroma continua `l2`.

Como o chunking continua por página (150/20), um chunk nunca cruza página (exceto os de cronograma, que reúnem linhas das págs. 3 a 5); `secao` é a seção do início do chunk e `secoes` lista as demais. O `_split_overlap`/`source_id` do Haystack deixaram de ser gravados.

### Cobertura por seção de nível 1 (índice final, `chroma_exp_fase2`)

| seção | nº de chunks | páginas | status | título |
|---|---|---|---|---|
| 1 | 7 | 1-3 | ok | INFORMAÇÕES GERAIS |
| 2 | 9 | 3-5 | ok | CRONOGRAMA DO PROCESSO SELETIVO |
| 3 | 3 | 5-6 | ok | QUEM PODE PARTICIPAR DO PROCESSO SELETIVO |
| 4 | 4 | 6-7 | ok | INFORMAÇÕES DAS VAGAS E COTAS |
| 5 | 1 | 7 | ok | DOS CURSOS, VAGAS E MATÉRIAS PARA PROVA |
| 6 | 2 | 8 | ok | DA ISENÇÃO (NÃO PAGAMENTO) DA TAXA DE INSCRIÇÃO |
| 7 | 7 | 8-10 | ok | DAS INSCRIÇÕES |
| 8 | 2 | 10 | ok | DA CONFIRMAÇÃO DA INSCRIÇÃO |
| 9 | 3 | 11-12 | ok | COMO SOLICITAR CONDIÇÃO ESPECIAL PARA A REALIZAÇÃO DO E |
| 10 | 6 | 14-16 | ok | REALIZAÇÃO DA COMISSÃO DE HETEROIDENTIFICAÇÃO |
| 11 | 12 | 16-20 | ok | REALIZAÇÃO DA PROVA |
| 12 | 1 | 20 | ok | RECURSO DO GABARITO DA PROVA |
| 13 | 2 | 20 | ok | DO SORTEIO PÚBLICO |
| 14 | 5 | 20-21 | ok | COMO SERÁ FEITA A CLASSIFICAÇÃO |
| 15 | 4 | 21-22 | ok | COMO SERÁ O PREENCHIMENTO DAS VAGAS |
| 16 | 6 | 22-23 | ok | COMO SERÁ A MATRÍCULA |
| 17 | 4 | 24-25 | ok | O QUE É E COMO FUNCIONA A CHAMADA PÚBLICA |
| 18 | 3 | 25-26 | ok | DOCUMENTAÇÃO PARA MATRÍCULA |
| 19 | 5 | 26-27 | ok | COMPROVAÇÃO DE REQUISITOS PARA OCUPAÇÃO DE VAGA RESERVA |
| 20 | 3 | 27-29 | ok | ENDEREÇO E E-MAIL DOS CAMPI |
| 20b | 1 | 31 | ok | DAS DISPOSIÇÕES GERAIS E FINAIS |
| ANEXO I | 2 | 32 | ok | LEGENDA PARA OS QUADROS DE VAGAS |
| ANEXO II | 14 | 44-45 | ok | QUADROS DE VAGAS - CURSOS INTEGRADOS, CONCOMITANTES E |
| ANEXO III | 0 | — | sem oferta do campus (esperado) | QUADROS DE VAGAS - CURSOS EJA |
| ANEXO IV | 6 | 69 | ok | QUADROS DE VAGAS - CURSOS SUBSEQUENTES |
| ANEXO V | 21 | 84 | ok | QUADROS DE VAGAS - CURSOS DE GRADUAÇÃO |
| ANEXO VI | 9 | 94-97 | ok | SISTEMA DE RESERVA DE VAGAS |
| ANEXO VII | 7 | 98-100 | ok | CONTEÚDOS PROGRAMÁTICOS - CURSOS INTEGRADOS, CONCOMITAN |
| ANEXO VIII | 15 | 101-105 | ok | CONTEÚDOS PROGRAMÁTICOS - CURSOS SUBSEQUENTES E DE GRAD |
| ANEXO IX | 2 | 106,108 | ok | TABELA DE PESOS PARA PROVA E ENEM PARA OS CURSOS DE GRA |
| ANEXO X | 4 | 110-111 | ok | FORMULÁRIO DE SOLICITAÇÃO DE CONDIÇÃO ESPECIAL |
| ANEXO XI | 3 | 112 | ok | FLUXO PARA PREENCHIMENTO DAS VAGAS RESTANTES |
| ANEXO XII | 13 | 113-117 | ok | DOCUMENTOS OBRIGATÓRIOS |
| ANEXO XIII | 1 | 118 | ok | FORMULÁRIO DE DECLARAÇÃO DE MEMBRO DE COMUNIDADE INDÍGE |
| ANEXO XIV | 2 | 119 | ok | DECLARAÇÃO DE PERTENCIMENTO ÉTNICO QUILOMBOLA |
| ANEXO XV | 2 | 120 | ok | SOLICITAÇÃO DE USO DE NOME SOCIAL PARA MENOR DE IDADE |

Nenhuma seção sem chunk. "ANEXO III" (quadros de EJA) não tem oferta do Ibirubá no PDF; as demais têm. Os chunks de "ANEXO II/IV/V" são as linhas de tabela e as legendas do Ibirubá.

### Contagem de chunks

| tipo | linha de base (fase 1) | fase 2 |
|---|---|---|
| texto bruto (corpo + anexos embaralhados) | 167 | — |
| corpo | — | 66 |
| anexo_geral | — | 60 |
| legenda | — | 32 |
| cronograma | — | 6 |
| tabela | 9 | 9 |
| **total** | **176** | **173** |

Qualidade (fase 2): 0 chunks ilegíveis (eram 128 de 167); 0 chunks > 200 palavras (eram 2); chunks de corpo com cabeçalho "Campus <outro>:": 0 (eram 4); `page_number` com 59 valores distintos (era sempre 1). 36 dos 66 chunks de corpo começam em letra minúscula (corte no meio de frase), igual em natureza à fase 1 porque o chunker não foi alterado.

## F2.3 Parte B — cronograma estruturado

**Método.** `pdfplumber.find_tables()` nas páginas do item 2 (da página do título "2." até a do título "3."): a tabela de 2 colunas ("Data/Período" | "Evento") é reconstruída célula a célula, então a data fica na **mesma faixa vertical** do evento (sem agrupar por linha de texto). Linha com a célula de data vazia é continuação do evento anterior, inclusive através da quebra de página (caso de "14 a 18/11/2026": o evento termina na pág. 4). O evento é mantido **inteiro**, o que preserva o complemento que diferencia as fases repetidas ("Período para realização de recursos … quanto ao gabarito das provas" / "… ao resultado das comissões de heteroidentificação…" / "… da análise da documentação de matrícula…"). O texto da tabela é removido das páginas (`outside_bbox`), então o texto bruto fragmentado não duplica; o título "2. CRONOGRAMA…" e a nota "*As informações referentes ao cronograma…" ficam no corpo.

**Resultado: 26 linhas.** Cada chunk traz linhas `Data: <célula de data> | Evento: <evento>`.

**Chunks.** 6 chunks `tipo="cronograma"` (linhas por chunk: 6, 3, 5, 6, 3 e 3; 66 a 169 palavras; 134 a 341 tokens do E5, abaixo do limite de 512). O primeiro agrupamento ("~6 linhas") gerou um chunk de 530 tokens, que seria truncado no embedding; passou a valer o teto de 180 palavras. Um chunk com 1 só linha não satisfaria a heurística "cronograma-like" de `nucleo.py` (≥ 3 datas), por isso todo chunk tem pelo menos 3 linhas.

**Confirmação manual.** As páginas 3, 4 e 5 foram renderizadas e lidas visualmente; as 26 linhas abaixo batem com o PDF (data na mesma linha da tabela do evento, na mesma ordem), incluindo o evento de 3 trechos "14 a 18/11/2026" que continua na pág. 4.

| # | pág. | Data/Período (célula) | Evento (início) |
|---|---|---|---|
| 1 | 3 | De 30/09 a 19/10/2026 | Período de inscrições e solicitação de isenção da taxa de inscrição (não pagamento da taxa de i… |
| 2 | 3 | 21/10/2026 | Data limite para pagamento da taxa de inscrição no sistema bancário. |
| 3 | 3 | 22/10/2026 | Divulgação da Lista Preliminar das/dos candidatas/candidatos inscritas/inscritos. A publicação … |
| 4 | 3 | 23 a 25/10/2026 | Recurso quanto a não processamento da inscrição. A/O candidata/candidato que o nome não constar… |
| 5 | 3 | 29/10/2026 | Divulgação da Lista Final das/dos candidatas/candidatos inscritas/inscritos. A publicação será … |
| 6 | 3 | 10/11/2026 | Publicação da chamada às/aos candidatas/candidatos autodeclarados negras/negros (pretas/pretos … |
| 7 | 3 | 14 a 18/11/2026 | Período para as/os candidatas/candidatos autodeclaradas/autodeclarados negra/negro (preto ou pa… |
| 8 | 4 | De 16 a 18/11/2026 | Realização das comissões de heteroidentificação para as/os candidatas/candidatos autodeclarados… |
| 9 | 4 | 19/11/2026 | Publicação do resultado preliminar das comissões de heteroidentificação para as/os candidatas/c… |
| 10 | 4 | De 20 a 22/11/2026 | Período para realização de recursos (pedidos de revisão, questionamentos) quanto ao resultado d… |
| 11 | 4 | 24/11/2026 | Publicação do resultado final das comissões de heteroidentificação para as/os candidatas/candid… |
| 12 | 4 | 27/11/2026 | Divulgação dos locais de prova. A publicação será realizada no site ingresso.ifrs.edu.br. |
| 13 | 4 | 06/12/2026 | Aplicação das provas conforme divulgação dos locais de prova. |
| 14 | 4 | 07/12/2026 | Divulgação do gabarito preliminar das provas. A publicação será realizada no site ingresso.ifrs… |
| 15 | 4 | 07 e 08/12/2026 | Período para realização de recursos (pedidos de revisão, questionamentos) quanto ao gabarito da… |
| 16 | 4 | 10/12/2026 | Realização do Sorteio Público. O sorteio será transmitido pelo canal oficial do IFRS no youtube… |
| 17 | 5 | 16/12/2026 | Divulgação do resultado dos recursos e gabarito definitivo da prova. A publicação será realizad… |
| 18 | 5 | 05/01/2027 | Publicação da Classificação Geral e Aprovados em 1ª Chamada (Sorteio e Prova). A data e cronogr… |
| 19 | 5 | De 06 a 12/01/2027 | Envio da documentação para todas/todos candidatas/candidatos inscritas/inscritos nos modos de i… |
| 20 | 5 | 18/01/2027 | Publicação do resultado preliminar da matrícula em Primeira Chamada (Sorteio e Prova). A public… |
| 21 | 5 | 19 e 20/01/2027 | Período para realização de recursos (pedidos de revisão, questionamentos) da análise da documen… |
| 22 | 5 | 22/01/2027 | Previsão de publicação da Classificação e da Chamada Pública para o modo de ingresso ENEM. A pu… |
| 23 | 5 | 23 a 27/01/2027 | Previsão de envio da documentação para todas/todos candidatas/candidatos inscritas/inscritos no… |
| 24 | 5 | 25/01/2027 | Publicação do resultado final das matrículas em Primeira Chamada (Sorteio e Prova). A publicaçã… |
| 25 | 5 | 27/01/2027 | Publicação da Chamada Pública (Sorteio e Prova). A publicação será realizada no site ingresso.i… |
| 26 | 5 | Até 03/02/2027 | Publicação do resultado final da matrícula na Chamada Pública (Sorteio, Prova e ENEM). A public… |

**Antes / depois (6 linhas).** "Antes" é o texto do índice da fase 1 (`chroma_exp_prod`); a data aparece afastada do evento por causa das duas colunas.

| Evento | Antes (fase 1) | Depois (fase 2) |
|---|---|---|
| Inscrições | `… 2. CRONOGRAMA DO PROCESSO SELETIVO Evento Data/Período De 30/09 a Período de inscrições e solicitação de isenção da taxa de inscrição (não 19/10/2026 pagamento da taxa de inscrição). 21/10/2026 Data…` | `Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de isenção da taxa de inscrição (não pagamento da taxa de inscrição).` |
| Recurso do gabarito | `… Período para realização de recursos (pedidos de revisão, 07 e questionamentos) quanto ao gabarito das provas por meio do 08/12/2026 preenchimento do Formulário de Recursos no link…` | `Data: 07 e 08/12/2026 \| Evento: Período para realização de recursos (pedidos de revisão, questionamentos) quanto ao gabarito das provas por meio do preenchimento do Formulário de Recursos no link: https://forms.gle/JAygrEBB9ZNMHpMa7.` |
| Recurso (heteroidentificação) | `… autodeclarados De 20 a negras/negros (pretas/pretos e pardas/pardos), indígenas e quilombolas 22/11/2026 inscritos em todos os modos…` | `Data: De 20 a 22/11/2026 \| Evento: Período para realização de recursos (pedidos de revisão, questionamentos) quanto ao resultado das comissões de heteroidentificação para as/os candidatas/candidatos autodeclarados negras/negros…` |
| Recurso (matrícula, 1ª chamada) | `… da análise da documentação de matrícula na Primeira 19 e Chamada (Sorteio e Prova) e eventual complementação da 20/01/2027 documentação…` | `Data: 19 e 20/01/2027 \| Evento: Período para realização de recursos (pedidos de revisão, questionamentos) da análise da documentação de matrícula na Primeira Chamada (Sorteio e Prova) e eventual complementação da documentação…` |
| Chamada Pública | `… Publicação da Chamada Pública (Sorteio e Prova). A publicação será 27/01/2027 realizada no site ingresso.ifrs.edu.br. Até Publicação do resultado final da matrícula…` | `Data: 27/01/2027 \| Evento: Publicação da Chamada Pública (Sorteio e Prova). A publicação será realizada no site ingresso.ifrs.edu.br.` |
| Resultado final | `… Até Publicação do resultado final da matrícula na Chamada Pública (Sorteio, 03/02/2027 Prova e ENEM)…` (texto do PDF; ficava no fim do chunk anterior) | `Data: Até 03/02/2027 \| Evento: Publicação do resultado final da matrícula na Chamada Pública (Sorteio, Prova e ENEM). A publicação será realizada no site ingresso.ifrs.edu.br.` |

## F2.4 Parte C — gabarito (`PIPELINE.md`)

| Pergunta | Mudança |
|---|---|
| **D02** | Resposta esperada corrigida: R$ 40,00 (Sorteio ou ENEM), R$ 80,00 (Prova, técnicos) e R$ 110,00 (Prova, Graduação), item 7.4; isenção pelo CadÚnico (seção 6) e sem taxa em EJA (7.5). Evidência inalterada. |
| **E03** | Agora "O que preciso para me inscrever no processo seletivo?" (internet em ingresso.ifrs.edu.br, login gov.br com CPF; itens 7.1 e 7.2). Evidência: `exclusivamente via INTERNET no endereço ingresso.ifrs.edu.br` e `para autenticar-se com seu CPF e senha`. |
| **E05** (nova) | "Quais documentos preciso apresentar para a matrícula?" (Anexo XII; seção 18). Evidência: título do bloco "CONVOCADOS PARA MATRÍCULA… DEVEM APRESENTAR", `Carteira de identidade ou documento equivalente`, `Documento onde conste o nº do CPF`. |
| **E04** | Evidência = 4 marcos do cronograma no formato "Data \| Evento" (inscrição, prova, classificação, resultado final); só é respondível com o cronograma todo no contexto. |
| **C03, C04, C06** (e C01, C02, C05, C07) | Evidência = **linha inteira** "Data: … \| Evento: …", única no índice. A evidência da fase 1 foi preservada na coluna `EVIDENCIA_LEGADO` para avaliar a linha de base (formato antigo). |
| **D01, D03** | `EVIDENCIA = NEGATIVA`: perguntas negativas, avaliadas só pela resposta "não consta"; fora das métricas de retrieval. |
| **F10a–F18c** (novas, 24) | Seção 10: 3 · 11: 4 · 12: 2 · 13: 2 · 14: 3 · 15: 2 · 16: 3 · 17: 2 · 18: 3. Cada `EVIDENCIA` é texto literal do PDF com **uma única ocorrência** (conferido por busca no texto normalizado; a primeira versão de F11d tinha uma evidência com 2 ocorrências e foi trocada por `V. Passaporte; ou VI. Carteira Nacional de Habilitação`). |

Os grupos são separados em todas as saídas do avaliador: **legado** (as 30 antigas: A01–A08, B01–B12, C01–C07, D02, E01, E02) e **novas** (24 do bloco F mais E03, E04 e E05 = 27).

**Dúvidas / ressalvas do gabarito**
- "Resposta esperada" das perguntas novas são resumos que escrevi a partir do texto das seções; as evidências foram conferidas, as palavras-chave de resposta **não** foram testadas com um LLM.
- E04 mede 4 marcos, não "todas as datas"; E05 cobre só o bloco comum a todos os candidatos do Anexo XII (há blocos por tipo de curso e cota).
- `teste_modelos.py` (não alterado) usa o ID `E03`, que agora tem outro texto e outras palavras-chave; as IDs `F…` e `E05` não são lidas por ele (o regex só aceita `[A-E]\d{2}`).
- D02: o item 7.4 informa valores, mas o enunciado antigo pedia "isenção ou ausência de taxa". Se a intenção era testar a resposta sobre isenção, vale uma pergunta separada.

## F2.5 Parte D — teste

Comandos: `python avaliar_retrieval.py --run base=./chroma_exp_prod --legado base --run fase2=./chroma_exp_fase2` (oficial; saída `retrieval_resultado.md/.csv`) e o mesmo com `--crono-por-meta --saida retrieval_resultado_meta` (hipótese). `base` usa `EVIDENCIA_LEGADO` nas perguntas antigas porque o índice antigo não tem o formato "Data | Evento"; a linha de base reproduz exatamente os números da fase 1.

### Grupo legado (30 perguntas antigas): fluxo atual do `nucleo.py`

| coleção | modo | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |
|---|---|---|---|---|---|---|
| base (fase 1) | semântico | 67% | 70% | 80% | 0,539 | 80% |
| base (fase 1) | híbrido | 57% | **80%** | 90% | 0,493 | 97% |
| fase 2 | semântico | 60% | 63% | 70% | 0,489 | 70% |
| fase 2 | híbrido | 50% | 67% | 73% | 0,434 | **97%** |

### Grupo novas (27 perguntas; a linha de base não tem como respondê-las)

| coleção | modo | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |
|---|---|---|---|---|---|---|
| base | semântico / híbrido | 0% | 0% | 4% | 0,004 | 4% |
| fase 2 | semântico | 63% | 78% | 81% | 0,499 | 81% |
| fase 2 | híbrido | 37% | 48% | 63% | 0,346 | **85%** |

(Na linha de base, 26 das 27 perguntas novas têm evidência **inexistente no índice**; na fase 2, 0.)

### Hipótese: injeção de cronograma por metadado (`--crono-por-meta`, não é o `nucleo.py`)

| coleção | grupo | modo | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |
|---|---|---|---|---|---|---|---|
| fase 2 | legado | híbrido | 57% | 77% | **90%** | 0,465 | 97% |
| fase 2 | novas | híbrido | 37% | 52% | **81%** | 0,364 | 85% |

### Tamanho do contexto do híbrido (vai inteiro para o LLM; `num_ctx=8192`)

| coleção | grupo | chunks (média / máx) | caracteres (média / máx) | tokens estimados (média / máx) |
|---|---|---|---|---|
| base | legado | 10,8 / 16 | 7.153 / 16.857 | 2.044 / 4.816 |
| fase 2 | legado | 11,9 / 24 | 7.195 / 20.437 | 2.056 / 5.839 |
| fase 2 | novas | 12,3 / 24 | 10.666 / 20.400 | 3.047 / 5.829 |

O pior caso (≈ 5,8 mil tokens estimados) fica abaixo de `num_ctx` menos a resposta (1024) e o prompt, mas com folga pequena. Com `--crono-por-meta` o máximo cai para 16 chunks / 3,9 mil tokens.

### Perguntas que ainda falham em k = 10 (fase 2, fluxo atual)

**Híbrido (fluxo atual de `nucleo.responder()`) — 18 perguntas falham em k = 10**

| id | grupo | menor k em que a evidência apareceu | chunks no contexto | 3 primeiros chunks (id · tipo · seção · 80 caracteres) |
|---|---|---|---|---|
| B08 | legado | k=19 | 24 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| C01 | legado | k=18 | 20 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`2aba6920` · cronograma · 2 · Data: 14 a 18/11/2026 \| Evento: Período para as/os candidatas/candidatos autodec<br>`4b48f8b1` · cronograma · 2 · Data: De 20 a 22/11/2026 \| Evento: Período para realização de recursos (pedidos  |
| C02 | legado | k=14 | 23 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| C04 | legado | k=11 | 20 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| C06 | legado | k=14 | 22 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| C07 | legado | k=13 | 22 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| D02 | legado | k=14 | 20 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`2aba6920` · cronograma · 2 · Data: 14 a 18/11/2026 \| Evento: Período para as/os candidatas/candidatos autodec |
| E02 | legado | não recuperada | 6 | `e075f0fe` · corpo · 20 · 20.1.10 Campus Ibirubá: Endereço: Rua Nelsi Ribas Fritsch, 1111- Bairro Esperanç<br>`b5b87fcf` · corpo · 2 · *As informações referentes ao cronograma estarão disponíveis na página ingresso. |
| E03 | novas | não recuperada | 10 | `8f4be260` · corpo · 7 · possível realizar o pagamento da taxa de inscrição após o prazo. 7.6.1. Caso dei<br>`dbddab29` · corpo · 11 · 11.6. A/O candidata/candidato poderá levar uma garrafa de água para seu consumo <br>`35de5046` · corpo · 1 · DO PROCESSO SELETIVO • 4. INFORMAÇÕES DAS VAGAS E COTAS • 5. DOS CURSOS, VAGAS E |
| E04 | novas | k=14 | 20 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`2aba6920` · cronograma · 2 · Data: 14 a 18/11/2026 \| Evento: Período para as/os candidatas/candidatos autodec<br>`4b48f8b1` · cronograma · 2 · Data: De 20 a 22/11/2026 \| Evento: Período para realização de recursos (pedidos  |
| F10a | novas | k=14 | 16 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| F10c | novas | não recuperada | 18 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| F11a | novas | não recuperada | 10 | `dcd678b2` · corpo · 11 · horário de Brasília para os cursos Subsequentes e de Graduação. 11.4. O portão d<br>`af2163e6` · corpo · 11 · os cursos Integrados, Concomitantes e Concomitantes/Subsequente; 11.25.2. 3 hora<br>`6c6d3c7d` · corpo · 11 · III. Selecionar, relacionar, organizar e interpretar informações, fatos, opiniõe |
| F12a | novas | k=11 | 20 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| F13a | novas | k=13 | 19 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| F15a | novas | k=15 | 24 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| F16a | novas | k=18 | 21 | `bbe12232` · corpo · 1 · 1.7 A/o candidata/candidato poderá escolher apenas um modo de ingresso. Quando h<br>`f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`ec9cfb75` · cronograma · 2 · Data: De 30/09 a 19/10/2026 \| Evento: Período de inscrições e solicitação de ise |
| F17b | novas | não recuperada | 10 | `b66f3f19` · corpo · 16 · em duas etapas: 16.5.1. Primeira Chamada: serão analisadas a documentação de mat<br>`875f34c3` · corpo · 17 · utilizará na Chamada Pública os Formulários Preenchidos e a documentação enviada<br>`3206b5f5` · corpo · 17 · 17. O QUE É E COMO FUNCIONA A CHAMADA PÚBLICA 17.1. A Chamada Pública é a última |

**Semântico puro — 14 perguntas falham em k = 10**

| id | grupo | menor k em que a evidência apareceu | chunks no contexto | 3 primeiros chunks (id · tipo · seção · 80 caracteres) |
|---|---|---|---|---|
| A01 | legado | não recuperada | 10 |  |
| B01 | legado | não recuperada | 10 |  |
| B02 | legado | não recuperada | 10 |  |
| B03 | legado | não recuperada | 10 | `e075f0fe` · corpo · 20 · 20.1.10 Campus Ibirubá: Endereço: Rua Nelsi Ribas Fritsch, 1111- Bairro Esperanç |
| B09 | legado | não recuperada | 10 |  |
| B12 | legado | não recuperada | 10 |  |
| C03 | legado | não recuperada | 10 | `66766ecd` · corpo · 11 · solicitado prova em condição especial). A Comissão Permanente de Seleção poderá <br>`6c6d3c7d` · corpo · 11 · III. Selecionar, relacionar, organizar e interpretar informações, fatos, opiniõe<br>`913d47d6` · corpo · 11 · 11.33. Ao concluir a prova, a/o última/último candidata/candidato de cada sala d |
| C05 | legado | não recuperada | 10 | `556238a0` · corpo · 15 · se classificado até o limite da quantidade de vagas destinadas ao acesso univers<br>`18f76e9c` · corpo · 14 · por curso, encontram-se no Anexo IX. 14.4. A Classificação será realizada: 14.4.<br>`35de5046` · corpo · 1 · DO PROCESSO SELETIVO • 4. INFORMAÇÕES DAS VAGAS E COTAS • 5. DOS CURSOS, VAGAS E |
| E02 | legado | não recuperada | 10 |  |
| E03 | novas | não recuperada | 10 | `8f4be260` · corpo · 7 · possível realizar o pagamento da taxa de inscrição após o prazo. 7.6.1. Caso dei<br>`dbddab29` · corpo · 11 · 11.6. A/O candidata/candidato poderá levar uma garrafa de água para seu consumo <br>`35de5046` · corpo · 1 · DO PROCESSO SELETIVO • 4. INFORMAÇÕES DAS VAGAS E COTAS • 5. DOS CURSOS, VAGAS E |
| E04 | novas | não recuperada | 10 | `f88b2371` · corpo · 1 · primeiro semestre de 2027 nos cursos dos campi do IFRS, não sendo válido para in<br>`35de5046` · corpo · 1 · DO PROCESSO SELETIVO • 4. INFORMAÇÕES DAS VAGAS E COTAS • 5. DOS CURSOS, VAGAS E<br>`00532c5d` · corpo · 6 · finalizar da inscrição aparecerá uma mensagem informando se sua solicitação foi  |
| F10c | novas | não recuperada | 10 | `b8f83e71` · corpo · 10 · 10. REALIZAÇÃO DA COMISSÃO DE HETEROIDENTIFICAÇÃO 10.1. De acordo com a Instruçã<br>`de13242b` · corpo · 10 · II. declaração de pertencimento à Comunidade Quilombola (Anexo XIV) assinado por<br>`76ef12cf` · corpo · 17 · candidata/candidato que não comprovar os requisitos para a ocupação de vaga rese |
| F11a | novas | não recuperada | 10 | `dcd678b2` · corpo · 11 · horário de Brasília para os cursos Subsequentes e de Graduação. 11.4. O portão d<br>`af2163e6` · corpo · 11 · os cursos Integrados, Concomitantes e Concomitantes/Subsequente; 11.25.2. 3 hora<br>`6c6d3c7d` · corpo · 11 · III. Selecionar, relacionar, organizar e interpretar informações, fatos, opiniõe |
| F17b | novas | não recuperada | 10 | `b66f3f19` · corpo · 16 · em duas etapas: 16.5.1. Primeira Chamada: serão analisadas a documentação de mat<br>`875f34c3` · corpo · 17 · utilizará na Chamada Pública os Formulários Preenchidos e a documentação enviada<br>`3206b5f5` · corpo · 17 · 17. O QUE É E COMO FUNCIONA A CHAMADA PÚBLICA 17.1. A Chamada Pública é a última |

Quase todas as falhas do híbrido são perguntas sobre datas ou sobre assuntos cuja pergunta contém palavras de `_PALAVRAS_CRONOGRAMA` ("inscrição", "quando", "prazo", "período", "data"), que disparam a injeção do cronograma: o chunk certo está no contexto, mas em posição > 10 (a coluna "menor k" mostra onde). As que **não** estão no contexto (E02, E03, F10c, F11a, F17b: `hit@ctx` do grupo novo = 85%; do legado, só E02) são falhas de ranking semântico: a pergunta tem pouca sobreposição lexical com o trecho (ex.: "A que horas começa a prova?" não recupera o item 11.3). E02 ("Liste todos os cursos…") falha igual à fase 1: nenhum termo dispara o híbrido de tabelas.

## F2.6 Diagnóstico da queda de ranking no grupo legado

Medido com variantes temporárias da primeira construção da fase 2 (cópia dos embeddings, sem reembedar):

| Variante | Resultado (grupo legado) | Conclusão |
|---|---|---|
| fase 2, sem os 32 chunks de legenda | semântico hit@10 70% → 73%; híbrido sem mudança | legenda pesa pouco |
| gravação em ordem de página (adotada) | híbrido hit@10 70% → 73%; C03 10→5, C05 12→6 | ajuda; aplicada no `indexar.py` |
| sem o chunk do Anexo IX do Ibirubá | MRR +0,005 | efeito pequeno; mantido |
| injeção de cronograma por metadado (`tipo`) | híbrido 57 / 83 / 90, MRR 0,465 (final: 57 / 77 / 90) | **explica a queda** |

**Mecanismo.** `nucleo.responder()` injeta *todos* os chunks "cronograma-like" (contêm "cronograma" ou ≥ 3 datas) **antes** dos resultados semânticos. Na fase 1 eram 6 (todos do próprio cronograma). Agora são 14: 6 de cronograma e 8 de corpo que apenas citam "conforme o cronograma do item 2" (seções 1, 2, 10, 11, 13, 16). O chunk certo desce para as posições 11 a 19 e o contexto cresce de ~11 para ~20 a 24 chunks. O teto (`hit@ctx`) não muda; é custo de ranking e de tokens.

Também contribui o tamanho do chunk: uma linha de cronograma passou a ficar em chunks de ~6 linhas (embedding mais "diluído" que o chunk de 150 palavras antigo), o que explica as perdas do modo semântico em C03, C05 e C06. Isso é efeito do que foi pedido (chunks de ~6 linhas) e só importa no modo semântico puro, que a produção não usa sozinho.

## F2.7 Riscos e decisões pendentes (para você)

1. **`nucleo.py` (fora do escopo desta fase).** Recomendo duas mudanças pequenas e independentes do índice: (a) `cronograma_docs` por `meta["tipo"] == "cronograma"` (com a heurística atual só como fallback para índices antigos), o que restaura o híbrido legado para 57 / 77 / 90 e leva o grupo novo a 81% em k=10; (b) o pós-filtro de cotas (`nucleo.py:502-510`) descarta, na consulta, todo chunk não-tabela com `C<n>:`, ou seja, os **32 chunks com esse padrão (30 de legenda e 2 do Anexo I, a legenda C1–C10 do edital)** nunca chegam ao LLM. Enquanto isso não mudar, as legendas ocupam posições do top-k semântico sem benefício.
2. **Texto oculto das legendas** (branco, 1 pt): confirmar que pode ser indexado (F2.2).
3. **Filtro de cotas removido do indexador.** Era o que mantinha os parágrafos "C1:…" fora do índice. Foi removido para preservar as legendas pedidas. Se a decisão for não indexar a "descrição das vagas", basta descartar chunks `tipo=legenda` que contenham `C<n>:`.
4. **Anexo IX** sem cabeçalho de colunas (F2.2).
5. **Extração dependente do formato do edital:** cabeçalhos de campus nos dois formatos conhecidos, títulos de nível 1 em caixa alta, lista de cursos do Ibirubá (`_CURSOS_IBIRUBA`, a atualizar a cada edital). Se o próximo edital mudar de formato, o portão de cobertura falha em vez de indexar menos.
6. **Reindexação de produção não foi executada.** `python indexar.py` grava em `PASTA_CHROMA` (padrão `./chroma_data`); se a cobertura falhar ele sai com código 1 e preserva o índice atual. O app (`documentos_page.py`) captura a exceção e mostra o erro.
7. As versões usadas foram haystack-ai 3.4.0, chroma-haystack 5.0.0, pdfplumber 0.11.10 (venv fora do repositório). `requirements.txt` só fixa mínimos; o `page_number` sobrescrito e o texto extraído podem variar em outra versão.
8. `PIPELINE.md` (seções 1 a 3, descrição do pipeline) ainda descreve a indexação anterior; não foi reescrito nesta fase.

## F2.8 Fase 3 (não feita)

Mudança de chunker (por seção/subitem, sem corte no meio de frase), prefixos `passage:`/`query:`, `distance_function="cosine"` e contexto (título da seção) prependido por chunk. Pré-requisitos já atendidos pelas Partes A a D: índice sem lixo, metadados de seção e página, cronograma estruturado e gabarito com grupos separados. As falhas semânticas restantes (F2.5) são o alvo natural. Decisão sobre F2.7 (1) antes de medir a fase 3, para não confundir efeito de chunking com efeito da injeção.

## F2.9 Reprodução

```
# coleção temporária (nunca ./chroma_data)
python -c "import indexar; indexar.indexar_documentos('./documentos', './chroma_exp_fase2')"
# avaliação
python avaliar_retrieval.py --run base=./chroma_exp_prod --legado base --run fase2=./chroma_exp_fase2
python avaliar_retrieval.py --run base=./chroma_exp_prod --legado base --run fase2=./chroma_exp_fase2 --crono-por-meta --saida retrieval_resultado_meta
```


---

# Fase 2C — cronograma por metadado e diagnóstico de ranking

Escopo (commit `07b9f50`): `nucleo.py` passa a injetar o cronograma pelo metadado `tipo="cronograma"` e o critério antigo ("cronograma" no texto ou ≥ 3 datas) é removido; `avaliar_retrieval.py` ganha ranking completo, posição do chunk com a evidência e diagnóstico de tabelas. Nada de produção foi reindexado.

## 2C.1 O que mudou

| Arquivo | Mudança |
|---|---|
| `nucleo.py` | `cronograma_docs = selecionar_cronograma(...)` (meta `tipo="cronograma"`); `CRONOGRAMA_MODO` = `todos` (padrão, cronograma inteiro) ou `topk` (os `CRONOGRAMA_TOP_K`=3 chunks mais similares, via `cronograma_mais_similares`, que usa o mesmo retriever filtrado por tipo); `responder()` passa o embedding da pergunta (`include_outputs_from`); `_PADRAO_DATA_RECENTE` removido |
| `avaliar_retrieval.py` | modos `semantico`, `hib-top3`, `hib-todos`; rank do chunk com a evidência no ranking completo; auditoria do critério antigo; diagnóstico de `_buscar_tabelas_por_keyword`; a avaliação de referência do índice novo usa a EVIDENCIA atual (sem `--legado`) |
| `avaliar.py`, `teste_timing.py`, `indexar.py`, `PIPELINE.md` | imports e comentários ajustados à seleção por metadado |

## 2C.2 Duas variantes de cronograma (índice da fase 2, EVIDENCIA atual)

| grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |
|---|---|---|---|---|---|---|---|---|
| legado | semantico | 30 | 60% | 63% | 70% | 0.489 | 70% | 6.198 |
| legado | hib-top3 | 30 | 70% | 83% | 93% | 0.530 | 93% | 4.205 |
| legado | hib-todos | 30 | 57% | 77% | 90% | 0.465 | 97% | 5.075 |
| novas | semantico | 27 | 63% | 78% | 81% | 0.499 | 81% | 9.039 |
| novas | hib-top3 | 27 | 41% | 63% | 81% | 0.398 | 81% | 8.359 |
| novas | hib-todos | 27 | 37% | 52% | 81% | 0.364 | 85% | 9.063 |

Nas 57 perguntas: `hit@10` top3 = 50, todos = 49; `hit@ctx` top3 = 50, todos = 52. **Escolhido `todos`** (padrão do `nucleo.py`): o `hit@ctx` é o que o LLM recebe; `todos` cobre C03 (o chunk certo não está entre os 3 mais similares) e E04 (4 datas de chunks diferentes). Custo: o cronograma inteiro tem 5.937 caracteres (≈ 1.696 tokens); contexto híbrido máximo 13.656 caracteres (≈ 3.900 tokens).

## 2C.3 Critério antigo removido

No índice da fase 2 ele selecionava 14 chunks: os 6 do cronograma e 8 de corpo que apenas citam o cronograma (seções 1, 1, 2, 10, 10, 11, 13 e 16). Nas perguntas que disparam o cronograma o fluxo antigo cobre 15/16 (igual a `hib-todos`) e, nas 57, 52 contra 52; nenhuma pergunta depende só dos 8 chunks. Contexto médio do fluxo antigo nessas perguntas: 20,8 chunks, e só 2 de 16 acertavam em k=10.

## 2C.4 Diagnóstico B08 / E02 e achados que viraram a Fase 3

- **E02** ("Liste todos os cursos"): 0 das 9 tabelas chegavam ao contexto. Critério 1 (tipo de curso) vazio e critério 2 (nome) só recebia "liste"/"todos".
- **B08**: as 4 tabelas de graduação chegavam pelo top-10 semântico; a de Ciência da Computação caía na posição 11 porque o cronograma entrava na frente (falso positivo do gatilho em "período noturno").
- **Gatilho do cronograma**: 5 das 16 perguntas que o disparavam não são de data (B08, D02, F10a, F10c, F15a): `data` casava dentro de "candi**data**", "inscrição" em "taxa de inscrição". → Fase 3, item 1.
- **Ids**: os ids de chunk do servidor diferiam dos locais porque o hash do Haystack inclui o `file_path` (`documentos\...` no Windows, `documentos/...` no Linux). → Fase 3, item 4.
- **Reprodutibilidade**: o top-10 aproximado do Chroma (HNSW) pode trocar o chunk da 10ª posição entre execuções (B05, B08); só o `chars_ctx` oscila, não os acertos nem o MRR.

---

# Fase 3 — chunking e recuperação no índice completo

Escopo: `indexar.py`, `nucleo.py`, `avaliar_retrieval.py`, `avaliar.py`/`teste_timing.py` (prefixo de consulta), `PIPELINE.md`. Validação **só em coleções temporárias** `./chroma_exp_f3_*`; a produção não foi reindexada (não há `./chroma_data` neste checkout). Modelo de embedding e base vetorial inalterados. Cada item tem uma variante controlada por variável de ambiente; os grupos legado (30) e novas (27) ficam separados em todas as tabelas. Relatório completo: `retrieval_resultado_fase3.md` (ignorado pelo git, como os demais `retrieval_resultado_*`).

## F3.0 Resumo

| Item | Resultado |
|---|---|
| 1 — gatilho do cronograma | **Concluído.** Palavra inteira + termos ambíguos só com evento do processo. 16 → 11 perguntas disparam; deixam de disparar B08, D02, F10a, F10c, F15a (nenhuma passa a disparar). Híbrido legado hit@10 90% → 93%, MRR 0,465 → 0,497 |
| 2 — cabeçalho nas tabelas | **Concluído.** Semântico legado hit@10 70% → 87%. B01, B09, B12, B03, A01 entram no top-10 sem injeção (ex.: B01 de 151 para 5). **Regressão:** B02 (17 → 63) |
| 2 — E02 | **Resolvido** com o critério "listar todos os cursos": custo de 9 tabelas = 999 caracteres (2.152 com cabeçalho) |
| 3 — chunker por item | **Concluído.** Chunks de corpo com 2+ seções: 17 → 0. Semântico novas hit@10 81% → 89%, MRR 0,50 → 0,75; **melhor variante: item + título** |
| 4 — ids estáveis | **Concluído e provado** (5 grafias de caminho, 2 indexações seguidas) |
| 5 — prefixos E5 | **Medido.** Ganho no semântico (+4 pp hit@10 nos dois grupos); híbrido misto. **Desligado por padrão** |
| 6 — metas | **Todas atingidas** por `final` e `final+e5` (tabela em F3.7) |
| 7 — prompt | **Reescrito** (diff em F3.8); sem LLM aqui, não medido em resposta |
| 8 — aceite F11a | **Passou** com os chunkers por item (8h30min e 14h no chunk 11.3, posição 1) e falha com o chunker atual |

## F3.1 Variantes e variáveis

| Variável | Onde | Valores | Padrão (após a Fase 3) |
|---|---|---|---|
| `IDX_CHUNKER` | `indexar.py` | `atual` · `item` · `item_titulo` | `item_titulo` |
| `IDX_TABELA_CABECALHO` | `indexar.py` | `0` · `1` | `1` |
| `IDX_IDS` | `indexar.py` | `estavel` · `legado` | `estavel` |
| `E5_PREFIXOS` | `indexar.py` e `nucleo.py` (têm que coincidir) | `0` · `1` | `0` |
| `CRONOGRAMA_GATILHO` | `nucleo.py` | `palavra` · `substring` | `palavra` |
| `TABELAS_LISTAR_TODAS` | `nucleo.py` | `0` · `1` | `1` |
| `CRONOGRAMA_MODO` | `nucleo.py` | `todos` · `topk` | `todos` |

No avaliador, cada `--run` aceita `@gatilho=palavra|substring,listar=0|1` para comparar as variantes de consulta sobre o mesmo índice. Coleções: `2C` = índice da fase 2 com o comportamento da 2C; `+gatilho`, `+listar` = mesmo índice com cada mudança de consulta; `+cab` = só o cabeçalho de tabela; `+item`, `+item+tit` = só o chunker; `final` = item + título + cabeçalho + gatilho + listar; `final+e5` = `final` com prefixos.

## F3.2 Item 1 — gatilho do cronograma

`dispara_cronograma()`: casamento por palavra inteira, sem acento. Termos fortes sozinhos: `cronograma`, `calendário`, `data(s)`, `prazo(s)`. Termos ambíguos (`quando`, `período(s)`, `etapa(s)`) só junto de um evento do processo (inscrição, prova, resultado, classificação, matrícula, recurso, chamada, sorteio, divulgação, publicação, gabarito, isenção, convocação, homologação, "processo seletivo"). "Inscrição" sozinha deixou de ser gatilho.

16 antes: B08, C01–C07, D02, E04, F10a, F10c, F12a, F13a, F15a, F16a. 11 depois: C01–C07, E04, F12a, F13a, F16a. Deixam de disparar: **B08, D02, F10a, F10c, F15a**. Efeito (mesmo índice, `2C` → `+gatilho`): híbrido legado hit@5 77% → 83%, hit@10 90% → 93%, MRR 0,465 → 0,497, `chars_ctx` 5.075 → 4.714; híbrido novas hit@3 37% → 44%, hit@5 52% → 59%, MRR 0,364 → 0,403, `chars_ctx` 9.063 → 8.441. `hit@ctx` não muda.

## F3.3 Item 2 — cabeçalho descritivo nas tabelas, e E02

Cada chunk de tabela ganha uma linha lida do PDF: número e título do anexo (agora completo: "…CONCOMITANTES E CONCOMITANTES/SUBSEQUENTES"), o rótulo da tabela (1ª célula do cabeçalho: "Cursos Técnicos Integrados ao Ensino Médio", "Cursos Superiores"…) e a formação, lida do prefixo do próprio curso. Exemplo:

```
Campus Ibirubá — Anexo V, Quadros de vagas - cursos de graduação: Cursos Superiores (bacharelado).
Curso: Bacharelado em Agronomia | Turnos: Manhã e Tarde | Duração (semestres): 10 semestres* | Total de Vagas: 30
```

O cabeçalho evita "Técnico em / Bacharelado em / Licenciatura em", pois `nucleo._tipo_tabela_chunk` classifica o chunk por esses prefixos. Rank semântico no ranking completo (sem injeção); célula = o maior dos melhores ranks de cada evidência, isto é, o menor top-k que cobre a pergunta; ✓ = cobre no top-10:

| id | pergunta | 2C (antes) | +cab | final | final+e5 |
|---|---|---|---|---|---|
| A01 | cursos técnicos disponíveis | 28 | **7** ✓ | **7** ✓ | **5** ✓ |
| B01 | cursos superiores oferecidos | 151 | **5** ✓ | **5** ✓ | **4** ✓ |
| B02 | bacharelados disponíveis | 17 | 63 | 75 | 30 |
| B03 | alguma licenciatura | 19 | **2** ✓ | **2** ✓ | **1** ✓ |
| B09 | cursos superiores do campus | 152 | **5** ✓ | **5** ✓ | **4** ✓ |
| B12 | quantos cursos superiores | 140 | **4** ✓ | **4** ✓ | **4** ✓ |
| B08 | superiores no período noturno | 5 ✓ | 3 ✓ | 3 ✓ | 3 ✓ |
| E02 | liste todos os cursos | 133 | 14 | 13 | 13 |

Cinco perguntas (A01, B01, B03, B09, B12) deixam de depender da injeção por palavra-chave. **B02 piora** com o cabeçalho (as 3 tabelas descem de 7–17 para 39–75; não investiguei a causa); no híbrido continua coberta pelo critério de tipo (`bacharelados` ∈ `_SINONIMOS_TIPO`).

**E02** continua fora do top-10 (rank 13), então entra o critério 3 de `_buscar_tabelas_por_keyword`: se a pergunta tem `liste/lista/listar/todos/todas/quais` + `cursos`, não tem tipo de curso e os critérios 1 e 2 não acharam nada, injeta as tabelas ainda ausentes. Não dispara em "O campus oferece curso de Direito?" (D01). **Custo:** as 9 tabelas somam 999 caracteres (≈ 285 tokens) sem cabeçalho e 2.152 (≈ 615 tokens) com cabeçalho; o contexto de E02 passa de 2.167 para 3.173 caracteres no índice antigo (`+listar`) e fica em 2.354 caracteres (10 chunks) no índice final. E02: evidências 0/8 → 8/8 no contexto.

## F3.4 Item 3 — chunker do corpo por item numerado

Item = linha que começa com `N.N.` cujo `N` é o número da seção de nível 1 em curso (guarda contra números soltos no texto). Subitens `N.N.N` ficam na unidade do item pai; um item acima de 200 palavras é cortado primeiro nos subitens (repetindo a 1ª linha do item, se curta) e, se um subitem sozinho passar do teto, em janelas de 200 palavras com sobreposição de 20. Nunca cruza seção de nível 1. Com `item_titulo` cada chunk começa com `[Seção 11 — REALIZAÇÃO DA PROVA]`. Metadados preservados (`secao`, `secoes`, `pagina_ini`, `pagina_fim`, `tipo`, `escopo_campus`); `pagina_ini/fim` agora podem diferir quando o item cruza a página.

| chunker | chunks de corpo | palavras (mediana / máx) | chunks com 2+ seções de nível 1 |
|---|---|---|---|
| atual (150/20 por página) | 66 | 165 / 170 | **17** (inclui `3b18b051`: seções 10 e 11; `9ccf77d8`: 17 e 18) |
| item | 160 | 41 / 200 | 0 |
| item + título | 160 | 47 / 211 | 0 |

Os portões foram preservados, com dois ajustes que o chunker fino exigiu: (a) **legibilidade**: formas com barra da escrita inclusiva ("da/do", "candidata/candidato") contam como um token na fração de tokens curtos (itens curtos como 11.15 chegavam a 44% e eram rejeitados); (b) o **mínimo avaliado** pelo portão passou de 20 para 30 tokens (11.15 tem 25 tokens e 40% de palavras funcionais, o limite do portão; os demais chunks de corpo chegam a 38%). Resultado: 0 rejeitados nos três chunkers; portão de cobertura sem falhas.

Resultado isolado (mesmo índice de tabelas, gatilho antigo): `+item` e `+item+tit` melhoram muito o grupo novas (semântico hit@10 81% → 89%, MRR 0,499 → 0,744/0,749; híbrido `hit@ctx` 85% → 93%) e **pioram o legado sozinhos** (semântico MRR 0,489 → 0,385; híbrido hit@10 90% → 83%/87%): muitos chunks curtos de corpo passam à frente das linhas de tabela. Isso é compensado pelo cabeçalho (F3.3). Entre `item` e `item+tit` o título ganha em hit@5 (novas semântico 81% → 89%), híbrido novas hit@10 (89% → 93%) e híbrido legado hit@10 (83% → 87%).

## F3.5 Item 4 — ids estáveis

`id = sha256(nome do arquivo | tipo | página | posição entre os chunks do mesmo tipo e página)`; `file_path` guarda só o nome do arquivo. Provado com a mesma PDF indexada com cinco grafias de caminho (relativo com `/`, relativo com `\`, absoluto com `\`, absoluto com `/`, cópia em outra pasta com espaço) e em duas indexações seguidas:

| `IDX_IDS` | ids iguais entre as 5 grafias | 2ª indexação seguida | `file_path` com separador | ids únicos |
|---|---|---|---|---|
| `estavel` | **sim (173/173)** | sim | 0 de 173 | sim |
| `legado` | **não (0/173 em comum)** | sim | 173 de 173 | sim |

Impacto: o `nucleo.py` só usa o id para evitar repetir chunk (`ids_semanticos`) e o `teste_timing.py` para rotular a origem no debug; nenhum id é persistido (`db.py` não guarda chunks). Como o id é posicional, ele muda quando muda a quantidade de chunks daquela página (por exemplo, ao trocar de chunker), não quando o texto muda; não há dependência externa de ids antigos.

## F3.6 Item 5 — prefixos do E5 (sobre o melhor chunker)

`E5_PREFIXOS=1`: `passage: ` no texto embedado dos chunks (o texto gravado não muda) e `query: ` na consulta. Medido só em cima de `final`:

| grupo | modo | | hit@3 | hit@5 | hit@10 | MRR | hit@ctx |
|---|---|---|---|---|---|---|---|
| legado | semantico | sem | 47% | 67% | 83% | 0.411 | 83% |
| legado | semantico | com | 53% | 73% | 87% | 0.457 | 87% |
| legado | híbrido | sem | 60% | 77% | 93% | 0.481 | 100% |
| legado | híbrido | com | 57% | 80% | 93% | 0.495 | 100% |
| novas | semantico | sem | 81% | 89% | 89% | 0.749 | 89% |
| novas | semantico | com | 85% | 85% | 93% | 0.820 | 93% |
| novas | híbrido | sem | 59% | 81% | 93% | 0.596 | 93% |
| novas | híbrido | com | 70% | 81% | 89% | 0.678 | 96% |

Ganho consistente no semântico (+4 pp em hit@10 nos dois grupos, MRR +0,05 a +0,07); no híbrido `hit@ctx` das novas sobe de 93% para 96% e o MRR sobe, mas o hit@10 das novas cai uma pergunta (93% → 89%). Contexto máximo sobe de 2.755 para 3.414 tokens estimados. São 57 perguntas: a diferença é de 1 a 2 perguntas por célula. **Recomendação:** ligar `E5_PREFIXOS=1` quando a produção for reindexada, **nos dois lados** (índice e consulta); por isso o padrão fica desligado (um índice sem prefixo consultado com `query: `, ou o contrário, degrada o ranking sem erro).

`distance_function` não foi alterado: os embeddings do E5 saem com norma 1,000000 (267 chunks e 59 consultas) e, com vetores normalizados, l2 e cosseno ordenam igual. A comparação completa das duas ordenações dos 267 chunks diferiu em 1 consulta de 59, numa troca nas posições 193/194 por uma diferença de distância de 1,4e-9 (empate numérico).

## F3.7 Item 6 — tabelas completas, falhas restantes e metas

Híbrido = `hib-todos` (cronograma inteiro). Legado: `n = 30`; novas: `n = 27`.

| coleção | grupo | modo | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |
|---|---|---|---|---|---|---|---|---|
| 2C | legado | semantico | 60% | 63% | 70% | 0.489 | 70% | 6.198 |
| 2C | legado | híbrido | 57% | 77% | 90% | 0.465 | 97% | 5.075 |
| +gatilho | legado | híbrido | 60% | 83% | 93% | 0.497 | 97% | 4.714 |
| +listar | legado | híbrido | 60% | 83% | 97% | 0.500 | 100% | 4.748 |
| +cab | legado | semantico | 57% | 70% | 87% | 0.515 | 87% | 6.533 |
| +cab | legado | híbrido | 53% | 73% | 93% | 0.443 | 97% | 5.357 |
| +item | legado | semantico | 53% | 53% | 70% | 0.385 | 70% | 4.543 |
| +item | legado | híbrido | 53% | 70% | 83% | 0.455 | 97% | 3.838 |
| +item+tit | legado | semantico | 50% | 57% | 67% | 0.383 | 67% | 4.642 |
| +item+tit | legado | híbrido | 57% | 73% | 87% | 0.485 | 97% | 3.962 |
| **final** | legado | semantico | 47% | 67% | **83%** | 0.411 | 83% | 4.798 |
| **final** | legado | híbrido | 60% | 77% | 93% | 0.481 | **100%** | 3.812 |
| final+e5 | legado | semantico | 53% | 73% | 87% | 0.457 | 87% | 5.228 |
| final+e5 | legado | híbrido | 57% | 80% | 93% | 0.495 | 100% | 3.922 |
| 2C | novas | semantico | 63% | 78% | 81% | 0.499 | 81% | 9.039 |
| 2C | novas | híbrido | 37% | 52% | 81% | 0.364 | 85% | 9.063 |
| +gatilho / +listar | novas | híbrido | 44% | 59% | 81% | 0.403 | 85% | 8.441 |
| +cab | novas | semantico | 63% | 78% | 81% | 0.493 | 81% | 8.896 |
| +cab | novas | híbrido | 41% | 56% | 81% | 0.370 | 85% | 9.135 |
| +item | novas | semantico | 78% | 81% | 89% | 0.744 | 89% | 3.896 |
| +item | novas | híbrido | 44% | 67% | 89% | 0.495 | 93% | 4.986 |
| +item+tit | novas | semantico | 81% | 89% | 89% | 0.749 | 89% | 3.927 |
| +item+tit | novas | híbrido | 48% | 70% | 93% | 0.500 | 93% | 5.178 |
| **final** | novas | semantico | 81% | 89% | **89%** | 0.749 | 89% | 3.911 |
| **final** | novas | híbrido | 59% | 81% | 93% | 0.596 | **93%** | 4.590 |
| final+e5 | novas | semantico | 85% | 85% | 93% | 0.820 | 93% | 4.782 |
| final+e5 | novas | híbrido | 70% | 81% | 89% | 0.678 | 96% | 5.178 |

Metas (tokens estimados = caracteres / 3,5):

| meta | alvo | `final` | `final+e5` |
|---|---|---|---|
| hit@ctx híbrido, legado | ≥ 97% | 100% ✓ | 100% ✓ |
| hit@ctx híbrido, novas | ≥ 90% | 93% ✓ | 96% ✓ |
| hit@10 semântico, legado | ≥ 80% | 83% ✓ | 87% ✓ |
| hit@10 semântico, novas | ≥ 80% | 89% ✓ | 93% ✓ |
| E02 resolvido (8/8 evidências no contexto) | sim | sim ✓ | sim ✓ |
| contexto máximo | ≤ 4.000 tokens | 9.641 caracteres ≈ **2.755** ✓ | 11.948 caracteres ≈ **3.414** ✓ |

O contexto máximo da 2C era 13.656 caracteres (≈ 3.902 tokens). Nenhuma meta depende de `E5_PREFIXOS`.

**Falhas restantes e posição do chunk com a evidência** (rank no ranking semântico completo de 267 chunks; "+N" = N posições além do top-10):

| coleção · modo | pergunta | chunk (seção) | rank | leitura |
|---|---|---|---|---|
| final · semântico | E02 | 3 tabelas técnicas (Anexos II/IV) | 11, 13, 14 | a 1–4 posições do top-10; a injeção "listar" cobre |
| final · semântico | C01 | `a8c3d0f0` (cronograma 2) | 12 (+2) | a 2 posições; o híbrido injeta o cronograma |
| final · semântico | F16a | `ac51fb01` (corpo 16) | 12 (+2) | a 2 posições do top-10 |
| final · semântico | B02 | 3 tabelas de bacharelado (Anexo V) | 42, 62, 75 | regressão do cabeçalho; híbrido cobre por tipo |
| final · semântico | C03, C05, E04 | chunks de cronograma | 28–208 | cronograma: o híbrido cobre |
| final · semântico/híbrido | E03 | `8c9240e7` (7) e `065bf643` (7) | 44 e 122 | **perdido**: duas evidências em itens diferentes (7.1 e 7.2); a pergunta genérica não se parece com nenhum |
| final · híbrido | F16a | `ac51fb01` (16) | 12 (+2) | não entra no contexto: fica a 2 posições; **ajuste de ranking** |
| final · híbrido | C02, C06 | cronograma | 9 e 10 (dentro) | cobertos no contexto, mas na posição 14–15 (cronograma na frente empurra o chunk certo) |

Com `final+e5`, E03 vira `8c9240e7` rank 15 (+5) e `065bf643` rank 68; o híbrido falha em k=10 em C01, C02, E04 e F16a só por posição no contexto (11 a 14; todos cobertos) e E03 não é coberto. Casos de foco da 2C, `k_min` (menor k que cobre a pergunta) de `2C` → `final`, semântico e híbrido: **F11a** — → 1; **F10c** — → 1; **F17b** — → 1; **F10a** 3/8 → 1/1; **F15a** 1/7 → 1/1 (em F10a e F15a o híbrido piorava por causa do cronograma injetado sem necessidade; agora a pergunta nem dispara). A posição do chunk no contexto não foi medida em resposta de LLM.

## F3.8 Item 7 — prompt do LLM (`nucleo.montar_prompt`)

Problema: a regra `COMPLETUDE OBRIGATÓRIA … Cada linha que começa com "Curso:" é um curso diferente` e o exemplo de registro só citavam "Curso:"; o `gemma3:4b` imitava o prefixo em respostas de data ("Curso: 30/09 a 19/10/2026 | …"). Antes: um texto fixo com 7 regras para toda pergunta. Depois: `montar_prompt()` monta as regras conforme a pergunta.

```
- COMPLETUDE OBRIGATÓRIA: ao listar cursos, percorra TODO o contexto … Cada linha que começa com "Curso:" é um curso diferente …   (sempre)
- Cada linha do contexto é um registro independente … Nunca combine o nome de um curso com turno, duração ou vagas …           (sempre)
- Se o contexto trouxer cursos de categorias diferentes (técnico integrado, …) …                                               (sempre)
```
passa a
```
COMPLETUDE OBRIGATÓRIA (a pergunta pede cursos) … "Isso vale só para listar cursos."   → só se a pergunta fala de cursos (curso, técnico, superiores, bacharelado, licenciatura, graduação, integrado…)
Categorias de curso                                                                    → idem
Registro: ex.: "Curso: … | Turnos: …"  (pergunta de cursos)  |  ex.: "Data: … | Evento: …"  (demais)
Frase própria: NÃO copie linhas do contexto e NÃO use os rótulos ("Curso:", "Turnos:", … | "Data:", "Evento:") no texto da resposta; para uma lista, um item por linha só com os nomes e os dados pedidos.   (sempre)
```

Como cada mudança evita o problema: (1) o termo "Curso:" **não aparece mais** no prompt de perguntas que não são de cursos (0 ocorrências no prompt de "Quando são as inscrições…", antes 2), então o modelo não recebe o prefixo para imitar; (2) o exemplo do registro e a lista de rótulos proibidos mudam com o tipo de pergunta ("Data:"/"Evento:" nas demais, "Curso:"/"Turnos:"… nas de cursos); (3) a nova regra pede resposta em frase própria e proíbe copiar linhas e rótulos, atacando também respostas de curso que repetiam "Curso:"; (4) a regra de completude continua para listas de cursos, que é onde ela evita omissões, e agora diz que vale só para listar cursos. **Não medido em resposta de LLM** (sem Ollama aqui); o avaliador do `teste_modelos.py` ganhou a coluna "observação" para pegar linhas cruas.

## F3.9 Item 8 — aceite F11a (só retrieval)

Contexto híbrido de "Que horas começa a prova?" (e da pergunta do gabarito, "A que horas começa a prova?") precisa conter `8h30min (oito horas e trinta minutos)` e `14h (catorze horas)`, e não só o chunk do portão (11.4, 7h45min / 13h15min).

| coleção | passou | onde |
|---|---|---|
| 2C, +gatilho, +listar, +cab (chunker atual) | **não** | só o chunk do portão, posição 1 |
| +item, +item+tit, final, final+e5 | **sim** | chunk `04bd8dd4` (item 11.3) na posição 1; portão (11.4) na posição 2 |

## F3.10 Riscos e pendências

1. **Produção não foi reindexada.** Os padrões do `indexar.py` agora são `item_titulo` + cabeçalho de tabela + ids estáveis; `python indexar.py` aplica tudo (e preserva o índice atual se o portão de cobertura falhar). O código novo continua funcionando com índices antigos (gatilho e "listar" são de consulta).
2. **E5:** ligar nos dois lados ao mesmo tempo (F3.6).
3. **57 perguntas:** diferenças de 1 a 2 perguntas (E5, entre `item` e `item+tit`) não são estatisticamente firmes; as mudanças grandes (cabeçalho, chunker) são.
4. **Cronograma na frente do contexto** empurra chunks corretos para as posições 11–15 (C01, C02, C06, E04, F16a). Está dentro do contexto, mas é candidato a reordenar (cronograma depois dos 3 primeiros semânticos).
5. **E03** segue perdida ("O que preciso para me inscrever…"): a resposta está em dois itens distintos da seção 7.
6. **B02** regrediu no semântico com o cabeçalho; a cobertura vem do critério de tipo.
7. **Reprodutibilidade:** o Chroma (HNSW) pode trocar o chunk da 10ª posição entre execuções; só o `chars_ctx` oscila.
8. `teste_modelos.py` (avaliador de respostas) foi reescrito à parte: datas por valor, contagem, observação de linhas cruas, `--todas`, `--reavaliar`, `--autoteste`.

## F3.11 Reprodução

```
# índices temporários (nunca ./chroma_data); IDX_* e E5_PREFIXOS são lidos na importação do indexar
IDX_CHUNKER=atual       IDX_TABELA_CABECALHO=1 python -c "import indexar; indexar.indexar_documentos('./documentos', './chroma_exp_f3_cab')"
IDX_CHUNKER=item        IDX_TABELA_CABECALHO=0 python -c "import indexar; indexar.indexar_documentos('./documentos', './chroma_exp_f3_item')"
IDX_CHUNKER=item_titulo IDX_TABELA_CABECALHO=0 python -c "import indexar; indexar.indexar_documentos('./documentos', './chroma_exp_f3_itemtit')"
IDX_CHUNKER=item_titulo IDX_TABELA_CABECALHO=1 python -c "import indexar; indexar.indexar_documentos('./documentos', './chroma_exp_f3_final')"
IDX_CHUNKER=item_titulo IDX_TABELA_CABECALHO=1 E5_PREFIXOS=1 python -c "import indexar; indexar.indexar_documentos('./documentos', './chroma_exp_f3_final_e5')"
# avaliação (o prefixo de consulta do E5 entra depois de ::, as opções de consulta depois de @)
python avaliar_retrieval.py --saida retrieval_resultado_fase3 \
  --run "2C=./chroma_exp_fase2@gatilho=substring,listar=0" --run "+gatilho=./chroma_exp_fase2@gatilho=palavra,listar=0" \
  --run "+listar=./chroma_exp_fase2@gatilho=palavra,listar=1" --run "+cab=./chroma_exp_f3_cab@gatilho=substring,listar=0" \
  --run "+item=./chroma_exp_f3_item@gatilho=substring,listar=0" --run "+item+tit=./chroma_exp_f3_itemtit@gatilho=substring,listar=0" \
  --run "final=./chroma_exp_f3_final@gatilho=palavra,listar=1" --run "final+e5=./chroma_exp_f3_final_e5::query: @gatilho=palavra,listar=1"
```
