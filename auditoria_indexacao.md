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
