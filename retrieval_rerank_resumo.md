# Reranking sobre o retrieval do IF Turing — resumo

Gerado em 2026-10-21 14:41 por `avaliar_rerank.py relatar` a partir de `rerank_cache/`. Índice: `./chroma_exp_f3_final` (267 chunks; item+título, cabeçalho de tabela, ids estáveis); opções de consulta {'gatilho': 'palavra', 'listar': '1'}; híbrido = `hib-todos` (cronograma inteiro). **Este experimento só lê índices temporários; o reranker existe em `nucleo.py` como opção desligada por padrão (`RERANKER_ATIVO`, ver GPUS.md).**

## Decisões, falhas e ressalvas (modo sem supervisão)

- Modelos: grande `BAAI/bge-reranker-v2-m3` (max_length 256 e 384); menor `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; substituto de reserva `BAAI/bge-reranker-base` (um único substituto por modelo). `trust_remote_code` **nunca** foi usado; nada foi instalado no venv (usei só `sentence-transformers` 6.1 / `transformers` 5.20 / `torch` 2.14 CPU já existentes).
- Cada pergunta tem os 30 primeiros candidatos do ranking semântico exato; pontuados UMA vez por (modelo, max_length). Os cenários 15 e 20 usam os 15/20 primeiros desses 30, sem repontuar; o reranker reordena e ficam os 10 primeiros.
- Baseline = ranking semântico exato (10 primeiros) sem reranker; ele pode diferir em ±1 chunk do retriever aproximado do Chroma usado em relatórios anteriores.
- **Tempos de 15 e 20 candidatos são DERIVADOS** (tempo medido com 30 × N/30), não medidos.
- Hardware desta medição: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, torch com 4 threads, **CPU, fp32**. Não há GPU nesta máquina.
- Perguntas do subconjunto foco: E03, F16a, F18a, A07, E01, A01, B08, C02, C06, C07, F11b, F12a, F13b, F14a, F18b (as 10 últimas são aleatórias com semente fixa 20261021).
- Nenhum erro de carregamento ou pontuação foi registrado.
- `BAAI/bge-reranker-v2-m3` ml=256: 57 perguntas pontuadas; estimativa da sonda 2291 s.
- `BAAI/bge-reranker-v2-m3` ml=384: 57 perguntas pontuadas; estimativa da sonda 2215 s.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` ml=256: 57 perguntas pontuadas; estimativa da sonda 101 s.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` ml=384: 57 perguntas pontuadas; estimativa da sonda 149 s.

## Métricas por configuração

`semantico` = lista reordenada pelo reranker (10 primeiros); `híbrido` = o mesmo fluxo de `nucleo.responder()` alimentado por essa lista. Grupos legado e novas separados.

| configuração | N cand. | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (sem reranker) | — | legado | semântico | 30 | 47% | 67% | 83% | 0.411 | 83% | 4,798 |
| baseline (sem reranker) | — | legado | híbrido | 30 | 60% | 77% | 93% | 0.481 | 100% | 3,812 |
| baseline (sem reranker) | — | novas | semântico | 27 | 81% | 89% | 89% | 0.749 | 89% | 3,911 |
| baseline (sem reranker) | — | novas | híbrido | 27 | 59% | 81% | 93% | 0.596 | 93% | 4,590 |
| bge-reranker-v2-m3 ml=256 | 15 | legado | semântico | 30 | 60% | 80% | 87% | 0.450 | 87% | 4,864 |
| bge-reranker-v2-m3 ml=256 | 15 | legado | híbrido | 30 | 63% | 83% | 100% | 0.508 | 100% | 3,672 |
| bge-reranker-v2-m3 ml=256 | 15 | novas | semântico | 27 | 89% | 93% | 93% | 0.843 | 93% | 4,461 |
| bge-reranker-v2-m3 ml=256 | 15 | novas | híbrido | 27 | 74% | 85% | 96% | 0.701 | 96% | 4,976 |
| bge-reranker-v2-m3 ml=256 | 20 | legado | semântico | 30 | 60% | 80% | 87% | 0.450 | 87% | 4,858 |
| bge-reranker-v2-m3 ml=256 | 20 | legado | híbrido | 30 | 63% | 83% | 100% | 0.508 | 100% | 3,703 |
| bge-reranker-v2-m3 ml=256 | 20 | novas | semântico | 27 | 89% | 93% | 93% | 0.843 | 93% | 4,821 |
| bge-reranker-v2-m3 ml=256 | 20 | novas | híbrido | 27 | 74% | 89% | 96% | 0.708 | 96% | 5,169 |
| bge-reranker-v2-m3 ml=256 | 30 | legado | semântico | 30 | 63% | 80% | 90% | 0.466 | 90% | 5,095 |
| bge-reranker-v2-m3 ml=256 | 30 | legado | híbrido | 30 | 60% | 83% | 100% | 0.506 | 100% | 3,805 |
| bge-reranker-v2-m3 ml=256 | 30 | novas | semântico | 27 | 89% | 93% | 93% | 0.843 | 93% | 5,295 |
| bge-reranker-v2-m3 ml=256 | 30 | novas | híbrido | 27 | 74% | 85% | 96% | 0.707 | 96% | 5,573 |
| bge-reranker-v2-m3 ml=384 | 15 | legado | semântico | 30 | 67% | 83% | 90% | 0.493 | 90% | 4,965 |
| bge-reranker-v2-m3 ml=384 | 15 | legado | híbrido | 30 | 60% | 83% | 100% | 0.506 | 100% | 3,698 |
| bge-reranker-v2-m3 ml=384 | 15 | novas | semântico | 27 | 89% | 93% | 93% | 0.843 | 93% | 4,562 |
| bge-reranker-v2-m3 ml=384 | 15 | novas | híbrido | 27 | 74% | 85% | 96% | 0.701 | 96% | 5,077 |
| bge-reranker-v2-m3 ml=384 | 20 | legado | semântico | 30 | 67% | 83% | 90% | 0.493 | 90% | 4,987 |
| bge-reranker-v2-m3 ml=384 | 20 | legado | híbrido | 30 | 60% | 83% | 100% | 0.506 | 100% | 3,696 |
| bge-reranker-v2-m3 ml=384 | 20 | novas | semântico | 27 | 89% | 93% | 93% | 0.843 | 93% | 4,890 |
| bge-reranker-v2-m3 ml=384 | 20 | novas | híbrido | 27 | 74% | 89% | 96% | 0.708 | 96% | 5,197 |
| bge-reranker-v2-m3 ml=384 | 30 | legado | semântico | 30 | 70% | 87% | 93% | 0.504 | 93% | 5,195 |
| bge-reranker-v2-m3 ml=384 | 30 | legado | híbrido | 30 | 60% | 80% | 100% | 0.506 | 100% | 3,756 |
| bge-reranker-v2-m3 ml=384 | 30 | novas | semântico | 27 | 89% | 93% | 93% | 0.841 | 93% | 5,455 |
| bge-reranker-v2-m3 ml=384 | 30 | novas | híbrido | 27 | 74% | 85% | 96% | 0.706 | 96% | 5,733 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | legado | semântico | 30 | 40% | 60% | 87% | 0.367 | 87% | 4,987 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | legado | híbrido | 30 | 53% | 87% | 90% | 0.480 | 100% | 3,621 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | novas | semântico | 27 | 93% | 93% | 93% | 0.852 | 93% | 4,368 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | novas | híbrido | 27 | 67% | 85% | 96% | 0.668 | 96% | 4,906 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | legado | semântico | 30 | 40% | 57% | 83% | 0.360 | 83% | 5,015 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | legado | híbrido | 30 | 57% | 87% | 93% | 0.508 | 100% | 3,451 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | novas | semântico | 27 | 93% | 93% | 93% | 0.852 | 93% | 4,372 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | novas | híbrido | 27 | 67% | 89% | 96% | 0.669 | 96% | 4,871 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | legado | semântico | 30 | 43% | 60% | 77% | 0.364 | 77% | 5,146 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | legado | híbrido | 30 | 57% | 90% | 100% | 0.517 | 100% | 3,561 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | novas | semântico | 27 | 93% | 93% | 93% | 0.852 | 93% | 4,772 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | novas | híbrido | 27 | 67% | 81% | 96% | 0.666 | 96% | 5,232 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | legado | semântico | 30 | 43% | 67% | 90% | 0.386 | 90% | 5,051 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | legado | híbrido | 30 | 50% | 87% | 93% | 0.477 | 100% | 3,610 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | novas | semântico | 27 | 93% | 93% | 93% | 0.852 | 93% | 4,368 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | novas | híbrido | 27 | 67% | 85% | 96% | 0.668 | 96% | 4,929 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | legado | semântico | 30 | 43% | 63% | 87% | 0.380 | 87% | 5,078 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | legado | híbrido | 30 | 53% | 87% | 97% | 0.504 | 100% | 3,509 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | novas | semântico | 27 | 93% | 93% | 93% | 0.852 | 93% | 4,497 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | novas | híbrido | 27 | 67% | 89% | 96% | 0.669 | 96% | 4,989 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | legado | semântico | 30 | 47% | 67% | 83% | 0.387 | 83% | 5,235 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | legado | híbrido | 30 | 50% | 87% | 100% | 0.505 | 100% | 3,564 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | novas | semântico | 27 | 93% | 93% | 93% | 0.852 | 93% | 4,922 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | novas | híbrido | 27 | 67% | 81% | 96% | 0.666 | 96% | 5,383 |

## Comparação no MESMO subconjunto foco (15 perguntas: 7 do legado, 8 das novas)

O reranker grande só foi pontuado nestas perguntas (estimativa > 20 min para as 57); aqui o menor é restrito ao mesmo conjunto para a comparação ser justa. n pequeno: leia como indício, não como medida.

| configuração | N cand. | grupo | modo | n | hit@3 | hit@5 | hit@10 | MRR | hit@ctx | chars_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (sem reranker) | — | legado | semântico | 7 | 43% | 43% | 100% | 0.257 | 100% | 4,008 |
| baseline (sem reranker) | — | legado | híbrido | 7 | 29% | 29% | 71% | 0.203 | 100% | 5,193 |
| baseline (sem reranker) | — | novas | semântico | 8 | 75% | 75% | 75% | 0.625 | 75% | 4,620 |
| baseline (sem reranker) | — | novas | híbrido | 8 | 38% | 62% | 75% | 0.390 | 75% | 5,545 |
| bge-reranker-v2-m3 ml=256 | 15 | legado | semântico | 7 | 57% | 71% | 86% | 0.386 | 86% | 3,582 |
| bge-reranker-v2-m3 ml=256 | 15 | legado | híbrido | 7 | 43% | 71% | 100% | 0.268 | 100% | 5,105 |
| bge-reranker-v2-m3 ml=256 | 15 | novas | semântico | 8 | 75% | 88% | 88% | 0.781 | 88% | 4,892 |
| bge-reranker-v2-m3 ml=256 | 15 | novas | híbrido | 8 | 38% | 75% | 88% | 0.470 | 88% | 5,677 |
| bge-reranker-v2-m3 ml=256 | 20 | legado | semântico | 7 | 57% | 71% | 86% | 0.386 | 86% | 3,758 |
| bge-reranker-v2-m3 ml=256 | 20 | legado | híbrido | 7 | 43% | 71% | 100% | 0.268 | 100% | 5,126 |
| bge-reranker-v2-m3 ml=256 | 20 | novas | semântico | 8 | 75% | 88% | 88% | 0.781 | 88% | 4,931 |
| bge-reranker-v2-m3 ml=256 | 20 | novas | híbrido | 8 | 38% | 75% | 88% | 0.470 | 88% | 5,489 |
| bge-reranker-v2-m3 ml=256 | 30 | legado | semântico | 7 | 57% | 71% | 86% | 0.386 | 86% | 4,246 |
| bge-reranker-v2-m3 ml=256 | 30 | legado | híbrido | 7 | 43% | 71% | 100% | 0.279 | 100% | 5,148 |
| bge-reranker-v2-m3 ml=256 | 30 | novas | semântico | 8 | 75% | 88% | 88% | 0.781 | 88% | 5,569 |
| bge-reranker-v2-m3 ml=256 | 30 | novas | híbrido | 8 | 38% | 75% | 88% | 0.472 | 88% | 6,077 |
| bge-reranker-v2-m3 ml=384 | 15 | legado | semântico | 7 | 71% | 86% | 100% | 0.529 | 100% | 3,724 |
| bge-reranker-v2-m3 ml=384 | 15 | legado | híbrido | 7 | 29% | 57% | 100% | 0.244 | 100% | 5,079 |
| bge-reranker-v2-m3 ml=384 | 15 | novas | semântico | 8 | 75% | 88% | 88% | 0.781 | 88% | 5,001 |
| bge-reranker-v2-m3 ml=384 | 15 | novas | híbrido | 8 | 38% | 75% | 88% | 0.470 | 88% | 5,786 |
| bge-reranker-v2-m3 ml=384 | 20 | legado | semântico | 7 | 71% | 86% | 100% | 0.529 | 100% | 3,900 |
| bge-reranker-v2-m3 ml=384 | 20 | legado | híbrido | 7 | 29% | 57% | 100% | 0.244 | 100% | 5,100 |
| bge-reranker-v2-m3 ml=384 | 20 | novas | semântico | 8 | 75% | 88% | 88% | 0.781 | 88% | 5,258 |
| bge-reranker-v2-m3 ml=384 | 20 | novas | híbrido | 8 | 38% | 75% | 88% | 0.470 | 88% | 5,678 |
| bge-reranker-v2-m3 ml=384 | 30 | legado | semântico | 7 | 71% | 86% | 100% | 0.529 | 100% | 4,422 |
| bge-reranker-v2-m3 ml=384 | 30 | legado | híbrido | 7 | 29% | 57% | 100% | 0.255 | 100% | 4,984 |
| bge-reranker-v2-m3 ml=384 | 30 | novas | semântico | 8 | 75% | 88% | 88% | 0.775 | 88% | 5,855 |
| bge-reranker-v2-m3 ml=384 | 30 | novas | híbrido | 8 | 38% | 75% | 88% | 0.470 | 88% | 6,363 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | legado | semântico | 7 | 14% | 43% | 86% | 0.250 | 86% | 4,026 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | legado | híbrido | 7 | 14% | 86% | 86% | 0.200 | 100% | 4,944 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | novas | semântico | 8 | 88% | 88% | 88% | 0.750 | 88% | 5,162 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | novas | híbrido | 8 | 38% | 75% | 88% | 0.412 | 88% | 5,750 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | legado | semântico | 7 | 14% | 29% | 86% | 0.237 | 86% | 4,278 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | legado | híbrido | 7 | 14% | 71% | 86% | 0.192 | 100% | 4,742 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | novas | semântico | 8 | 88% | 88% | 88% | 0.750 | 88% | 5,057 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | novas | híbrido | 8 | 38% | 75% | 88% | 0.412 | 88% | 5,641 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | legado | semântico | 7 | 14% | 29% | 57% | 0.204 | 57% | 4,416 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | legado | híbrido | 7 | 14% | 71% | 100% | 0.213 | 100% | 4,663 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | novas | semântico | 8 | 88% | 88% | 88% | 0.750 | 88% | 5,302 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | novas | híbrido | 8 | 38% | 75% | 88% | 0.415 | 88% | 5,774 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | legado | semântico | 7 | 14% | 57% | 100% | 0.279 | 100% | 4,168 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | legado | híbrido | 7 | 0% | 71% | 86% | 0.167 | 100% | 4,917 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | novas | semântico | 8 | 88% | 88% | 88% | 0.750 | 88% | 5,162 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | novas | híbrido | 8 | 38% | 75% | 88% | 0.412 | 88% | 5,829 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | legado | semântico | 7 | 14% | 43% | 100% | 0.266 | 100% | 4,476 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | legado | híbrido | 7 | 0% | 57% | 86% | 0.159 | 100% | 4,772 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | novas | semântico | 8 | 88% | 88% | 88% | 0.750 | 88% | 5,252 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | novas | híbrido | 8 | 38% | 75% | 88% | 0.412 | 88% | 5,814 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | legado | semântico | 7 | 14% | 43% | 71% | 0.232 | 71% | 4,636 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | legado | híbrido | 7 | 0% | 57% | 100% | 0.180 | 100% | 4,714 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | novas | semântico | 8 | 88% | 88% | 88% | 0.750 | 88% | 5,494 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | novas | híbrido | 8 | 38% | 75% | 88% | 0.415 | 88% | 5,965 |

## Perguntas que mudaram em relação ao baseline (menor k que cobre a evidência, híbrido)

| configuração | N | melhoraram | pioraram | iguais |
|---|---|---|---|---|
| bge-reranker-v2-m3 ml=256 | 15 | 20 | 3 | 34 |
| bge-reranker-v2-m3 ml=256 | 20 | 21 | 3 | 33 |
| bge-reranker-v2-m3 ml=256 | 30 | 20 | 5 | 32 |
| bge-reranker-v2-m3 ml=384 | 15 | 20 | 3 | 34 |
| bge-reranker-v2-m3 ml=384 | 20 | 21 | 3 | 33 |
| bge-reranker-v2-m3 ml=384 | 30 | 20 | 5 | 32 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | 14 | 5 | 38 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | 15 | 4 | 38 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | 16 | 6 | 35 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | 15 | 4 | 38 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | 16 | 3 | 38 |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | 16 | 6 | 35 |

## Tempo do reranker por pergunta (CPU, fp32)

Medido: `predict` de 30 pares (pergunta, chunk), lote 16, média e mediana por pergunta. **15 e 20 são DERIVADOS** proporcionalmente (tempo(30) × N/30).

| configuração | perguntas medidas | tokens/pergunta (30 pares) | 30 cand. (medido) média / mediana | 20 cand. (derivado) | 15 cand. (derivado) |
|---|---|---|---|---|---|
| bge-reranker-v2-m3 ml=256 | 57 | 4,168 | 40.48 s / 36.94 s | 26.99 s | 20.24 s |
| bge-reranker-v2-m3 ml=384 | 57 | 4,565 | 34.45 s / 34.61 s | 22.96 s | 17.22 s |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 57 | 4,168 | 2.22 s / 2.19 s | 1.48 s | 1.11 s |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 57 | 4,565 | 2.92 s / 2.84 s | 1.95 s | 1.46 s |

## Estimativa para uma NVIDIA T1000 (fp32) — **ESTIMATIVA, não medida**

Não há GPU nesta máquina. Método: FLOPs ≈ 2 × parâmetros do corpo do transformador (sem embeddings) × tokens processados (contados com o tokenizer, truncados em max_length; ignora atenção e o custo de lotes com padding); tempo = FLOPs / (pico fp32 da T1000 ≈ 2.5 TFLOPS × utilização 25–40%). A utilização é uma suposição típica de inferência fp32 de transformers; transferência de dados e carga do modelo não entram. O 'efeito CPU' é o mesmo cálculo com o tempo medido, para calibrar a conta.

| configuração | pesos fp32 (GB) | GFLOPs por pergunta (30 pares) | CPU medida (GFLOPS efetivos) | T1000 30 cand. (estimativa) | T1000 20 cand. | T1000 15 cand. |
|---|---|---|---|---|---|---|
| bge-reranker-v2-m3 ml=256 | 2.27 | 2,529 | 62 | ≈ 2.53–4.05 s | ≈ 1.69–2.70 s | ≈ 1.26–2.02 s |
| bge-reranker-v2-m3 ml=384 | 2.27 | 2,770 | 80 | ≈ 2.77–4.43 s | ≈ 1.85–2.95 s | ≈ 1.38–2.22 s |
| mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 0.47 | 179 | 80 | ≈ 0.18–0.29 s | ≈ 0.12–0.19 s | ≈ 0.09–0.14 s |
| mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 0.47 | 196 | 67 | ≈ 0.20–0.31 s | ≈ 0.13–0.21 s | ≈ 0.10–0.16 s |

Memória: a T1000 tem 4 GB ou 8 GB (conferir); os pesos fp32 da tabela acima mais ativações de lote 16 precisam caber nisso, e o e5-base (~1,1 GB fp32) soma se `EMBEDDER_DEVICE=cuda`. Ver GPUS.md.

## Evidência por pergunta foco (E03, F16a, F18a, A07, E01)

`sem.` = menor k no semântico reordenado; `híb.` = menor k no contexto híbrido (— = nunca cobre); `posição das evidências` = posição, no ranking reordenado dos N candidatos, do melhor chunk de cada evidência (✗ = fora dos N candidatos).

| pergunta | configuração | N | k sem. | k híb. | posição das evidências |
|---|---|---|---|---|---|
| E03 | baseline | 10 | — | — | ✗, ✗ (ranking semântico) |
| E03 | bge-reranker-v2-m3 ml=256 | 15 | — | — | ✗, ✗ |
| E03 | bge-reranker-v2-m3 ml=256 | 20 | — | — | ✗, ✗ |
| E03 | bge-reranker-v2-m3 ml=256 | 30 | — | — | ✗, ✗ |
| E03 | bge-reranker-v2-m3 ml=384 | 15 | — | — | ✗, ✗ |
| E03 | bge-reranker-v2-m3 ml=384 | 20 | — | — | ✗, ✗ |
| E03 | bge-reranker-v2-m3 ml=384 | 30 | — | — | ✗, ✗ |
| E03 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | — | — | ✗, ✗ |
| E03 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | — | — | ✗, ✗ |
| E03 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | — | — | ✗, ✗ |
| E03 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | — | — | ✗, ✗ |
| E03 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | — | — | ✗, ✗ |
| E03 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | — | — | ✗, ✗ |
| F16a | baseline | 10 | — | — | 12 (ranking semântico) |
| F16a | bge-reranker-v2-m3 ml=256 | 15 | 4 | 9 | 4 |
| F16a | bge-reranker-v2-m3 ml=256 | 20 | 4 | 9 | 4 |
| F16a | bge-reranker-v2-m3 ml=256 | 30 | 4 | 8 | 4 |
| F16a | bge-reranker-v2-m3 ml=384 | 15 | 4 | 9 | 4 |
| F16a | bge-reranker-v2-m3 ml=384 | 20 | 4 | 9 | 4 |
| F16a | bge-reranker-v2-m3 ml=384 | 30 | 5 | 9 | 5 |
| F16a | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | 2 | 7 | 2 |
| F16a | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | 2 | 7 | 2 |
| F16a | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | 2 | 6 | 2 |
| F16a | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | 2 | 7 | 2 |
| F16a | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | 2 | 7 | 2 |
| F16a | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | 2 | 6 | 2 |
| F18a | baseline | 10 | 2 | 2 | 2 (ranking semântico) |
| F18a | bge-reranker-v2-m3 ml=256 | 15 | 1 | 1 | 1 |
| F18a | bge-reranker-v2-m3 ml=256 | 20 | 1 | 1 | 1 |
| F18a | bge-reranker-v2-m3 ml=256 | 30 | 1 | 1 | 1 |
| F18a | bge-reranker-v2-m3 ml=384 | 15 | 1 | 1 | 1 |
| F18a | bge-reranker-v2-m3 ml=384 | 20 | 1 | 1 | 1 |
| F18a | bge-reranker-v2-m3 ml=384 | 30 | 1 | 1 | 1 |
| F18a | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | 2 | 2 | 2 |
| F18a | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | 2 | 2 | 2 |
| F18a | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | 2 | 2 | 2 |
| F18a | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | 2 | 2 | 2 |
| F18a | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | 2 | 2 | 2 |
| F18a | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | 2 | 2 | 2 |
| A07 | baseline | 10 | 2 | 2 | 2, 1 (ranking semântico) |
| A07 | bge-reranker-v2-m3 ml=256 | 15 | 2 | 2 | 1, 2 |
| A07 | bge-reranker-v2-m3 ml=256 | 20 | 2 | 2 | 1, 2 |
| A07 | bge-reranker-v2-m3 ml=256 | 30 | 2 | 2 | 1, 2 |
| A07 | bge-reranker-v2-m3 ml=384 | 15 | 2 | 2 | 1, 2 |
| A07 | bge-reranker-v2-m3 ml=384 | 20 | 2 | 2 | 1, 2 |
| A07 | bge-reranker-v2-m3 ml=384 | 30 | 2 | 2 | 1, 2 |
| A07 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | 8 | 5 | 3, 8 |
| A07 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | 8 | 5 | 3, 8 |
| A07 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | 8 | 5 | 3, 8 |
| A07 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | 8 | 5 | 3, 8 |
| A07 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | 8 | 5 | 3, 8 |
| A07 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | 8 | 5 | 3, 8 |
| E01 | baseline | 10 | 9 | 7 | 9, 5, 1, 4 (ranking semântico) |
| E01 | bge-reranker-v2-m3 ml=256 | 15 | 6 | 6 | 5, 2, 1, 6 |
| E01 | bge-reranker-v2-m3 ml=256 | 20 | 6 | 6 | 5, 2, 1, 6 |
| E01 | bge-reranker-v2-m3 ml=256 | 30 | 6 | 6 | 5, 2, 1, 6 |
| E01 | bge-reranker-v2-m3 ml=384 | 15 | 6 | 6 | 5, 2, 1, 6 |
| E01 | bge-reranker-v2-m3 ml=384 | 20 | 6 | 6 | 5, 2, 1, 6 |
| E01 | bge-reranker-v2-m3 ml=384 | 30 | 6 | 6 | 5, 2, 1, 6 |
| E01 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 15 | 8 | 5 | 1, 2, 3, 8 |
| E01 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 20 | 9 | 5 | 1, 2, 3, 9 |
| E01 | mmarco-mMiniLMv2-L12-H384-v1 ml=256 | 30 | 10 | 5 | 1, 2, 3, 10 |
| E01 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 15 | 8 | 5 | 1, 2, 3, 8 |
| E01 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 20 | 9 | 5 | 1, 2, 3, 9 |
| E01 | mmarco-mMiniLMv2-L12-H384-v1 ml=384 | 30 | 10 | 5 | 1, 2, 3, 10 |


## Leitura dos resultados (escrita à mão após a execução completa de 57 perguntas; `relatar` não gera esta seção — se rodá-lo de novo, reanexe)

Esta versão substitui a leitura anterior, que tinha o bge só em 15 perguntas. Agora o bge-reranker-v2-m3 está pontuado nas **57 perguntas** nas duas larguras (256 e 384) e nos três N (15, 20, 30).

**O ganho do bge nas 57 existe, mas é menor e menos "limpo" do que no subconjunto.**
- Híbrido (o que a produção usa), N = 15, ml = 256, contra o baseline:
  - **legado (30):** hit@3 60% → 63%, hit@5 77% → 83%, hit@10 93% → 100%, MRR 0,481 → 0,508; hit@ctx continua 100%.
  - **novas (27):** hit@3 59% → 74%, hit@5 81% → 85%, hit@10 93% → 96%, MRR 0,596 → 0,701; hit@ctx 93% → 96%.
- Semântico puro: legado hit@10 83% → 87% (ml=256) ou 90% (ml=384); novas 89% → 93%, MRR 0,749 → 0,843.
- **Difere do subconjunto:** nas 15 perguntas foco o bge tinha 8 melhoradas e **0 pioradas**; nas 57 são **20 melhoradas, 3 pioradas, 34 iguais** (N = 15). Ou seja, a direção se confirma, mas "não piorou nenhuma" era artefato do subconjunto: ele foi montado em torno de perguntas que já falhavam, então o ganho lá (híbrido legado hit@10 71% → 100%) foi bem maior que o ganho real (93% → 100%).
- **As 3 pioradas (k mínimo no híbrido):** C01 (1 → 8, a pior; a evidência continua no contexto), E02 (8 → 10) e F15a (1 → 2). Com N = 30 pioram também C03 e E04 (5 pioradas).

**Escolha de configuração**
- **N = 15 é suficiente; N = 30 é pior.** N = 15 e N = 20 empatam em quase tudo (N = 20 dá +4 pontos em hit@5 das novas, 1 pergunta); N = 30 tem 5 pioradas e hit@3/hit@5 do legado iguais ou menores. O tempo cresce linear com N. Sugestão: `RERANKER_CANDIDATOS=15`.
- **ml = 256 × 384:** no híbrido, empate (legado hit@3 63% × 60%, novas idênticas). Só o semântico puro do legado melhora com 384 (hit@3 60% → 67%). Como o ml = 384 processa ~10% mais tokens, sugestão: `RERANKER_MAX_LENGTH=256`.
- **mmarco-mMiniLM (menor), 57 perguntas:** ajuda o híbrido (novas hit@3 59% → 67%, MRR 0,596 → 0,668; legado hit@5 77% → 87%), mas fica **abaixo do bge** (legado hit@3 50–57% contra 60–63%; MRR novas 0,666–0,669 contra 0,701–0,708), tem 3–6 pioradas e piora o semântico do legado com N = 30 (hit@10 83% → 77%). Seu atrativo é o custo (≈ 1,1 s para 15 candidatos em CPU, derivado), não a qualidade.

**Perguntas foco (bge, híbrido)**
- **F16a:** evidência no rank semântico 12 (fora do top-10) → posição 4 no reordenado; entra no contexto em k = 9.
- **E01:** 9 → 6 (semântico) e 7 → 6 (híbrido). **F18a:** 2 → 1. **A07:** igual (o menor piora de 2 para 8).
- **E03 não muda em nenhuma configuração:** as evidências estão nos ranks semânticos 44 e 122, fora dos 30 candidatos. Reranking só reordena candidatos; E03 é problema de recall (chunking/consulta), não de ordem.

**Tempo (CPU, 4 threads, fp32) — leia com cuidado**
- Sem interferência (primeira execução, 15 perguntas): bge ≈ **28,8 s (ml=256)** e **33,4 s (ml=384)** para 30 candidatos. A tabela de tempos acima mostra 40,5 s para ml=256 porque **a execução completa dividiu a CPU com as minhas provas offline da Fase 4** (14:16 é o fim do ml=256); o ml=384 (34,5 s) rodou quase sozinho. Use ≈ 29–34 s para 30 candidatos e ≈ 14–17 s para 15 (esse último derivado).
- **T1000 (ESTIMATIVA por FLOPs, não medida)**, bge ml = 256: ≈ 1,3–2,0 s para 15 candidatos e ≈ 2,5–4,1 s para 30, supondo pico fp32 de 2,5 TFLOPS e 25–40% de utilização. Se o pico real for 1,53 TFLOPS (valor informado para fp32), multiplique por ~1,6 (≈ 2,1–3,3 s para 15 candidatos).

**Limites**
- Mede só retrieval, não respostas de LLM. O baseline é o ranking exato, que pode diferir em ±1 chunk do retriever aproximado do Chroma (`nucleo.py` com o reranker pede `RERANKER_CANDIDATOS` ao retriever aproximado; na prova offline o top-10 e o contexto coincidiram com este experimento nas 15 perguntas foco, mas isso não foi repetido nas 57).
- Com 57 perguntas e diferenças de 1 a 3 perguntas entre configurações, só o efeito "bge melhora o ranking médio" é robusto; as diferenças entre N = 15/20 e ml = 256/384 estão dentro do ruído.
