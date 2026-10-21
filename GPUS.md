# GPUs no IF Turing

Legenda: **[medido]** = medido por mim nesta máquina de desenvolvimento (só CPU, sem GPU nem Ollama);
**[informado]** = dado que veio de você/dos logs do servidor e que não pude conferir aqui;
**[estimativa]** = cálculo, não medição; **[hipótese]** = ideia ainda sem teste.

## Divisão de trabalho

| Peça | GPU | Como |
|---|---|---|
| LLM (Ollama) | RX 9060 XT 16 GB | backend **Vulkan** do Ollama. Escolha automática (`library=Vulkan`, `available_gpu_count=2`), não fixada **[informado]** |
| Embedder (`multilingual-e5-base`) | T1000 | PyTorch CUDA, **fp32**: `EMBEDDER_DEVICE=cuda` |
| Reranker (opcional) | T1000 | PyTorch CUDA, **fp32**: `RERANKER_ATIVO=1 RERANKER_DEVICE=cuda` |

A RX 9060 XT é AMD: o PyTorch CUDA não a enxerga, então embedder/reranker ficam na T1000.

## T1000: fp32 sempre

- fp16 na T1000 é inviável: ~0,36 TFLOPS contra ~1,53 em fp32 **[informado]**. `nucleo.py` força `.float()` no
  reranker e o embedder usa o padrão (fp32); nunca configure half/bfloat16.
- **Memória (4 GB ou 8 GB: conferir com `nvidia-smi`; existem as duas versões).** Pesos em fp32 **[estimativa por nº de
  parâmetros; 568 M parâmetros do bge medidos]**: reranker bge-reranker-v2-m3 ≈ 2,3 GB; embedder (278 M) ≈ 1,1 GB;
  contexto CUDA + ativações ≈ 0,5 GB. Somados ≈ 3,9 GB: **não cabe com folga em 4 GB**, cabe em 8 GB. Em 4 GB use o
  reranker no CPU, ou o reranker menor (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, ~118 M, ~0,5 GB).
- **Latência do reranker (estimativa, não medida na T1000).** Cálculo por FLOPs (2 × parâmetros do corpo × tokens)
  com 25–40 % do pico: com pico de 2,5 TFLOPS, bge (ml=256) ≈ 1,3–2,0 s para 15 candidatos e 2,5–4,1 s para 30 (valores
  do relatório das 57 perguntas). Se o pico real for 1,53 TFLOPS (valor acima), multiplique por ~1,6: ≈ 2,1–3,3 s e
  4,1–6,6 s. Em CPU (4 threads) **[medido]**: ≈ 29–34 s para 30 candidatos (15 candidatos ≈ metade, derivado; a média
  de 40 s do relatório inclui um período em que a CPU foi dividida com outras provas).
- O PyTorch deste venv é a build **CPU**; usar CUDA exige instalar a build `+cu12x` no venv do servidor (não feito aqui).
- Embedder: a 1ª pergunta paga ~5,4 s de carga do modelo **[informado]**; `preaquecer()` (chamado na partida em
  `app_publico.py`) move isso para o início.

## Variáveis

| Variável | Padrão | Efeito |
|---|---|---|
| `EMBEDDER_DEVICE` | `cpu` | dispositivo do embedder (`cuda`, `cuda:0`) |
| `RERANKER_ATIVO` | `0` | liga o reranker (desligado = comportamento antigo, idêntico) |
| `RERANKER_MODELO` | `BAAI/bge-reranker-v2-m3` | cross-encoder (sem `trust_remote_code`) |
| `RERANKER_CANDIDATOS` | `15` | chunks buscados antes de reordenar (fica com `TOP_K`=10) |
| `RERANKER_MAX_LENGTH` | `256` | tokens por par (pergunta, chunk) |
| `RERANKER_DEVICE` | `cpu` | `cuda` na T1000; se falhar, cai para a ordem semântica com aviso no log |
| `OLLAMA_URLS` | (vazio → `OLLAMA_URL`) | lista de instâncias; menos requisições em andamento + failover |

## Como medir

- **NVIDIA (T1000):** `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1` durante o teste.
- **LLM na RX 9060 XT:** `ollama ps` (coluna PROCESSOR deve dizer `100% GPU`) e o log do serviço
  (`library=Vulkan`); uso/VRAM no Gerenciador de Tarefas → Desempenho → GPU.
- **Fluxo completo:** `python teste_concorrencia.py --modelo gemma3:4b --concorrencia 1,2,4,8 --rodadas 3`
  (com `RERANKER_ATIVO=1 RERANKER_DEVICE=cuda` para incluir o reranker). Gera `resultado_concorrencia.csv/.md` (fora do git).
- O `responder()` devolve `tempo_rerank_s` e `rerank_aplicado` por pergunta.
- `responder()`, `teste_modelos.py` e `teste_timing.py` usam o MESMO retrieval (`nucleo.recuperar_contexto`), então
  `RERANKER_ATIVO=1` vale nos três. `teste_modelos.py` imprime `RERANKER_ATIVO=… aplicado em N/total perguntas`,
  grava `tempo_rerank_s`/`rerank_aplicado` no CSV/MD e **termina com erro** se o reranker não puder ser usado
  (ou nunca for aplicado); se for aplicado só em parte, avisa no relatório. Procure `[timing] reranker:` no log.

## Segunda instância do Ollama na T1000 [hipótese]

Ideia: uma 2ª instância atendendo parte das requisições, usando `OLLAMA_URLS`. Não mexe no serviço existente (porta 11434):

```
set CUDA_VISIBLE_DEVICES=0            # índice da T1000 em nvidia-smi (conferir)
set OLLAMA_HOST=127.0.0.1:11435
ollama serve
```
e depois `OLLAMA_URLS=http://localhost:11434,http://localhost:11435`. Pontos a testar antes de adotar: (1) um modelo
pequeno cabe na T1000 junto com embedder/reranker? (4 GB: dificilmente; 8 GB: só modelos ≤ ~3B em Q4); (2) a T1000
é bem mais lenta que a RX 9060 XT, então a instância dela pode piorar o p95 em vez de melhorar a vazão; (3) a
instância extra baixa/guarda os modelos em outra pasta se `OLLAMA_MODELS` não for compartilhada. Só vale se
`teste_concorrencia.py` mostrar ganho de vazão sem piorar p95.
