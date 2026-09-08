# Comparação entre 3 rodadas de avaliação

Cada valor é **média ± desvio padrão** entre as rodadas (100 perguntas do Golden Dataset por rodada).

| Modelo | LLM (s) | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness |
|---|---|---|---|---|---|---|
| phi3:mini | 14.897 ± 0.2033 | 0.6502 ± 0.0466 | 0.3973 ± 0.0175 | 0.6533 ± 0.0451 | 0.52 ± 0.0186 | 0.3671 ± 0.004 |
| qwen3:0.6b | 5.352 ± 0.0877 | 0.5489 ± 0.0406 | 0.345 ± 0.0365 | 0.65 ± 0.01 | 0.5444 ± 0.0077 | 0.3072 ± 0.0072 |
| smollm2:1.7b | 8.8797 ± 0.3229 | 0.4607 ± 0.0593 | 0.5242 ± 0.029 | 0.6433 ± 0.0208 | 0.5575 ± 0.0229 | 0.3415 ± 0.0088 |
