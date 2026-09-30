"""Script one-shot: roda a mesma pergunta 5x isolado (sem histórico de chat)."""
import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from teste_timing import carregar_pipeline, perguntar
from nucleo import MODELOS, MODELO_PADRAO_SIMPLES

pipeline, tabela_docs = carregar_pipeline()
modelo_cfg = MODELOS[MODELO_PADRAO_SIMPLES]
pergunta = "Quais sao as datas do processo seletivo?"

for i in range(5):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"TENTATIVA {i+1}: {pergunta}")
    print(sep)
    perguntar(pipeline, tabela_docs, pergunta, modelo_cfg)
