"""
debug_linhas.py — Mostra TODAS as linhas geradas por extrair_tabelas_como_texto().
"""
import sys
from pathlib import Path
from indexar import extrair_tabelas_como_texto

PDFS = sorted(Path("./documentos").glob("*.pdf"))

if not PDFS:
    print("ERRO: nenhum PDF em ./documentos")
    sys.exit(1)

total_geral = 0
for pdf in PDFS:
    print(f"\n{'='*70}")
    print(f"PDF: {pdf.name}")
    print('='*70)
    linhas = extrair_tabelas_como_texto(str(pdf))
    print(f"Total de linhas extraídas: {len(linhas)}")
    for i, linha in enumerate(linhas, 1):
        print(f"  {i:3}. {linha}")
    total_geral += len(linhas)

print(f"\n{'='*70}")
print(f"TOTAL GERAL: {total_geral} linhas extraídas de {len(PDFS)} PDF(s)")
