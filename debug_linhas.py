"""
debug_linhas.py — Inspeciona linhas geradas por extrair_tabelas_como_texto().
"""
import sys
from pathlib import Path
from indexar import extrair_tabelas_como_texto

PDF = Path("./documentos/EDITAL Nº XX-2026 - EDITAL DE INGRESSO DE ESTUDANTES NO PRIMEIRO SEMESTRE DE 2027 (1).pdf")

if not PDF.exists():
    print(f"ERRO: {PDF} não encontrado.")
    sys.exit(1)

linhas = extrair_tabelas_como_texto(str(PDF))
print(f"Total de linhas extraídas: {len(linhas)}\n")

TERMOS = ("Ciência da Computação", "Matemática", "Agropecuária", "Eletrotécnica")

print(f"=== Linhas que contêm: {TERMOS} ===")
encontrados = {t: [] for t in TERMOS}
for linha in linhas:
    for t in TERMOS:
        if t in linha:
            encontrados[t].append(linha)

for termo, matches in encontrados.items():
    print(f"\n--- {termo} ({len(matches)} ocorrência(s)) ---")
    for m in matches:
        print(f"  {m}")
