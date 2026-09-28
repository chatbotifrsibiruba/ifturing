"""
debug_linhas.py — Verifica linhas geradas por extrair_tabelas_como_texto().
Mostra as primeiras 20 linhas totais e as primeiras 10 linhas de curso
(que contenham "Técnico", "Graduação" ou "Superior").
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

print("=== Primeiras 20 linhas (qualquer tipo) ===")
for i, linha in enumerate(linhas[:20], 1):
    print(f"[{i:02d}] {linha}")

print("\n=== Primeiras 10 linhas de curso (contêm 'Técnico', 'Graduação' ou 'Superior') ===")
cursos = [l for l in linhas if any(k in l for k in ("Técnico", "Graduação", "Superior"))]
for i, linha in enumerate(cursos[:10], 1):
    print(f"[C{i:02d}] {linha}")
