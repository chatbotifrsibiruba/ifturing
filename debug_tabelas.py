"""
debug_tabelas.py — Inspeciona tabelas brutas extraídas pelo pdfplumber.
Uso: python debug_tabelas.py
"""

import pdfplumber
from pathlib import Path

PDF = Path("./documentos/EDITAL Nº XX-2026 - EDITAL DE INGRESSO DE ESTUDANTES NO PRIMEIRO SEMESTRE DE 2027 (1).pdf")

if not PDF.exists():
    print(f"ERRO: arquivo não encontrado em {PDF.resolve()}")
    print("Arquivos disponíveis em ./documentos:")
    for f in sorted(Path("./documentos").glob("*.pdf")):
        print(f"  {f.name}")
    raise SystemExit(1)

print(f"=== Abrindo: {PDF.name} ===\n")

with pdfplumber.open(PDF) as pdf:
    print(f"Total de páginas: {len(pdf.pages)}\n")
    for n_pag, page in enumerate(pdf.pages, start=1):
        tabelas = page.extract_tables()
        if not tabelas:
            continue
        print(f"{'='*60}")
        print(f"PÁGINA {n_pag}  —  {len(tabelas)} tabela(s) encontrada(s)")
        print(f"{'='*60}")
        for n_tab, tabela in enumerate(tabelas):
            print(f"\n  [Tabela {n_tab}]  {len(tabela)} linhas x {len(tabela[0]) if tabela else 0} colunas")
            print(f"  {'─'*50}")
            for i, linha in enumerate(tabela[:5]):
                print(f"  linha[{i}]: {linha}")
            if len(tabela) > 5:
                print(f"  ... (+{len(tabela) - 5} linhas omitidas)")
        print()
