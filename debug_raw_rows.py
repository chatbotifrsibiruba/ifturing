"""
debug_raw_rows.py — Imprime todas as linhas brutas das tabelas em páginas
específicas para diagnosticar células mescladas e estrutura real do PDF.
"""
import sys
import pdfplumber
from pathlib import Path

# (página 68 = índice 67, página 59 = índice 58)
ALVOS = [
    ("EDITAL Nº XX-2026 - EDITAL DE INGRESSO DE ESTUDANTES NO PRIMEIRO SEMESTRE DE 2027 (1).pdf", [67, 68]),
    ("EDITAL-No-78-2025-EDITAL-DE-INGRESSO-DE-ESTUDANTES-NO-PRIMEIRO-SEMESTRE-DE-2026-1.pdf", [58, 59]),
]

DOCS = Path("./documentos")

for nome_pdf, indices_pagina in ALVOS:
    caminho = DOCS / nome_pdf
    if not caminho.exists():
        print(f"ARQUIVO NÃO ENCONTRADO: {caminho}")
        continue

    print(f"\n{'='*72}")
    print(f"PDF: {nome_pdf}")
    print(f"{'='*72}")

    with pdfplumber.open(str(caminho)) as pdf:
        for idx_pag in indices_pagina:
            if idx_pag >= len(pdf.pages):
                continue
            page = pdf.pages[idx_pag]
            tabelas = page.extract_tables()
            print(f"\n--- Página {idx_pag + 1}: {len(tabelas)} tabela(s) ---")
            for t_num, tabela in enumerate(tabelas):
                print(f"\n  Tabela {t_num + 1} ({len(tabela)} linhas totais, {len(tabela[0]) if tabela else 0} colunas):")
                for r_num, linha in enumerate(tabela):
                    # Mostra só as primeiras 6 colunas para não poluir
                    colunas_repr = []
                    for c_num, cel in enumerate(linha[:6]):
                        if cel is None:
                            colunas_repr.append(f"[{c_num}]=None")
                        else:
                            val = " ".join(str(cel).replace("\n", " ").split())
                            colunas_repr.append(f"[{c_num}]={repr(val[:40])}")
                    print(f"    linha {r_num:2d}: {' | '.join(colunas_repr)}")
