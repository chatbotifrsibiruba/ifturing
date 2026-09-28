"""
debug_campus_scan.py — Varre o PDF inteiro e mostra:
  1. Cada página onde aparece um cabeçalho "Campus X:" (ativa/desativa estado)
  2. Estado ibiruba_ativo para cada página com tabela
  3. Col[0] de cada linha de dado em tabelas capturadas e rejeitadas (± 2 páginas em torno de
     mudanças de estado)
"""
import re
import sys
import pdfplumber
from pathlib import Path

_RE_CAMPUS_IBIRUBA = re.compile(r"campus\s+ibirub[aá]\s*:", re.IGNORECASE)
_RE_CAMPUS_OUTRO   = re.compile(r"campus\s+\w+(?:\s+\w+)*\s*:", re.IGNORECASE)

PDFS = sorted(Path("./documentos").glob("*.pdf"))
if not PDFS:
    print("Nenhum PDF em ./documentos"); sys.exit(1)

for caminho in PDFS:
    print(f"\n{'='*72}")
    print(f"PDF: {caminho.name}")
    print(f"{'='*72}")

    ibiruba_ativo = False
    eventos = []  # (pagina, tipo, detalhe)

    with pdfplumber.open(str(caminho)) as pdf:
        for i, page in enumerate(pdf.pages):
            texto = page.extract_text() or ""
            tabelas = page.extract_tables()

            # Detecta mudanças de estado
            novo_estado = ibiruba_ativo
            cabecalho_encontrado = None
            if _RE_CAMPUS_IBIRUBA.search(texto):
                novo_estado = True
                cabecalho_encontrado = _RE_CAMPUS_IBIRUBA.search(texto).group(0)
            elif _RE_CAMPUS_OUTRO.search(texto):
                novo_estado = False
                cabecalho_encontrado = _RE_CAMPUS_OUTRO.search(texto).group(0)

            if cabecalho_encontrado or tabelas:
                eventos.append({
                    "pagina": i + 1,
                    "cabecalho": cabecalho_encontrado,
                    "estado_antes": ibiruba_ativo,
                    "estado_depois": novo_estado,
                    "n_tabelas": len(tabelas),
                    "tabelas": tabelas,
                })

            ibiruba_ativo = novo_estado

    # Imprime eventos relevantes
    for ev in eventos:
        pag = ev["pagina"]
        cab = ev["cabecalho"]
        antes = ev["estado_antes"]
        depois = ev["estado_depois"]
        n_tab = ev["n_tabelas"]

        if cab:
            seta = "→ ATIVO" if depois else "→ inativo"
            print(f"\n  [pág {pag:3d}] Cabeçalho: {repr(cab[:60])}  {seta}")

        if n_tab > 0:
            estado_str = "ATIVO  " if depois else "inativo"
            for t_num, tabela in enumerate(ev["tabelas"]):
                n_linhas = len(tabela)
                # Extrai col[0] de cada linha de dado
                col0s = []
                for linha in tabela[1:]:
                    if linha:
                        val = " ".join(str(linha[0]).replace("\n"," ").split()) if linha[0] else "None"
                        col0s.append(val[:50])
                print(f"    [pág {pag:3d}] [{estado_str}] tabela {t_num+1}/{n_tab}: "
                      f"{n_linhas} linhas, col0s={col0s}")
