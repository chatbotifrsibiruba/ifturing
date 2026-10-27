"""
teste_sobrecarga_offline.py — Provas offline do teste_sobrecarga.py contra um Ollama FALSO (sem Ollama, GPU nem índice
de produção). Prova que o script classifica, mede, para e grava corretamente; NÃO mede desempenho real.

O servidor falso atende /api/generate com tempo de serviço simulado, paralelismo configurável e fila com limite que
devolve 503, e /api/ps com uma VRAM fixa. Cada prova roda o script em um subprocesso e confere o CSV/relatório.

Uso:
    python teste_sobrecarga_offline.py                 # provas rápidas (~3 min)
    python teste_sobrecarga_offline.py -k a,b,f        # só algumas
    python teste_sobrecarga_offline.py --lento         # inclui --rapido completo (h) e soak (l): +8 min
A prova (i) usa PASTA_CHROMA=./chroma_exp_f3_final e carrega o embedder (precisa do venv do projeto).
"""
import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).parent
SCRIPT = AQUI / "teste_sobrecarga.py"
BASE_ARGS = ["--pausa", "0", "--pausa-cenario", "0", "--intervalo-monitor", "0.3", "--confirmo"]


# ── Servidor falso ─────────────────────────────────────────────────────────

class OllamaFalso:
    """/api/generate com tempo de serviço, paralelismo e fila limitada (503); /api/ps com VRAM fixa.
    `lento_apos=(n, s)`: da requisição n+1 em diante o serviço passa a levar s segundos."""

    def __init__(self, servico_s=0.1, paralelismo=1, fila_max=1000, tokens=60, lento_apos=None):
        self.servico_s, self.fila_max, self.tokens, self.lento_apos = servico_s, fila_max, tokens, lento_apos
        self.lock = threading.Lock()
        self.livres, self.fila = paralelismo, deque()      # fila FIFO (Semaphore não garante a ordem)
        self.n = self.rejeitadas = 0
        falso = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, codigo, obj):
                corpo = json.dumps(obj).encode()
                try:
                    self.send_response(codigo)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(corpo)))
                    self.end_headers()
                    self.wfile.write(corpo)
                except (ConnectionError, OSError):
                    pass                                  # o cliente desistiu (timeout)

            def do_GET(self):
                if self.path == "/api/ps":
                    self._json(200, {"models": [{"name": "gemma3:4b", "size": 5_500_000_000,
                                                 "size_vram": 5_000_000_000}]})
                else:
                    self._json(404, {"error": "não existe"})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path != "/api/generate":
                    return self._json(404, {"error": "não existe"})
                with falso.lock:
                    falso.n += 1
                    n = falso.n
                    if falso.livres > 0 and not falso.fila:
                        falso.livres -= 1
                        vez = None
                    elif len(falso.fila) >= falso.fila_max:
                        falso.rejeitadas += 1
                        return self._json(503, {"error": "server busy, please try again. maximum pending "
                                                         "requests exceeded"})
                    else:
                        vez = threading.Event()
                        falso.fila.append(vez)
                if vez is not None:
                    vez.wait()                            # a vaga é passada direto pelo release, em ordem
                try:
                    s = falso.servico_s
                    if falso.lento_apos and n > falso.lento_apos[0]:
                        s = falso.lento_apos[1]
                    time.sleep(s)
                finally:
                    with falso.lock:
                        if falso.fila:
                            falso.fila.popleft().set()
                        else:
                            falso.livres += 1
                self._json(200, {"model": "gemma3:4b", "response": "Resposta simulada. " * 20, "done": True,
                                 "eval_count": falso.tokens, "prompt_eval_count": 1100,
                                 "load_duration": 0, "prompt_eval_duration": int(1e8),
                                 "eval_duration": int(s * 1e9)})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def parar(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ── Execução do script ─────────────────────────────────────────────────────

def executar(args: list, falso: OllamaFalso, env_extra=None, timeout=600, sinal_apos=None,
             sem_psutil=False, base=True) -> dict:
    """Roda teste_sobrecarga.py num subprocesso, com saída num diretório temporário; devolve rc, md, linhas do CSV,
    linhas do CSV da linha do tempo e o texto impresso. `sinal_apos`: segundos até o sinal de interrupção."""
    saida = Path(tempfile.mkdtemp(prefix="sobrecarga_"))
    cmd_args = ["--ollama-url", falso.url, "--saida-dir", str(saida)] + (BASE_ARGS if base else []) + args
    if sem_psutil:                                        # psutil=None em sys.modules faz `import psutil` falhar
        cmd = [sys.executable, "-c", "import sys, runpy; sys.modules['psutil'] = None; "
               "sys.argv = ['teste_sobrecarga.py'] + sys.argv[1:]; "
               "runpy.run_path('teste_sobrecarga.py', run_name='__main__')"] + cmd_args
    else:
        cmd = [sys.executable, str(SCRIPT)] + cmd_args
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OLLAMA_URL": falso.url, **(env_extra or {})}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=AQUI, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", creationflags=flags)
    if sinal_apos is not None:
        time.sleep(sinal_apos)
        p.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
    try:
        texto, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        texto, _ = p.communicate()
        texto += "\n[TIMEOUT DA PROVA]"
    lerlinhas = lambda padrao: [dict(r) for f in sorted(saida.glob(padrao)) for r in csv.DictReader(open(f, encoding="utf-8"))]
    mds = sorted(saida.glob("resultado_sobrecarga_*.md"))
    return {"rc": p.returncode, "texto": texto, "segundos": time.monotonic() - t0, "saida": saida,
            "md": mds[0].read_text(encoding="utf-8") if mds else "",
            "linhas": lerlinhas("resultado_sobrecarga_*.csv"), "timeline": lerlinhas("linha_do_tempo_*.csv")}


def lat(linhas, cenario, rotulo, papel="carga", validos=True):
    return [float(r["latencia_s"]) for r in linhas if r["cenario"] == cenario and r["rotulo"] == rotulo
            and r["papel"] == papel and (not validos or r["status"] in ("ok", "truncada"))]


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else 0.0


def vazao_md(md, cenario_titulo, rotulo):
    """Lê a coluna 'vazão (resp/min)' da linha `rotulo` na tabela da seção `cenario_titulo` do relatório."""
    sec = md.split(f"## {cenario_titulo}")[1].split("\n## ")[0]
    for l in sec.splitlines():
        c = [x.strip() for x in l.strip("|").split("|")]
        if c and c[0] == rotulo and len(c) >= 12:
            return float(c[-3])
    raise ValueError(f"{rotulo} não achado em {cenario_titulo}")


# ── Provas ─────────────────────────────────────────────────────────────────

def prova_a():
    """(a) rampa contra paralelismo 1: a vazão satura e o p95 cresce."""
    f = OllamaFalso(servico_s=0.3, paralelismo=1)
    try:
        r = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "1,2,4", "--duracao", "6"], f)
    finally:
        f.parar()
    v = {c: vazao_md(r["md"], "Rampa (laço fechado)", f"c={c}") for c in (1, 2, 4)}
    p1, p4 = p95(lat(r["linhas"], "rampa", "c=1")), p95(lat(r["linhas"], "rampa", "c=4"))
    ok = (r["rc"] == 0 and v[2] < 1.10 * v[1] and v[4] < 1.10 * v[2] and p4 > 2 * p1
          and "+ saturação" in r["md"])
    return ok, f"vazão c=1/2/4 = {v[1]:.0f}/{v[2]:.0f}/{v[4]:.0f} resp/min | p95 c=1 {p1:.2f}s → c=4 {p4:.2f}s | rc={r['rc']}"


def _aberto_sobrecarregado():
    f = OllamaFalso(servico_s=0.5, paralelismo=1, fila_max=4)
    try:
        r = executar(["--alvo", "ollama", "--cenario", "aberto", "--taxas", "1,3,6,9", "--duracao", "6"], f, timeout=240)
        rejeitadas = f.rejeitadas
    finally:
        f.parar()
    return r, rejeitadas


def prova_b():
    """(b) aberto acima da capacidade chega ao colapso e a varredura aborta sozinha (2 taxas seguidas)."""
    r, _ = _aberto_sobrecarregado()
    rotulos = {l["rotulo"] for l in r["linhas"] if l["cenario"] == "aberto"}
    ok = ("aberto interrompido após 6 req/s" in r["md"] and "9 req/s" not in rotulos and "3 req/s" in rotulos
          and "COLAPSO" in r["md"] and r["rc"] == 0)
    return ok, f"rotulos executados={sorted(rotulos)} | rc={r['rc']} | {r['segundos']:.0f}s"


def prova_c():
    """(c) 503 do Ollama é contado como http_5xx (com o código) e não trava o teste."""
    r, rej = _aberto_sobrecarregado()
    h = [l for l in r["linhas"] if l["status"] == "http_5xx"]
    ok = bool(h) and all(l["codigo_http"] == "503" for l in h) and r["rc"] == 0 and r["segundos"] < 200 and "HTTP 503" in r["md"]
    return ok, f"{len(h)} linhas http_5xx (código 503), {rej} rejeitadas pelo falso, terminou em {r['segundos']:.0f}s"


def prova_d():
    """(d) pico e recuperação com o falso lento por alguns segundos (fila de 12 x 0,3 s)."""
    f = OllamaFalso(servico_s=0.3, paralelismo=1)
    try:
        r = executar(["--alvo", "ollama", "--cenario", "pico", "--pico", "12", "--n-serie", "3",
                      "--intervalo-sonda", "1", "--janela-recuperacao", "25"], f)
    finally:
        f.parar()
    m = re.search(r"voltou a <= 1,5x a linha de base (\d+) s após o disparo", r["md"])
    t = int(m.group(1)) if m else None
    ok = r["rc"] == 0 and t is not None and 3 <= t <= 12 and "sonda 1" in "".join(l["rotulo"] for l in r["linhas"])
    return ok, f"recuperou em {t} s após o disparo (fila teórica ≈ 3,6 s) | rc={r['rc']}"


def prova_d2():
    """(d') se o falso nunca volta ao normal dentro da janela, o relatório diz 'não voltou'."""
    f = OllamaFalso(servico_s=0.2, paralelismo=1, lento_apos=(4, 1.5))   # 1 aq. + 3 base rápidos; depois 1,5 s
    try:
        r = executar(["--alvo", "ollama", "--cenario", "pico", "--pico", "4", "--n-serie", "3",
                      "--intervalo-sonda", "1", "--janela-recuperacao", "8", "--timeout-cliente", "30"], f, timeout=200)
    finally:
        f.parar()
    ok = r["rc"] == 0 and "não voltou" in r["md"]
    return ok, f"rc={r['rc']} | 'não voltou' no relatório: {'não voltou' in r['md']}"


def prova_e():
    """(e) timeout do cliente: classificado como timeout; aquecimento que estoura o timeout aborta com motivo."""
    f = OllamaFalso(servico_s=0.1, paralelismo=4, lento_apos=(6, 3.0))   # aquecimento + 5 de base rápidos
    try:
        r = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "2", "--duracao", "5",
                      "--timeout-cliente", "1"], f, timeout=120)
    finally:
        f.parar()
    st = {l["status"] for l in r["linhas"] if l["papel"] == "carga"}
    ok1 = r["rc"] == 0 and "timeout" in st and "timeout:" in r["md"]
    lat_to = [float(l["latencia_s"]) for l in r["linhas"] if l["status"] == "timeout"]
    ok1 = ok1 and lat_to and all(0.9 <= x <= 1.6 for x in lat_to)
    f2 = OllamaFalso(servico_s=3.0, paralelismo=4)
    try:
        r2 = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "1", "--duracao", "3",
                       "--timeout-cliente", "1"], f2, timeout=60)
    finally:
        f2.parar()
    ok2 = r2["rc"] == 3 and "aquecimento falhou (status=timeout)" in r2["md"]
    return ok1 and ok2, (f"carga: status={sorted(st)}, latência dos timeouts≈{min(lat_to, default=0):.2f}–"
                         f"{max(lat_to, default=0):.2f}s | aquecimento lento: rc={r2['rc']}, abortou={ok2}")


def prova_f():
    """(f) Ctrl+C (CTRL_BREAK no Windows) grava o parcial: CSV com respostas, relatório 'ABORTADO', rc 130."""
    f = OllamaFalso(servico_s=0.2, paralelismo=2)
    try:
        r = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "2", "--duracao", "40"], f,
                     sinal_apos=7, timeout=90)
    finally:
        f.parar()
    n_ok = sum(l["status"] == "ok" and l["papel"] == "carga" for l in r["linhas"])
    ok = (r["rc"] == 130 and "ABORTADO — interrompido pelo usuário" in r["md"] and n_ok > 5 and r["segundos"] < 40
          and any(l.get("evento", "").startswith("INICIO") for l in r["timeline"]))
    return ok, f"rc={r['rc']} | {n_ok} respostas de carga no CSV | terminou em {r['segundos']:.0f}s de 40s planejados"


def prova_g():
    """(g) sem sysfs, nvidia-smi nem psutil o script roda e avisa o que não leu."""
    f = OllamaFalso(servico_s=0.1, paralelismo=2)
    try:
        r = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "1", "--duracao", "3",
                      "--drm-raiz", str(Path(tempfile.gettempdir()) / "nao_existe_drm"),
                      "--nvidia-smi", "nvidia-smi-que-nao-existe"], f, sem_psutil=True, timeout=90)
    finally:
        f.parar()
    avisos = [x for x in ("amd_sysfs: NÃO LIDA", "nvidia_smi: NÃO LIDA", "cpu_ram: NÃO LIDA") if x in r["md"]]
    ok = r["rc"] == 0 and len(avisos) == 3 and "ollama_ps: NÃO LIDA" not in r["md"]
    return ok, f"rc={r['rc']} | avisos no relatório: {len(avisos)}/3 | ollama_ps lido: {'ollama_ps: NÃO' not in r['md']}"


def _fake_gpu(vram_pct: float, temp_nv=None):
    """sysfs AMD falso (+ nvidia-smi falso opcional) num diretório temporário."""
    raiz = Path(tempfile.mkdtemp(prefix="drm_"))
    dev = raiz / "card0" / "device"
    (dev / "hwmon" / "hwmon0").mkdir(parents=True)
    total = 16 * 1024 ** 3
    for nome, val in (("vendor", "0x1002"), ("mem_info_vram_total", total), ("mem_info_vram_used", int(total * vram_pct / 100)),
                      ("gpu_busy_percent", 40)):
        (dev / nome).write_text(str(val))
    (dev / "hwmon" / "hwmon0" / "temp1_input").write_text("61000")
    extra = []
    if temp_nv is not None:
        smi = raiz / "smi_falso.py"
        smi.write_text(f"print('1024, 30, {temp_nv}')\n")
        extra = ["--nvidia-smi", f"{Path(sys.executable).as_posix()} {smi.as_posix()}"]
    return str(raiz), extra


def prova_j():
    """(j) sysfs lido de verdade (VRAM, temperatura) e aborto se VRAM da AMD > 97% ou GPU > 95 °C."""
    resultados = []
    for vram, temp_nv, quer_abortar, trecho in ((50, None, False, ""), (98, None, True, "VRAM da AMD em 98.0%"),
                                                 (50, 96, True, "temperatura da GPU NVIDIA em 96")):
        raiz, extra = _fake_gpu(vram, temp_nv)
        f = OllamaFalso(servico_s=0.1, paralelismo=2)
        try:
            r = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "1", "--duracao", "8",
                          "--drm-raiz", raiz] + extra, f, timeout=90)
        finally:
            f.parar()
        abortou = r["rc"] == 3 and trecho in r["md"]
        leu = any(l.get("amd_vram_pct") not in (None, "") for l in r["timeline"])
        marcou = sum(1 for l in r["timeline"] if l.get("evento", "").startswith(("INICIO", "FIM")))
        resultados.append((abortou if quer_abortar else (r["rc"] == 0 and leu and marcou >= 2 and "61 " in r["md"] + " "), vram, temp_nv, r["rc"]))
    return all(x[0] for x in resultados), " | ".join(f"vram={v}% nv={t} rc={rc} ok={o}" for o, v, t, rc in resultados)


def prova_k():
    """(k) trava de segurança: sem --confirmo (c>8 ou >1 req/s) e sem --permitir-64 o script se recusa e mostra o plano."""
    f = OllamaFalso()
    try:
        r1 = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "1,16"], f, base=False, timeout=30)
        r2 = executar(["--alvo", "ollama", "--cenario", "aberto", "--taxas", "2"], f, base=False, timeout=30)
        r3 = executar(["--alvo", "ollama", "--cenario", "rampa", "--niveis", "64", "--confirmo"], f, base=False, timeout=30)
    finally:
        f.parar()
    ok = (r1["rc"] == r2["rc"] == r3["rc"] == 2 and "Plano de carga" in r1["texto"] and "exigem --confirmo" in r1["texto"]
          and "--permitir-64" in r3["texto"] and f.n == 0 and not r1["linhas"])
    return ok, f"rc={r1['rc']}/{r2['rc']}/{r3['rc']} | requisições recebidas pelo falso: {f.n}"


def prova_i():
    """(i) alvo pipeline de ponta a ponta: PASTA_CHROMA=./chroma_exp_f3_final, Ollama falso (precisa do venv do projeto)."""
    f = OllamaFalso(servico_s=0.2, paralelismo=2)
    try:
        r = executar(["--alvo", "pipeline", "--cenario", "rampa", "--niveis", "1,2", "--duracao", "8"], f,
                     env_extra={"PASTA_CHROMA": "./chroma_exp_f3_final", "RERANKER_ATIVO": "0"}, timeout=600)
    finally:
        f.parar()
    cargas = [l for l in r["linhas"] if l["papel"] == "carga"]
    ok_ = [l for l in cargas if l["status"] == "ok"]
    ok = (r["rc"] == 0 and len(ok_) >= 4 and all(l["tempo_retrieval_s"] for l in ok_) and all(l["tempo_llm_s"] for l in ok_)
          and len({l["id"] for l in cargas}) > 1 and "alvo `pipeline`" in r["md"])
    return ok, (f"rc={r['rc']} | {len(ok_)}/{len(cargas)} ok | ids distintos={len({l['id'] for l in cargas})} | "
                f"retrieval médio {sum(float(l['tempo_retrieval_s']) for l in ok_) / max(1, len(ok_)):.2f}s")


def prova_m():
    """(m) lógica pura: critérios (limite/saturação/colapso), regressão da deriva e sorteio determinístico."""
    sys.path.insert(0, str(AQUI))
    import teste_sobrecarga as ts
    mk = lambda vazao, p95_, pct_erro=0.0, n=10: {"n": n, "descartadas": 0, "vazao": vazao, "p95": p95_,
                                                    "pct_erro": pct_erro, "n_erro": round(n * pct_erro / 100)}
    rs = [mk(40, 3), mk(45, 8), mk(46, 14), mk(46, 40), mk(40, 30, 20), mk(20, None, 90)]
    c = ts.criterios(rs, 15.0)
    perguntas = {f"Q{i:02d}": f"pergunta {i}" for i in range(40)} | {"E04": "e4", "E02": "e2", "D02": "d2"}
    s1, s2 = ts.Sorteador(perguntas, 42, "c=4"), ts.Sorteador(perguntas, 42, "c=4")
    a = [s1.proximo()[0] for _ in range(14)]
    b = [s2.proximo()[0] for _ in range(14)]
    slope = ts.regressao([1, 2, 3, 4], [10, 12, 14, 16])
    # rs[3]: p95 de 40 s sem erros = acima do SLO, mas ainda não é colapso (> 60 s ou > 5% de erros); rs[4] colapsa
    ok = (c == {"limite": 2, "saturacao": 2, "colapso": 4, "contigua": True} and a == b
          and {"E04", "E02", "D02"} <= set(a) and abs(slope - 2.0) < 1e-9 and ts.colapsou(rs[5]) and not ts.colapsou(rs[2])
          and len(ts.prompt_fixo(7)) == 4500 and ts.prompt_fixo(7) != ts.prompt_fixo(8))
    return ok, f"critérios={c} | longas no 1º baralho={ {'E04', 'E02', 'D02'} <= set(a) } | inclinação={slope}"


def prova_h():
    """(h) --rapido completo (rampa+aberto+pico) com as pausas padrão termina em poucos minutos (< 8 min)."""
    f = OllamaFalso(servico_s=0.4, paralelismo=4)
    try:
        r = executar(["--alvo", "ollama", "--rapido", "--pausa", "5", "--pausa-cenario", "30"], f, base=False,
                     timeout=900)
    finally:
        f.parar()
        # BASE_ARGS sem pausa não é usado aqui; precisamos de --confirmo (pico 8 e c=8 não exigem, taxa 1 não exige)
    ok = r["rc"] == 0 and r["segundos"] < 480 and "Pico e recuperação" in r["md"] and "Aberto" in r["md"]
    return ok, f"rc={r['rc']} | duração {r['segundos'] / 60:.1f} min | seções: " + ", ".join(
        s for s in ("Rampa", "Aberto", "Pico e recuperação") if s in r["md"])


def prova_l():
    """(l) soak de 70% da saturação por ~75 s: janelas de 1 min com p50 e deriva."""
    f = OllamaFalso(servico_s=0.2, paralelismo=2)
    try:
        r = executar(["--alvo", "ollama", "--cenario", "soak", "--soak-min", "1.25", "--vazao-saturacao", "60"], f, timeout=300)
    finally:
        f.parar()
    ok = r["rc"] == 0 and "## Soak" in r["md"] and "Deriva do p50" in r["md"]
    return ok, f"rc={r['rc']} | duração {r['segundos']:.0f}s | janelas={r['md'].count('| 1 |') + r['md'].count('| 2 |')}"


PROVAS = {"a": prova_a, "b": prova_b, "c": prova_c, "d": prova_d, "d2": prova_d2, "e": prova_e, "f": prova_f,
          "g": prova_g, "j": prova_j, "k": prova_k, "m": prova_m, "i": prova_i}
LENTAS = {"h": prova_h, "l": prova_l}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="provas separadas por vírgula (ex.: a,b,f); padrão: todas as rápidas")
    ap.add_argument("--lento", action="store_true", help="inclui (h) --rapido completo e (l) soak")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    todas = {**PROVAS, **(LENTAS if a.lento or any(x in LENTAS for x in a.k.split(",")) else {})}
    quais = [x for x in a.k.split(",") if x] or list(todas)
    falhas = 0
    for nome in quais:
        t0 = time.monotonic()
        try:
            ok, detalhe = todas[nome]()
        except Exception as exc:
            ok, detalhe = False, f"EXCEÇÃO {type(exc).__name__}: {exc}"
        falhas += not ok
        print(f"[{'PASSOU' if ok else 'FALHOU'}] {todas[nome].__doc__.strip().splitlines()[0]}\n"
              f"          {detalhe}  [{time.monotonic() - t0:.0f}s]", flush=True)
    print(f"\n{len(quais) - falhas}/{len(quais)} provas passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
