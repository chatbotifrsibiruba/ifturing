"""
teste_sobrecarga.py — Até onde o gemma3:4b (Ollama) aguenta: onde a latência passa do aceitável, onde a vazão
satura, onde começam erros/travamentos e se o servidor se recupera depois de um pico.

Complementa teste_concorrencia.py (que só vai até 8 usuários); não altera nucleo.py nem o Ollama.

Alvos (--alvo):
    pipeline   fluxo completo de nucleo.responder() (reranker só se RERANKER_ATIVO=1 no ambiente)
    ollama     POST direto em /api/generate com prompt fixo de ~4.500 caracteres (isola o limite do Ollama)

Cenários (--cenario):
    rampa    laço fechado: C trabalhadores, cada um manda a próxima pergunta ao terminar a anterior
    aberto   chegadas Poisson (req/s) sem esperar as anteriores
    pico     5 em série (linha de base), dispara --pico de uma vez, mede 5 em série a cada 10 s e reporta
             em quantos segundos a latência (média da sonda) volta a <= 1,5x a linha de base (média)
    soak     carga contínua a 70% da vazão de saturação da rampa por --soak-min minutos (desligado por padrão)
    todos    rampa + aberto + pico (+ soak se --soak-min > 0)

Uso (no servidor, dentro do venv do projeto, em tmux):
    python teste_sobrecarga.py --rapido --confirmo --alvo ollama
    python teste_sobrecarga.py --cenario rampa --alvo ollama --confirmo
    python teste_sobrecarga.py --cenario aberto --alvo pipeline --confirmo

Concorrência > 8 ou taxa > 1 req/s exigem --confirmo (o Ollama do servidor é compartilhado com outro app).
O plano de carga é sempre mostrado antes de começar. Ctrl+C grava os resultados parciais.

Saídas (fora do git, em --saida-dir): resultado_sobrecarga_<AAAAMMDD_HHMM>.csv (uma linha por requisição),
linha_do_tempo_<...>.csv (monitoramento a cada 2 s) e resultado_sobrecarga_<...>.md (relatório).
O monitoramento (/api/ps, sysfs AMD, nvidia-smi, psutil ou /proc) é opcional: o que faltar é listado no relatório.
"""
import argparse
import csv
import glob
import itertools
import math
import os
import random
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

from teste_concorrencia import percentil

try:                                   # opcional: sem psutil o monitor tenta /proc e, faltando, avisa
    import psutil
except ImportError:
    psutil = None

AQUI = Path(__file__).parent
_OUT = sys.stdout                      # o pipeline silencia sys.stdout durante a carga; o log usa este
_ERR = sys.stderr
_LOG_LOCK = threading.Lock()

STATUS_SUCESSO = ("ok", "truncada")    # resposta recebida; "truncada" = bateu em num_predict (qualidade, não carga)
TOKENS_TRUNCADA = 1000                 # num_predict é 1024; >= 1000 tokens de saída conta como truncada
LONGAS = ("E04", "E02", "D02")         # perguntas de resposta longa que sempre entram cedo no sorteio
PROMPT_CHARS = 4500                    # tamanho do prompt do alvo ollama
GRACA_ABORT_S = 10.0                   # ao abortar, espera tantos segundos pelas requisições em voo
MIB = 1024 * 1024

COLUNAS_CSV = ["cenario", "rotulo", "papel", "id", "t_ini_s", "t_fim_s", "latencia_s", "status", "codigo_http",
               "tokens_entrada", "tokens_saida", "tempo_retrieval_s", "tempo_rerank_s", "tempo_llm_s",
               "tempo_geracao_s", "em_voo_no_inicio", "detalhe"]
COLUNAS_TL = ["ts", "t_s", "evento", "cenario", "degrau", "em_voo", "ollama_modelos", "ollama_vram_mib",
              "amd_vram_mib", "amd_vram_total_mib", "amd_vram_pct", "amd_busy_pct", "amd_temp_c",
              "nv_mem_mib", "nv_util_pct", "nv_temp_c", "cpu_pct", "ram_pct", "ram_mib"]


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(msg, file=_OUT, flush=True)


def fmt(x, casas: int = 2, sufixo: str = "") -> str:
    return "—" if x is None else f"{x:.{casas}f}{sufixo}"


def tabela(cabecalho: list, linhas: list) -> list:
    return (["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
            + ["| " + " | ".join(str(c) for c in l) + " |" for l in linhas])


class _Nulo:
    """Descarta o que o nucleo imprime ([timing] a cada pergunta) e os tracebacks que responder() engole."""
    def write(self, s):
        return len(s)

    def flush(self):
        pass


# ── Perguntas e prompt ─────────────────────────────────────────────────────

class Sorteador:
    """Baralho de perguntas do gabarito, reembaralhado a cada volta (semente fixa). As respostas longas
    (E04, E02, D02) entram sempre nas primeiras posições, para que até degraus curtos as incluam."""

    def __init__(self, perguntas: dict, semente: int, sal: str):
        self.itens = list(perguntas.items())
        self.rng = random.Random(f"{semente}:{sal}")
        self.fila, self.lock = [], threading.Lock()

    def _novo_baralho(self) -> list:
        outras = [i for i in self.itens if i[0] not in LONGAS]
        self.rng.shuffle(outras)
        for it in [i for i in self.itens if i[0] in LONGAS]:
            outras.insert(self.rng.randrange(0, min(12, len(outras)) + 1), it)
        return outras[::-1]                                  # pop() tira do fim

    def proximo(self) -> tuple:
        with self.lock:
            if not self.fila:
                self.fila = self._novo_baralho()
            return self.fila.pop()


class SorteadorFixo:
    """Alvo ollama: o prompt é fixo, não há pergunta a sortear."""
    def proximo(self) -> tuple:
        return ("FIXO", "")


_TRECHO = ("Art. {n}. O candidato inscrito no processo seletivo deverá comparecer ao campus escolhido, munido de "
           "documento de identificação com foto, nos prazos e horários divulgados no cronograma do edital. A "
           "classificação observará as cotas, o desempenho nas provas e os critérios de desempate previstos no "
           "edital, e a matrícula dependerá da entrega dos documentos listados no anexo correspondente. ")
_INSTRUCAO = "\nResuma, em tópicos detalhados, o que cada artigo acima exige do candidato."


def prompt_fixo(n_req: int) -> str:
    """Prompt de exatamente PROMPT_CHARS caracteres. O cabeçalho '[req N]' muda a cada requisição para o Ollama
    não reaproveitar o prefixo já processado (cache de prompt) e subestimar o prefill sob carga."""
    cab = f"[req {n_req % 10**6:06d}]\n"
    corpo = "".join(_TRECHO.format(n=i) for i in range(1, 30))
    return (cab + corpo)[:PROMPT_CHARS - len(_INSTRUCAO)] + _INSTRUCAO


# ── Classificação ──────────────────────────────────────────────────────────

def status_por_codigo(codigo: int) -> str:
    if codigo == 429:
        return "http_429"
    if codigo >= 500:
        return "http_5xx"
    return "http_4xx"


def status_por_excecao(exc: BaseException) -> str:
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "conexao"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.RequestException):
        return "conexao"
    return "erro_outro"


def resultado_vazio() -> dict:
    return {"status": "erro_outro", "codigo": None, "tokens_entrada": None, "tokens_saida": None,
            "t_retrieval": None, "t_rerank": None, "t_llm": None, "t_geracao": None, "detalhe": ""}


# ── Alvos ──────────────────────────────────────────────────────────────────

class AlvoOllama:
    """POST direto em /api/generate (mesmo payload do nucleo.chamar_ollama), prompt fixo de ~4.500 caracteres."""
    nome = "ollama"

    def __init__(self, url: str, modelo: str, timeout: float):
        self.url, self.modelo, self.timeout = url, modelo, timeout
        self._n, self._lock = itertools.count(1), threading.Lock()

    def executar(self, qid: str, pergunta: str) -> dict:
        res = resultado_vazio()
        with self._lock:
            n = next(self._n)
        payload = {"model": self.modelo, "prompt": prompt_fixo(n), "stream": False, "think": False,
                   "options": {"temperature": 0.0, "num_predict": 1024, "num_ctx": 8192}}
        try:
            r = requests.post(f"{self.url}/api/generate", json=payload,
                              timeout=(min(10.0, self.timeout), self.timeout))
        except requests.exceptions.RequestException as exc:
            res.update(status=status_por_excecao(exc), detalhe=f"{type(exc).__name__}: {exc}"[:120])
            return res
        res["codigo"] = r.status_code
        if r.status_code != 200:
            res.update(status=status_por_codigo(r.status_code), detalhe=r.text[:120].replace("\n", " "))
            return res
        try:
            d = r.json()
        except ValueError:
            res.update(status="erro_outro", detalhe="JSON inválido na resposta 200")
            return res
        ns = 1e9
        res.update(tokens_entrada=d.get("prompt_eval_count", 0), tokens_saida=d.get("eval_count", 0),
                   t_geracao=round(d.get("eval_duration", 0) / ns, 3))
        if not str(d.get("response", "")).strip():
            res["status"] = "resposta_vazia"
        elif (res["tokens_saida"] or 0) >= TOKENS_TRUNCADA:
            res["status"] = "truncada"
        else:
            res["status"] = "ok"
        return res


class _ProxyRequests:
    """Entra no lugar de nucleo.requests: impõe o timeout do cliente à chamada ao Ollama e guarda, por thread, o
    código HTTP ou a exceção (responder() engole os erros e só devolve erro=True)."""

    def __init__(self, real, timeout: float):
        self._real, self._timeout = real, timeout
        self.ultimo = threading.local()

    def __getattr__(self, nome):
        return getattr(self._real, nome)

    def post(self, url, **kw):
        kw["timeout"] = (min(10.0, self._timeout), self._timeout)
        self.ultimo.info = {}
        try:
            r = self._real.post(url, **kw)
        except Exception as exc:
            self.ultimo.info = {"exc": exc}
            raise
        self.ultimo.info = {"codigo": r.status_code}
        return r


class AlvoPipeline:
    """Fluxo completo de nucleo.responder(): embedding, busca, reranker opcional, híbrido e Ollama."""
    nome = "pipeline"

    def __init__(self, nucleo, modelo: str, timeout: float):
        self.nucleo, self.cfg = nucleo, {"tipo": "ollama", "modelo": modelo}
        self.proxy = _ProxyRequests(nucleo.requests, timeout)
        nucleo.requests = self.proxy

    def executar(self, qid: str, pergunta: str) -> dict:
        res = resultado_vazio()
        self.proxy.ultimo.info = {}
        r = self.nucleo.responder(pergunta, self.cfg)
        info = getattr(self.proxy.ultimo, "info", {})
        res.update(tokens_entrada=r.get("tokens_entrada"), tokens_saida=r.get("tokens_saida"),
                   t_retrieval=r.get("tempo_retrieval_s"), t_rerank=r.get("tempo_rerank_s"),
                   t_llm=r.get("tempo_llm_s"), t_geracao=r.get("tempo_geracao_s"), codigo=info.get("codigo"))
        if r.get("erro"):
            if info.get("exc") is not None:
                res.update(status=status_por_excecao(info["exc"]),
                           detalhe=f"{type(info['exc']).__name__}: {info['exc']}"[:120])
            elif info.get("codigo") and info["codigo"] >= 400:
                res["status"] = status_por_codigo(info["codigo"])
            else:
                res.update(status="erro_pipeline", detalhe=str(r.get("resposta", ""))[:120])
        elif str(r.get("resposta", "")).startswith("O modelo não conseguiu gerar"):
            res["status"] = "resposta_vazia"
        elif (res["tokens_saida"] or 0) >= TOKENS_TRUNCADA:
            res["status"] = "truncada"
        else:
            res["status"] = "ok"
        return res


# ── Monitoramento (opcional, tolera a ausência de cada fonte) ──────────────

class Monitor:
    """Amostra a cada --intervalo-monitor s para a linha do tempo: /api/ps do Ollama, VRAM/uso/temperatura da AMD
    (sysfs, sem sudo), NVIDIA (nvidia-smi), CPU/RAM (psutil ou /proc). Nunca falha o teste; aborta-o se a VRAM da
    AMD ou a temperatura de qualquer GPU passar do limite."""
    FONTES = ("ollama_ps", "amd_sysfs", "nvidia_smi", "cpu_ram")

    def __init__(self, a, t_rel, em_voo, abortar, caminho_csv: Path):
        self.a, self.t_rel, self.em_voo, self.abortar = a, t_rel, em_voo, abortar
        self.url = a.ollama_url
        self.intervalo = a.intervalo_monitor
        self.limite_vram, self.limite_temp = a.max_vram_pct, a.max_temp
        self.linhas, self.lock, self.fim = [], threading.Lock(), threading.Event()
        self.contexto = ("", "")
        self.fontes = {n: {"ok": 0, "falha": 0, "motivo": ""} for n in self.FONTES}
        self.thread = None
        self.arq = open(caminho_csv, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.arq, fieldnames=COLUNAS_TL)
        self.w.writeheader()
        self.arq.flush()
        self.amd = self._descobrir_amd(Path(a.drm_raiz))
        self.nvidia = shlex.split(a.nvidia_smi)
        self.proc_cpu_anterior = None
        if psutil is not None:
            psutil.cpu_percent(interval=None)            # a primeira chamada devolve 0.0; prepara a seguinte

    # descoberta ----------------------------------------------------------
    def _descobrir_amd(self, raiz: Path):
        """Card AMD (vendor 0x1002) com a maior VRAM (ignora iGPU); None se o sysfs não existir/não servir."""
        melhor = None
        for dev in sorted(glob.glob(str(raiz / "card*" / "device"))):
            dev = Path(dev)
            try:
                if (dev / "vendor").read_text().strip().lower() != "0x1002":
                    continue
                total = int((dev / "mem_info_vram_total").read_text())
            except (OSError, ValueError):
                continue
            if melhor is None or total > melhor[1]:
                melhor = (dev, total)
        if melhor is None:
            self.fontes["amd_sysfs"]["motivo"] = f"nenhum card 0x1002 com mem_info_vram_total em {raiz}/card*/device"
        return melhor[0] if melhor else None

    # leituras -----------------------------------------------------------
    def _ler_ollama(self) -> dict:
        r = requests.get(f"{self.url}/api/ps", timeout=2)
        r.raise_for_status()
        ms = r.json().get("models", [])
        return {"ollama_modelos": len(ms), "ollama_vram_mib": round(sum(m.get("size_vram", 0) for m in ms) / MIB)}

    def _ler_amd(self) -> dict:
        if self.amd is None:
            raise RuntimeError(self.fontes["amd_sysfs"]["motivo"])
        usado = int((self.amd / "mem_info_vram_used").read_text())
        total = int((self.amd / "mem_info_vram_total").read_text())
        d = {"amd_vram_mib": round(usado / MIB), "amd_vram_total_mib": round(total / MIB),
             "amd_vram_pct": round(100 * usado / total, 1) if total else None}
        try:
            d["amd_busy_pct"] = int((self.amd / "gpu_busy_percent").read_text())
        except (OSError, ValueError):
            pass
        for t in sorted(glob.glob(str(self.amd / "hwmon" / "hwmon*" / "temp1_input"))):
            try:
                d["amd_temp_c"] = int(Path(t).read_text()) / 1000
                break
            except (OSError, ValueError):
                continue
        return d

    def _ler_nvidia(self) -> dict:
        saida = subprocess.run(self.nvidia + ["--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                                              "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5, check=True).stdout
        mem, util, temp = 0.0, [], []
        for linha in saida.strip().splitlines():
            m, u, t = [x.strip() for x in linha.split(",")]
            mem += float(m)
            util.append(float(u))
            temp.append(float(t))
        if not temp:
            raise RuntimeError("nvidia-smi não listou nenhuma GPU")
        return {"nv_mem_mib": mem, "nv_util_pct": max(util), "nv_temp_c": max(temp)}

    def _ler_cpu_ram(self) -> dict:
        if psutil is not None:
            vm = psutil.virtual_memory()
            return {"cpu_pct": psutil.cpu_percent(interval=None), "ram_pct": vm.percent,
                    "ram_mib": round(vm.used / MIB)}
        # sem psutil: /proc (Linux)
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        usado = mi["MemTotal"] - mi["MemAvailable"]
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        ocioso, total = v[3] + v[4], sum(v)
        cpu = None
        if self.proc_cpu_anterior:
            dt, di = total - self.proc_cpu_anterior[0], ocioso - self.proc_cpu_anterior[1]
            cpu = round(100 * (1 - di / dt), 1) if dt > 0 else None
        self.proc_cpu_anterior = (total, ocioso)
        return {"cpu_pct": cpu, "ram_pct": round(100 * usado / mi["MemTotal"], 1), "ram_mib": round(usado / 1024)}

    # laço ---------------------------------------------------------------
    def amostrar(self) -> dict:
        linha = {}
        for nome, leitor in (("ollama_ps", self._ler_ollama), ("amd_sysfs", self._ler_amd),
                             ("nvidia_smi", self._ler_nvidia), ("cpu_ram", self._ler_cpu_ram)):
            f = self.fontes[nome]
            try:
                linha.update(leitor())
                f["ok"] += 1
            except Exception as exc:                     # fonte ausente/instável: registra e segue
                f["falha"] += 1
                f["motivo"] = f["motivo"] or f"{type(exc).__name__}: {exc}"[:160]
        return linha

    def _gravar(self, linha: dict) -> None:
        linha = {"ts": datetime.now().isoformat(timespec="seconds"), "t_s": round(self.t_rel(), 2),
                 "cenario": self.contexto[0], "degrau": self.contexto[1], **linha}
        with self.lock:
            self.linhas.append(linha)
            self.w.writerow({k: linha.get(k, "") for k in COLUNAS_TL})
            self.arq.flush()

    def marcar(self, evento: str) -> None:
        self._gravar({"evento": evento, "em_voo": self.em_voo()})

    def _verificar_limites(self, l: dict) -> None:
        if l.get("amd_vram_pct") is not None and l["amd_vram_pct"] > self.limite_vram:
            self.abortar(f"VRAM da AMD em {l['amd_vram_pct']:.1f}% (> {self.limite_vram:g}%)")
        for chave, nome in (("amd_temp_c", "AMD"), ("nv_temp_c", "NVIDIA")):
            if l.get(chave) is not None and l[chave] > self.limite_temp:
                self.abortar(f"temperatura da GPU {nome} em {l[chave]:.0f} °C (> {self.limite_temp:g} °C)")

    def _laco(self) -> None:
        while True:
            try:
                l = self.amostrar()
                l["em_voo"] = self.em_voo()
                self._gravar(l)
                self._verificar_limites(l)
            except Exception:
                pass                                     # o monitor nunca derruba o teste
            if self.fim.wait(self.intervalo):
                return

    def iniciar(self) -> None:
        self.thread = threading.Thread(target=self._laco, daemon=True)
        self.thread.start()

    def parar(self) -> None:
        self.fim.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            self.arq.close()

    def maximos(self, t_a: float, t_b: float) -> dict:
        """Máximos das amostras (não dos marcadores) entre t_a e t_b, em segundos desde o início."""
        with self.lock:
            sel = [l for l in self.linhas if not l.get("evento") and t_a <= l["t_s"] <= t_b]
        pega = lambda k: [l[k] for l in sel if l.get(k) is not None]
        temps = pega("amd_temp_c") + pega("nv_temp_c")
        return {"n": len(sel), "ollama_vram_mib": max(pega("ollama_vram_mib"), default=None),
                "amd_vram_mib": max(pega("amd_vram_mib"), default=None),
                "amd_vram_pct": max(pega("amd_vram_pct"), default=None),
                "nv_mem_mib": max(pega("nv_mem_mib"), default=None), "temp_c": max(temps, default=None)}

    def ausentes(self) -> list:
        """Fontes que nunca foram lidas (ou falharam em parte), com o motivo do primeiro erro."""
        out = []
        for nome, f in self.fontes.items():
            if f["ok"] == 0:
                out.append(f"{nome}: NÃO LIDA — {f['motivo'] or 'sem amostras'}")
            elif f["falha"]:
                out.append(f"{nome}: {f['falha']} falha(s) em {f['ok'] + f['falha']} amostras — {f['motivo']}")
        if self.fontes["amd_sysfs"]["ok"] and not any(l.get("amd_temp_c") is not None for l in self.linhas):
            out.append("temperatura da AMD: não lida (hwmon temp1_input ausente); o limite de temperatura só vale "
                       "para a NVIDIA")
        return out


# ── Execução ───────────────────────────────────────────────────────────────

class Teste:
    def __init__(self, a, alvo, perguntas: dict, caminho_csv: Path, caminho_tl: Path):
        self.a, self.alvo, self.perguntas = a, alvo, perguntas
        self.t0 = time.monotonic()
        self.inicio = datetime.now()
        self.prazo = self.t0 + a.max_total_min * 60
        self.parar = threading.Event()
        self.motivo_parada = ""
        self.lock = threading.Lock()
        self.em_voo = 0
        self.degraus = []                               # rampa, aberto, soak e pico, na ordem em que rodaram
        self.base = None
        self.pico = None
        self.soak = None
        self.notas = []
        self.erro_fatal = ""
        self.arq = open(caminho_csv, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.arq)
        self.w.writerow(COLUNAS_CSV)
        self.arq.flush()
        self.monitor = Monitor(a, self.rel, lambda: self.em_voo, self.abortar, caminho_tl)

    # infraestrutura ------------------------------------------------------
    def rel(self) -> float:
        return time.monotonic() - self.t0

    def abortar(self, motivo: str) -> None:
        with self.lock:
            if self.parar.is_set():
                return
            self.motivo_parada = motivo
            self.parar.set()
        log(f"!! ABORTANDO: {motivo}")

    def esperar(self, s: float) -> bool:
        """Dorme s segundos ou até pararem o teste; True se foi interrompido."""
        fim = time.monotonic() + s
        while not self.parar.is_set():
            if time.monotonic() > self.prazo:
                self.abortar(f"teto de duração de {self.a.max_total_min:g} min atingido")
                break
            rem = fim - time.monotonic()
            if rem <= 0:
                return False
            self.parar.wait(min(rem, 0.5))
        return True

    def cabe(self, segundos: float, o_que: str) -> bool:
        if self.prazo - time.monotonic() < segundos:
            self.abortar(f"teto de duração de {self.a.max_total_min:g} min: {o_que} ({segundos:.0f} s) não cabe")
            return False
        return True

    def requisitar(self, qid: str, pergunta: str, cenario: str, rotulo: str, papel: str,
                   contado: bool = False) -> dict:
        """Uma requisição, com tempo medido e linha gravada no CSV. `contado`: o chamador já incrementou em_voo."""
        if not contado:
            with self.lock:
                self.em_voo += 1
        with self.lock:
            n_voo = self.em_voo
        t_ini, p0 = self.rel(), time.perf_counter()
        try:
            res = self.alvo.executar(qid, pergunta)
        except Exception as exc:                          # rede de segurança: o alvo já classifica o esperado
            res = resultado_vazio()
            res["detalhe"] = f"{type(exc).__name__}: {exc}"[:120]
        lat = time.perf_counter() - p0
        with self.lock:
            self.em_voo -= 1
        reg = {"cenario": cenario, "rotulo": rotulo, "papel": papel, "id": qid, "t_ini": t_ini,
               "t_fim": t_ini + lat, "latencia_s": lat, "em_voo_ini": n_voo, **res}
        self.gravar(reg)
        return reg

    def gravar(self, reg: dict) -> None:
        r = lambda x: "" if x is None else x
        with self.lock:
            self.w.writerow([reg["cenario"], reg["rotulo"], reg["papel"], reg["id"], f"{reg['t_ini']:.3f}",
                             f"{reg['t_fim']:.3f}", f"{reg['latencia_s']:.3f}", reg["status"], r(reg["codigo"]),
                             r(reg["tokens_entrada"]), r(reg["tokens_saida"]), r(reg["t_retrieval"]),
                             r(reg["t_rerank"]), r(reg["t_llm"]), r(reg["t_geracao"]), reg["em_voo_ini"],
                             reg["detalhe"]])
            self.arq.flush()

    def descartar(self, d: dict) -> None:
        """Chegada que o gerador não enviou por já haver --max-em-voo requisições em voo."""
        t = self.rel()
        reg = {"cenario": d["cenario"], "rotulo": d["rotulo"], "papel": "carga", "id": "-", "t_ini": t, "t_fim": t,
               "latencia_s": 0.0, "em_voo_ini": self.em_voo, **resultado_vazio()}
        reg.update(status="descartada", detalhe=f"teto de {self.a.max_em_voo} em voo")
        self.gravar(reg)
        d["descartadas"] += 1

    def novo_degrau(self, cenario: str, rotulo: str, carga: float, unidade: str, janela_s: float) -> dict:
        d = {"cenario": cenario, "rotulo": rotulo, "carga": carga, "unidade": unidade, "janela_s": janela_s,
             "t_ini": self.rel(), "t_fim": None, "regs": [], "descartadas": 0, "chegadas": 0,
             "interrompido": False, "sem_resposta": 0}
        with self.lock:
            self.degraus.append(d)
        self.monitor.contexto = (cenario, rotulo)
        self.monitor.marcar(f"INICIO {cenario} {rotulo}")
        return d

    def drenar(self, threads: list) -> int:
        """Espera as requisições em voo terminarem (todas têm timeout). Ao abortar, dá GRACA_ABORT_S e desiste das
        restantes (threads daemon). Devolve quantas ficaram sem terminar."""
        limite = None
        while any(t.is_alive() for t in threads):
            if self.parar.is_set():
                limite = limite or time.monotonic() + GRACA_ABORT_S
                if time.monotonic() > limite:
                    break
            time.sleep(0.1)
        return sum(t.is_alive() for t in threads)

    def fechar_degrau(self, d: dict, janela_real: float, threads: list) -> None:
        d["sem_resposta"] = self.drenar(threads)
        d["janela_s"] = max(min(d["janela_s"], janela_real), 1e-6)
        d["interrompido"] = self.parar.is_set()
        d["t_fim"] = self.rel()
        self.monitor.marcar(f"FIM {d['cenario']} {d['rotulo']}")

    def sorteador(self, sal: str):
        return Sorteador(self.perguntas, self.a.semente, sal) if self.perguntas else SorteadorFixo()

    # aquecimento e linha de base ------------------------------------------
    def serie(self, n: int, cenario: str, rotulo: str, papel: str, sorteio) -> list:
        regs = []
        for _ in range(n):
            if self.parar.is_set():
                break
            qid, perg = sorteio.proximo()
            regs.append(self.requisitar(qid, perg, cenario, rotulo, papel))
        return regs

    def aquecer_e_medir_base(self, com_base: bool) -> bool:
        """1 pergunta de aquecimento (carrega o modelo) e, se pedido, --n-serie em série como linha de base."""
        log("aquecimento: 1 pergunta em série (carrega o modelo no Ollama)")
        self.monitor.contexto = ("base", "aquecimento")
        aq = self.serie(1, "base", "aquecimento", "aquecimento", self.sorteador("aq"))
        if not aq or aq[0]["status"] not in STATUS_SUCESSO:
            st = aq[0]["status"] if aq else "interrompido"
            det = aq[0]["detalhe"] if aq else ""
            self.abortar(f"aquecimento falhou (status={st}) {det}".strip())
            return False
        if not com_base:
            return True
        self.monitor.contexto = ("base", "serie")
        regs = self.serie(self.a.n_serie, "base", "serie", "base", self.sorteador("base"))
        ok = [r for r in regs if r["status"] in STATUS_SUCESSO]
        if ok:
            lat = [r["latencia_s"] for r in ok]
            toks = sum(r["tokens_saida"] or 0 for r in ok)
            self.base = {"n": len(ok), "p50": percentil(lat, 50), "p95": percentil(lat, 95), "max": max(lat),
                         "tok_s": toks / sum(lat) if sum(lat) else None}
            log(f"linha de base (série, n={len(ok)}): p50 {self.base['p50']:.2f}s | {fmt(self.base['tok_s'], 1)} tok/s")
        else:
            self.notas.append("linha de base em série sem nenhuma resposta válida: 'espera na fila' não calculada")
        return not self.parar.is_set()

    @property
    def base_p50(self):
        return self.base["p50"] if self.base else None

    # rampa (laço fechado) -------------------------------------------------
    def rodar_degrau_fechado(self, c: int) -> dict:
        d = self.novo_degrau("rampa", f"c={c}", c, "c", self.a.duracao)
        sorteio = self.sorteador(d["rotulo"])
        inicio = time.monotonic()
        fim = inicio + self.a.duracao

        def trabalhador():
            while not self.parar.is_set() and time.monotonic() < min(fim, self.prazo):
                qid, perg = sorteio.proximo()
                reg = self.requisitar(qid, perg, "rampa", d["rotulo"], "carga")
                with self.lock:
                    d["regs"].append(reg)

        threads = [threading.Thread(target=trabalhador, daemon=True) for _ in range(c)]
        for t in threads:
            t.start()
        self.esperar(self.a.duracao)
        self.fechar_degrau(d, time.monotonic() - inicio, threads)
        return d

    def rodar_rampa(self) -> None:
        colapsos = 0
        for c in self.a.niveis:
            if self.parar.is_set() or not self.cabe(self.a.duracao, f"degrau c={c}"):
                break
            d = self.rodar_degrau_fechado(c)
            res = resumir(d, self.base_p50, self.monitor)
            log(linha_resumo(d, res))
            colapsos = colapsos + 1 if colapsou(res) else 0
            if colapsos >= 2:
                self.notas.append(f"rampa interrompida após o degrau c={c}: 2 degraus seguidos em colapso")
                log(self.notas[-1])
                break
            self.esperar(self.a.pausa)

    # aberto (chegadas Poisson) -------------------------------------------
    def rodar_degrau_aberto(self, taxa: float, duracao: float, cenario: str, rotulo: str) -> dict:
        d = self.novo_degrau(cenario, rotulo, taxa, "req/s", duracao)
        sorteio = self.sorteador(rotulo)
        rng = random.Random(f"{self.a.semente}:{rotulo}:chegadas")
        inicio = time.monotonic()
        fim = inicio + duracao
        proxima, threads = inicio, []

        def uma(qid, perg):
            reg = self.requisitar(qid, perg, cenario, rotulo, "carga", contado=True)
            with self.lock:
                d["regs"].append(reg)

        while not self.parar.is_set():
            proxima += rng.expovariate(taxa)
            if proxima >= fim or proxima > self.prazo:
                break
            if self.esperar(proxima - time.monotonic()):
                break
            d["chegadas"] += 1
            with self.lock:
                cheio = self.em_voo >= self.a.max_em_voo
                if not cheio:
                    self.em_voo += 1
            if cheio:
                self.descartar(d)
                continue
            qid, perg = sorteio.proximo()
            t = threading.Thread(target=uma, args=(qid, perg), daemon=True)
            t.start()
            threads.append(t)
        if not self.parar.is_set():
            self.esperar(max(0.0, fim - time.monotonic()))     # janela completa mesmo se a última chegada foi antes
        self.fechar_degrau(d, time.monotonic() - inicio, threads)
        return d

    def rodar_aberto(self) -> None:
        colapsos = 0
        for taxa in self.a.taxas:
            if self.parar.is_set() or not self.cabe(self.a.duracao, f"taxa {taxa:g} req/s"):
                break
            d = self.rodar_degrau_aberto(taxa, self.a.duracao, "aberto", f"{taxa:g} req/s")
            res = resumir(d, self.base_p50, self.monitor)
            log(linha_resumo(d, res))
            colapsos = colapsos + 1 if colapsou(res) else 0
            if colapsos >= 2:
                self.notas.append(f"aberto interrompido após {taxa:g} req/s: 2 taxas seguidas em colapso")
                log(self.notas[-1])
                break
            self.esperar(self.a.pausa)

    # pico e recuperação ---------------------------------------------------
    def rodar_pico(self) -> None:
        a = self.a
        self.monitor.contexto = ("pico", "base")
        sorteio = self.sorteador("pico")
        base = [r for r in self.serie(a.n_serie, "pico", "base", "base", sorteio) if r["status"] in STATUS_SUCESSO]
        if not base:
            self.notas.append("pico não executado: a linha de base em série não teve respostas válidas")
            return
        # Média, não mediana: numa sonda em série só a primeira requisição espera atrás do pico na fila, e a
        # mediana das demais esconderia isso (a prova offline (d) pegou exatamente esse erro).
        base_ref = statistics.fmean(r["latencia_s"] for r in base)
        limiar = 1.5 * base_ref
        if not self.cabe(a.janela_recuperacao, "pico + recuperação"):
            return
        d = self.novo_degrau("pico", f"pico {a.pico}", a.pico, "simultâneas", a.janela_recuperacao)
        largada = threading.Event()

        def uma(qid, perg):
            largada.wait()
            reg = self.requisitar(qid, perg, "pico", d["rotulo"], "pico", contado=True)
            with self.lock:
                d["regs"].append(reg)

        threads = []
        for _ in range(a.pico):
            qid, perg = sorteio.proximo()
            with self.lock:
                self.em_voo += 1
            t = threading.Thread(target=uma, args=(qid, perg), daemon=True)
            t.start()
            threads.append(t)
        t_disp = time.monotonic()
        largada.set()
        log(f"pico: {a.pico} requisições disparadas de uma vez; base (média) {base_ref:.2f}s, limiar {limiar:.2f}s")

        sondas, t_recuperou, k = [], None, 1
        while not self.parar.is_set():
            alvo_t = t_disp + k * a.intervalo_sonda
            if alvo_t - t_disp > a.janela_recuperacao or self.esperar(max(0.0, alvo_t - time.monotonic())):
                break
            ini = time.monotonic() - t_disp
            regs = self.serie(a.n_serie, "pico", f"sonda {k}", "sonda", sorteio)
            ok = [r for r in regs if r["status"] in STATUS_SUCESSO]
            media_s = statistics.fmean(r["latencia_s"] for r in ok) if ok else None
            volta = len(regs) == a.n_serie and len(ok) == len(regs) and media_s <= limiar
            sondas.append({"t": ini, "media": media_s, "ok": len(ok), "n": len(regs), "volta": volta})
            log(f"  sonda {k} (t={ini:.0f}s após o disparo): média {fmt(media_s)}s, {len(ok)}/{len(regs)} ok"
                f"{' → RECUPEROU' if volta else ''}")
            if volta:
                t_recuperou = ini
                break
            k = max(k + 1, int((time.monotonic() - t_disp) // a.intervalo_sonda) + 1)
        self.fechar_degrau(d, a.janela_recuperacao, threads)
        t_dren = max((r["t_fim"] for r in d["regs"]), default=d["t_ini"]) - (d["t_ini"] + 0.0)
        d["janela_s"] = max(t_dren, 1e-6)                 # vazão do pico = respostas / tempo até o último término
        self.pico = {"n": a.pico, "base_ref": base_ref, "limiar": limiar, "sondas": sondas,
                     "t_recuperou": t_recuperou, "t_dren": t_dren, "degrau": d,
                     "interrompido": self.parar.is_set() and t_recuperou is None}

    # soak ------------------------------------------------------------------
    def vazao_saturacao(self):
        """resp/min de saturação: a maior vazão da rampa sem colapso (ou --vazao-saturacao)."""
        if self.a.vazao_saturacao:
            return self.a.vazao_saturacao
        rs = [resumir(d, None, None) for d in self.degraus if d["cenario"] == "rampa"]
        bons = [r["vazao"] for r in rs if r["n"] and not colapsou(r)] or [r["vazao"] for r in rs if r["n"]]
        return max(bons, default=None)

    def rodar_soak(self) -> None:
        v = self.vazao_saturacao()
        if not v:
            self.notas.append("soak não executado: sem referência de saturação (rode a rampa junto ou passe "
                              "--vazao-saturacao)")
            return
        taxa = 0.7 * v / 60
        if taxa > 1.0 and not self.a.confirmo:
            self.notas.append(f"soak não executado: taxa de {taxa:.2f} req/s exige --confirmo")
            return
        segundos = self.a.soak_min * 60
        if not self.cabe(segundos, "soak"):
            return
        log(f"soak: {taxa:.3f} req/s (70% de {v:.1f} resp/min) por {self.a.soak_min:g} min")
        d = self.rodar_degrau_aberto(taxa, segundos, "soak", f"{taxa:.2f} req/s")
        self.soak = {"taxa": taxa, "vazao_ref": v, "degrau": d}
        log(linha_resumo(d, resumir(d, self.base_p50, self.monitor)))


# ── Resumo e critérios ─────────────────────────────────────────────────────

def resumir(d: dict, base_p50, monitor) -> dict:
    """Métricas de um degrau/taxa. Vazão e tokens/s contam as respostas concluídas DENTRO da janela de carga;
    percentis e erros contam todas as requisições iniciadas no degrau (inclusive as concluídas na drenagem)."""
    regs = d["regs"]
    n = len(regs)
    ok = [r for r in regs if r["status"] in STATUS_SUCESSO]
    erros = Counter(r["status"] for r in regs if r["status"] not in STATUS_SUCESSO)
    codigos = Counter(r["codigo"] for r in regs if r["status"].startswith("http_") and r["codigo"])
    lat = [r["latencia_s"] for r in ok]
    janela = d["janela_s"]
    na_janela = [r for r in ok if r["t_fim"] <= d["t_ini"] + janela]
    n_erro = sum(erros.values()) + d["descartadas"]
    total = n + d["descartadas"]
    media_de = lambda k: statistics.fmean([r[k] for r in ok if r[k] is not None]) if any(
        r[k] is not None for r in ok) else None
    p50, p95 = (percentil(lat, 50), percentil(lat, 95)) if lat else (None, None)
    res = {"n": n, "descartadas": d["descartadas"], "ok": len(ok), "truncadas": sum(r["status"] == "truncada" for r in regs),
           "pct_ok": 100 * len(ok) / n if n else 0.0, "erros": erros, "codigos": codigos,
           "pct_erro": 100 * n_erro / total if total else 0.0, "n_erro": n_erro,
           "p50": p50, "p95": p95, "p99": percentil(lat, 99) if lat else None, "max": max(lat) if lat else None,
           "vazao": 60 * len(na_janela) / janela, "tok_s": sum(r["tokens_saida"] or 0 for r in na_janela) / janela,
           "oferecida": 60 * d["chegadas"] / janela if d["chegadas"] else None,
           "retrieval": media_de("t_retrieval"), "rerank": media_de("t_rerank"), "llm": media_de("t_llm"),
           "fila_p50": max(0.0, p50 - base_p50) if p50 is not None and base_p50 is not None else None,
           "fila_p95": max(0.0, p95 - base_p50) if p95 is not None and base_p50 is not None else None,
           "recursos": monitor.maximos(d["t_ini"], d["t_fim"] if d["t_fim"] is not None else d["t_ini"] + janela)
           if monitor else {}}
    return res


def colapsou(r: dict) -> bool:
    """COLAPSO: erros > 5% ou p95 > 60 s (sem nenhuma resposta válida também conta)."""
    if r["n"] + r["descartadas"] == 0:
        return False
    return r["pct_erro"] > 5 or r["p95"] is None or r["p95"] > 60


def aceitavel(r: dict, slo_p95: float) -> bool:
    """LIMITE ACEITÁVEL: 0 erros e p95 <= SLO."""
    return r["n"] > 0 and r["n_erro"] == 0 and r["p95"] is not None and r["p95"] <= slo_p95


def criterios(resumos: list, slo_p95: float) -> dict:
    """Índices (na lista, em ordem de carga crescente) do limite aceitável, da saturação e do colapso."""
    limite = max((i for i, r in enumerate(resumos) if aceitavel(r, slo_p95)), default=None)
    sat = next((i for i in range(1, len(resumos))
                if resumos[i - 1]["vazao"] > 0 and resumos[i]["vazao"] < 1.10 * resumos[i - 1]["vazao"]), None)
    col = next((i for i, r in enumerate(resumos) if colapsou(r)), None)
    contigua = limite is None or all(aceitavel(r, slo_p95) for r in resumos[:limite + 1])
    return {"limite": limite, "saturacao": sat, "colapso": col, "contigua": contigua}


def estado_de(r: dict, slo_p95: float) -> str:
    if r["n"] + r["descartadas"] == 0:
        return "sem dados"
    if colapsou(r):
        return "COLAPSO"
    return "ok" if aceitavel(r, slo_p95) else "acima do SLO"


def txt_erros(r: dict) -> str:
    partes = [f"{k}:{v}" for k, v in sorted(r["erros"].items())]
    if r["descartadas"]:
        partes.append(f"descartada:{r['descartadas']}")
    cods = ",".join(str(c) for c in sorted(r["codigos"]))
    return (" ".join(partes) + (f" (HTTP {cods})" if cods else "")) or "0"


def linha_resumo(d: dict, r: dict) -> str:
    return (f"[{d['cenario']} {d['rotulo']}] n={r['n']} ok={r['pct_ok']:.0f}% erros={txt_erros(r)} | "
            f"p50 {fmt(r['p50'])}s p95 {fmt(r['p95'])}s máx {fmt(r['max'])}s | {r['vazao']:.1f} resp/min | "
            f"{r['tok_s']:.1f} tok/s{' | INTERROMPIDO' if d['interrompido'] else ''}")


# ── Relatório ──────────────────────────────────────────────────────────────

def tabelas_degraus(t: Teste, cenario: str, resumos: list, degraus: list) -> list:
    aberto = cenario in ("aberto", "soak")
    cab = ["carga"] + (["oferecida (req/min)", "descartadas"] if aberto else []) + [
        "n", "% ok", "trunc.", "erros", "p50 (s)", "p95 (s)", "p99 (s)", "máx (s)", "vazão (resp/min)", "tokens/s",
        "estado"]
    cri = criterios(resumos, t.a.slo_p95)
    linhas = []
    for i, (d, r) in enumerate(zip(degraus, resumos)):
        marca = estado_de(r, t.a.slo_p95)
        if cri["saturacao"] == i:
            marca += " + saturação"
        linhas.append([d["rotulo"]] + ([fmt(r["oferecida"], 1), r["descartadas"]] if aberto else [])
                      + [r["n"], f"{r['pct_ok']:.0f}%", r["truncadas"], txt_erros(r), fmt(r["p50"]), fmt(r["p95"]),
                         fmt(r["p99"]), fmt(r["max"]), f"{r['vazao']:.1f}", f"{r['tok_s']:.1f}",
                         marca + (" (interrompido)" if d["interrompido"] else "")])
    L = tabela(cab, linhas) + [""]
    cab2 = ["carga", "retrieval (s)", "rerank (s)", "llm (s)", "espera p50 (s)", "espera p95 (s)", "VRAM Ollama máx (MiB)",
            "VRAM AMD máx (MiB / %)", "VRAM NVIDIA máx (MiB)", "temp. máx (°C)"]
    l2 = []
    for d, r in zip(degraus, resumos):
        rc = r["recursos"]
        vram = (f"{rc['amd_vram_mib']:.0f} / {rc['amd_vram_pct']:.1f}%" if rc.get("amd_vram_mib") is not None else "—")
        l2.append([d["rotulo"], fmt(r["retrieval"]), fmt(r["rerank"]), fmt(r["llm"]), fmt(r["fila_p50"]),
                   fmt(r["fila_p95"]), fmt(rc.get("ollama_vram_mib"), 0), vram, fmt(rc.get("nv_mem_mib"), 0),
                   fmt(rc.get("temp_c"), 0)])
    return L + tabela(cab2, l2)


def regressao(xs: list, ys: list):
    """Inclinação (mínimos quadrados) de ys em xs; None com menos de 2 pontos."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def janelas_soak(t: Teste) -> list:
    d = t.soak["degrau"]
    por_min = {}
    for r in d["regs"]:
        por_min.setdefault(int((r["t_ini"] - d["t_ini"]) // 60), []).append(r)
    out = []
    for m in sorted(por_min):
        rs = por_min[m]
        ok = [r["latencia_s"] for r in rs if r["status"] in STATUS_SUCESSO]
        rec = t.monitor.maximos(d["t_ini"] + 60 * m, d["t_ini"] + 60 * (m + 1))
        out.append({"min": m + 1, "n": len(rs), "erros": len(rs) - len(ok),
                    "p50": percentil(ok, 50) if ok else None, "p95": percentil(ok, 95) if ok else None,
                    "vram": rec.get("amd_vram_mib") if rec.get("amd_vram_mib") is not None else rec.get("ollama_vram_mib")})
    return out


def deriva_soak(janelas: list) -> dict:
    com = [j for j in janelas if j["p50"] is not None]
    if len(com) < 2:
        return {"inclinacao": None, "razao": None}
    return {"inclinacao": regressao([j["min"] for j in com], [j["p50"] for j in com]),
            "razao": com[-1]["p50"] / com[0]["p50"] if com[0]["p50"] else None}


def secoes_por_cenario(t: Teste) -> dict:
    out = {}
    for cen in ("rampa", "aberto"):
        ds = [d for d in t.degraus if d["cenario"] == cen]
        if ds:
            rs = [resumir(d, t.base_p50, t.monitor) for d in ds]
            out[cen] = (ds, rs, criterios(rs, t.a.slo_p95))
    return out


def gerar_leitura(t: Teste, secoes: dict) -> list:
    """8 a 12 linhas só com o que os números mostram; causas só como itens a verificar, nunca afirmadas."""
    a, L = t.a, []
    todas = [r for d in t.degraus for r in d["regs"]]
    st = Counter(r["status"] for r in todas)
    n_err = sum(v for k, v in st.items() if k not in STATUS_SUCESSO)
    L.append(f"Foram registradas {len(todas)} requisições de carga em {len(t.degraus)} degrau(s)/taxa(s) "
             f"(alvo {t.alvo.nome}, modelo {a.modelo}); {n_err} com erro"
             + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(st.items()) if k not in STATUS_SUCESSO)})." if n_err else "."))
    if t.base:
        geracao = f" e a geração roda a {t.base['tok_s']:.1f} tokens/s" if t.base["tok_s"] else ""
        L.append(f"Em série (sem concorrência) a latência p50 é {t.base['p50']:.2f} s{geracao} "
                 f"(n={t.base['n']}); é a referência da 'espera na fila'.")
    if "rampa" in secoes:
        ds, rs, c = secoes["rampa"]
        v1 = next((r["vazao"] for d, r in zip(ds, rs) if d["carga"] == 1), None)
        melhor = max(range(len(rs)), key=lambda i: rs[i]["vazao"])
        L.append(f"Rampa: a vazão máxima foi {rs[melhor]['vazao']:.1f} resp/min em {ds[melhor]['rotulo']}"
                 + (f", {rs[melhor]['vazao'] / v1:.2f}x a de c=1 ({v1:.1f} resp/min). Razão perto de 1 é compatível "
                    f"com o Ollama atendendo uma requisição por vez (não prova a causa; ver OLLAMA_NUM_PARALLEL)."
                    if v1 else "."))
        if c["limite"] is not None:
            L.append(f"Limite aceitável (0 erros e p95 <= {a.slo_p95:g} s): {ds[c['limite']]['rotulo']} "
                     f"(p95 {rs[c['limite']]['p95']:.2f} s)"
                     + ("" if c["contigua"] else "; atenção: algum degrau abaixo dele não cumpriu o critério")
                     + (f"; o degrau seguinte {ds[c['limite'] + 1]['rotulo']} já tem p95 {fmt(rs[c['limite'] + 1]['p95'])} s "
                        f"e erros {txt_erros(rs[c['limite'] + 1])}." if c["limite"] + 1 < len(rs) else "."))
        else:
            L.append(f"Limite aceitável não encontrado: nem o degrau mais leve ({ds[0]['rotulo']}) teve 0 erros e "
                     f"p95 <= {a.slo_p95:g} s (p95 {fmt(rs[0]['p95'])} s, erros {txt_erros(rs[0])}).")
        if c["saturacao"] is not None:
            i = c["saturacao"]
            L.append(f"Saturação em {ds[i]['rotulo']}: vazão {rs[i]['vazao']:.1f} resp/min contra "
                     f"{rs[i - 1]['vazao']:.1f} no degrau anterior (ganho < 10%).")
        else:
            L.append("Saturação não atingida: a vazão ainda crescia >= 10% a cada degrau.")
        ult = rs[-1]
        L.append(f"A latência p95 foi de {fmt(rs[0]['p95'])} s em {ds[0]['rotulo']} para {fmt(ult['p95'])} s em "
                 f"{ds[-1]['rotulo']}; a espera na fila (p50 menos a latência em série) foi de {fmt(ult['fila_p50'])} s "
                 f"no último degrau.")
        L.append(f"Colapso (erros > 5% ou p95 > 60 s) na rampa: "
                 + (f"{ds[c['colapso']]['rotulo']} (erros {txt_erros(rs[c['colapso']])}, p95 {fmt(rs[c['colapso']]['p95'])} s)."
                    if c["colapso"] is not None else "não atingido nos degraus testados."))
    if "aberto" in secoes:
        ds, rs, c = secoes["aberto"]
        bons = [d["rotulo"] for d, r in zip(ds, rs) if aceitavel(r, a.slo_p95)]
        L.append(f"Chegadas Poisson: aceitáveis até {bons[-1] if bons else 'nenhuma taxa testada'}; "
                 + (f"colapso em {ds[c['colapso']]['rotulo']} (vazão {rs[c['colapso']]['vazao']:.1f} resp/min contra "
                    f"{rs[c['colapso']]['oferecida'] or 0:.1f} oferecidas)." if c["colapso"] is not None
                    else "colapso não atingido."))
    if t.pico:
        p = t.pico
        L.append(f"Pico de {p['n']} simultâneas: "
                 + (f"a latência voltou a <= 1,5x a linha de base ({p['limiar']:.2f} s) {p['t_recuperou']:.0f} s após o "
                    f"disparo; o último pico terminou em {p['t_dren']:.0f} s." if p["t_recuperou"] is not None
                    else f"a latência NÃO voltou a <= {p['limiar']:.2f} s em {a.janela_recuperacao:g} s "
                         f"(o último pico terminou em {p['t_dren']:.0f} s)."))
    if t.soak:
        dv = deriva_soak(janelas_soak(t))
        L.append(f"Soak a {t.soak['taxa']:.2f} req/s: "
                 + (f"p50 variou {dv['inclinacao']:+.2f} s por minuto (último/primeiro minuto = {dv['razao']:.2f}x)."
                    if dv["inclinacao"] is not None and dv["razao"] else "janelas insuficientes para medir deriva."))
    rc = [r["recursos"] for ds, rs, _ in secoes.values() for r in rs]
    vram = max([x["amd_vram_pct"] for x in rc if x.get("amd_vram_pct") is not None], default=None)
    temp = max([x["temp_c"] for x in rc if x.get("temp_c") is not None], default=None)
    L.append("Recursos: " + (f"VRAM da AMD chegou a {vram:.1f}%; " if vram is not None else "VRAM da AMD não lida; ")
             + (f"temperatura máxima {temp:.0f} °C." if temp is not None else "temperatura não lida."))
    L.append("Execução " + (f"ABORTADA: {t.motivo_parada}." if t.motivo_parada else "concluída sem abortar."))
    extras = ["Cada carga foi medida uma vez (semente " + str(a.semente) + "); para ver variação, repita a execução.",
              "O script mede e não diagnostica: itens como OLLAMA_NUM_PARALLEL e OLLAMA_MAX_QUEUE são hipóteses a "
              "conferir no servidor (seção abaixo), não conclusões.",
              "Os números valem para este modelo, esta versão do Ollama e o hardware onde o script rodou."]
    while len(L) < 8 and extras:
        L.append(extras.pop(0))
    return L[:12]


def gerar_relatorio(t: Teste, caminho: Path) -> None:
    a = t.a
    secoes = secoes_por_cenario(t)
    dur = t.rel()
    L = [f"# Teste de sobrecarga — {t.inicio:%Y-%m-%d %H:%M}", "",
         f"**Resultado:** {'ABORTADO — ' + t.motivo_parada if t.motivo_parada else 'concluído'}"
         f" · duração {dur / 60:.1f} min · alvo `{t.alvo.nome}` · modelo `{a.modelo}`", ""]
    if t.erro_fatal:
        L += ["**Erro inesperado do script** (resultados parciais abaixo):", "", "```", t.erro_fatal.rstrip(), "```", ""]
    L += ["## Leitura", ""] + [f"{i}. {s}" for i, s in enumerate(gerar_leitura(t, secoes), 1)] + [""]

    L += ["## Parâmetros", "", "```", "python " + " ".join(shlex.quote(x) for x in sys.argv), "```", "",
          f"- Ollama: `{a.ollama_url}` · cenários: {', '.join(a.cenarios)} · semente {a.semente}",
          f"- Rampa (laço fechado): níveis {a.niveis}, {a.duracao:g} s por degrau · Aberto: taxas {a.taxas} req/s, "
          f"{a.duracao:g} s, teto de {a.max_em_voo} em voo",
          f"- Pico: {a.pico} simultâneas; sondas de {a.n_serie} em série a cada {a.intervalo_sonda:g} s por até "
          f"{a.janela_recuperacao:g} s · Soak: {a.soak_min:g} min",
          f"- Timeout do cliente {a.timeout_cliente:g} s (por chamada HTTP) · teto total {a.max_total_min:g} min · "
          f"abortar se VRAM AMD > {a.max_vram_pct:g}% ou GPU > {a.max_temp:g} °C",
          ("- Reranker (pipeline): " + os.environ.get("RERANKER_ATIVO", "0")) if a.alvo == "pipeline" else
          f"- Prompt fixo de {PROMPT_CHARS} caracteres, `num_predict` 1024, `num_ctx` 8192, `temperature` 0, `stream` falso; "
          f"o cabeçalho `[req N]` muda a cada requisição para o Ollama não reaproveitar o prefixo (cache de prompt).",
          "", "### Critérios", "",
          f"- **Limite aceitável** = maior carga com 0 erros e p95 <= `--slo-p95` ({a.slo_p95:g} s). Erro é qualquer "
          f"status diferente de `ok`/`truncada` (truncada = bateu em `num_predict`, não é erro de carga) mais as "
          f"chegadas descartadas pelo teto de requisições em voo.",
          "- **Saturação** = primeiro degrau cuja vazão cresce menos de 10% sobre o anterior.",
          "- **Colapso** = erros > 5% ou p95 > 60 s (ou nenhuma resposta válida). A rampa e o aberto param sozinhos "
          "após 2 degraus seguidos em colapso.",
          "- **Vazão** e **tokens/s** contam as respostas concluídas dentro da janela de carga; percentis, erros e "
          "% ok contam todas as requisições iniciadas no degrau (as que terminam na drenagem entram). Percentis só "
          "das respostas válidas. **Espera na fila** = p50/p95 do degrau menos o p50 da linha de base em série.", ""]

    if t.base:
        L += ["## Linha de base (série)", "",
              f"n={t.base['n']} · p50 {t.base['p50']:.2f} s · p95 {t.base['p95']:.2f} s · máx {t.base['max']:.2f} s · "
              f"{fmt(t.base['tok_s'], 1)} tokens/s", ""]
    for cen, titulo, alvo_txt in (("rampa", "Rampa (laço fechado)", "c = trabalhadores simultâneos"),
                                  ("aberto", "Aberto (chegadas Poisson)", "carga = taxa de chegada")):
        if cen in secoes:
            ds, rs, _ = secoes[cen]
            L += [f"## {titulo}", "", alvo_txt, ""] + tabelas_degraus(t, cen, rs, ds) + [""]

    if t.pico:
        p = t.pico
        L += ["## Pico e recuperação", "",
              f"Linha de base (média de {a.n_serie} em série) {p['base_ref']:.2f} s → limiar {p['limiar']:.2f} s (1,5x). "
              f"Cada sonda são {a.n_serie} requisições em série; vale a média delas (só a primeira espera atrás do "
              f"pico, então a mediana esconderia a fila) e todas precisam responder. Pico de {p['n']} "
              f"simultâneas; o último terminou {p['t_dren']:.0f} s após o disparo.", ""]
        r = resumir(p["degrau"], p["base_ref"], t.monitor)
        L += [f"Requisições do pico: n={r['n']}, ok {r['pct_ok']:.0f}%, erros {txt_erros(r)}, p50 {fmt(r['p50'])} s, "
              f"p95 {fmt(r['p95'])} s, máx {fmt(r['max'])} s.", ""]
        L += tabela(["sonda", "t após o disparo (s)", "média (s)", "ok/n", "voltou?"],
                    [[i, f"{s['t']:.0f}", fmt(s["media"]), f"{s['ok']}/{s['n']}", "sim" if s["volta"] else "não"]
                     for i, s in enumerate(p["sondas"], 1)]) + [""]
        L += [("**Recuperação:** a latência voltou a <= 1,5x a linha de base " + f"{p['t_recuperou']:.0f} s após o disparo."
               if p["t_recuperou"] is not None else
               f"**Recuperação:** não voltou em {a.janela_recuperacao:g} s após o disparo."), ""]
    if t.soak:
        js = janelas_soak(t)
        dv = deriva_soak(js)
        r = resumir(t.soak["degrau"], t.base_p50, t.monitor)
        L += ["## Soak", "",
              f"{t.soak['taxa']:.3f} req/s = 70% de {t.soak['vazao_ref']:.1f} resp/min (saturação de referência) por "
              f"{a.soak_min:g} min · n={r['n']} · ok {r['pct_ok']:.0f}% · erros {txt_erros(r)}", ""]
        L += tabela(["minuto", "n", "erros", "p50 (s)", "p95 (s)", "VRAM máx (MiB)"],
                    [[j["min"], j["n"], j["erros"], fmt(j["p50"]), fmt(j["p95"]), fmt(j["vram"], 0)] for j in js]) + [""]
        L += [f"**Deriva do p50:** " + (f"{dv['inclinacao']:+.3f} s/min (regressão linear sobre as janelas); último/primeiro "
              f"minuto = {dv['razao']:.2f}x." if dv["inclinacao"] is not None and dv["razao"] else "janelas insuficientes."), ""]

    L += ["## Critérios aplicados", ""]
    for cen, (ds, rs, c) in secoes.items():
        nome = lambda i: ds[i]["rotulo"] if i is not None else None
        L.append(f"- **{cen}** — limite aceitável: {nome(c['limite']) or 'nenhum'} · saturação: "
                 f"{nome(c['saturacao']) or 'não atingida'} · colapso: {nome(c['colapso']) or 'não atingido'}")
    if not secoes:
        L.append("- (rampa/aberto não executados)")
    L += ["", "## Sem leitura / observações", ""]
    amb = t.monitor.ausentes()
    L += [f"- {x}" for x in amb] or ["- todas as fontes de monitoramento foram lidas"]
    L += [f"- {n}" for n in t.notas]
    if t.motivo_parada:
        L.append(f"- execução interrompida: {t.motivo_parada}")
    L += ["", "## A verificar no servidor (não aplicado por este script)", "",
          "O script não altera variáveis nem o serviço do Ollama (compartilhado com outro app, porta 8509). Para "
          "interpretar a saturação, confira a configuração efetiva:", "",
          "```", 'sudo journalctl -u ollama | grep "server config"', "```", "",
          "- `OLLAMA_NUM_PARALLEL`: quantas requisições o Ollama processa ao mesmo tempo por modelo carregado; o "
          "valor efetivo (e se o KV cache cresce com ele, pois `num_ctx` é multiplicado por esse número) deve ser lido "
          "do log, não presumido.",
          "- `OLLAMA_MAX_QUEUE`: tamanho da fila de espera; passando dele o Ollama responde 503 (aparece aqui como "
          "`http_5xx`, com o código nas colunas de erro).",
          "- Também vale ver `OLLAMA_MAX_LOADED_MODELS` e `OLLAMA_KEEP_ALIVE`, porque o outro app usa o mesmo servidor.", ""]
    caminho.write_text("\n".join(L), encoding="utf-8")


# ── Plano, argumentos e main ───────────────────────────────────────────────

def lista(txt: str, tipo) -> list:
    return [tipo(x) for x in txt.split(",") if x.strip()]


def montar_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", choices=("pipeline", "ollama"), default="pipeline")
    ap.add_argument("--cenario", choices=("rampa", "aberto", "pico", "soak", "todos"), default="todos")
    ap.add_argument("--modelo", default=None, help="id no Ollama (padrão: MODELO_PADRAO do ambiente ou gemma3:4b)")
    ap.add_argument("--ollama-url", default=None, help="padrão: OLLAMA_URL do ambiente ou http://localhost:11434")
    ap.add_argument("--niveis", default=None, help="rampa: trabalhadores por degrau (padrão 1,2,4,8,16,32)")
    ap.add_argument("--duracao", type=float, default=None, help="segundos por degrau/taxa (padrão 60)")
    ap.add_argument("--taxas", default=None, help="aberto: req/s por degrau (padrão 0.2,0.5,1,2,4)")
    ap.add_argument("--max-em-voo", type=int, default=128, help="aberto/soak: teto de requisições em voo (padrão 128)")
    ap.add_argument("--pico", type=int, default=None, help="requisições simultâneas do pico (padrão 32)")
    ap.add_argument("--n-serie", type=int, default=5, help="requisições em série na linha de base e em cada sonda")
    ap.add_argument("--intervalo-sonda", type=float, default=10.0)
    ap.add_argument("--janela-recuperacao", type=float, default=None, help="segundos de sondas após o pico (padrão 120)")
    ap.add_argument("--soak-min", type=float, default=0.0, help="minutos do soak (0 = desligado)")
    ap.add_argument("--vazao-saturacao", type=float, default=None,
                    help="resp/min de saturação para o soak, se a rampa não rodar junto")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--timeout-cliente", type=float, default=120.0)
    ap.add_argument("--slo-p95", type=float, default=15.0)
    ap.add_argument("--max-total-min", type=float, default=45.0)
    ap.add_argument("--confirmo", action="store_true", help="exigido para concorrência > 8 ou taxa > 1 req/s")
    ap.add_argument("--permitir-64", action="store_true", help="libera níveis acima de 32 (ex.: 64)")
    ap.add_argument("--rapido", action="store_true",
                    help="ensaio: 30 s por degrau, níveis 1,2,4,8, taxas 0.5,1, pico de 8 e recuperação em 60 s")
    ap.add_argument("--max-vram-pct", type=float, default=97.0, help="aborta se a VRAM da AMD passar disto")
    ap.add_argument("--max-temp", type=float, default=95.0, help="aborta se qualquer GPU passar disto (°C)")
    ap.add_argument("--intervalo-monitor", type=float, default=2.0)
    ap.add_argument("--drm-raiz", default="/sys/class/drm", help="onde procurar o sysfs da AMD")
    ap.add_argument("--nvidia-smi", default="nvidia-smi", help="comando do nvidia-smi")
    ap.add_argument("--pausa", type=float, default=5.0, help="segundos entre degraus")
    ap.add_argument("--pausa-cenario", type=float, default=30.0, help="segundos entre cenários")
    ap.add_argument("--saida-dir", default=str(AQUI))
    return ap


def resolver(a) -> None:
    """Preenche os padrões (que dependem de --rapido) e valida; explícito na linha de comando sempre vence."""
    r = a.rapido
    a.niveis = lista(a.niveis, int) if a.niveis else ([1, 2, 4, 8] if r else [1, 2, 4, 8, 16, 32])
    a.taxas = lista(a.taxas, float) if a.taxas else ([0.5, 1.0] if r else [0.2, 0.5, 1.0, 2.0, 4.0])
    a.duracao = a.duracao if a.duracao is not None else (30.0 if r else 60.0)
    a.pico = a.pico if a.pico is not None else (8 if r else 32)
    a.janela_recuperacao = a.janela_recuperacao if a.janela_recuperacao is not None else (60.0 if r else 120.0)
    a.ollama_url = (a.ollama_url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    a.modelo = a.modelo or os.environ.get("MODELO_PADRAO", "").strip() or "gemma3:4b"
    a.cenarios = ["rampa", "aberto", "pico"] + (["soak"] if a.soak_min > 0 else []) if a.cenario == "todos" \
        else [a.cenario]
    if "soak" in a.cenarios and a.soak_min <= 0:
        raise SystemExit("--cenario soak exige --soak-min > 0")


def plano(a) -> tuple:
    """Linhas do plano de carga e (maior concorrência, maior taxa req/s, tempo nominal s, tempo máximo s)."""
    L, conc, taxa, nominal, maximo = [], 1, 0.0, 0.0, 0.0
    for cen in a.cenarios:
        if cen == "rampa":
            conc = max(conc, max(a.niveis))
            L.append(f"rampa : laço fechado, c = {a.niveis}, {a.duracao:g} s por degrau (≥ {sum(a.niveis)} "
                     f"requisições; o total depende da vazão medida)")
            nominal += len(a.niveis) * (a.duracao + a.pausa)
            maximo += len(a.niveis) * (a.duracao + a.timeout_cliente + a.pausa)
        elif cen == "aberto":
            taxa = max(taxa, max(a.taxas))
            L.append(f"aberto: chegadas Poisson {a.taxas} req/s, {a.duracao:g} s cada "
                     f"(≈ {sum(t * a.duracao for t in a.taxas):.0f} requisições; teto de {a.max_em_voo} em voo)")
            nominal += len(a.taxas) * (a.duracao + a.pausa)
            maximo += len(a.taxas) * (a.duracao + a.timeout_cliente + a.pausa)
        elif cen == "pico":
            conc = max(conc, a.pico)
            L.append(f"pico  : {a.pico} simultâneas + sondas de {a.n_serie} em série a cada {a.intervalo_sonda:g} s por até "
                     f"{a.janela_recuperacao:g} s")
            nominal += a.janela_recuperacao + 30
            maximo += a.janela_recuperacao + a.timeout_cliente + 30 + a.n_serie * a.timeout_cliente
        elif cen == "soak":
            if a.vazao_saturacao:
                taxa = max(taxa, 0.7 * a.vazao_saturacao / 60)
            L.append(f"soak  : 70% da vazão de saturação por {a.soak_min:g} min")
            nominal += a.soak_min * 60
            maximo += a.soak_min * 60 + a.timeout_cliente
    n = len(a.cenarios)
    nominal += (n - 1) * a.pausa_cenario
    maximo += (n - 1) * a.pausa_cenario
    return L, conc, taxa, nominal, maximo


def instalar_sinais(t: Teste) -> None:
    def tratador(signum, frame):
        if t.parar.is_set():
            raise KeyboardInterrupt                      # segundo sinal: sai da espera e vai direto gravar
        t.abortar(f"interrompido pelo usuário (sinal {signum})")

    for nome in ("SIGINT", "SIGTERM", "SIGBREAK"):
        s = getattr(signal, nome, None)
        if s is not None:
            try:
                signal.signal(s, tratador)
            except (ValueError, OSError):
                pass


def main() -> int:
    global _OUT, _ERR
    sys.stdout.reconfigure(encoding="utf-8")
    _OUT, _ERR = sys.stdout, sys.stderr
    a = montar_parser().parse_args()
    resolver(a)

    L, conc, taxa, nominal, maximo = plano(a)
    log(f"== Plano de carga | alvo={a.alvo} modelo={a.modelo} ollama={a.ollama_url} ==")
    for l in L:
        log("  " + l)
    log(f"  maior concorrência {conc} | maior taxa {taxa:g} req/s | tempo nominal ≈ {nominal / 60:.1f} min, "
        f"máximo teórico ≈ {maximo / 60:.1f} min (teto --max-total-min {a.max_total_min:g})")
    if any(n > 32 for n in a.niveis) and not a.permitir_64:
        log("RECUSADO: níveis acima de 32 (ex.: 64) exigem --permitir-64")
        return 2
    if (conc > 8 or taxa > 1.0) and not a.confirmo:
        log("RECUSADO: concorrência > 8 ou taxa > 1 req/s exigem --confirmo (o Ollama do servidor é compartilhado "
            "com outro app). Confira o plano acima e rode de novo com --confirmo.")
        return 2

    if a.alvo == "pipeline":
        if a.ollama_url != os.environ.get("OLLAMA_URL", "").rstrip("/"):
            os.environ["OLLAMA_URL"] = a.ollama_url      # antes de importar nucleo (lido na importação)
        if "," in os.environ.get("OLLAMA_URLS", ""):
            log("AVISO: OLLAMA_URLS tem várias instâncias; o monitor lê só " + a.ollama_url)
        import nucleo
        import avaliar_retrieval as ar
        perguntas = {q: g["pergunta"] for q, g in ar.carregar_gabarito().items()}
        alvo = AlvoPipeline(nucleo, a.modelo, a.timeout_cliente)
    else:
        perguntas, alvo = {}, AlvoOllama(a.ollama_url, a.modelo, a.timeout_cliente)

    carimbo = datetime.now().strftime("%Y%m%d_%H%M")
    saida = Path(a.saida_dir)
    saida.mkdir(parents=True, exist_ok=True)
    c_csv, c_tl, c_md = (saida / f"resultado_sobrecarga_{carimbo}.csv", saida / f"linha_do_tempo_{carimbo}.csv",
                         saida / f"resultado_sobrecarga_{carimbo}.md")
    t = Teste(a, alvo, perguntas, c_csv, c_tl)
    instalar_sinais(t)
    t.monitor.iniciar()
    silencio = a.alvo == "pipeline"
    if silencio:
        if hasattr(alvo.nucleo, "preaquecer"):
            alvo.nucleo.preaquecer()
        sys.stdout = sys.stderr = _Nulo()
    try:
        try:
            if t.aquecer_e_medir_base(com_base=a.cenarios != ["pico"]):
                for i, cen in enumerate(a.cenarios):
                    if t.parar.is_set():
                        break
                    if i and t.esperar(a.pausa_cenario):
                        break
                    {"rampa": t.rodar_rampa, "aberto": t.rodar_aberto, "pico": t.rodar_pico,
                     "soak": t.rodar_soak}[cen]()
        except KeyboardInterrupt:
            t.abortar("interrompido pelo usuário (segundo Ctrl+C)")
        except Exception:
            t.erro_fatal = traceback.format_exc()
            t.abortar("erro inesperado do script (ver relatório)")
    finally:
        if silencio:
            sys.stdout, sys.stderr = _OUT, _ERR
        t.monitor.parar()
        t.arq.close()
        gerar_relatorio(t, c_md)
    log(f"\nsalvo:\n  {c_md}\n  {c_csv}\n  {c_tl}")
    if t.erro_fatal:
        return 1
    if t.motivo_parada:
        return 130 if "usuário" in t.motivo_parada else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
