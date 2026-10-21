"""
nucleo.py — Lógica compartilhada entre as páginas do IF Turing.
Backend puro: constantes, MODELOS, pipeline RAG, Ollama, persistência.
Sem layout ou CSS do Streamlit.
"""
import os
import re
import threading
import time
import unicodedata
import traceback
import psutil
import requests
import streamlit as st
from pathlib import Path
from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from dotenv import load_dotenv
from db import upsert_modelo, inserir_consulta

load_dotenv()

PASTA_CHROMA   = os.getenv("PASTA_CHROMA",  "./chroma_data")
OLLAMA_URL     = os.getenv("OLLAMA_URL",    "http://localhost:11434")
# Várias instâncias do Ollama (ex.: uma por GPU), separadas por vírgula. Ausente → só OLLAMA_URL (comportamento
# de sempre). Com a lista, OLLAMA_URL passa a ser a primeira (usada por quem lê o catálogo de modelos).
OLLAMA_URLS    = [u.strip().rstrip("/") for u in os.getenv("OLLAMA_URLS", "").split(",") if u.strip()] or [OLLAMA_URL]
OLLAMA_URL     = OLLAMA_URLS[0]
MODELO_OLLAMA  = os.getenv("MODELO_OLLAMA", "llama3:latest")
MARITACA_URL   = "https://chat.maritaca.ai/api/chat/completions"
MODELO_EMB      = "intfloat/multilingual-e5-base"
TOP_K           = 10
EMBEDDER_DEVICE = os.getenv("EMBEDDER_DEVICE", "cpu")

# Detecta chunks que descrevem distribuição de cotas (ex: "C1:", "C10:")
_PADRAO_COTA = re.compile(r"C\d{1,2}:")

# Injeção do cronograma (perguntas de data/prazo). Os chunks são os de meta tipo="cronograma"
# gravados por indexar.py (nenhuma heurística de texto/data):
#   "todos"  injeta o cronograma inteiro (todos os chunks tipo="cronograma")
#   "topk"   injeta só os CRONOGRAMA_TOP_K mais similares à pergunta (ranking semântico dentro do tipo)
CRONOGRAMA_MODO  = os.getenv("CRONOGRAMA_MODO", "todos").strip().lower()
CRONOGRAMA_TOP_K = 3

def _flag(nome: str, padrao: str = "") -> bool:
    return os.getenv(nome, padrao).strip().lower() not in ("", "0", "false", "no", "nao", "não")


# Prefixos do E5 ("query: " na consulta; o indexador usa "passage: " nos chunks). Só têm efeito se o ÍNDICE
# tiver sido construído com o mesmo E5_PREFIXOS (ver indexar.py); desligado por padrão.
E5_PREFIXOS     = _flag("E5_PREFIXOS")
PREFIXO_CONSULTA = "query: " if E5_PREFIXOS else ""

# Reranker (cross-encoder) OPCIONAL, desligado por padrão. Com RERANKER_ATIVO=1: busca semântica de
# RERANKER_CANDIDATOS chunks → reordena pelo score do CrossEncoder → fica com TOP_K → injeção híbrida de sempre
# (mesmo fluxo do modo hib-todos de avaliar_rerank.py). SEMPRE fp32 (a T1000 tem fp16 a ~0,36 TFLOPS contra
# ~1,53 em fp32) e nunca trust_remote_code. Se falhar (sem CUDA, sem memória…), segue com a ordem semântica.
RERANKER_ATIVO      = _flag("RERANKER_ATIVO", "0")
RERANKER_MODELO     = os.getenv("RERANKER_MODELO", "BAAI/bge-reranker-v2-m3").strip()
RERANKER_CANDIDATOS = max(TOP_K, int(os.getenv("RERANKER_CANDIDATOS", "15") or 15))
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "256") or 256)
RERANKER_DEVICE     = os.getenv("RERANKER_DEVICE", "cpu").strip() or "cpu"

# Gatilho do cronograma (perguntas de data/prazo):
#   "palavra"    casamento por PALAVRA INTEIRA, sem acento (padrão): termos fortes sozinhos; termos ambíguos
#                ("quando", "período", "etapa") só junto de um evento do processo (inscrição, prova, recurso…)
#   "substring"  comportamento antigo (fase 2C): qualquer _PALAVRAS_CRONOGRAMA dentro do texto — "data" dispara
#                em "candidata", "inscrição" em "taxa de inscrição", "período" em "período noturno"
CRONOGRAMA_GATILHO = os.getenv("CRONOGRAMA_GATILHO", "palavra").strip().lower()

# Palavras do gatilho antigo (modo "substring")
_PALAVRAS_CRONOGRAMA = frozenset({
    "data", "datas", "cronograma", "prazo", "prazos",
    "inscrição", "inscricao", "quando", "período", "periodo",
    "calendário", "calendario", "etapa", "etapas",
})

_RE_CRONO_FORTE = re.compile(r"\b(?:cronograma|calendario|datas?|prazos?)\b")
_RE_CRONO_AMBIGUO = re.compile(r"\b(?:quando|periodos?|etapas?)\b")
_RE_EVENTO_PROCESSO = re.compile(
    r"\b(?:inscri\w*|provas?|resultados?|classifica\w*|matriculas?|pre-matriculas?|recursos?|chamadas?|"
    r"sorteios?|divulga\w*|publica\w*|gabaritos?|isencao|isencoes|convoca\w*|homologa\w*|"
    r"processo\s+seletivo|locais\s+de\s+prova)\b"
)


def _sem_acento(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def dispara_cronograma(pergunta: str, modo: str = None) -> bool:
    """A pergunta é de data/prazo (injeta o cronograma)? `modo` = CRONOGRAMA_GATILHO ('palavra'|'substring')."""
    modo = (modo or CRONOGRAMA_GATILHO)
    if modo == "substring":
        pl = pergunta.lower()
        return any(kw in pl for kw in _PALAVRAS_CRONOGRAMA)
    if modo != "palavra":
        raise ValueError(f"CRONOGRAMA_GATILHO inválido: {modo!r} (use 'palavra' ou 'substring')")
    t = _sem_acento(pergunta)
    return bool(_RE_CRONO_FORTE.search(t) or (_RE_CRONO_AMBIGUO.search(t) and _RE_EVENTO_PROCESSO.search(t)))

# Palavras que disparam a lógica de retrieval híbrido de tabelas
_PALAVRAS_TABELA = frozenset({
    "curso", "cursos", "técnico", "técnicos", "tecnologia", "tecnologias",
    "licenciatura", "licenciaturas", "superior", "superiores", "graduação",
    "bacharelado", "bacharelados",
    "vagas", "vaga", "turno", "turnos", "duração", "semestres",
    "integrado", "subsequente", "concomitante",
    "oferta", "disponível", "disponíveis",
    "agropecuária", "informática", "administração", "matemática",
    "computação", "ciência", "química", "agronomia", "engenharia",
})

# "Listar todos os cursos do campus": sem tipo de curso nem nome específico na pergunta, injeta as tabelas
# inteiras (custo ≈ 2 mil caracteres por 9 tabelas; ver auditoria_indexacao.md, Fase 3).
TABELAS_LISTAR_TODAS = _flag("TABELAS_LISTAR_TODAS", "1")
_RE_LISTAR_CURSOS = re.compile(r"\b(?:liste|listar|lista|todos|todas|quais)\b")
_RE_CURSOS = re.compile(r"\bcursos?\b")

_STOP_HIBRIDO = frozenset({
    "quais", "qual", "são", "está", "estão", "para", "com", "dos", "das",
    "nos", "nas", "uma", "como", "pode", "podem", "esse", "essa", "pelo",
    "pela", "sobre", "mais", "onde", "quando", "também", "ainda", "muito",
    "campus", "ibirubá",
})

# Termos genéricos que aparecem em qualquer pergunta sobre cursos mas não identificam um
# chunk específico (ex: "cursos" aparece em todos os chunks como "Curso: …")
_TERMOS_GENERICOS = frozenset({
    "curso", "cursos", "vaga", "vagas", "turno", "turnos",
    "duração", "semestre", "semestres", "total",
    "oferta", "oferece", "oferecer", "oferecidos", "oferecidas",
    "disponível", "disponíveis", "quantas", "quantos",
})

# Mapeamento de termos da pergunta → tipos internos de chunk de tabela que devem ser incluídos.
# Permite que "superiores"/"graduação" encontrem chunks com "Bacharelado em" ou "Licenciatura em".
_SINONIMOS_TIPO: dict = {
    "técnico":       frozenset({"tecnico"}),
    "técnicos":      frozenset({"tecnico"}),
    "superior":      frozenset({"licenciatura", "bacharelado", "superior"}),
    "superiores":    frozenset({"licenciatura", "bacharelado", "superior"}),
    "graduação":     frozenset({"licenciatura", "bacharelado", "superior"}),
    "graduacao":     frozenset({"licenciatura", "bacharelado", "superior"}),
    "licenciatura":  frozenset({"licenciatura"}),
    "licenciaturas": frozenset({"licenciatura"}),
    "bacharelado":   frozenset({"bacharelado", "superior"}),
    "bacharelados":  frozenset({"bacharelado", "superior"}),
    "tecnologia":    frozenset({"tecnologia"}),
    "tecnologias":   frozenset({"tecnologia"}),
    "integrado":     frozenset({"integrado"}),
    "integrados":    frozenset({"integrado"}),
    "subsequente":   frozenset({"subsequente"}),
    "subsequentes":  frozenset({"subsequente"}),
    "concomitante":  frozenset({"concomitante"}),
    "concomitantes": frozenset({"concomitante"}),
}


def selecionar_cronograma(todos: list) -> list:
    """Chunks do cronograma = os de meta tipo="cronograma" (ordem de gravação = ordem do documento)."""
    return [d for d in todos if (d.meta or {}).get("tipo") == "cronograma"]


def cronograma_mais_similares(pipeline, embedding: list, k: int = CRONOGRAMA_TOP_K) -> list:
    """Os k chunks tipo="cronograma" mais similares à consulta, do mais para o menos similar.

    Mesma busca vetorial do retriever do pipeline (mesmo índice e métrica), restrita ao tipo.
    """
    retriever = pipeline.get_component("retriever")
    return retriever.run(
        query_embedding=embedding,
        filters={"field": "meta.tipo", "operator": "==", "value": "cronograma"},
        top_k=k,
    )["documents"]


def _buscar_cronograma_por_keyword(
    pergunta: str,
    cronograma_docs: list,
    docs_semanticos: list,
    ranquear=None,
    modo: str = None,
    gatilho: str = None,
) -> list:
    """Injeta chunks de cronograma (meta tipo="cronograma") quando a pergunta menciona datas/prazos.

    Garante que o cronograma do processo seletivo entre no contexto mesmo quando o
    score semântico não é suficiente para colocá-lo no top-K. Chunks já presentes no
    contexto semântico não são repetidos.

    modo (padrão CRONOGRAMA_MODO):
      "todos"  todos os chunks de cronograma, na ordem do documento
      "topk"   só os CRONOGRAMA_TOP_K mais similares à pergunta, do mais para o menos similar;
               exige `ranquear`, função k -> chunks ordenados por similaridade (ver
               cronograma_mais_similares), chamada só quando a pergunta dispara o cronograma
    gatilho (padrão CRONOGRAMA_GATILHO): ver dispara_cronograma().
    """
    if not cronograma_docs:
        return []
    if not dispara_cronograma(pergunta, gatilho):
        if _DEBUG_HIBRIDO:
            print("[debug_hibrido] pergunta não dispara o cronograma → skip")
        return []
    modo = modo or CRONOGRAMA_MODO
    if modo == "topk":
        if ranquear is None:
            raise ValueError("CRONOGRAMA_MODO='topk' exige `ranquear` (ranking semântico dentro do tipo)")
        candidatos = ranquear(CRONOGRAMA_TOP_K)
    elif modo == "todos":
        candidatos = cronograma_docs
    else:
        raise ValueError(f"CRONOGRAMA_MODO inválido: {modo!r} (use 'todos' ou 'topk')")
    ids_semanticos = {d.id for d in docs_semanticos if d.id}
    extras = [d for d in candidatos if d.id not in ids_semanticos]
    if _DEBUG_HIBRIDO:
        print(f"[debug_hibrido] cronograma ({modo}): {len(extras)} chunks injetados")
    return extras


def _tipo_tabela_chunk(content: str) -> str:
    """Infere o tipo de curso a partir do conteúdo do chunk de tabela.

    Usa prefixos canônicos do IFRS: 'Técnico em', 'Licenciatura em',
    'Bacharelado em', 'Tecnologia em'. Demais cursos são tratados como
    superiores sem prefixo explícito (bacharelado implícito).
    """
    lower = content.lower()
    if "técnico em" in lower:
        return "tecnico"
    if "licenciatura em" in lower:
        return "licenciatura"
    if "bacharelado em" in lower:
        return "bacharelado"
    if "tecnologia em" in lower:
        return "tecnologia"
    return "superior"


_DEBUG_HIBRIDO   = os.getenv("DEBUG_HIBRIDO",   "").strip() not in ("", "0", "false", "no")
_DEBUG_CONTEXTO  = os.getenv("DEBUG_CONTEXTO",  "").strip() not in ("", "0", "false", "no")


def _buscar_tabelas_por_keyword(
    pergunta: str,
    tabela_docs: list,
    docs_semanticos: list,
    listar_todas: bool = None,
) -> list:
    """Retorna chunks de tabela relevantes à pergunta por tipo-de-curso e por nome.

    Estratégia dual:
    1. Tipo-de-curso: termos em _SINONIMOS_TIPO (ex: 'superiores') são expandidos para
       os tipos reais usados nos chunks ('bacharelado', 'licenciatura', 'superior'),
       e TODOS os chunks desses tipos são incluídos.
    2. Nome-específico: demais termos não-genéricos são buscados literalmente no
       conteúdo do chunk, com variante singular/plural simples (±s final).

    3. Listar todos (TABELAS_LISTAR_TODAS): se a pergunta pede lista/"todos"/"quais" cursos e os critérios
       1 e 2 não acharam nada (sem tipo de curso nem nome específico), injeta TODAS as tabelas.

    Só dispara se a pergunta contiver alguma palavra de _PALAVRAS_TABELA.
    Remove documentos já presentes em docs_semanticos (dedup por id).
    Ative DEBUG_HIBRIDO=1 para ver trace detalhado do matching.
    """
    if not tabela_docs:
        return []
    pergunta_lower = pergunta.lower()
    if not any(kw in pergunta_lower for kw in _PALAVRAS_TABELA):
        if _DEBUG_HIBRIDO:
            print("[debug_hibrido] pergunta não contém palavras de _PALAVRAS_TABELA → skip")
        return []

    termos_pergunta = {
        w.strip("?!.,;:\"'()[]").lower()
        for w in pergunta.split()
        if len(w) > 3
    } - _STOP_HIBRIDO

    # Critério 1: tipos de curso requisitados (expansão semântica de categoria)
    tipos_solicitados: set = set()
    for t in termos_pergunta:
        if t in _SINONIMOS_TIPO:
            tipos_solicitados |= _SINONIMOS_TIPO[t]

    # Critério 2: termos específicos de nome de curso, com variante ±s
    termos_match: set = set()
    for t in termos_pergunta:
        if t in _SINONIMOS_TIPO or t in _TERMOS_GENERICOS:
            continue
        termos_match.add(t)
        if t.endswith("s") and len(t) > 4:
            termos_match.add(t[:-1])   # "técnicos" → "técnico" (fallback genérico)
        else:
            termos_match.add(t + "s")  # "informática" → "informáticas"

    if _DEBUG_HIBRIDO:
        print(f"[debug_hibrido] termos_pergunta  = {sorted(termos_pergunta)}")
        print(f"[debug_hibrido] tipos_solicitados= {sorted(tipos_solicitados)}")
        print(f"[debug_hibrido] termos_match     = {sorted(termos_match)}")
        print(f"[debug_hibrido] total tabela_docs = {len(tabela_docs)}")

    ids_semanticos = {d.id for d in docs_semanticos if d.id}
    resultado = []
    for doc in tabela_docs:
        content = doc.content or ""
        tipo = _tipo_tabela_chunk(content)
        ja_semantico = doc.id in ids_semanticos
        bate_tipo = bool(tipos_solicitados and tipo in tipos_solicitados)
        bate_nome = bool(termos_match and any(t in content.lower() for t in termos_match))

        if _DEBUG_HIBRIDO:
            print(f"[debug_hibrido]   id={doc.id!r:<38} tipo={tipo:<12} "
                  f"semantico={ja_semantico}  bate_tipo={bate_tipo}  bate_nome={bate_nome}"
                  f"\n                  conteudo={content[:80]!r}")

        if ja_semantico:
            continue
        if bate_tipo or bate_nome:
            resultado.append(doc)

    if listar_todas is None:
        listar_todas = TABELAS_LISTAR_TODAS
    if listar_todas and not tipos_solicitados and not resultado:
        t = _sem_acento(pergunta)
        if _RE_CURSOS.search(t) and _RE_LISTAR_CURSOS.search(t):
            resultado = [d for d in tabela_docs if d.id not in ids_semanticos]
            if _DEBUG_HIBRIDO:
                print(f"[debug_hibrido] listar todos os cursos → {len(resultado)} tabelas injetadas")

    return resultado[:20]

MODELOS = {
    "🖥️ LLaMA 3 (Ollama local)": {
        "tipo": "ollama", "modelo": MODELO_OLLAMA,
        "descricao": "Local no servidor", "params_b": 8.0, "quantizacao": "Q4_0",
    },
    "🖥️ Mistral 7B (Ollama local)": {
        "tipo": "ollama", "modelo": "mistral",
        "descricao": "Local no servidor", "params_b": 7.2, "quantizacao": "Q4_0",
    },
    "🖥️ Gemma 2 27B (Ollama local)": {
        "tipo": "ollama", "modelo": "gemma2:27b",
        "descricao": "Local no servidor", "params_b": 27.2, "quantizacao": "Q4_0",
    },
    "🖥️ Qwen3 0.6B (Ollama local)": {
        "tipo": "ollama", "modelo": "qwen3:0.6b",
        "descricao": "Local no servidor", "params_b": 0.6, "quantizacao": "Q4_0",
    },
    "🖥️ Gemma 3 4B (Ollama local)": {
        "tipo": "ollama", "modelo": "gemma3:4b",
        "descricao": "Local no servidor", "params_b": 4.3, "quantizacao": "Q4_K_M",
    },
    "🌐 Sabiá-4 (Maritaca)": {
        "tipo": "maritaca", "modelo": "sabia-4",
        "descricao": "API Maritaca (nuvem)", "params_b": 0.0, "quantizacao": "—",
    },
}

# Modelo padrão do app: gemma3:4b, sobrescrevível pela variável de ambiente MODELO_PADRAO (id do Ollama, ex.:
# MODELO_PADRAO=qwen3:0.6b). Se ele não estiver instalado no Ollama, o app usa o primeiro modelo da lista
# (ver indice_modelo_padrao / chave_modelo_padrao).
_MODELO_PADRAO_ID = os.getenv("MODELO_PADRAO", "").strip() or "gemma3:4b"


def _mesmo_modelo(a: str, b: str) -> bool:
    """Mesmo modelo do Ollama; "gemma3" equivale a "gemma3:latest"."""
    if a == b:
        return True
    return (":" not in b and a == f"{b}:latest") or (":" not in a and b == f"{a}:latest")


def _chave_por_id(modelos: dict, modelo_id: str) -> str:
    return next((k for k, v in modelos.items() if _mesmo_modelo(v.get("modelo", ""), modelo_id)), "")


if not _chave_por_id(MODELOS, _MODELO_PADRAO_ID):      # id da variável que não está na lista fixa
    MODELOS[f"🖥️ {_MODELO_PADRAO_ID} (Ollama local)"] = {
        "tipo": "ollama", "modelo": _MODELO_PADRAO_ID,
        "descricao": "Local no servidor", "params_b": 0.0, "quantizacao": "?",
    }

MODELO_PADRAO_SIMPLES = _chave_por_id(MODELOS, _MODELO_PADRAO_ID)    # chave em MODELOS (scripts de linha de comando)


def indice_modelo_padrao(modelos: dict, padrao_id: str = None) -> int:
    """Índice do modelo padrão em `modelos` (para st.selectbox(index=...)). Se o padrão não estiver na lista
    (não instalado no Ollama), 0 = primeiro modelo disponível."""
    chave = _chave_por_id(modelos, padrao_id or _MODELO_PADRAO_ID)
    return list(modelos).index(chave) if chave else 0


def chave_modelo_padrao(modelos: dict, padrao_id: str = None) -> str:
    """Chave do modelo padrão em `modelos`; se não estiver na lista, a do primeiro modelo disponível."""
    return list(modelos)[indice_modelo_padrao(modelos, padrao_id)]


@st.cache_data(ttl=60)
def listar_modelos_ollama() -> dict:
    """Consulta GET /api/tags no Ollama e retorna dict no mesmo formato de MODELOS.
    Sempre inclui Maritaca ao final. Fallback para MODELOS fixo se o Ollama estiver inacessível."""
    _maritaca_entry = {"🌐 Sabiá-4 (Maritaca)": MODELOS["🌐 Sabiá-4 (Maritaca)"]}
    try:
        for _i, _url in enumerate(OLLAMA_URLS):          # catálogo da primeira instância que responder
            try:
                r = requests.get(f"{_url}/api/tags", timeout=5)
                r.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if _i == len(OLLAMA_URLS) - 1:
                    raise
        itens = r.json().get("models", [])
        if not itens:
            return {**MODELOS, **_maritaca_entry}
        resultado = {}
        for m in itens:
            nome = m["name"]
            details = m.get("details", {})
            params_str = details.get("parameter_size", "0B")
            try:
                params_b = float(params_str.upper().rstrip("B"))
            except (ValueError, AttributeError):
                params_b = 0.0
            quant = details.get("quantization_level", "?")
            resultado[f"🖥️ {nome} (Ollama local)"] = {
                "tipo": "ollama",
                "modelo": nome,
                "descricao": "Local no servidor",
                "params_b": params_b,
                "quantizacao": quant,
            }
        resultado.update(_maritaca_entry)
        return resultado
    except Exception:
        return {**MODELOS, **_maritaca_entry}


def registrar_modelos() -> None:
    for nome, cfg in MODELOS.items():
        upsert_modelo(
            modelo_id=cfg["modelo"],
            nome_bonito=nome,
            tipo=cfg.get("tipo", "ollama"),
            params_b=cfg.get("params_b"),
            quantizacao=cfg.get("quantizacao"),
        )


@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def carregar_pipeline():
    chroma_path = Path(PASTA_CHROMA)
    if not chroma_path.exists():
        return None, 0, []
    document_store = ChromaDocumentStore(collection_name="ifturing", persist_path=str(chroma_path))
    n_chunks = document_store.count_documents()
    if n_chunks == 0:
        return None, 0, []

    # Pré-carrega chunks de tabela e cronograma para retrieval híbrido (feito uma vez, cacheado)
    try:
        todos = document_store.filter_documents()
        tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
        cronograma_docs = selecionar_cronograma(todos)
    except Exception as _e:
        print(f"[setup] aviso ao pré-carregar docs: {_e}")
        tabela_docs = []
        cronograma_docs = []
    print(f"[setup] {len(tabela_docs)} chunks de tabela pré-carregados para retrieval híbrido")
    print(f"[setup] {len(cronograma_docs)} chunks de cronograma pré-carregados para retrieval híbrido")

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(
            model=MODELO_EMB,
            device=ComponentDevice.from_str(EMBEDDER_DEVICE),
            prefix=PREFIXO_CONSULTA,
        ),
    )
    pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=TOP_K))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    return pipeline, n_chunks, tabela_docs, cronograma_docs


# ── Reranker (opcional) ────────────────────────────────────────────────────
_reranker_modelo = None            # singleton: CrossEncoder já em fp32
_reranker_falhou = False           # falha ao carregar: não tenta de novo a cada pergunta
_reranker_lock_carga = threading.Lock()
_reranker_lock_uso = threading.Lock()      # um predict por vez (tokenizer/GPU única)


def _reranker():
    """CrossEncoder único, carregado na 1ª chamada; None se desligado ou se não carregou (aviso no log)."""
    global _reranker_modelo, _reranker_falhou
    if _reranker_modelo is not None or _reranker_falhou or not RERANKER_ATIVO:
        return _reranker_modelo
    with _reranker_lock_carga:
        if _reranker_modelo is not None or _reranker_falhou:
            return _reranker_modelo
        try:
            from sentence_transformers import CrossEncoder
            t0 = time.perf_counter()
            rr = CrossEncoder(RERANKER_MODELO, max_length=RERANKER_MAX_LENGTH,
                              device=RERANKER_DEVICE, trust_remote_code=False)
            rr.model.float()                       # fp32 sempre, qualquer que seja o dtype do checkpoint
            _reranker_modelo = rr
            print(f"[reranker] {RERANKER_MODELO} carregado em {RERANKER_DEVICE} (fp32, "
                  f"max_length={RERANKER_MAX_LENGTH}) em {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            _reranker_falhou = True
            print(f"[reranker] AVISO: não carregou ({type(exc).__name__}: {str(exc)[:200]}) — "
                  f"seguindo com a ordem semântica")
    return _reranker_modelo


def reordenar_por_reranker(pergunta: str, docs: list, top_k: int = TOP_K) -> tuple:
    """(docs, segundos, aplicado): reordena `docs` pelo score do CrossEncoder e fica com `top_k`.
    Empate → ordem semântica. Qualquer falha devolve os `top_k` primeiros na ordem semântica."""
    rr = _reranker()
    if rr is None or len(docs) <= 1:
        return docs[:top_k], 0.0, False
    t0 = time.perf_counter()
    try:
        with _reranker_lock_uso:
            scores = rr.predict([(pergunta, d.content or "") for d in docs], batch_size=16,
                                show_progress_bar=False)
        ordem = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), i))
        return [docs[i] for i in ordem][:top_k], time.perf_counter() - t0, True
    except Exception as exc:
        print(f"[reranker] AVISO: falhou ao pontuar ({type(exc).__name__}: {str(exc)[:200]}) — "
              f"seguindo com a ordem semântica")
        return docs[:top_k], time.perf_counter() - t0, False


def preaquecer(pipeline=None) -> dict:
    """Carrega embedder e (se ativo) reranker de uma vez, para a 1ª pergunta não pagar a carga (~5 s o embedder).
    `pipeline`: o pipeline que vai responder (scripts que montam o próprio); None → o de carregar_pipeline().
    Idempotente. Devolve os segundos gastos em cada parte."""
    t0 = time.perf_counter()
    if pipeline is None:
        pipeline = carregar_pipeline()[0]
    if pipeline is not None:
        pipeline.get_component("embedder").run(text="aquecimento")     # warm_up + 1ª inferência
    t_emb = time.perf_counter() - t0
    t1 = time.perf_counter()
    rr = _reranker()
    aquecido = rr is not None
    if rr is not None:
        try:
            with _reranker_lock_uso:
                rr.predict([("aquecimento", "texto de aquecimento")], batch_size=1, show_progress_bar=False)
        except Exception as exc:          # ex.: CUDA sem memória: o app sobe e cada pergunta cai na ordem semântica
            aquecido = False
            print(f"[reranker] AVISO: falhou no aquecimento ({type(exc).__name__}: {str(exc)[:200]})")
    t_rr = time.perf_counter() - t1
    estado = "desligado" if not RERANKER_ATIVO else ("ok" if aquecido else "indisponivel")
    print(f"[preaquecer] embedder {t_emb:.1f}s | reranker "
          f"{estado if estado != 'ok' else f'{t_rr:.1f}s'}")
    return {"embedder_s": round(t_emb, 2), "reranker_s": round(t_rr, 2), "reranker": estado}


# ── Ollama: escolha da instância ───────────────────────────────────────────
_ollama_ativas = {u: 0 for u in OLLAMA_URLS}      # requisições em andamento por instância
_ollama_fora_ate = {}                              # url → instante (monotonic) até o qual está "fora"
_ollama_lock = threading.Lock()
_OLLAMA_QUARENTENA_S = 30.0                        # após uma falha, a instância é preterida por este tempo


def _ordem_ollama() -> list:
    """Instâncias da menos para a mais ocupada (empate → ordem da lista); as que falharam há pouco vão por último."""
    agora = time.monotonic()
    with _ollama_lock:
        return sorted(OLLAMA_URLS, key=lambda u: (_ollama_fora_ate.get(u, 0) > agora, _ollama_ativas[u],
                                                  OLLAMA_URLS.index(u)))


class _OllamaOffline(Exception):
    pass


class _MaritacaError(Exception):
    pass


_MSG_OLLAMA_OFFLINE = (
    "⚠️ Estou com instabilidade no momento. Por favor, tente novamente "
    "em alguns instantes ou entre em contato pelo site oficial do processo seletivo."
)


def chamar_ollama(prompt: str, modelo: str) -> dict:
    print(f"[timing] ollama_prompt_chars: {len(prompt)}")
    # think:false desativa o modo de raciocínio do qwen3 (que consome tokens em <think>
    # sem emitir resposta, deixando o campo "response" vazio)
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 1024, "num_ctx": 8192},
    }
    t_http_inicio = time.perf_counter()
    r, url_usada, ultimo_erro = None, None, None
    for url in _ordem_ollama():                      # failover: se uma instância falha, tenta a próxima
        with _ollama_lock:
            _ollama_ativas[url] += 1
        try:
            r = requests.post(f"{url}/api/generate", json=payload,
                              timeout=300 if len(OLLAMA_URLS) == 1 else (10, 300))
            r.raise_for_status()
            url_usada = url
            break
        except requests.exceptions.RequestException as _exc:      # conexão, timeout ou HTTP de erro
            ultimo_erro = _exc
            with _ollama_lock:
                _ollama_fora_ate[url] = time.monotonic() + _OLLAMA_QUARENTENA_S
            if len(OLLAMA_URLS) > 1:
                print(f"[ollama] AVISO: {url} falhou ({type(_exc).__name__}) — tentando outra instância")
        finally:
            with _ollama_lock:
                _ollama_ativas[url] -= 1
    if url_usada is None:
        raise _OllamaOffline() from ultimo_erro
    t_http = time.perf_counter() - t_http_inicio

    d = r.json()
    texto = d.get("response", "")
    if not texto.strip():
        # Log do JSON completo para diagnóstico — remove se for muito verboso em produção
        print(f"[ollama] AVISO: response vazio! done_reason={d.get('done_reason')!r} "
              f"done={d.get('done')} eval_count={d.get('eval_count')} "
              f"thinking={str(d.get('thinking',''))[:120]!r}")
        texto = "O modelo não conseguiu gerar uma resposta, tente novamente ou reformule a pergunta."

    ns = 1_000_000_000
    t_load    = d.get("load_duration", 0) / ns
    t_prefill = d.get("prompt_eval_duration", 0) / ns
    t_gen     = d.get("eval_duration", 0) / ns
    tok_entrada = d.get("prompt_eval_count", 0)
    tok_saida   = d.get("eval_count", 0)

    print(f"[timing] ollama_http_total:   {t_http:.3f}s")
    print(f"[timing]   └─ load_model:     {t_load:.3f}s")
    print(f"[timing]   └─ prefill:        {t_prefill:.3f}s  ({tok_entrada} tokens)")
    print(f"[timing]   └─ geração:        {t_gen:.3f}s  ({tok_saida} tokens)")
    print(f"[timing] ollama_resposta_chars: {len(texto)}")
    if tok_saida > 0 and t_gen > 0:
        print(f"[timing] ollama_throughput:   {tok_saida / t_gen:.1f} tok/s")

    return {
        "texto": texto,
        "tokens_entrada": tok_entrada,
        "tokens_saida": tok_saida,
        "tokens_total": tok_entrada + tok_saida,
        "tempo_carga_s": round(t_load, 3),
        "tempo_prefill_s": round(t_prefill, 3),
        "tempo_geracao_s": round(t_gen, 3),
        "ollama_url": url_usada,
    }


def chamar_maritaca(prompt: str, modelo: str) -> dict:
    api_key = os.getenv("MARITACA_API_KEY", "").strip()
    if not api_key:
        return {
            "texto": "Modelo Sabiá-4 indisponível: chave de API não configurada.",
            "tokens_entrada": 0, "tokens_saida": 0, "tokens_total": 0,
            "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
    }
    print(f"[timing] maritaca_prompt_chars: {len(prompt)}")
    t_http_inicio = time.perf_counter()
    try:
        r = requests.post(MARITACA_URL, json=payload, headers=headers, timeout=120)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise _MaritacaError("Timeout ou falha de conexão com a API Maritaca") from exc
    t_http = time.perf_counter() - t_http_inicio

    if r.status_code == 401:
        raise _MaritacaError("Chave de API inválida ou sem permissão (HTTP 401)")
    if r.status_code == 429:
        raise _MaritacaError("Limite de requisições atingido na API Maritaca (HTTP 429)")
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise _MaritacaError(f"Erro HTTP {r.status_code} na API Maritaca") from exc

    d = r.json()
    texto = d["choices"][0]["message"]["content"]
    uso = d.get("usage", {})
    tok_entrada = uso.get("prompt_tokens", 0)
    tok_saida   = uso.get("completion_tokens", 0)

    print(f"[timing] maritaca_http_total:   {t_http:.3f}s")
    print(f"[timing] maritaca_tokens:       {tok_entrada}→{tok_saida}")
    print(f"[timing] maritaca_resposta_chars: {len(texto)}")

    return {
        "texto": texto,
        "tokens_entrada": tok_entrada,
        "tokens_saida": tok_saida,
        "tokens_total": tok_entrada + tok_saida,
        "tempo_carga_s": 0.0,
        "tempo_prefill_s": 0.0,
        "tempo_geracao_s": round(t_http, 3),
    }


_RE_PEDE_CURSOS = re.compile(
    r"\b(?:cursos?|tecnic\w+|superiores|bacharelad\w+|licenciatur\w+|graduacao|tecnolog\w+|"
    r"integrados?|subsequentes?|concomitantes?)\b"
)

_PROMPT_REGRAS_COMUNS = """Você é um assistente especializado nos documentos do processo seletivo do IFRS.
ESCOPO: Todas as informações fornecidas no contexto abaixo referem-se exclusivamente ao IFRS – Campus Ibirubá (Instituto Federal do Rio Grande do Sul, Campus Ibirubá). Ao responder, trate o contexto como sendo sempre sobre esse campus específico — não presuma nem mencione outros campi do IFRS, a menos que o usuário pergunte explicitamente sobre outro campus.
Responda à pergunta usando SOMENTE as informações do contexto abaixo.
Regras obrigatórias:
- Responda diretamente, em português, sem introduções nem rótulos como "Resposta:", "Inferência:" ou similares.
- Nunca comente sobre como a resposta foi construída, deduzida ou inferida."""

_PROMPT_REGRAS_CURSOS = """- COMPLETUDE OBRIGATÓRIA (a pergunta pede cursos): percorra TODO o contexto e inclua TODOS os cursos do tipo perguntado. Cada linha do contexto que começa com "Curso:" é um curso diferente — não pule nem omita nenhum. Só encerre a lista depois de verificar todas as linhas "Curso:" presentes. Isso vale só para listar cursos.
- Se o contexto trouxer cursos de categorias diferentes (técnico integrado, técnico subsequente, superior), responda apenas com os cursos que pertencem à categoria perguntada, sem misturar categorias."""

# Os exemplos de rótulo dependem do tipo de pergunta: citar "Curso:" numa pergunta de data induziria o modelo a
# imitar justamente esse prefixo ("Curso: 30/09 a 19/10/2026 | …").
_ROTULOS_CURSOS = '"Curso:", "Turnos:", "Duração (semestres):", "Total de Vagas:"'
_ROTULOS_DATAS = '"Data:", "Evento:"'

_PROMPT_REGRA_REGISTRO = """- Cada linha do contexto no formato "Rótulo: valor | Rótulo: valor" (ex.: {exemplo}) é um registro independente: use APENAS os dados que aparecem juntos na MESMA linha. Nunca combine dados de linhas diferentes."""

_PROMPT_REGRA_FRASE = """- Escreva a resposta em frases próprias. NÃO copie linhas do contexto e NÃO use os rótulos do contexto ({rotulos}) no texto da resposta; para uma lista, escreva um item por linha só com os nomes e os dados pedidos."""

_PROMPT_REGRAS_FINAIS = """- Use exatamente os nomes, turnos, datas e dados que aparecem no contexto — não invente nem complete informações ausentes.
{registro}
{frase}
- Se a informação não estiver no contexto, responda apenas: "Não encontrei essa informação nos documentos.\""""


def montar_prompt(pergunta: str, contexto: str) -> str:
    """Prompt do LLM. As regras de listagem de cursos ("Curso:") e os exemplos de rótulo de curso só entram quando a
    pergunta fala de cursos; a de responder em frases próprias, sem copiar linhas cruas nem rótulos do contexto,
    vale sempre (com os rótulos "Data:"/"Evento:" nas demais perguntas)."""
    pede_cursos = bool(_RE_PEDE_CURSOS.search(_sem_acento(pergunta)))
    regras = [_PROMPT_REGRAS_COMUNS]
    if pede_cursos:
        regras.append(_PROMPT_REGRAS_CURSOS)
    exemplo = '"Curso: … | Turnos: … | Total de Vagas: …"' if pede_cursos else '"Data: … | Evento: …"'
    regras.append(_PROMPT_REGRAS_FINAIS.format(
        registro=_PROMPT_REGRA_REGISTRO.format(exemplo=exemplo),
        frase=_PROMPT_REGRA_FRASE.format(rotulos=_ROTULOS_CURSOS if pede_cursos else _ROTULOS_DATAS)))
    return "\n".join(regras) + f"\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}"


def perguntar_llm(pergunta: str, contexto: str, modelo_cfg: dict) -> dict:
    t0 = time.perf_counter()
    prompt = montar_prompt(pergunta, contexto)
    print(f"[timing] montagem_prompt:     {time.perf_counter() - t0:.6f}s  ({len(prompt)} chars total)")
    if modelo_cfg.get("tipo") == "maritaca":
        return chamar_maritaca(prompt, modelo_cfg["modelo"])
    return chamar_ollama(prompt, modelo_cfg["modelo"])


def recuperar_contexto(pergunta: str, pipeline, tabela_docs: list, cronograma_docs: list) -> dict:
    """Retrieval completo de UMA pergunta, até o texto do contexto. É o ÚNICO caminho de retrieval: responder(),
    teste_modelos.py e teste_timing.py chamam esta função (não copie este fluxo).

    Busca semântica (RERANKER_CANDIDATOS chunks se RERANKER_ATIVO=1, senão TOP_K) → reranker (opcional) → TOP_K →
    pós-filtro de cotas → injeção híbrida de tabelas → injeção híbrida de cronograma → contexto.
    Devolve: docs, contexto, t_retrieval (embed+busca), t_rerank, rerank_aplicado, t_contexto, scores,
    ids_tabela_extra e ids_crono_extra (chunks injetados, para depuração)."""
    # 1. Embedding da query + busca vetorial no ChromaDocumentStore
    t0 = time.perf_counter()
    entrada = {"embedder": {"text": pergunta}}
    if RERANKER_ATIVO:
        entrada["retriever"] = {"top_k": RERANKER_CANDIDATOS}      # candidatos para o reranker
    resultado = pipeline.run(entrada, include_outputs_from={"embedder"})
    docs = resultado["retriever"]["documents"]
    embedding_pergunta = resultado["embedder"]["embedding"]
    t_retrieval = time.perf_counter() - t0
    print(f"[timing] embed+retrieve:      {t_retrieval:.3f}s  ({len(docs)} docs recuperados)")

    # Reranker (opcional): reordena os candidatos e fica com TOP_K; antes da injeção híbrida
    t_rerank, rerank_aplicado = 0.0, False
    if RERANKER_ATIVO:
        docs, t_rerank, rerank_aplicado = reordenar_por_reranker(pergunta, docs)
        print(f"[timing] reranker:            {t_rerank:.3f}s  "
              f"({'aplicado' if rerank_aplicado else 'NÃO aplicado — ordem semântica'}, {len(docs)} docs)")

    # Pós-filtro: descarta chunks de texto que descrevem distribuição de cotas
    docs_antes = len(docs)
    docs = [
        d for d in docs
        if (d.meta or {}).get("tipo") == "tabela" or not _PADRAO_COTA.search(d.content or "")
    ]
    n_removidos = docs_antes - len(docs)
    if n_removidos:
        print(f"[timing] pos_filtro_cota:     {n_removidos} chunk(s) removido(s)")

    # Retrieval híbrido: injeta chunks de tabela relevantes por palavra-chave
    t0_hib = time.perf_counter()
    docs_tabela_extra = _buscar_tabelas_por_keyword(pergunta, tabela_docs, docs)
    if docs_tabela_extra:
        # Tabelas primeiro → LLM vê dados estruturados antes do texto livre
        docs_tabela_ja  = [d for d in docs if (d.meta or {}).get("tipo") == "tabela"]
        docs_texto_final = [d for d in docs if (d.meta or {}).get("tipo") != "tabela"]
        # Quando há injeção de tabelas, reduz chunks de texto para 6 para preservar
        # espaço de geração sem sacrificar o contexto estruturado mais confiável
        docs_texto_final = docs_texto_final[:6]
        docs = docs_tabela_ja + docs_tabela_extra + docs_texto_final
        n_tab = len(docs_tabela_ja) + len(docs_tabela_extra)
        print(f"[timing] hibrido_tabela:      {time.perf_counter() - t0_hib:.6f}s  "
              f"({len(docs_tabela_extra)} tabelas injetadas → "
              f"{n_tab} tab + {len(docs_texto_final)} texto = {len(docs)} docs)")
    else:
        print(f"[timing] hibrido_tabela:      {time.perf_counter() - t0_hib:.6f}s  "
              f"(0 injetados)")

    # Retrieval híbrido: injeta chunks de cronograma quando a pergunta menciona datas/prazos
    t0_crono = time.perf_counter()
    docs_crono_extra = _buscar_cronograma_por_keyword(
        pergunta, cronograma_docs, docs,
        ranquear=lambda k: cronograma_mais_similares(pipeline, embedding_pergunta, k),
    )
    if docs_crono_extra:
        # Cronograma no início → LLM lê as datas antes do restante do contexto
        docs = docs_crono_extra + docs
        print(f"[timing] hibrido_cronograma:  {time.perf_counter() - t0_crono:.6f}s  "
              f"({len(docs_crono_extra)} cronograma injetados, {len(docs)} docs total)")
    else:
        print(f"[timing] hibrido_cronograma:  {time.perf_counter() - t0_crono:.6f}s  (0 injetados)")

    # Montagem do contexto (join dos documentos recuperados)
    t0 = time.perf_counter()
    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join([d.content for d in docs])
    t_contexto = time.perf_counter() - t0
    if docs:
        n_tab_ctx = sum(1 for d in docs if (d.meta or {}).get("tipo") == "tabela")
        print(f"[timing] montagem_contexto:   {t_contexto:.6f}s  "
              f"({len(contexto)} chars | {n_tab_ctx} tabela + {len(docs)-n_tab_ctx} texto)")
    return {
        "docs": docs, "contexto": contexto, "scores": scores,
        "t_retrieval": t_retrieval, "t_rerank": t_rerank, "rerank_aplicado": rerank_aplicado,
        "t_contexto": t_contexto,
        "ids_tabela_extra": {d.id for d in docs_tabela_extra if d.id},
        "ids_crono_extra": {d.id for d in docs_crono_extra if d.id},
    }


def responder(pergunta: str, modelo_cfg: dict, propagar_erros: bool = False) -> dict:
    try:
        pipeline, _, tabela_docs, cronograma_docs = carregar_pipeline()
        if pipeline is None:
            raise RuntimeError("Pipeline não carregado — execute indexar.py primeiro.")

        proc = psutil.Process()
        ram_antes = proc.memory_info().rss / 1024**2
        t_inicio = time.perf_counter()

        print(f"\n[timing] ═══ início | modelo={modelo_cfg['modelo']} | pergunta={len(pergunta)} chars ═══")

        # 1. Retrieval completo (busca, reranker opcional, pós-filtro, híbrido, contexto): recuperar_contexto()
        rc = recuperar_contexto(pergunta, pipeline, tabela_docs, cronograma_docs)
        docs, contexto, scores = rc["docs"], rc["contexto"], rc["scores"]
        t_retrieval, t_rerank, rerank_aplicado = rc["t_retrieval"], rc["t_rerank"], rc["rerank_aplicado"]

        if not docs:
            t_total = time.perf_counter() - t_inicio
            print(f"[timing] TOTAL (sem docs):    {t_total:.3f}s")
            return {
                "resposta": "⚠️ Nenhum trecho relevante encontrado nos documentos.",
                "tempo_total_s": round(t_total, 3),
                "tempo_retrieval_s": round(t_retrieval, 3), "tempo_llm_s": 0.0,
                "tempo_rerank_s": round(t_rerank, 3), "rerank_aplicado": rerank_aplicado,
                "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
                "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
                "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
                "ram_delta_mb": 0.0, "nao_encontrado": True, "erro": False,
                "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
            }

        if _DEBUG_CONTEXTO:
            print("\n" + "─" * 72)
            print(f"[DEBUG_CONTEXTO] {len(docs)} docs no contexto:")
            for i, d in enumerate(docs, 1):
                tipo = (d.meta or {}).get("tipo", "texto")
                src  = (d.meta or {}).get("source", (d.meta or {}).get("file_path", "?"))
                print(f"  [{i:02d}] tipo={tipo}  src={src}")
                print(f"        {(d.content or '')!r}")
            print("─" * 72 + "\n")

        # 2. LLM: montagem do prompt + chamada ao Ollama (detalhado dentro de perguntar_llm/chamar_ollama)
        t1 = time.perf_counter()
        try:
            r = perguntar_llm(pergunta, contexto, modelo_cfg)
        except (_OllamaOffline, _MaritacaError) as exc:
            t_llm = time.perf_counter() - t1
            t_total = time.perf_counter() - t_inicio
            is_maritaca = isinstance(exc, _MaritacaError)
            tag = "Maritaca" if is_maritaca else "Ollama"
            print(f"[timing] llm_total:           {t_llm:.3f}s  (FALHA — {tag} erro: {exc})")
            print(f"[timing] TOTAL:               {t_total:.3f}s")
            if is_maritaca:
                msg_erro = f"⚠️ Erro ao consultar Maritaca: {exc}."
            else:
                msg_erro = _MSG_OLLAMA_OFFLINE
            return {
                "resposta": msg_erro,
                "tempo_total_s": round(t_total, 3),
                "tempo_retrieval_s": round(t_retrieval, 3), "tempo_llm_s": 0.0,
                "tempo_rerank_s": round(t_rerank, 3), "rerank_aplicado": rerank_aplicado,
                "docs_recuperados": len(docs),
                "score_max": round(max(scores), 4) if scores else 0.0,
                "score_medio": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "chars_contexto": len(contexto), "tokens_entrada": 0, "tokens_saida": 0,
                "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
                "ram_delta_mb": 0.0, "nao_encontrado": False, "erro": True,
                "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
            }
        t_llm = time.perf_counter() - t1
        print(f"[timing] llm_total:           {t_llm:.3f}s")

        t_total = time.perf_counter() - t_inicio
        ram_depois = proc.memory_info().rss / 1024**2
        texto = r["texto"]
        tok_saida = r["tokens_saida"]

        print(f"[timing] ram_delta:           {ram_depois - ram_antes:+.1f} MB")
        print(f"[timing] TOTAL:               {t_total:.3f}s  ════════════════")

        return {
            "resposta": texto,
            "tempo_total_s": round(t_total, 3),
            "tempo_retrieval_s": round(t_retrieval, 3),
            "tempo_rerank_s": round(t_rerank, 3),
            "rerank_aplicado": rerank_aplicado,
            "ollama_url": r.get("ollama_url"),
            "tempo_llm_s": round(t_llm, 3),
            "tempo_carga_s": r.get("tempo_carga_s", 0.0),
            "tempo_prefill_s": r.get("tempo_prefill_s", 0.0),
            "tempo_geracao_s": r.get("tempo_geracao_s", 0.0),
            "docs_recuperados": len(docs),
            "score_max": round(max(scores), 4) if scores else 0.0,
            "score_medio": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "chars_contexto": len(contexto),
            "tokens_entrada": r["tokens_entrada"],
            "tokens_saida": tok_saida,
            "tokens_total": r["tokens_total"],
            "tokens_por_s": round(tok_saida / t_llm, 2) if t_llm > 0 else 0.0,
            "chars_resposta": len(texto),
            "ram_delta_mb": round(ram_depois - ram_antes, 1),
            "nao_encontrado": "não encontrei essa informação" in texto.lower(),
            "erro": False,
        }
    except Exception:
        if propagar_erros:
            raise
        traceback.print_exc()
        return {
            "resposta": "Ocorreu um erro temporário ao processar sua pergunta. Tente novamente em instantes.",
            "tempo_total_s": 0.0, "tempo_retrieval_s": 0.0, "tempo_llm_s": 0.0,
            "tempo_rerank_s": 0.0, "rerank_aplicado": False,
            "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
            "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
            "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
            "ram_delta_mb": 0.0, "nao_encontrado": False, "erro": True,
            "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
        }


def salvar_consulta_no_banco(sessao_id: int, modelo_cfg: dict, pergunta: str, m: dict) -> int | None:
    try:
        # Garante que o modelo está registrado antes de inserir a consulta.
        # Necessário para modelos dinâmicos do Ollama que não constam no dict MODELOS estático.
        upsert_modelo(
            modelo_id=modelo_cfg["modelo"],
            nome_bonito=modelo_cfg.get("descricao"),
            tipo=modelo_cfg.get("tipo", "ollama"),
            params_b=modelo_cfg.get("params_b"),
            quantizacao=modelo_cfg.get("quantizacao"),
        )
        return inserir_consulta(
            sessao_id=sessao_id,
            modelo_id=modelo_cfg["modelo"],
            pergunta=pergunta,
            resposta_gerada=m["resposta"],
            tempo_retrieval_s=m["tempo_retrieval_s"],
            tempo_llm_s=m["tempo_llm_s"],
            tempo_total_s=m["tempo_total_s"],
            tokens_entrada=m["tokens_entrada"],
            tokens_saida=m["tokens_saida"],
            docs_recuperados=m["docs_recuperados"],
            score_max=m["score_max"],
            score_medio=m["score_medio"],
            nao_encontrado=m["nao_encontrado"],
            erro=m["erro"],
        )
    except Exception as e:
        print(f"[db] erro ao gravar consulta: {e}")
        return None


def formatar_metricas(m: dict, modelo_id: str) -> str:
    return (
        f"⏱ **{m['tempo_total_s']}s** total "
        f"(retrieval {m['tempo_retrieval_s']}s · LLM {m['tempo_llm_s']}s) · "
        f"🔢 {m['tokens_entrada']}→{m['tokens_saida']} tokens · "
        f"⚡ {m['tokens_por_s']} tok/s · "
        f"📄 {m['docs_recuperados']} trechos (score máx {m['score_max']}) · "
        f"🤖 {modelo_id}"
    )
