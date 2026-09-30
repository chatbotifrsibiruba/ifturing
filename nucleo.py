"""
nucleo.py — Lógica compartilhada entre as páginas do IF Turing.
Backend puro: constantes, MODELOS, pipeline RAG, Ollama, persistência.
Sem layout ou CSS do Streamlit.
"""
import os
import re
import time
import psutil
import requests
import streamlit as st
from pathlib import Path
from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from dotenv import load_dotenv
from db import upsert_modelo, inserir_consulta

load_dotenv()

PASTA_CHROMA  = os.getenv("PASTA_CHROMA",  "./chroma_data")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3:latest")
MODELO_EMB    = "intfloat/multilingual-e5-base"
TOP_K         = 10

# Detecta chunks que descrevem distribuição de cotas (ex: "C1:", "C10:")
_PADRAO_COTA = re.compile(r"C\d{1,2}:")

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


def _buscar_tabelas_por_keyword(
    pergunta: str,
    tabela_docs: list,
    docs_semanticos: list,
) -> list:
    """Retorna chunks de tabela relevantes à pergunta por tipo-de-curso e por nome.

    Estratégia dual:
    1. Tipo-de-curso: termos em _SINONIMOS_TIPO (ex: 'superiores') são expandidos para
       os tipos reais usados nos chunks ('bacharelado', 'licenciatura', 'superior'),
       e TODOS os chunks desses tipos são incluídos.
    2. Nome-específico: demais termos não-genéricos são buscados literalmente no
       conteúdo do chunk, com variante singular/plural simples (±s final).

    Só dispara se a pergunta contiver alguma palavra de _PALAVRAS_TABELA.
    Remove documentos já presentes em docs_semanticos (dedup por id).
    """
    if not tabela_docs:
        return []
    pergunta_lower = pergunta.lower()
    if not any(kw in pergunta_lower for kw in _PALAVRAS_TABELA):
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

    ids_semanticos = {d.id for d in docs_semanticos if d.id}
    resultado = []
    for doc in tabela_docs:
        if doc.id in ids_semanticos:
            continue
        content = doc.content or ""
        content_lower = content.lower()
        if tipos_solicitados and _tipo_tabela_chunk(content) in tipos_solicitados:
            resultado.append(doc)
        elif termos_match and any(t in content_lower for t in termos_match):
            resultado.append(doc)

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
}

MODELO_PADRAO_SIMPLES = "🖥️ Qwen3 0.6B (Ollama local)"


@st.cache_data(ttl=60)
def listar_modelos_ollama() -> dict:
    """Consulta GET /api/tags no Ollama e retorna dict no mesmo formato de MODELOS.
    Fallback para MODELOS fixo se o Ollama estiver inacessível."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        itens = r.json().get("models", [])
        if not itens:
            return MODELOS
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
        return resultado
    except Exception:
        return MODELOS


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

    # Pré-carrega todos os chunks de tabela para o retrieval híbrido (feito uma vez, cacheado)
    try:
        todos = document_store.filter_documents()
        tabela_docs = [d for d in todos if (d.meta or {}).get("tipo") == "tabela"]
    except Exception as _e:
        print(f"[setup] aviso ao pré-carregar tabelas: {_e}")
        tabela_docs = []
    print(f"[setup] {len(tabela_docs)} chunks de tabela pré-carregados para retrieval híbrido")

    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_EMB))
    pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=TOP_K))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    return pipeline, n_chunks, tabela_docs


class _OllamaOffline(Exception):
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
        "options": {"temperature": 0.3, "num_predict": 1024, "num_ctx": 4096},
    }
    t_http_inicio = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
            requests.exceptions.RequestException) as _exc:
        raise _OllamaOffline() from _exc
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
    }


def perguntar_llm(pergunta: str, contexto: str, modelo_cfg: dict) -> dict:
    t0 = time.perf_counter()
    prompt = f"""Você é um assistente especializado nos documentos do processo seletivo do IFRS.
Responda à pergunta usando SOMENTE as informações do contexto abaixo.
Regras obrigatórias:
- Responda diretamente, em português, sem introduções nem rótulos como "Resposta:", "Inferência:" ou similares.
- Nunca comente sobre como a resposta foi construída, deduzida ou inferida.
- Ao listar cursos, use exatamente os nomes, turnos e dados que aparecem no contexto — não invente nem complete informações ausentes.
- Cada linha do contexto é um registro independente: use APENAS os dados que aparecem juntos na MESMA linha. Nunca combine o nome de um curso com turno, duração ou vagas de uma linha diferente.
- Se o contexto trouxer cursos de categorias diferentes (técnico integrado, técnico subsequente, superior), responda apenas com os cursos que pertencem à categoria perguntada, sem misturar categorias.
- Se a informação não estiver no contexto, responda apenas: "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}"""
    print(f"[timing] montagem_prompt:     {time.perf_counter() - t0:.6f}s  ({len(prompt)} chars total)")
    return chamar_ollama(prompt, modelo_cfg["modelo"])


def responder(pergunta: str, modelo_cfg: dict) -> dict:
    pipeline, _, tabela_docs = carregar_pipeline()
    if pipeline is None:
        raise RuntimeError("Pipeline não carregado — execute indexar.py primeiro.")

    proc = psutil.Process()
    ram_antes = proc.memory_info().rss / 1024**2
    t_inicio = time.perf_counter()

    print(f"\n[timing] ═══ início | modelo={modelo_cfg['modelo']} | pergunta={len(pergunta)} chars ═══")

    # 1. Embedding da query + busca vetorial no ChromaDocumentStore
    t0 = time.perf_counter()
    resultado = pipeline.run({"embedder": {"text": pergunta}})
    docs = resultado["retriever"]["documents"]
    t_retrieval = time.perf_counter() - t0
    print(f"[timing] embed+retrieve:      {t_retrieval:.3f}s  ({len(docs)} docs recuperados)")

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
        docs_tabela_ja = [d for d in docs if (d.meta or {}).get("tipo") == "tabela"]
        docs_texto_final = [d for d in docs if (d.meta or {}).get("tipo") != "tabela"]
        docs = docs_tabela_ja + docs_tabela_extra + docs_texto_final
        print(f"[timing] hibrido_tabela:      {time.perf_counter() - t0_hib:.6f}s  "
              f"({len(docs_tabela_extra)} chunks de tabela injetados, "
              f"{len(docs)} docs no contexto)")
    else:
        print(f"[timing] hibrido_tabela:      {time.perf_counter() - t0_hib:.6f}s  "
              f"(0 injetados)")

    if not docs:
        t_total = time.perf_counter() - t_inicio
        print(f"[timing] TOTAL (sem docs):    {t_total:.3f}s")
        return {
            "resposta": "⚠️ Nenhum trecho relevante encontrado nos documentos.",
            "tempo_total_s": round(t_total, 3),
            "tempo_retrieval_s": round(t_retrieval, 3), "tempo_llm_s": 0.0,
            "docs_recuperados": 0, "score_max": 0.0, "score_medio": 0.0,
            "chars_contexto": 0, "tokens_entrada": 0, "tokens_saida": 0,
            "tokens_total": 0, "tokens_por_s": 0.0, "chars_resposta": 0,
            "ram_delta_mb": 0.0, "nao_encontrado": True, "erro": False,
            "tempo_carga_s": 0.0, "tempo_prefill_s": 0.0, "tempo_geracao_s": 0.0,
        }

    # 2. Montagem do contexto (join dos documentos recuperados)
    t0 = time.perf_counter()
    scores = [d.score for d in docs if d.score is not None]
    contexto = "\n\n---\n\n".join([d.content for d in docs])
    t_contexto = time.perf_counter() - t0
    print(f"[timing] montagem_contexto:   {t_contexto:.6f}s  ({len(contexto)} chars)")

    # 3. LLM: montagem do prompt + chamada ao Ollama (detalhado dentro de perguntar_llm/chamar_ollama)
    t1 = time.perf_counter()
    try:
        r = perguntar_llm(pergunta, contexto, modelo_cfg)
    except _OllamaOffline:
        t_llm = time.perf_counter() - t1
        t_total = time.perf_counter() - t_inicio
        print(f"[timing] llm_total:           {t_llm:.3f}s  (FALHA — Ollama offline)")
        print(f"[timing] TOTAL:               {t_total:.3f}s")
        print(f"[ollama] falha de conexão ao consultar modelo {modelo_cfg['modelo']}")
        return {
            "resposta": _MSG_OLLAMA_OFFLINE,
            "tempo_total_s": round(t_total, 3),
            "tempo_retrieval_s": round(t_retrieval, 3), "tempo_llm_s": 0.0,
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


def salvar_consulta_no_banco(sessao_id: int, modelo_cfg: dict, pergunta: str, m: dict) -> int | None:
    try:
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
