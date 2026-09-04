"""
db.py — Schema e funções de acesso ao banco SQLite do IF Turing.

Guarda todas as consultas (pergunta/resposta), métricas de desempenho,
avaliações RAGAS e feedback do usuário num único banco, em vez de CSVs
soltos. Usado por app.py, avaliar.py e comparar_rodadas.py.

Uso:
    from db import get_conn, init_db, inserir_consulta, inserir_feedback

    init_db()  # roda uma vez, cria as tabelas se não existirem
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("./ifturing.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS modelos (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        modelo_id     TEXT UNIQUE NOT NULL,   -- ex: "llama3", "gemma2:27b"
        nome_bonito   TEXT,                   -- ex: "LLaMA 3"
        tipo          TEXT DEFAULT 'ollama',
        params_b      REAL,
        quantizacao   TEXT
    );

    CREATE TABLE IF NOT EXISTS sessoes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        sessao_uid    TEXT UNIQUE NOT NULL,   -- ex: timestamp da sessão do Streamlit
        origem        TEXT,                   -- 'app' (uso real) ou 'avaliar' (Golden Dataset)
        modo          TEXT,                   -- 'simples' ou 'detalhado', quando origem='app'
        branch        TEXT,                   -- nome da rodada/branch, quando origem='avaliar'
        criado_em     TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS consultas (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        sessao_id           INTEGER REFERENCES sessoes(id),
        modelo_id           TEXT REFERENCES modelos(modelo_id),
        pergunta            TEXT NOT NULL,
        resposta_gerada     TEXT,
        resposta_esperada   TEXT,             -- só preenchido quando vem do Golden Dataset
        tempo_retrieval_s   REAL,
        tempo_llm_s         REAL,
        tempo_total_s       REAL,
        tokens_entrada      INTEGER,
        tokens_saida        INTEGER,
        docs_recuperados    INTEGER,
        score_max           REAL,
        score_medio         REAL,
        nao_encontrado      INTEGER DEFAULT 0,
        erro                INTEGER DEFAULT 0,
        criado_em           TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS avaliacoes_ragas (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        consulta_id           INTEGER UNIQUE REFERENCES consultas(id),
        faithfulness          REAL,
        answer_relevancy      REAL,
        context_precision     REAL,
        context_recall        REAL,
        answer_correctness    REAL,
        juiz_modelo           TEXT DEFAULT 'sabiazinho-4'
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        consulta_id   INTEGER REFERENCES consultas(id),
        util          INTEGER,               -- 1 = útil, 0 = não útil (thumbs up/down)
        comentario    TEXT,
        criado_em     TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_consultas_modelo ON consultas(modelo_id);
    CREATE INDEX IF NOT EXISTS idx_consultas_sessao ON consultas(sessao_id);
    """)
    conn.commit()
    conn.close()
    print(f"✅ Banco inicializado em {DB_PATH.resolve()}")


def upsert_modelo(modelo_id: str, nome_bonito: str = None, tipo: str = "ollama",
                   params_b: float = None, quantizacao: str = None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO modelos (modelo_id, nome_bonito, tipo, params_b, quantizacao)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(modelo_id) DO UPDATE SET
            nome_bonito=excluded.nome_bonito,
            tipo=excluded.tipo,
            params_b=excluded.params_b,
            quantizacao=excluded.quantizacao
    """, (modelo_id, nome_bonito, tipo, params_b, quantizacao))
    conn.commit()
    conn.close()


def get_or_create_sessao(sessao_uid: str, origem: str, modo: str = None, branch: str = None) -> int:
    conn = get_conn()
    cur = conn.execute("SELECT id FROM sessoes WHERE sessao_uid = ?", (sessao_uid,))
    row = cur.fetchone()
    if row:
        conn.close()
        return row["id"]

    cur = conn.execute(
        "INSERT INTO sessoes (sessao_uid, origem, modo, branch) VALUES (?, ?, ?, ?)",
        (sessao_uid, origem, modo, branch),
    )
    conn.commit()
    sessao_id = cur.lastrowid
    conn.close()
    return sessao_id


def inserir_consulta(sessao_id: int, modelo_id: str, pergunta: str, resposta_gerada: str,
                      resposta_esperada: str = None, tempo_retrieval_s: float = None,
                      tempo_llm_s: float = None, tempo_total_s: float = None,
                      tokens_entrada: int = None, tokens_saida: int = None,
                      docs_recuperados: int = None, score_max: float = None,
                      score_medio: float = None, nao_encontrado: bool = False,
                      erro: bool = False,
                      ragas: dict = None) -> int:
    """
    Insere uma consulta e, se fornecido, as métricas RAGAS associadas.
    ragas = {"faithfulness":..., "answer_relevancy":..., "context_precision":...,
             "context_recall":..., "answer_correctness":...}
    Retorna o id da consulta inserida.
    """
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO consultas (
            sessao_id, modelo_id, pergunta, resposta_gerada, resposta_esperada,
            tempo_retrieval_s, tempo_llm_s, tempo_total_s,
            tokens_entrada, tokens_saida, docs_recuperados,
            score_max, score_medio, nao_encontrado, erro
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sessao_id, modelo_id, pergunta, resposta_gerada, resposta_esperada,
        tempo_retrieval_s, tempo_llm_s, tempo_total_s,
        tokens_entrada, tokens_saida, docs_recuperados,
        score_max, score_medio, int(nao_encontrado), int(erro),
    ))
    consulta_id = cur.lastrowid

    if ragas:
        conn.execute("""
            INSERT INTO avaliacoes_ragas (
                consulta_id, faithfulness, answer_relevancy,
                context_precision, context_recall, answer_correctness
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            consulta_id,
            ragas.get("faithfulness"), ragas.get("answer_relevancy"),
            ragas.get("context_precision"), ragas.get("context_recall"),
            ragas.get("answer_correctness"),
        ))

    conn.commit()
    conn.close()
    return consulta_id


def inserir_feedback(consulta_id: int, util: bool, comentario: str = None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO feedback (consulta_id, util, comentario) VALUES (?, ?, ?)",
        (consulta_id, int(util), comentario),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
