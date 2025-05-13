from collections import Counter
import json
import os

import pandas as pd
import numpy as np
import umap
import hdbscan
import openai
from dotenv import load_dotenv
from .models import SessionLocal, Complaint
from sklearn.cluster import DBSCAN


load_dotenv()

def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    snippet = text[:max_len]
    return snippet.rsplit(" ", 1)[0] + "…"

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    client = openai
    client.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str) -> list[float]:
    if hasattr(client, "embeddings"):
        resp = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return resp.data[0].embedding
    resp = client.Embedding.create(input=text, model="text-embedding-3-small")
    return resp["data"][0]["embedding"]


CLASSIFY_PROMPT = """
Você é um assistente que recebe o texto de uma reclamação e retorna apenas
o motivo principal em uma única palavra ou frase curta.
Retorne somente o texto limpo, sem JSON.
"""


def classify_motivation(text: str) -> str:
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT.strip()},
                {"role": "user",   "content": text}
            ],
            temperature=0.0,
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT.strip()},
            {"role": "user",   "content": text}
        ],
        temperature=0.0,
        max_tokens=10
    )
    return resp.choices[0].message.content.strip()


def save_complaint(protocol: str, text: str) -> Complaint:
    db = SessionLocal()
    emb = get_embedding(text)
    mot = classify_motivation(text)
    comp = Complaint(
        protocol=protocol,
        text=text,
        embedding=json.dumps(emb),
        motivation=mot
    )
    db.add(comp)
    db.commit()
    db.refresh(comp)
    db.close()
    return comp


def get_weekly_counts() -> list[dict]:
    db = SessionLocal()
    rows = (
        db.query(
            func.to_char(Complaint.created_at, 'IYYY-"W"IW').label("week"),
            func.count(Complaint.id).label("count")
        )
        .group_by("week")
        .all()
    )
    db.close()
    return [{"week": w, "count": c} for w, c in rows]


def process_bulk(ids: list[int], batch_size: int = 50):
    """
    Processa em background os embeddings e motivações para os IDs pendentes,
    fazendo chamadas de embedding em lotes para não ultrapassar o limite de tokens.
    """
    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.id.in_(ids)).all()

    # Extrai textos e prepara lista de embeddings vazia
    texts = [c.text for c in comps]

    # Percorre em lotes de tamanho batch_size
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_comps  = comps[start:end]
        batch_texts = texts[start:end]

        # Gera embeddings para o lote
        if hasattr(client, "embeddings"):
            resp = client.embeddings.create(input=batch_texts, model="text-embedding-3-small")
            batch_embs = [d.embedding for d in resp.data]
        else:
            resp = client.Embedding.create(input=batch_texts, model="text-embedding-3-small")
            batch_embs = [item["embedding"] for item in resp["data"]]

        # Atualiza cada objeto Complaint no lote
        for comp, emb in zip(batch_comps, batch_embs):
            comp.embedding = json.dumps(emb)
            comp.motivation = classify_motivation(comp.text)

        # Commit parcial para não acumular muita memória
        db.commit()

    db.close()

def cluster_and_recommend() -> pd.DataFrame:
    """
    - Carrega todas as reclamações com embedding
    - Reduz dimensão com UMAP e clusteriza com HDBSCAN
    - Gera um nome curto para cada cluster (usando trechos de até 200 chars)
    - Identifica os 10 textos mais recorrentes em cada cluster (até 100 chars)
    - Gera recomendações por cluster usando apenas o cluster_name
    Retorna DataFrame com colunas:
      protocol, text, motivation,
      cluster_id, cluster_name, top_cases, recommendation
    """
    def truncate(text: str, max_len: int = 200) -> str:
        if len(text) <= max_len:
            return text
        snippet = text[:max_len]
        return snippet.rsplit(" ", 1)[0] + "…"

    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.embedding != None).all()
    db.close()

    df = pd.DataFrame([{
        "protocol": c.protocol,
        "text": c.text,
        "motivation": c.motivation,
        "embedding": json.loads(c.embedding)
    } for c in comps])

    # UMAP
    X = np.vstack(df["embedding"].to_list())
    X_red = umap.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine"
    ).fit_transform(X)

    # HDBSCAN
    df["cluster_id"] = hdbscan.HDBSCAN(
        min_cluster_size=30, metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(X_red)

    # Nome curto de cada cluster
    summaries = {}
    for cl in sorted(df["cluster_id"].unique()):
        if cl == -1:
            summaries[cl] = "ruído"
            continue
        examples = df[df["cluster_id"] == cl]["text"].tolist()[:5]
        prompts = [f"- {truncate(t)}" for t in examples]
        prompt_name = (
            "Dado estes trechos de reclamações, "
            "resuma a causa comum em até 6 palavras:\n"
            + "\n".join(prompts)
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt_name}],
            temperature=0.0,
            max_tokens=10
        )
        summaries[cl] = resp.choices[0].message.content.strip()
    df["cluster_name"] = df["cluster_id"].map(summaries)

    # Top-10 casos recorrentes (até 100 chars cada)
    top_cases_map = {}
    for cl in sorted(df["cluster_id"].unique()):
        texts = df[df["cluster_id"] == cl]["text"].tolist()
        counter = Counter(texts)
        top10 = counter.most_common(10)
        formatted = [truncate(txt, 100) + f" ({cnt})" for txt, cnt in top10]
        top_cases_map[cl] = "; ".join(formatted)
    df["top_cases"] = df["cluster_id"].map(top_cases_map)

    # Recomendações por cluster (baseadas no nome)
    recommendations = {}
    for cl, name in summaries.items():
        if cl == -1:
            recommendations[cl] = ""
            continue
        prompt_rec = (
            f"Para o grupo de reclamações de causa “{name}”, "
            "sugira 3 ações de mitigação curtas e objetivas."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt_rec}],
            temperature=0.7,
            max_tokens=100
        )
        recommendations[cl] = resp.choices[0].message.content.strip()
    df["recommendation"] = df["cluster_id"].map(recommendations)

    return df[[
        "protocol",
        "text",
        "motivation",
        "cluster_id",
        "cluster_name",
        "top_cases",
        "recommendation"
    ]]
def group_and_suggest_by_motivation() -> pd.DataFrame:
    """
    Agrupa todas as reclamações pela motivação e, para cada grupo,
    gera:
     - lista de protocolos
     - contagem de casos
     - sugestão geral para resolver esse tipo de problema
    """
    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.motivation.isnot(None)).all()
    db.close()

    df = pd.DataFrame([{
        "protocol": c.protocol,
        "text": c.text,
        "motivation": c.motivation
    } for c in comps])

    groups = []
    for mot, grp in df.groupby("motivation"):
        prots = grp["protocol"].tolist()
        count = len(prots)
        # usa alguns exemplos truncados para ilustrar
        exemplos = grp["text"].tolist()[:5]
        exemplos = [truncate(t, 200) for t in exemplos]

        prompt = (
            f"Estes {count} casos foram classificados como “{mot}”.\n\n"
            "Aqui estão alguns exemplos:\n" +
            "\n".join(f"- {e}" for e in exemplos) +
            "\n\n"
            "Com base nisso, sugira UMA recomendação geral de política "
            "para o plano de saúde resolver esse tipo de problema (ex.: "
            "cobrir X, agilizar liberação de Y, etc)."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=60
        )
        suggestion = resp.choices[0].message.content.strip()

        groups.append({
            "motivation": mot,
            "count": count,
            "protocols": ",".join(prots),
            "suggestion": suggestion
        })

    return pd.DataFrame(groups)
def group_and_summarize_similarities() -> pd.DataFrame:
    """
    Para cada motivação, coleta até 5 exemplos de texto,
    e pede ao GPT para listar as principais semelhanças
    entre todos os casos desse grupo.
    Retorna DataFrame com colunas:
      motivation | protocols | similarities
    """
    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.motivation.isnot(None)).all()
    db.close()

    df = pd.DataFrame([{
        "protocol": c.protocol,
        "text": c.text,
        "motivation": c.motivation
    } for c in comps])

    groups = []
    for mot, grp in df.groupby("motivation"):
        prots    = grp["protocol"].tolist()
        exemplos = [truncate(t, 200) for t in grp["text"].tolist()[:5]]

        prompt = (
            f"Estes {len(prots)} casos foram classificados como “{mot}”.\n\n"
            "Aqui estão alguns exemplos:\n" +
            "\n".join(f"- {e}" for e in exemplos) +
            "\n\nListe as PRINCIPAIS semelhanças entre todos esses casos."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        similarities = resp.choices[0].message.content.strip()

        groups.append({
            "motivation": mot,
            "protocols": ",".join(prots),
            "similarities": similarities
        })

    return pd.DataFrame(groups)