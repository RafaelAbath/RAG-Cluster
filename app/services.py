import os
import umap
import hdbscan
import json
import openai
from dotenv import load_dotenv
from .models import SessionLocal, Complaint
import pandas as pd
import numpy as np
from sqlalchemy import func
from sklearn.cluster import DBSCAN

load_dotenv()

# Inicialização do cliente OpenAI (v1.x ou fallback)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    client = openai
    client.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str) -> list[float]:
    # Interface v1.x
    if hasattr(client, "embeddings"):
        resp = client.embeddings.create(
            input=[text],  # sempre em lista
            model="text-embedding-3-small"
        )
        return resp.data[0].embedding
    # Interface v0.x
    resp = client.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return resp["data"][0]["embedding"]


CLASSIFY_PROMPT = """
Você é um assistente que recebe o texto de uma reclamação e retorna apenas
o motivo principal em uma única palavra ou frase curta.
Retorne somente o texto limpo, sem JSON.
"""


def classify_motivation(text: str) -> str:
    # Interface v1.x
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
    # Interface v0.x
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


def process_bulk(ids: list[int]):
    """
    Chamada em background pelo FastAPI.
    Gera embeddings em batch e classifica todos.
    """
    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.id.in_(ids)).all()
    texts = [c.text for c in comps]

    # Batch de embeddings (v1.x ou v0.x)
    if hasattr(client, "embeddings"):
        resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = [d.embedding for d in resp.data]
    else:
        resp = client.Embedding.create(input=texts, model="text-embedding-3-small")
        embeddings = [item["embedding"] for item in resp["data"]]

    # Armazena embedding e classificação
    for comp, emb in zip(comps, embeddings):
        comp.embedding = json.dumps(emb)
        comp.motivation = classify_motivation(comp.text)

    db.commit()
    db.close()


def cluster_and_recommend() -> pd.DataFrame:
    # 1) Carrega todas as reclamações com embedding
    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.embedding != None).all()
    db.close()
    df = pd.DataFrame([{
        "protocol": c.protocol,
        "text": c.text,
        "motivation": c.motivation,
        "embedding": json.loads(c.embedding)
    } for c in comps])

    # 2) Matriz de embeddings
    X = np.vstack(df["embedding"].to_list())

    # 3) UMAP para 10 dimensões (preserva similaridades)
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine")
    X_red = reducer.fit_transform(X)

    # 4) HDBSCAN (descobre clusters de forma dinâmica)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_red)
    df["cluster_id"] = labels

    # 5) Gerar um nome curto para cada cluster via GPT
    summaries = {}
    for cl in sorted(df["cluster_id"].unique()):
        if cl == -1:
            summaries[cl] = "ruído"
            continue
        examples = df[df["cluster_id"] == cl]["text"].tolist()[:5]
        prompt_name = (
            "Dado estes exemplos de reclamações, resuma a causa comum "
            "em até 6 palavras:\n" + "\n".join(examples)
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt_name}],
            temperature=0.0,
            max_tokens=10
        )
        summaries[cl] = resp.choices[0].message.content.strip()

    df["cluster_name"] = df["cluster_id"].map(summaries)

    # 6) Gerar recomendações específicas por cluster
    recommendations = {}
    for cl, name in summaries.items():
        if cl == -1:
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

    df["recommendation"] = df["cluster_id"].map(lambda cl: recommendations.get(cl, ""))

    
    return df[["protocol", "text", "motivation", "cluster_id", "cluster_name", "recommendation"]]