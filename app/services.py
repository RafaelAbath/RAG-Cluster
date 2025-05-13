from collections import Counter
import json
import os

import pandas as pd
import openai
from dotenv import load_dotenv
from .models import SessionLocal, Complaint

load_dotenv()

def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    snippet = text[:max_len]
    return snippet.rsplit(" ", 1)[0] + "…"

# Cliente OpenAI (v1.x ou fallback)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    client = openai
    client.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str) -> list[float]:
    # v1.x
    if hasattr(client, "embeddings"):
        resp = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return resp.data[0].embedding
    # v0.x
    resp = client.Embedding.create(input=text, model="text-embedding-3-small")
    return resp["data"][0]["embedding"]


CLASSIFY_PROMPT = """
Você é um assistente que recebe o texto de uma reclamação e retorna apenas
o motivo principal em uma única palavra ou frase curta.
Retorne somente o texto limpo, sem JSON.
"""


def classify_motivation(text: str) -> str:
    # v1.x
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
    # v0.x
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


def process_bulk(ids: list[int], batch_size: int = 50):
    """
    Gera embeddings e motivações em lotes menores para não estourar tokens.
    """
    db = SessionLocal()
    comps = db.query(Complaint).filter(Complaint.id.in_(ids)).all()

    for i in range(0, len(comps), batch_size):
        batch = comps[i : i + batch_size]
        texts = [c.text for c in batch]

        # embeddings
        if hasattr(client, "embeddings"):
            resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
            embs = [d.embedding for d in resp.data]
        else:
            resp = client.Embedding.create(input=texts, model="text-embedding-3-small")
            embs = [item["embedding"] for item in resp["data"]]

        # salva no DB
        for comp, emb in zip(batch, embs):
            comp.embedding  = json.dumps(emb)
            comp.motivation = classify_motivation(comp.text)

        db.commit()

    db.close()


def group_and_summarize_similarities() -> pd.DataFrame:
    """
    Para cada motivação, coleta até 5 exemplos e pede ao GPT
    para listar as principais semelhanças de todos os casos.
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
        sims = resp.choices[0].message.content.strip()

        groups.append({
            "motivation": mot,
            "protocols": ",".join(prots),
            "similarities": sims
        })

    return pd.DataFrame(groups)
