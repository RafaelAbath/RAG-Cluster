from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import pandas as pd
import warnings
from sqlalchemy import func
from dotenv import load_dotenv

from .models import init_db, SessionLocal, Complaint
from .services import (
    process_bulk,
    cluster_and_recommend,
    truncate,
    client,
    group_and_suggest_by_motivation,
    group_and_summarize_similarities,
)

load_dotenv()
init_db()

app = FastAPI(title="NIP Insights MVP", version="0.1.0")


@app.post("/complaints/bulk")
async def bulk_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingestão em lote de Excel (.xls/.xlsx) ou CSV (.csv)
    grava raw no DB e dispara process_bulk em background.
    """
    fname = file.filename.lower()
    contents = await file.read()

    if fname.endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(contents), parse_dates=False)
    elif fname.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents), dtype=str)
    else:
        raise HTTPException(400, "Envie um arquivo Excel (.xls/.xlsx) ou CSV (.csv)")

    cols = {c.upper(): c for c in df.columns}
    if "PROTOCOLO" not in cols or "RECLAMAÇÃO DA NIP" not in cols:
        raise HTTPException(
            400,
            "Colunas obrigatórias não encontradas. "
            "São esperadas: 'PROTOCOLO' e 'RECLAMAÇÃO DA NIP'"
        )

    db = SessionLocal()
    ids = []
    for _, row in df.iterrows():
        proto = str(row[cols["PROTOCOLO"]])
        text  = str(row[cols["RECLAMAÇÃO DA NIP"]])
        comp = Complaint(protocol=proto, text=text)
        db.add(comp)
        db.flush()
        ids.append(comp.id)
    db.commit()
    db.close()

    background_tasks.add_task(process_bulk, ids)
    return {
        "ingested": len(ids),
        "message": "Processing started in background"
    }


@app.post("/complaints/reprocess")
def reprocess_pending(background_tasks: BackgroundTasks):
    """
    Reagenda em background o process_bulk para quem ainda não tem embedding.
    """
    db = SessionLocal()
    pending = [
        r[0]
        for r in db.query(Complaint.id)
                   .filter(Complaint.embedding.is_(None))
                   .all()
    ]
    db.close()

    if not pending:
        return {"scheduled": 0, "message": "Nenhum embedding pendente."}

    background_tasks.add_task(process_bulk, pending)
    return {
        "scheduled": len(pending),
        "message": "Processamento reagendado em background."
    }


@app.get("/complaints/status")
def complaints_status():
    """
    Retorna progresso: total, embeddings processados e motivações classificadas.
    """
    db = SessionLocal()
    total = db.query(func.count(Complaint.id)).scalar()
    emb   = db.query(func.count(Complaint.id)) \
              .filter(Complaint.embedding.isnot(None)).scalar()
    mot   = db.query(func.count(Complaint.id)) \
              .filter(Complaint.motivation.isnot(None)).scalar()
    db.close()
    return {
        "total": total,
        "embeddings": emb,
        "motivations": mot
    }


@app.get("/clusters/export")
def export_clusters():
    """
    Gera um Excel com 4 abas:
      - clusters:            protocolo, text, motivation, cluster_id, cluster_name, top_cases
      - groups:              cluster_id, protocolos (todos do mesmo cluster)
      - motivation_counts:   cluster_id, motivation, count
      - case_recommendations: protocolo, cluster_id, case_snippet, recommendation
    """
    # 1) DataFrame completo de clusters
    df = cluster_and_recommend()

    # 2) Aba “groups”: lista de protocolos por cluster
    groups_df = (
        df.groupby("cluster_id")["protocol"]
          .apply(lambda prots: ",".join(prots))
          .reset_index(name="protocols")
    )

    # 3) Aba “motivation_counts”: contagem de motivações por cluster
    mot_df = (
        df.groupby(["cluster_id", "motivation"])
          .size()
          .reset_index(name="count")
          .sort_values(["cluster_id", "count"], ascending=[True, False])
    )

    # 4) Aba “case_recommendations”: recomendação para cada caso
    rows = []
    for _, row in df.iterrows():
        prot    = row["protocol"]
        cl      = row["cluster_id"]
        snippet = truncate(row["text"], 200)
        prompt  = (
            f"Para esta reclamação (ID {prot}):\n\n"
            f"\"{snippet}\"\n\n"
            "sugira 3 ações de mitigação curtas e objetivas."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        rec = resp.choices[0].message.content.strip()
        rows.append({
            "protocol": prot,
            "cluster_id": cl,
            "case_snippet": snippet,
            "recommendation": rec
        })
    case_recs_df = pd.DataFrame(rows)

    # 5) Monta e retorna o Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="clusters", index=False)
        groups_df.to_excel(writer, sheet_name="groups", index=False)
        mot_df.to_excel(writer, sheet_name="motivation_counts", index=False)
        case_recs_df.to_excel(writer, sheet_name="case_recommendations", index=False)
    out.seek(0)

    headers = {
        "Content-Disposition": "attachment; filename=clusters_with_case_recs.xlsx"
    }
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )


@app.get("/motivations/export")
def export_motivation_suggestions():
    """
    Gera um Excel com uma aba única 'motivation_summary' contendo:
      - motivation
      - count
      - protocols (lista)
      - suggestion
    """
    df = group_and_suggest_by_motivation()
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="motivation_summary", index=False)
    out.seek(0)

    headers = {
        "Content-Disposition": "attachment; filename=motivation_suggestions.xlsx"
    }
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )
@app.get("/motivations/similarities")
def export_motivation_similarities():
    """
    Gera um Excel com uma aba única 'motivation_similarities' contendo:
      - motivation
      - protocols (lista)
      - similarities (texto com o que é comum em cada grupo)
    """
    df = group_and_summarize_similarities()
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="motivation_similarities", index=False)
    out.seek(0)

    headers = {
        "Content-Disposition":
            "attachment; filename=motivation_similarities.xlsx"
    }
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )