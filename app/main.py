from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from dotenv import load_dotenv

from .models import init_db, SessionLocal, Complaint
from .services import process_bulk, truncate, client, group_and_summarize_similarities

load_dotenv()
init_db()

app = FastAPI(title="NIP Insights MVP", version="0.1.0")


@app.post("/complaints/bulk")
async def bulk_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingestão em lote de Excel (.xls/.xlsx) ou CSV (.csv):
    grava protocol+text no DB e dispara process_bulk em background.
    """
    fname = file.filename.lower()
    contents = await file.read()

    if fname.endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(contents), parse_dates=False)
    elif fname.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents), dtype=str)
    else:
        raise HTTPException(400, "Envie um .xls/.xlsx ou .csv")

    cols = {c.upper(): c for c in df.columns}
    if "PROTOCOLO" not in cols or "RECLAMAÇÃO DA NIP" not in cols:
        raise HTTPException(
            400,
            "Colunas obrigatórias não encontradas: 'PROTOCOLO' e 'RECLAMAÇÃO DA NIP'"
        )

    db = SessionLocal()
    ids = []
    for _, row in df.iterrows():
        comp = Complaint(
            protocol=str(row[cols["PROTOCOLO"]]),
            text=str(row[cols["RECLAMAÇÃO DA NIP"]])
        )
        db.add(comp)
        db.flush()
        ids.append(comp.id)
    db.commit()
    db.close()

    background_tasks.add_task(process_bulk, ids)
    return {"ingested": len(ids), "message": "Processing started in background"}


@app.post("/complaints/reprocess")
def reprocess_pending(background_tasks: BackgroundTasks):
    """
    Reagenda process_bulk para todos os registros sem embedding.
    """
    db = SessionLocal()
    pending = [r[0] for r in db.query(Complaint.id)
                     .filter(Complaint.embedding.is_(None))
                     .all()]
    db.close()

    if not pending:
        return {"scheduled": 0, "message": "Nenhum embedding pendente."}

    background_tasks.add_task(process_bulk, pending)
    return {"scheduled": len(pending), "message": "Processamento reagendado."}


@app.get("/complaints/status")
def complaints_status():
    """
    Retorna progresso do processamento: total, embeddings e motivações.
    """
    db = SessionLocal()
    total = db.query(Complaint).count()
    emb   = db.query(Complaint).filter(Complaint.embedding.isnot(None)).count()
    mot   = db.query(Complaint).filter(Complaint.motivation.isnot(None)).count()
    db.close()
    return {"total": total, "embeddings": emb, "motivations": mot}


@app.get("/motivations/similarities")
def export_motivation_similarities():
    """
    Gera um Excel com 'motivation', 'protocols' e 'similarities'
    para cada grupo de motivação.
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
