# app/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
import pandas as pd
import warnings
from dotenv import load_dotenv

from .models import init_db, SessionLocal, Complaint
from .services import process_bulk, cluster_and_recommend

load_dotenv()
init_db()

app = FastAPI(title="NIP Insights MVP", version="0.1.0")


@app.post("/complaints/bulk")
async def bulk_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(400, "Envie um arquivo Excel (.xls ou .xlsx)")

    contents = await file.read()

    # Suprimir warnings de data inválida
    warnings.filterwarnings(
        "ignore",
        message="Cell .* is marked as a date but the serial value .* is outside the limits for dates"
    )

    # Lê o Excel sem converter colunas em datas
    df = pd.read_excel(io.BytesIO(contents), parse_dates=False)
    cols = {c.upper(): c for c in df.columns}
    if "PROTOCOLO" not in cols or "RECLAMAÇÃO DA NIP" not in cols:
        raise HTTPException(
            400,
            "Colunas obrigatórias não encontradas. "
            "São esperadas: 'PROTOCOLO' e 'RECLAMAÇÃO DA NIP'"
        )

    # Insere raw no banco e coleta IDs
    ids = []
    db = SessionLocal()
    for _, row in df.iterrows():
        proto = str(row[cols["PROTOCOLO"]])
        text  = str(row[cols["RECLAMAÇÃO DA NIP"]])
        comp = Complaint(protocol=proto, text=text)
        db.add(comp)
        db.flush()
        ids.append(comp.id)
    db.commit()
    db.close()

    # Processa embeddings e classificação em background
    background_tasks.add_task(process_bulk, ids)

    return {
        "ingested": len(ids),
        "message": "Processing started in background"
    }


@app.get("/clusters/export")
def export_clusters():
    """
    Agrupa por cluster e gera recomendações,
    então devolve um Excel para download.
    """
    df = cluster_and_recommend()
    out = io.BytesIO()
    df.to_excel(out, index=False, engine="openpyxl")
    out.seek(0)

    headers = {"Content-Disposition": "attachment; filename=clusters.xlsx"}
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )
