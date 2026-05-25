import csv
import io
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db, engine
from .models import Base, Mensaje
from . import classifier

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Clasificador de Mensajes Clínicos")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Carga BETO en background al arrancar para no bloquear el servidor
def _preload_beto():
    classifier.load_beto()


threading.Thread(target=_preload_beto, daemon=True).start()


# ── Schemas ──────────────────────────────────────────────────────────────────

class ClasificarRequest(BaseModel):
    texto: str


class OpinionRequest(BaseModel):
    opinion_medico: str
    notas: Optional[str] = None


# ── Rutas ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/clasificar")
def clasificar(req: ClasificarRequest, db: Session = Depends(get_db)):
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")

    texto = req.texto.strip()
    resultado = classifier.predict(texto)
    tfidf = resultado["tfidf"]
    beto = resultado["beto"]

    # Guardamos primero para excluirlo de la búsqueda de similitud
    mensaje = Mensaje(
        texto=texto,
        pred_tfidf=tfidf["prediccion"],
        conf_tfidf=tfidf["confianza"],
        pred_beto=beto.get("prediccion"),
        conf_beto=beto.get("confianza"),
    )
    db.add(mensaje)
    db.commit()
    db.refresh(mensaje)

    # Mensajes previos de la BD (excluimos el que acabamos de guardar)
    previos = [
        {"texto": m.texto, "pred_tfidf": m.pred_tfidf, "pred_beto": m.pred_beto,
         "opinion_medico": m.opinion_medico}
        for m in db.query(Mensaje).filter(Mensaje.id != mensaje.id).all()
    ]

    # Keywords TF-IDF
    keywords = []
    if tfidf["prediccion"]:
        try:
            keywords = classifier.top_keywords_tfidf(texto, tfidf["prediccion"])
        except Exception:
            pass

    # Mensaje más cercano (TF-IDF)
    cercano_tfidf = None
    try:
        cercano_tfidf = classifier.mensaje_mas_cercano_tfidf(texto, previos)
    except Exception:
        pass

    # Mensaje más cercano (BETO)
    cercano_beto = None
    try:
        cercano_beto = classifier.mensaje_mas_cercano_beto(texto, previos)
    except Exception:
        pass

    return {
        "id": mensaje.id,
        "tfidf": {**tfidf, "keywords": keywords, "cercano": cercano_tfidf},
        "beto": {**beto, "cercano": cercano_beto},
    }


@app.put("/api/mensajes/{mensaje_id}/opinion")
def guardar_opinion(mensaje_id: int, req: OpinionRequest, db: Session = Depends(get_db)):
    mensaje = db.query(Mensaje).filter(Mensaje.id == mensaje_id).first()
    if not mensaje:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")

    mensaje.opinion_medico = req.opinion_medico
    mensaje.notas = req.notas
    mensaje.tfidf_correcto = (mensaje.pred_tfidf == req.opinion_medico) if mensaje.pred_tfidf else None
    mensaje.beto_correcto = (mensaje.pred_beto == req.opinion_medico) if mensaje.pred_beto else None

    db.commit()
    db.refresh(mensaje)

    return {
        "id": mensaje.id,
        "tfidf_correcto": mensaje.tfidf_correcto,
        "beto_correcto": mensaje.beto_correcto,
    }


@app.get("/api/mensajes")
def listar_mensajes(db: Session = Depends(get_db)):
    mensajes = db.query(Mensaje).order_by(Mensaje.id.desc()).all()
    return [
        {
            "id": m.id,
            "texto": m.texto,
            "pred_tfidf": m.pred_tfidf,
            "conf_tfidf": m.conf_tfidf,
            "pred_beto": m.pred_beto,
            "conf_beto": m.conf_beto,
            "opinion_medico": m.opinion_medico,
            "notas": m.notas,
            "tfidf_correcto": m.tfidf_correcto,
            "beto_correcto": m.beto_correcto,
            "creado_en": m.creado_en.isoformat() if m.creado_en else None,
        }
        for m in mensajes
    ]


@app.get("/api/estadisticas")
def estadisticas(db: Session = Depends(get_db)):
    todos = db.query(Mensaje).all()
    con_opinion = [m for m in todos if m.opinion_medico is not None]

    total_valorados = len(con_opinion)

    def accuracy(lista, campo):
        if not lista:
            return None
        correctos = sum(1 for m in lista if getattr(m, campo) is True)
        return round(correctos / len(lista) * 100, 1)

    def por_clase(lista, campo_pred, campo_correcto):
        resultado = {}
        for label in ["leve", "moderada", "grave"]:
            del_medico = [m for m in lista if m.opinion_medico == label]
            if not del_medico:
                resultado[label] = {"total": 0, "tfidf_ok": None, "beto_ok": None}
                continue
            tfidf_ok = sum(1 for m in del_medico if m.tfidf_correcto is True)
            beto_ok = sum(1 for m in del_medico if m.beto_correcto is True)
            resultado[label] = {
                "total": len(del_medico),
                "tfidf_ok": round(tfidf_ok / len(del_medico) * 100, 1),
                "beto_ok": round(beto_ok / len(del_medico) * 100, 1) if any(m.pred_beto for m in del_medico) else None,
            }
        return resultado

    return {
        "total_clasificados": len(todos),
        "total_valorados": total_valorados,
        "tfidf_accuracy": accuracy(con_opinion, "tfidf_correcto"),
        "beto_accuracy": accuracy(
            [m for m in con_opinion if m.pred_beto is not None], "beto_correcto"
        ),
        "por_clase": por_clase(con_opinion, "pred_tfidf", "pred_beto"),
    }


@app.get("/api/mensajes/csv")
def exportar_csv(db: Session = Depends(get_db)):
    mensajes = db.query(Mensaje).order_by(Mensaje.id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "texto",
        "pred_tfidf", "conf_tfidf",
        "pred_beto", "conf_beto",
        "opinion_medico", "tfidf_correcto", "beto_correcto",
        "notas", "creado_en",
    ])
    for m in mensajes:
        writer.writerow([
            m.id, m.texto,
            m.pred_tfidf, round(m.conf_tfidf * 100, 1) if m.conf_tfidf else "",
            m.pred_beto or "", round(m.conf_beto * 100, 1) if m.conf_beto else "",
            m.opinion_medico or "",
            "Sí" if m.tfidf_correcto is True else ("No" if m.tfidf_correcto is False else ""),
            "Sí" if m.beto_correcto is True else ("No" if m.beto_correcto is False else ""),
            m.notas or "",
            m.creado_en.strftime("%Y-%m-%d %H:%M") if m.creado_en else "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=mensajes_clinicos.csv"},
    )


@app.delete("/api/mensajes/{mensaje_id}")
def borrar_uno(mensaje_id: int, db: Session = Depends(get_db)):
    mensaje = db.query(Mensaje).filter(Mensaje.id == mensaje_id).first()
    if not mensaje:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    db.delete(mensaje)
    db.commit()
    return {"eliminado": mensaje_id}


@app.delete("/api/mensajes")
def borrar_todos(db: Session = Depends(get_db)):
    eliminados = db.query(Mensaje).delete()
    db.commit()
    return {"eliminados": eliminados}


@app.get("/health")
def health():
    return {"status": "ok", "beto_ready": classifier._beto_ready}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
