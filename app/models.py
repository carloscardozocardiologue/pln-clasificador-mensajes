from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.sql import func
from .database import Base


class Mensaje(Base):
    __tablename__ = "mensajes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    texto = Column(Text, nullable=False)

    pred_tfidf = Column(String(20), nullable=False)
    conf_tfidf = Column(Float, nullable=False)

    pred_beto = Column(String(20))
    conf_beto = Column(Float)

    opinion_medico = Column(String(20))
    notas = Column(Text)

    tfidf_correcto = Column(Boolean)
    beto_correcto = Column(Boolean)

    creado_en = Column(DateTime, server_default=func.now())
