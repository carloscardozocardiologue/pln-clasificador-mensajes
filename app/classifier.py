import re
import os
import warnings
from typing import Optional
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "modelos"
BETO_ENABLED = os.getenv("BETO_ENABLED", "true").lower() == "true"

_tfidf_pipeline = None
_beto_clf = None
_beto_tokenizer = None
_beto_encoder = None
_beto_ready = False
_beto_error = None


def _clean_text(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def load_tfidf():
    global _tfidf_pipeline
    if _tfidf_pipeline is None:
        _tfidf_pipeline = joblib.load(MODELS_DIR / "modelo_tfidf.joblib")
    return _tfidf_pipeline


def load_beto():
    global _beto_clf, _beto_tokenizer, _beto_encoder, _beto_ready, _beto_error
    if _beto_ready:
        return True
    if not BETO_ENABLED:
        _beto_error = "BETO deshabilitado por configuración"
        return False
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        data = joblib.load(MODELS_DIR / "modelo_beto.joblib")
        _beto_clf = data["clf"]
        encoder_name = data["encoder_name"]

        _beto_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        _beto_encoder = AutoModel.from_pretrained(encoder_name)
        _beto_encoder.eval()
        _beto_ready = True
        return True
    except Exception as e:
        _beto_error = str(e)
        return False


def _beto_embedding(texto: str) -> np.ndarray:
    import torch

    inputs = _beto_tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        outputs = _beto_encoder(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)


def top_keywords_tfidf(texto: str, clase: str, n: int = 5) -> list:
    """Palabras/bigramas del mensaje que más empujaron hacia 'clase'."""
    pipeline = load_tfidf()
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    texto_limpio = _clean_text(texto)
    vec = vectorizer.transform([texto_limpio])
    feature_names = vectorizer.get_feature_names_out()
    class_idx = list(clf.classes_).index(clase)
    coefs = clf.coef_[class_idx]

    # Peso = presencia TF-IDF × coeficiente de la clase
    scores = vec.toarray()[0] * coefs
    top_idx = np.argsort(scores)[::-1][:n]
    return [feature_names[i] for i in top_idx if scores[i] > 0]


def mensaje_mas_cercano_tfidf(texto: str, mensajes_db: list) -> Optional[dict]:
    """Mensaje de la BD con mayor similitud coseno según TF-IDF."""
    if not mensajes_db:
        return None
    from sklearn.metrics.pairwise import cosine_similarity

    pipeline = load_tfidf()
    vectorizer = pipeline.named_steps["tfidf"]

    texto_limpio = _clean_text(texto)
    vec_query = vectorizer.transform([texto_limpio])
    textos_db = [_clean_text(m["texto"]) for m in mensajes_db]
    vecs_db = vectorizer.transform(textos_db)

    sims = cosine_similarity(vec_query, vecs_db)[0]
    idx = int(np.argmax(sims))
    similitud = float(sims[idx])

    if similitud < 0.05:  # demasiado distinto, no mostrar
        return None

    m = mensajes_db[idx]
    return {
        "texto": m["texto"],
        "pred_tfidf": m["pred_tfidf"],
        "opinion_medico": m.get("opinion_medico"),
        "similitud": round(similitud * 100, 1),
    }


def mensaje_mas_cercano_beto(texto: str, mensajes_db: list) -> Optional[dict]:
    """Mensaje de la BD con mayor similitud coseno según embeddings BETO."""
    if not _beto_ready or not mensajes_db:
        return None
    from sklearn.metrics.pairwise import cosine_similarity

    vec_query = _beto_embedding(texto)
    embeddings_db = np.vstack([_beto_embedding(m["texto"]) for m in mensajes_db])

    sims = cosine_similarity(vec_query, embeddings_db)[0]
    idx = int(np.argmax(sims))
    similitud = float(sims[idx])

    if similitud < 0.5:
        return None

    m = mensajes_db[idx]
    return {
        "texto": m["texto"],
        "pred_beto": m["pred_beto"],
        "opinion_medico": m.get("opinion_medico"),
        "similitud": round(similitud * 100, 1),
    }


def predict(texto: str) -> dict:
    resultado = {}

    # TF-IDF
    pipeline = load_tfidf()
    texto_limpio = _clean_text(texto)
    proba_tfidf = pipeline.predict_proba([texto_limpio])[0]
    classes = pipeline.classes_
    idx_tfidf = int(np.argmax(proba_tfidf))
    resultado["tfidf"] = {
        "prediccion": classes[idx_tfidf],
        "confianza": float(proba_tfidf[idx_tfidf]),
        "probabilidades": {c: float(p) for c, p in zip(classes, proba_tfidf)},
    }

    # BETO
    if load_beto():
        try:
            embedding = _beto_embedding(texto)
            proba_beto = _beto_clf.predict_proba(embedding)[0]
            classes_b = _beto_clf.classes_
            idx_beto = int(np.argmax(proba_beto))
            resultado["beto"] = {
                "prediccion": classes_b[idx_beto],
                "confianza": float(proba_beto[idx_beto]),
                "probabilidades": {c: float(p) for c, p in zip(classes_b, proba_beto)},
            }
        except Exception as e:
            resultado["beto"] = {"prediccion": None, "confianza": None, "error": str(e)}
    else:
        resultado["beto"] = {"prediccion": None, "confianza": None, "error": _beto_error}

    return resultado
