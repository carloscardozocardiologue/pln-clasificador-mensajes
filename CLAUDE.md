# CLAUDE.md — Clasificador de Mensajes Clínicos

## Resumen del proyecto

Aplicación web para comparar en tiempo real dos modelos de PLN (TF-IDF y BETO) clasificando mensajes de pacientes en tres niveles de prioridad clínica: leve, moderada y grave. El médico anota su valoración y la app genera métricas comparativas automáticamente.

Actividad académica: Unidad 6 — Máster IA en Salud, Universidad Europea de Madrid.

## Stack

- **Backend:** FastAPI + Uvicorn (Python 3.11)
- **Base de datos:** MySQL en Railway (SQLAlchemy + pymysql)
- **Modelos:** `modelos/modelo_tfidf.joblib` (pipeline completo) y `modelos/modelo_beto.joblib` (LogReg + encoder name para HuggingFace)
- **Frontend:** HTML + CSS + JS vanilla · Chart.js (CDN) · Web Speech API (voz)

## Arquitectura

```
Browser
  ├─ POST /api/clasificar      → classifier.py → ambos modelos → guarda en DB → devuelve predicciones
  ├─ PUT  /api/mensajes/{id}/opinion → guarda valoración médico + calcula correcto/incorrecto
  ├─ GET  /api/mensajes         → lista todos los mensajes
  └─ GET  /api/estadisticas     → métricas agregadas (accuracy, por clase)
```

## Variables de entorno (.env)

| Variable | Uso |
|---|---|
| `DATABASE_URL` | URL MySQL de Railway (ej: `mysql://user:pass@host:3306/db`) |
| `BETO_ENABLED` | `true`/`false` — deshabilitar si el servidor tiene < 2GB RAM |
| `PORT` | Puerto (Railway lo asigna automáticamente) |

## Comandos

```bash
# Instalar dependencias
pip install -r requirements.txt

# Arrancar en local (desde la raíz del proyecto)
uvicorn app.main:app --reload --port 8000

# La app abre en: http://localhost:8000
```

## Comportamientos clave

- **BETO se carga en background** al arrancar el servidor. El primer arranque descarga ~500MB de HuggingFace (se cachea localmente). En Railway, esto ocurre en cada deploy.
- **Si BETO falla** (poca RAM, error de red), la tarjeta BETO muestra "No disponible" y TF-IDF sigue funcionando.
- **Logo:** el archivo `app/static/logo_ue.png` debe colocarse manualmente (no está en el repo por tamaño).
- **Versión sklearn:** los modelos fueron entrenados con sklearn 1.6.1 — usar esa versión exacta en requirements.txt.
- **MySQL en Railway:** la URL que proporciona Railway empieza por `mysql://`; database.py la convierte a `mysql+pymysql://` automáticamente.

## Estado (mayo 2026)

Primera versión funcional. Pendiente: despliegue en Railway con la URL de DB real.

## Reglas heredadas del global

- No tocar lo que ya funciona. Cambios mínimos por bug.
- Interfaz en español. Tildes obligatorias.
- Diagnosticar antes de optimizar.
