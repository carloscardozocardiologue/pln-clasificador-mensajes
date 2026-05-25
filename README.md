# Clasificador de Mensajes Clínicos
### TF-IDF vs BETO · Validación clínica en tiempo real

**Actividad académica — Unidad 6**  
Máster Universitario en Inteligencia Artificial en Ciencias de la Salud  
Universidad Europea de Madrid · Mayo 2026  
**Autor:** Dr. Carlos Cardozo

---

## ¿Qué hace esta aplicación?

Esta aplicación web permite clasificar mensajes escritos por pacientes en tres niveles de prioridad clínica:

| Nivel | Color | Descripción |
|---|---|---|
| **Leve** | 🟢 Verde | Síntoma no urgente, puede esperar |
| **Moderada** | 🟡 Amarillo | Requiere atención en horas |
| **Grave** | 🔴 Rojo | Urgencia vital, atención inmediata |

El mismo mensaje es clasificado simultáneamente por **dos modelos de PLN** entrenados sobre un corpus de 2.000 mensajes clínicos simulados, permitiendo comparar su comportamiento en condiciones reales.

---

## Modelos comparados

### TF-IDF + Regresión Logística
- Vectorización con unigramas y bigramas (`ngram_range=(1,2)`)
- Lista de stopwords reducida que **conserva negaciones** (`no`, `sin`, `nunca`) — fundamentales para detectar urgencias como *"no puede respirar"*
- F1-macro en test interno: **0.99** · F1-macro OOD (mensajes nuevos): **0.87**

### BETO + Regresión Logística
- [BETO](https://github.com/dccuchile/beto) — BERT preentrenado en español (`dccuchile/bert-base-spanish-wwm-uncased`)
- Usado como extractor de características congelado (mean pooling, 768 dimensiones)
- F1-macro en test interno: **0.93** · F1-macro OOD: **0.74**

> El modelo TF-IDF supera a BETO en este escenario porque el dataset sintético es separable por palabras clave. El verdadero potencial de BETO requeriría fine-tuning sobre datos clínicos reales.

---

## Arquitectura

```
┌─────────────────────────────────────────┐
│            Navegador (Frontend)          │
│  HTML + CSS + JavaScript vanilla        │
│  Web Speech API (entrada por voz)       │
│  Chart.js (gráficas de métricas)        │
└────────────────┬────────────────────────┘
                 │ HTTP / REST API
┌────────────────▼────────────────────────┐
│            Backend (FastAPI)            │
│  POST /api/clasificar  → ambos modelos  │
│  PUT  /api/mensajes/{id}/opinion        │
│  GET  /api/estadisticas                 │
│  GET  /api/mensajes/csv                 │
└──────────┬──────────────┬───────────────┘
           │              │
┌──────────▼───┐  ┌───────▼──────────────┐
│  TF-IDF      │  │  BETO                │
│  Pipeline    │  │  (HuggingFace +      │
│  sklearn     │  │   LogReg guardado)   │
└──────────────┘  └──────────────────────┘
           │
┌──────────▼──────────────────────────────┐
│         MySQL · Railway                 │
│  Tabla: mensajes                        │
│  (predicciones + valoración médico)     │
└─────────────────────────────────────────┘
```

---

## Funcionalidades

### Pestaña — Asistente de Mensajes Clínicos
- **Entrada de texto** o **dictado por voz** (Web Speech API, español)
- Clasificación instantánea con los dos modelos en paralelo
- Badge de color con nivel de prioridad y porcentaje de confianza
- Barras de probabilidad para las tres clases
- Palabras clave detectadas (interpretabilidad TF-IDF)
- Mensaje más similar en la base de datos (similitud coseno)
- **Valoración del médico**: el clínico anota la prioridad real y sus observaciones

### Pestaña — Estadísticas
- Precisión global de cada modelo (solo sobre mensajes valorados por el médico)
- Precisión por clase: leve / moderada / grave
- Registro completo de mensajes con filtros de edición y eliminación
- **Exportación CSV** con todas las columnas para análisis externo

---

## Estructura del proyecto

```
pln-clasificador-mensajes/
├── app/
│   ├── main.py          # FastAPI — rutas y lógica principal
│   ├── classifier.py    # Carga de modelos, inferencia, similitud
│   ├── database.py      # Conexión SQLAlchemy (MySQL / SQLite)
│   ├── models.py        # Modelo ORM tabla mensajes
│   └── static/
│       ├── index.html   # Interfaz completa (dos pestañas)
│       ├── style.css    # Estilos
│       ├── app.js       # Lógica frontend
│       └── logo_ue.png  # Logo Universidad Europea
├── modelos/
│   ├── modelo_tfidf.joblib   # Pipeline TF-IDF + LogReg (sklearn 1.6.1)
│   └── modelo_beto.joblib    # LogReg + nombre encoder (HuggingFace)
├── requirements.txt
├── Procfile             # Para Railway: uvicorn app.main:app
└── .env.example         # Plantilla de variables de entorno
```

---

## Instalación local

### Requisitos
- Python 3.9+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/carloscardozocardiologue/pln-clasificador-mensajes.git
cd pln-clasificador-mensajes

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con la URL de tu base de datos MySQL
# Si no tienes MySQL, la app usa SQLite automáticamente en local

# 4. Arrancar
uvicorn app.main:app --reload --port 8000
```

Abre el navegador en **http://localhost:8000**

> **Nota:** el primer arranque descarga el modelo BETO desde HuggingFace (~500 MB). Los siguientes arranques usan la caché local.

---

## Despliegue en Railway

1. Fork o clona este repositorio en tu cuenta de GitHub
2. En [Railway](https://railway.app) → tu proyecto con el plugin MySQL → **New Service → GitHub Repo**
3. Selecciona este repositorio
4. En **Variables** del nuevo servicio, añade:

```
DATABASE_URL   →  (copia el valor del plugin MySQL de Railway)
BETO_ENABLED   →  true
```

5. Railway detecta el `Procfile` automáticamente y lanza el deploy

---

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `DATABASE_URL` | URL MySQL de Railway | `sqlite:///./local.db` |
| `BETO_ENABLED` | Activar modelo BETO | `true` |
| `PORT` | Puerto de la aplicación | `8000` |

---

## Contexto académico

Este proyecto forma parte de la **Actividad Evaluable 1 de la Unidad 6** del Máster en IA en Ciencias de la Salud (Universidad Europea de Madrid).

El objetivo es diseñar, entrenar y evaluar críticamente un modelo de PLN que clasifique mensajes de pacientes en niveles de prioridad clínica. La aplicación va más allá del requisito mínimo al implementar una **validación clínica en vivo**: un médico especialista (cardiólogo intervencionista) anota en tiempo real la prioridad real de cada mensaje, generando métricas de rendimiento sobre datos reales — no sintéticos.

### Hallazgos principales
- TF-IDF supera a BETO en este escenario de dataset pequeño y sintético
- El principal riesgo clínico identificado es el **infra-triaje** (clasificar como leve un caso grave)
- Casos como *"pierna muy hinchada después de un viaje largo"* (TVP → riesgo de TEP) son clasificados como moderados por ambos modelos, evidenciando la limitación del razonamiento puramente léxico

---

## Aviso

> ⚠️ Esta herramienta ha sido desarrollada con **fines exclusivamente educativos**.  
> **No es un dispositivo médico** y no debe utilizarse para diagnóstico clínico.  
> Cualquier decisión asistencial debe tomarse siempre por un profesional sanitario cualificado.

---

## Referencias

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- Cañete et al. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data (BETO)*. PML4DC, ICLR.
- Manning, Raghavan & Schütze (2008). *Introduction to Information Retrieval*. Cambridge University Press.
