# Documentación Técnica — Clasificador de Mensajes Clínicos
**Dr. Carlos CARDOZO · Cardiólogo · Exp. 225K3417**  
Máster Universitario en Inteligencia Artificial en Ciencias de la Salud  
Universidad Europea de Madrid · Mayo 2026

---

## 1. Descripción general

Se ha desarrollado una aplicación web que permite clasificar en tiempo real mensajes escritos por pacientes en tres niveles de prioridad clínica: **Leve**, **Moderada** y **Grave**. El mismo mensaje es analizado simultáneamente por dos modelos de PLN entrenados previamente, lo que permite comparar su comportamiento sobre datos reales anotados por un médico especialista.

El objetivo doble es:
- **Académico:** demostrar el ciclo completo de PLN (entrenamiento → evaluación → interpretabilidad → validación clínica)
- **Clínico:** generar evidencia real sobre el rendimiento de cada modelo ante mensajes auténticos de pacientes, más allá del dataset sintético de entrenamiento

---

## 2. Stack tecnológico

| Componente | Tecnología |
|---|---|
| Backend / API | Python 3.11 · FastAPI · Uvicorn |
| Base de datos | MySQL (Railway) · SQLAlchemy · PyMySQL |
| Modelo TF-IDF | scikit-learn 1.6.1 (Pipeline: TfidfVectorizer + LogisticRegression) |
| Modelo BETO | HuggingFace Transformers · BETO (`dccuchile/bert-base-spanish-wwm-uncased`) · LogisticRegression |
| Frontend | HTML5 · CSS3 · JavaScript vanilla · Chart.js |
| Entrada de voz | Web Speech API (nativa del navegador) |
| Despliegue | Railway (backend + MySQL) · GitHub (repositorio) |

---

## 3. Arquitectura de la aplicación

```
┌──────────────────────────────────────────────────────┐
│                  NAVEGADOR (Cliente)                  │
│                                                       │
│  ┌─────────────────────┐  ┌─────────────────────┐    │
│  │  Pestaña Asistente  │  │  Pestaña Estadísticas│    │
│  │  - Input texto/voz  │  │  - Métricas en vivo  │    │
│  │  - Resultados TF-IDF│  │  - Gráficas Chart.js │    │
│  │  - Resultados BETO  │  │  - Tabla de mensajes │    │
│  │  - Valoración médico│  │  - Exportar CSV      │    │
│  └──────────┬──────────┘  └──────────┬───────────┘   │
│             │ REST API (fetch)        │               │
└─────────────┼────────────────────────┼───────────────┘
              │                        │
┌─────────────▼────────────────────────▼───────────────┐
│                  FASTAPI (Backend)                     │
│                                                       │
│  POST /api/clasificar      → inferencia ambos modelos │
│  PUT  /api/mensajes/{id}/opinion → guardar valoración │
│  GET  /api/mensajes        → listar registro          │
│  GET  /api/mensajes/csv    → exportar CSV             │
│  GET  /api/estadisticas    → métricas agregadas       │
│  DELETE /api/mensajes/{id} → eliminar mensaje         │
│  GET  /health              → estado del servidor      │
└───────────┬─────────────────────────┬─────────────────┘
            │                         │
┌───────────▼──────────┐  ┌───────────▼──────────────────┐
│  MODELO TF-IDF        │  │  MODELO BETO                  │
│                       │  │                               │
│  Pipeline sklearn:    │  │  1. Carga encoder HuggingFace │
│  TfidfVectorizer      │  │     (bert-base-spanish)       │
│  + LogisticRegression │  │  2. Mean pooling → embedding  │
│                       │  │     768 dimensiones           │
│  Archivo:             │  │  3. LogisticRegression        │
│  modelo_tfidf.joblib  │  │                               │
│                       │  │  Archivo: modelo_beto.joblib  │
└───────────────────────┘  └───────────────────────────────┘
            │
┌───────────▼───────────────────────────────────────────┐
│              MySQL · Railway (nube)                    │
│                                                       │
│  Tabla: mensajes                                      │
│  id · texto · pred_tfidf · conf_tfidf                 │
│  pred_beto · conf_beto · opinion_medico               │
│  tfidf_correcto · beto_correcto · notas · creado_en   │
└───────────────────────────────────────────────────────┘
```

---

## 4. Modelos de PLN

### 4.1 TF-IDF + Regresión Logística

**Archivo:** `modelos/modelo_tfidf.joblib`  
**Formato:** Pipeline completo de scikit-learn (vectorizador + clasificador)

Parámetros del vectorizador:
- `ngram_range = (1, 2)` — unigramas y bigramas
- `min_df = 2` — términos que aparecen en al menos 2 documentos
- `sublinear_tf = True` — suavizado logarítmico de frecuencias
- `lowercase = True` — conversión automática a minúsculas
- Stopwords personalizadas: lista reducida que **conserva negaciones** (`no`, `sin`, `nunca`) e intensificadores (`muy`, `mucho`, `insoportable`) — términos críticos para la detección de urgencias

Rendimiento (dataset sintético aumentado, 2.000 mensajes):
| Conjunto | Accuracy | F1-macro |
|---|---|---|
| Test interno (mismo corpus) | 0.99 | 0.99 |
| OOD — mensajes nuevos parafraseados | 0.87 | 0.87 |

**Interpretabilidad:** los coeficientes de la Regresión Logística son directamente legibles. Los términos de mayor peso para la clase `grave` incluyen: `no`, `insoportable`, `ahogo`, `sangre`, `infarto`. La negación `no` es el término de mayor peso individual para detectar urgencias.

### 4.2 BETO + Regresión Logística

**Archivo:** `modelos/modelo_beto.joblib`  
**Formato:** diccionario con `{'clf': LogisticRegression, 'encoder_name': str, 'labels': list}`

BETO es el modelo BERT preentrenado en español desarrollado por el Departamento de Ciencias de la Computación de la Universidad de Chile (`dccuchile/bert-base-spanish-wwm-uncased`). Se usa como **extractor de características congelado** (sin fine-tuning), calculando el embedding de cada mensaje mediante *mean pooling* sobre los vectores de los tokens (768 dimensiones). Ese embedding se alimenta a la Regresión Logística guardada en el archivo joblib.

Rendimiento:
| Conjunto | Accuracy | F1-macro |
|---|---|---|
| Test interno | 0.93 | 0.93 |
| OOD — mensajes nuevos parafraseados | 0.73 | 0.74 |

**Observación:** TF-IDF supera a BETO en este escenario porque el dataset sintético es separable por palabras clave concretas. El verdadero potencial de BETO requeriría fine-tuning sobre datos clínicos reales y variados.

---

## 5. Base de datos

**Motor:** MySQL 8 (plugin Railway)  
**ORM:** SQLAlchemy 2.0

### Esquema — tabla `mensajes`

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | INT AUTO_INCREMENT PK | Identificador interno |
| `texto` | TEXT | Mensaje original del paciente |
| `pred_tfidf` | VARCHAR(20) | Predicción TF-IDF: `leve`, `moderada` o `grave` |
| `conf_tfidf` | FLOAT | Probabilidad de la clase predicha (0–1) |
| `pred_beto` | VARCHAR(20) | Predicción BETO (puede ser NULL si BETO no está disponible) |
| `conf_beto` | FLOAT | Probabilidad BETO |
| `opinion_medico` | VARCHAR(20) | Prioridad real anotada por el médico |
| `notas` | TEXT | Comentario clínico del médico |
| `tfidf_correcto` | BOOLEAN | `true` si pred_tfidf == opinion_medico |
| `beto_correcto` | BOOLEAN | `true` si pred_beto == opinion_medico |
| `creado_en` | DATETIME | Fecha y hora del registro |

La tabla se crea automáticamente al arrancar la aplicación si no existe (`Base.metadata.create_all`).

---

## 6. API — endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/clasificar` | Clasifica un mensaje con ambos modelos, guarda en BD, devuelve predicciones + keywords + mensaje más cercano |
| `PUT` | `/api/mensajes/{id}/opinion` | Guarda la valoración del médico y calcula `tfidf_correcto` / `beto_correcto` |
| `GET` | `/api/mensajes` | Lista todos los mensajes ordenados por fecha desc |
| `GET` | `/api/mensajes/csv` | Devuelve todos los mensajes como archivo CSV descargable |
| `GET` | `/api/estadisticas` | Accuracy global y por clase (solo mensajes valorados) |
| `DELETE` | `/api/mensajes/{id}` | Elimina un mensaje individual |
| `GET` | `/health` | Estado del servidor y disponibilidad de BETO |

---

## 7. Interfaz de usuario

### Pestaña 1 — Asistente de Mensajes Clínicos

1. **Campo de entrada:** el usuario escribe el mensaje del paciente o lo dicta por voz (botón micrófono con Web Speech API, español)
2. **Clasificar:** llama a `POST /api/clasificar` — en menos de 2 segundos aparecen los resultados
3. **Tarjeta TF-IDF:** badge con color (🟢/🟡/🔴) + porcentaje de confianza + barras de probabilidad de las tres clases + palabras clave detectadas + mensaje más similar en la BD (similitud coseno)
4. **Tarjeta BETO:** misma estructura, sin palabras clave (modelo de caja negra) + mensaje más similar según embeddings
5. **Valoración del médico:** el clínico selecciona la prioridad real de una lista desplegable, añade notas opcionales y pulsa "Guardar valoración" — la BD se actualiza y las métricas se recalculan

### Pestaña 2 — Estadísticas

- **4 tarjetas resumen:** mensajes clasificados, valorados por médico, precisión TF-IDF, precisión BETO
- **Gráfica 1:** comparativa de precisión global entre ambos modelos (barras)
- **Gráfica 2:** precisión por clase (leve / moderada / grave) para cada modelo
- **Tabla de registro:** todos los mensajes con mensaje completo + notas del médico en segunda línea, predicciones con confianza integrada en el badge, valoración médica, correctos/incorrectos, fecha y botones de edición (✏️) y eliminación (🗑)
- **Exportar CSV:** descarga directa del archivo `mensajes_clinicos.csv`

---

## 8. Despliegue en Railway

Railway es una plataforma de despliegue en la nube que permite alojar aplicaciones web y bases de datos con configuración mínima. El proyecto usa dos servicios dentro del mismo proyecto Railway:

### 8.1 Plugin MySQL
- Motor: MySQL 8
- Aloja la tabla `mensajes` con todos los registros de la validación clínica
- Proporciona la variable de entorno `DATABASE_URL` con el formato `mysql://usuario:contraseña@host:puerto/nombre_db`

### 8.2 Servicio web (FastAPI)
- Conectado al repositorio GitHub: `carloscardozocardiologue/pln-clasificador-mensajes`
- Railway detecta el archivo `Procfile` y ejecuta:
  ```
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```
- Python 3.11 fijado mediante el archivo `.python-version` (necesario para compatibilidad de wheels de `tokenizers`)

### 8.3 Variables de entorno configuradas en Railway

| Variable | Valor |
|---|---|
| `DATABASE_URL` | URL MySQL interna del plugin Railway |
| `BETO_ENABLED` | `true` |

### 8.4 Proceso de build

Railway usa **Nixpacks** como sistema de build. Al detectar `requirements.txt`, instala automáticamente todas las dependencias. El primer deploy descarga el modelo BETO desde HuggingFace (~500 MB); los deploys posteriores reutilizan la caché de Nixpacks.

### 8.5 Versiones críticas de dependencias

```
scikit-learn==1.6.1    # debe coincidir con la versión usada para guardar los modelos
torch==2.5.1           # versión compatible con Python 3.11 y Railway
tokenizers>=0.15.0     # wheel precompilado disponible para Python 3.11
transformers==4.41.0
```

---

## 9. Repositorio GitHub

**URL pública:** https://github.com/carloscardozocardiologue/pln-clasificador-mensajes

### Estructura de archivos

```
pln-clasificador-mensajes/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI: rutas, arranque, tablas
│   ├── classifier.py    # Carga de modelos, inferencia, similitud coseno, keywords
│   ├── database.py      # Conexión SQLAlchemy, conversión URL MySQL
│   ├── models.py        # ORM: tabla mensajes
│   └── static/
│       ├── index.html   # Interfaz completa (SPA dos pestañas)
│       ├── style.css    # Estilos (variables CSS, diseño responsive)
│       ├── app.js       # Lógica frontend: tabs, voz, fetch, charts, modal
│       └── logo_ue.png  # Logo Universidad Europea Madrid
├── modelos/
│   ├── modelo_tfidf.joblib   # Pipeline TF-IDF completo
│   └── modelo_beto.joblib    # LogReg + nombre encoder BETO
├── docs/
│   ├── 0MSR001107_UA6_AA1.pdf            # Enunciado de la actividad
│   ├── Memoria_Clasificador_Mensajes_Clinicos.docx  # Memoria académica
│   └── Documentacion_Tecnica.md          # Este documento
├── .env.example         # Plantilla de variables de entorno
├── .gitignore           # Excluye .env, __pycache__, *.db
├── .python-version      # Fija Python 3.11 para Railway
├── CLAUDE.md            # Instrucciones para el asistente de desarrollo
├── Procfile             # Comando de arranque para Railway
├── README.md            # Documentación pública del repositorio
└── requirements.txt     # Dependencias Python
```

---

## 10. Ejecución local

```bash
# Clonar el repositorio
git clone https://github.com/carloscardozocardiologue/pln-clasificador-mensajes.git
cd pln-clasificador-mensajes

# Instalar dependencias (recomendado: entorno virtual)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env: pegar DATABASE_URL de Railway
# Sin DATABASE_URL, la app usa SQLite local automáticamente

# Arrancar
uvicorn app.main:app --reload --port 8000
```

Abrir en el navegador: **http://localhost:8000**

> El primer arranque descarga BETO desde HuggingFace (~500 MB, se cachea localmente en `~/.cache/huggingface`).

---

## 11. Flujo de validación clínica

El flujo diseñado para la validación real del modelo es el siguiente:

```
Médico introduce mensaje del paciente
         ↓
App clasifica con TF-IDF y BETO en paralelo
         ↓
Se muestra: nivel de prioridad + confianza + palabras clave + mensaje similar
         ↓
Médico anota la prioridad real + notas clínicas
         ↓
BD guarda: predicciones + valoración + correcto/incorrecto
         ↓
Métricas se actualizan automáticamente en la pestaña Estadísticas
         ↓
CSV exportable con todos los datos para análisis posterior
```

Este flujo permite obtener métricas de rendimiento real (no de laboratorio) sobre mensajes auténticos redactados espontáneamente, revelando limitaciones que el dataset sintético no detecta — como el caso de *"pierna muy hinchada después de un viaje largo"* (TVP → riesgo de TEP), clasificado como Moderada por ambos modelos cuando la prioridad clínica real es Grave.

---

## 12. Limitaciones identificadas durante la validación

| Limitación | Descripción |
|---|---|
| **Dependencia léxica** | Ambos modelos fallan ante sinónimos o expresiones no vistas en entrenamiento |
| **Infra-triaje** | El error más frecuente y peligroso: clasificar como leve un caso grave |
| **Razonamiento clínico ausente** | Ningún modelo infiere consecuencias (ej. TVP → TEP) |
| **Erratas** | Las faltas de ortografía reducen la confianza de TF-IDF |
| **BETO congelado** | Confianza siempre próxima al 100% (artefacto del LogReg sobre embeddings fijos) |
| **Dataset sintético** | Las métricas de laboratorio (0.99) no reflejan el rendimiento real (0.73–0.87) |
