# 2025 NASA Space Apps Challenge A World Away: Hunting for Exoplanets with AI

## Integrantes 

1. Juan Mario Sosa Romo - juan.mario.sosa.p@gmail.com

***

Este repositorio contiene una solución mínima y reproducible para entrenar y usar modelos de detección de exoplanetas (AutoGluon) sobre la BDD KOI/Kepler. Incluye scripts para preparar datos, una pequeña UI con Streamlit para predecir y otra para reentrenar un predictor.

A continuación tienes qué hay, cómo desplegarlo y qué hace cada cosa.

***

## Estructura del proyecto

```
.
├── Back
│   ├── AutogluonModels/           # modelos previamente entrenados (ej.: ag-20251004_015342)
│   └── Exoplanet-Detection-with-ML.ipynb
├── Data
│   ├── 1_cumulative_2025.csv      # dataset principal (KOI cumulative)
│   ├── 2_TOI_2025.csv
│   └── 3_k2pandc_2025.csv
├── Front
│   ├── app_streamlit_min.py       # app Streamlit mínima (demo predictor)
│   ├── create_test_from_csv.py    # genera test_input_real.csv (transformaciones + features)
│   ├── pagina_predictor.py        # página predictor (transformaciones + predict UI)
│   ├── pagina_modelo.py           # página para re-entrenar un predictor (UI)
│   ├── transform_input.py         # lógica reutilizable para transformar CSV -> features del predictor
│   ├── test_input_real.csv        # CSV de ejemplo ya preparado
│   ├── preds_test.csv             # ejemplo de salida de predict
│   └── static/ etc.
├── requirements.txt
└── README.md
```
***

## Requisitos

- Linux 
- Python 3.9+
- 8 Gb de Ram 
- Venv

## Pasos

1.  **Clona el repositorio:**
    ```bash
    git clone git@github.com:JSR-Mario/NASA.git 
    cd NASA
    ```

2.  **Crea y activa un entorno virtual:**
    Esto aísla las dependencias del proyecto para evitar conflictos con otros proyectos o con las librerías del sistema.

    ```bash
    # Crea el entorno virtual
    python3 -m venv .venv

    # Activa el entorno (Linux/macOS)
    source .venv/bin/activate
    ```

3.  **Instala las dependencias:**
    El archivo `requirements.txt` contiene todas las librerías de Python necesarias.

    ```bash
    pip install -r requirements.txt
    ```

4. **Muevete al Front y ejecuta:**
   Esto va a levantar un servidor en local (http://localhost:8501) para que puedas interactuar.
    ```bash
    streamlit run app_streamlit_min.py
    ```
