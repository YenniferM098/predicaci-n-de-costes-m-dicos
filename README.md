# Predicción de costes médicos anuales

Aplicación Flask que utiliza un pipeline de Gradient Boosting para estimar el
coste médico anual.

## Ejecución local

```bash
pip install -r requirements.txt
python app.py
```

Abrir: http://127.0.0.1:5000

## Configuración en Render

- Runtime: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

## Variables solicitadas

- insurance_coverage_pct
- hospital_admissions
- medication_count
- heart_disease
- previous_year_cost
- smoker


## Corrección de predicciones negativas

La aplicación valida los datos utilizando los rangos observados en el dataset.
Además, cuando el modelo genera una estimación inferior al coste mínimo
observado, la salida se ajusta a 404.95 USD. Esta restricción evita mostrar
costes negativos, que no tienen interpretación válida en el problema.


## Mejoras del formulario

- El campo de coste del año anterior indica que debe escribirse sin comas.
- Se agregó un botón para limpiar todos los campos y el resultado mostrado.
