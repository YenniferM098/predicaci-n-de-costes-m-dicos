from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline_costes_medicos_gradient_boosting.joblib"

pipeline = joblib.load(MODEL_PATH)

# Rangos reales observados en el dataset utilizado para entrenar el modelo.
RANGOS = {
    "insurance_coverage_pct": (0, 94),
    "hospital_admissions": (0, 6),
    "medication_count": (0, 7),
    "heart_disease": (0, 1),
    "previous_year_cost": (500, 19996),
}

# El coste médico anual mínimo observado en el dataset fue 404.95 USD.
# Este límite evita mostrar predicciones negativas, que no tienen sentido
# en el contexto del problema.
COSTE_MINIMO_VALIDO = 404.95


def convertir_entero(nombre, valor, minimo, maximo):
    try:
        numero = int(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{nombre} debe ser un número entero.")

    if numero < minimo or numero > maximo:
        raise ValueError(
            f"{nombre} debe estar entre {minimo} y {maximo}, "
            "de acuerdo con el rango observado en el dataset."
        )
    return numero


def convertir_decimal(nombre, valor, minimo, maximo):
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{nombre} debe ser un número válido.")

    if numero < minimo or numero > maximo:
        raise ValueError(
            f"{nombre} debe estar entre {minimo} y {maximo}, "
            "de acuerdo con el rango observado en el dataset."
        )
    return numero


@app.route("/", methods=["GET", "POST"])
def inicio():
    prediccion = None
    prediccion_ajustada = False
    error = None
    datos = {}

    if request.method == "POST":
        datos = request.form.to_dict()

        try:
            cobertura = convertir_decimal(
                "El porcentaje de cobertura",
                datos.get("insurance_coverage_pct"),
                *RANGOS["insurance_coverage_pct"],
            )
            hospitalizaciones = convertir_entero(
                "Los ingresos hospitalarios",
                datos.get("hospital_admissions"),
                *RANGOS["hospital_admissions"],
            )
            medicamentos = convertir_entero(
                "La cantidad de medicamentos",
                datos.get("medication_count"),
                *RANGOS["medication_count"],
            )
            enfermedad_cardiaca = convertir_entero(
                "La enfermedad cardíaca",
                datos.get("heart_disease"),
                *RANGOS["heart_disease"],
            )
            coste_anterior = convertir_decimal(
                "El coste del año anterior",
                datos.get("previous_year_cost"),
                *RANGOS["previous_year_cost"],
            )

            fumador = datos.get("smoker")
            if fumador not in {"Yes", "No"}:
                raise ValueError(
                    "Selecciona una opción válida para la condición de fumador."
                )

            entrada = pd.DataFrame([{
                "insurance_coverage_pct": cobertura,
                "hospital_admissions": hospitalizaciones,
                "medication_count": medicamentos,
                "heart_disease": enfermedad_cardiaca,
                "previous_year_cost": coste_anterior,
                "smoker": fumador,
            }])

            prediccion_original = float(pipeline.predict(entrada)[0])

            # Restricción de dominio: un coste médico no puede ser negativo.
            if prediccion_original < COSTE_MINIMO_VALIDO:
                prediccion = COSTE_MINIMO_VALIDO
                prediccion_ajustada = True
            else:
                prediccion = prediccion_original

        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = (
                "No fue posible generar la predicción. "
                "Verifica los datos e inténtalo nuevamente."
            )

    return render_template(
        "index.html",
        prediccion=prediccion,
        prediccion_ajustada=prediccion_ajustada,
        error=error,
        datos=datos,
    )


@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True)
