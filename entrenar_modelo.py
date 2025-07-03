import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime

print("🌿 === SISTEMA DE ENTRENAMIENTO DE MODELO DE RIEGO CON DATOS REALES ===")

def obtener_datos_meteo():
    """
    Obtiene datos climáticos reales de la API Open-Meteo por hora para el día actual en Huánuco.
    """
    print("📡 Obteniendo datos reales desde Open-Meteo...")

    lat = -9.93
    lon = -76.24
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation&timezone=America/Lima"

    try:
        response = requests.get(url)
        data = response.json()

        horas = data['hourly']['time']
        temps = data['hourly']['temperature_2m']
        humedades = data['hourly']['relative_humidity_2m']
        precipitaciones = data['hourly']['precipitation']

        registros = []
        for i in range(len(horas)):
            fecha = horas[i][:10]  # Extraer solo la fecha
            temp = temps[i]
            humedad = humedades[i]
            precip = precipitaciones[i]

            # Fórmula simple de riego estimado
            factor_temp = (temp - 15) / 10
            factor_humedad = (100 - humedad) / 100
            factor_precipitacion = max(0, 1 - precip / 10)

            riego_base = 2.5
            riego_estimado = riego_base * (1 + factor_temp * 0.8 + factor_humedad * 0.6 + factor_precipitacion * 0.4)
            riego_estimado = max(0.5, riego_estimado + np.random.normal(0, 0.3))

            registros.append({
                'fecha': fecha,
                'temp_media': round(temp, 1),
                'humedad_media': round(humedad, 1),
                'precipitacion': round(precip, 1),
                'riego_estimado': round(riego_estimado, 2)
            })

        df = pd.DataFrame(registros)
        df.to_csv("riego_historico_huanuco.csv", index=False)
        print(f"✅ Datos reales guardados en 'riego_historico_huanuco.csv'")
        return df

    except Exception as e:
        print(f"❌ Error al obtener datos de la API: {e}")
        return None

def entrenar_modelo(df):
    print("🤖 Entrenando modelo...")

    columnas = ['temp_media', 'humedad_media', 'precipitacion', 'riego_estimado']
    df_clean = df[columnas].dropna()

    if len(df_clean) < 10:
        print("❌ No hay suficientes datos.")
        return None

    X = df_clean[['temp_media', 'humedad_media', 'precipitacion']]
    y = df_clean['riego_estimado']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"🎯 R²: {r2:.3f}")

    joblib.dump(modelo, "modelo_riego.pkl")
    print("✅ Modelo guardado como 'modelo_riego.pkl'")
    return modelo

def main():
    df = obtener_datos_meteo()
    if df is None:
        print("❌ No se pudo obtener datos.")
        return

    modelo = entrenar_modelo(df)
    if modelo is None:
        return

    # Verificación rápida
    ejemplo = pd.DataFrame({
        'temp_media': [20.0],
        'humedad_media': [65.0],
        'precipitacion': [3.0]
    })
    pred = modelo.predict(ejemplo)[0]
    print(f"🔍 Predicción de prueba: {pred:.2f} L/m²")

    print("🚀 Listo para usar con Streamlit")

if __name__ == "__main__":
    main()

