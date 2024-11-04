# src/preprocess.py
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
import yaml

def preprocess(input_file, output_file, features, target):
    # Cargar el dataset
    df = pd.read_csv(input_file)

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Seleccionar solo las columnas necesarias
    columns = features + [target]
    df = df[columns]

    # Separar características y variable objetivo
    X = df[features]
    y = df[target]

    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Crear un DataFrame con las características normalizadas
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled[target] = y

    # Guardar el dataset limpio y normalizado
    df_scaled.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Datos normalizados guardados en {output_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo de salida, archivo de parámetros
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    # Leer parámetros desde params.yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)
