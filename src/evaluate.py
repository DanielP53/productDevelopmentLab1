# src/evaluate.py
import os
import pandas as pd
import joblib
import json
import sys
import yaml
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(input_file, models_dir, metrics_file, params_file, output_csv, output_md):
    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Leer los parámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Separar características y variable objetivo
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    # Lista para almacenar las métricas de todos los modelos
    metrics_list = []

    # Evaluar cada modelo en la carpeta de modelos
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)

            # Realizar predicciones
            predictions = model.predict(X)

            # Calcular métricas de rendimiento
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            model_name = model_file.replace('.pkl', '')

            # Imprimir las métricas en la consola
            print(f"{model_name} - MSE: {mse}, R2: {r2}")

            # Añadir las métricas a la lista
            metrics_list.append({
                'model': model_name,
                'MSE': mse,
                'R2-Score': r2
            })

    # Convertir la lista de métricas a un DataFrame de pandas
    metrics_df = pd.DataFrame(metrics_list)

    # Guardar todas las métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    print(f"Métricas exportadas a {metrics_file}")

    # Exportar las métricas a un archivo CSV
    metrics_df.to_csv(output_csv, index=False)
    print(f"Métricas exportadas a {output_csv}")

    # Exportar las métricas a un archivo Markdown
    metrics_df.to_markdown(output_md, index=False)
    print(f"Métricas exportadas a {output_md}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, directorio de modelos, archivo de métricas, archivo de parámetros, archivo de salida CSV, archivo de salida Markdown
    input_file = sys.argv[1]
    models_dir = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]
    output_csv = sys.argv[5]
    output_md = sys.argv[6]

    evaluate(input_file, models_dir, metrics_file, params_file, output_csv, output_md)
