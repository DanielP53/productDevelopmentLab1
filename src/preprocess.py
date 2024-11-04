# src/preprocess.py
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
import yaml

def preprocess(input_file, output_file, features, target):
    df = pd.read_csv(input_file)

    df = df.dropna()

    columns = features + [target]
    df = df[columns]

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled[target] = y

    df_scaled.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Datos normalizados guardados en {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)
