import os
import pandas as pd
import joblib
import sys
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train(input_file, output_dir, params_file):
    df = pd.read_csv(input_file)

    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = {
        'linear_regression': LinearRegression(**params['train']['models']['linear_regression']),
        'random_forest': RandomForestRegressor(**params['train']['models']['random_forest']),
        'gradient_boosting': GradientBoostingRegressor(**params['train']['models']['gradient_boosting'])
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        model_path = f"{output_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Modelo {model_name} entrenado y guardado en {model_path}")

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results[model_name] = mse
        print(f"{model_name} MSE: {mse}")

    with open(f"{output_dir}/results.txt", 'w') as f:
        for model_name, mse in results.items():
            f.write(f"{model_name}: MSE = {mse}\n")


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, output_dir, params_file)
