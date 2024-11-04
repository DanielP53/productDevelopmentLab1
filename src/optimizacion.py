# src/optimizacion.py
import os
import pandas as pd
import joblib
import sys
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer

def hyperparameter_tuning(input_file, output_dir, params_file):
    df = pd.read_csv(input_file)

    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models_and_grids = {
        'random_forest': (RandomForestRegressor(), params['train']['hyperparameters']['random_forest']),
        'gradient_boosting': (GradientBoostingRegressor(), params['train']['hyperparameters']['gradient_boosting'])
    }

    best_models = {}

    for model_name, (model, param_grid) in models_and_grids.items():
        search = GridSearchCV(model, param_grid, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=5)
        search.fit(X, y)

        model_path = f"{output_dir}/{model_name}_best.pkl"
        joblib.dump(search.best_estimator_, model_path)
        best_models[model_name] = search.best_params_
        print(f"Mejor modelo para {model_name} guardado en {model_path}")

    if not os.path.exists('results/'):
        os.makedirs('results/')

    with open('results/best_hyperparameters.json', 'w') as f:
        import json
        json.dump(best_models, f, indent=4)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    params_file = sys.argv[3]

    hyperparameter_tuning(input_file, output_dir, params_file)
