# SMART CANTEEN DEMAND PREDICTOR
# Module: train_model.py
# Purpose: Train and compare multiple ML regression models, evaluate metrics, and serialize the best model.
# Technology: 100% Python (scikit-learn, pandas, numpy, joblib).

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import get_train_test_split, get_preprocessor, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def train_and_evaluate_models(csv_path='data/canteen_sales.csv', model_save_path='models/demand_model.pkl'):
    """
    Trains Linear Regression, Decision Tree, and Random Forest models.
    Evaluates MAE, MSE, RMSE, and R2 on the test set.
    Selects the best performing model and exports it using joblib.
    """
    print('=' * 75)
    print('           SMART CANTEEN DEMAND PREDICTOR - MODEL TRAINING & EVALUATION')
    print('=' * 75)
    
    # 1. Split data
    X_train, X_test, y_train, y_test = get_train_test_split(csv_path)
    print(f'Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}')
    
    # 2. Define Candidate Models
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=12, min_samples_split=4, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_split=3, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_pipelines = {}
    
    print('\n[TRAINING] Fitting models on training dataset...')
    for model_name, model_estimator in candidate_models.items():
        preprocessor = get_preprocessor()
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_estimator)
        ])
        
        # Fit pipeline
        pipe.fit(X_train, y_train)
        trained_pipelines[model_name] = pipe
        
        # Predict on test set
        y_pred = pipe.predict(X_test)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'MAE': round(mae, 3),
            'MSE': round(mse, 3),
            'RMSE': round(rmse, 3),
            'R2_Score': round(r2, 4)
        })
        
    results_df = pd.DataFrame(results)
    
    print('\n' + '=' * 75)
    print('                           MODEL BENCHMARK COMPARISON')
    print('=' * 75)
    print(results_df.to_string(index=False))
    print('=' * 75)
    
    # 3. Select Best Model based on highest R2 score and lowest RMSE
    best_row = results_df.sort_values(by=['R2_Score', 'RMSE'], ascending=[False, True]).iloc[0]
    best_model_name = best_row['Model']
    best_pipeline = trained_pipelines[best_model_name]
    
    print(f'\n>>> WINNING MODEL SELECTED: {best_model_name}')
    print(f'>>> Test R² Score: {best_row["R2_Score"]} | MAE: {best_row["MAE"]} | RMSE: {best_row["RMSE"]}')
    
    # 4. Save best pipeline, benchmark results, and metadata
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    artifact_payload = {
        'model_name': best_model_name,
        'pipeline': best_pipeline,
        'benchmark_table': results_df,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'best_metrics': best_row.to_dict()
    }
    
    joblib.dump(artifact_payload, model_save_path)
    print(f'[SAVED] Best model pipeline successfully saved to: {model_save_path}\n')
    
    return results_df, artifact_payload

if __name__ == '__main__':
    train_and_evaluate_models()
