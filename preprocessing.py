# SMART CANTEEN DEMAND PREDICTOR
# Module: preprocessing.py
# Purpose: Feature engineering, data cleaning, and scikit-learn preprocessing pipeline.
# Technology: 100% Python (pandas, scikit-learn).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = [
    'Food_Item', 'Category', 'Weather', 'Day', 'Exam_Day', 'Holiday', 'Weekend'
]

NUMERICAL_FEATURES = [
    'Students_Present', 'Temperature', 'Food_Price', 
    'Previous_Day_Sales', 'Previous_Week_Sales', 'Current_Inventory'
]

TARGET_FEATURE = 'Quantity_Sold'

def get_preprocessor():
    """Constructs a scikit-learn ColumnTransformer for categorical and numerical features."""
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, CATEGORICAL_FEATURES),
            ('num', num_transformer, NUMERICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_data(df):
    """
    Cleans DataFrame, ensures correct data types, and extracts feature matrix X and target y.
    """
    # Create clean copy
    data = df.copy()
    
    # Drop duplicates if any
    data = data.drop_duplicates()
    
    # Handle missing values
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].fillna('Unknown')
    for num_col in NUMERICAL_FEATURES + [TARGET_FEATURE]:
        if num_col in data.columns:
            data[num_col] = pd.to_numeric(data[num_col], errors='coerce').fillna(data[num_col].median())
            
    # Ensure types
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(str)
        
    for col in NUMERICAL_FEATURES:
        data[col] = data[col].astype(float)
        
    X = data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = data[TARGET_FEATURE].astype(float)
    
    return X, y

def get_train_test_split(csv_path='data/canteen_sales.csv', test_size=0.2, random_state=42):
    """Loads CSV dataset, prepares features, and performs train-test split."""
    df = pd.read_csv(csv_path)
    X, y = prepare_data(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f'[PREPROCESSING] Train set shape: {X_train.shape}, Test set shape: {X_test.shape}')
