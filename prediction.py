# SMART CANTEEN DEMAND PREDICTOR
# Module: prediction.py
# Purpose: Real-time inference engine, safety buffer calculation, and database persistence.
# Technology: 100% Python (joblib, pandas, scikit-learn).

import os
import joblib
import pandas as pd
from datetime import datetime
from database import save_prediction

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'demand_model.pkl')

_CACHED_MODEL_PAYLOAD = None

def load_prediction_model(model_path=MODEL_PATH):
    """Loads and caches the serialized model artifact."""
    global _CACHED_MODEL_PAYLOAD
    if _CACHED_MODEL_PAYLOAD is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model artifact not found at: {model_path}. Please run train_model.py first.')
        _CACHED_MODEL_PAYLOAD = joblib.load(model_path)
    return _CACHED_MODEL_PAYLOAD

def get_benchmark_results():
    """Returns the benchmark dataframe of trained models."""
    payload = load_prediction_model()
    return payload.get('benchmark_table', pd.DataFrame()), payload.get('model_name', 'Unknown')

def predict_canteen_demand(input_dict, safety_margin_pct=6.0, save_to_db=True):
    """
    Takes single prediction input features as a dict, validates values,
    executes machine learning regression inference, calculates safety margin,
    and returns comprehensive actionable metrics.
    """
    payload = load_prediction_model()
    pipeline = payload['pipeline']
    
    # Input validation & sanitization
    food_item = str(input_dict.get('Food_Item', 'Samosa'))
    category = str(input_dict.get('Category', 'Snacks'))
    weather = str(input_dict.get('Weather', 'Sunny'))
    day = str(input_dict.get('Day', 'Monday'))
    exam_day = str(input_dict.get('Exam_Day', 'No'))
    holiday = str(input_dict.get('Holiday', 'No'))
    weekend = str(input_dict.get('Weekend', 'No'))
    
    students_present = max(0, int(input_dict.get('Students_Present', 850)))
    temperature = float(input_dict.get('Temperature', 30.0))
    food_price = max(1.0, float(input_dict.get('Food_Price', 20.0)))
    previous_day_sales = max(0, int(input_dict.get('Previous_Day_Sales', 165)))
    previous_week_sales = max(0, int(input_dict.get('Previous_Week_Sales', 150)))
    current_inventory = max(0, int(input_dict.get('Current_Inventory', 100)))
    prediction_date = str(input_dict.get('Date', datetime.now().strftime('%Y-%m-%d')))
    
    # Construct single-row DataFrame matching training schema
    row_df = pd.DataFrame([{
        'Food_Item': food_item,
        'Category': category,
        'Weather': weather,
        'Day': day,
        'Exam_Day': exam_day,
        'Holiday': holiday,
        'Weekend': weekend,
        'Students_Present': students_present,
        'Temperature': temperature,
        'Food_Price': food_price,
        'Previous_Day_Sales': previous_day_sales,
        'Previous_Week_Sales': previous_week_sales,
        'Current_Inventory': current_inventory
    }])
    
    # Run Machine Learning Pipeline Inference
    raw_pred = pipeline.predict(row_df)[0]
    predicted_demand = max(0, int(round(raw_pred)))
    
    # Calculate recommended preparation with safety margin buffer
    margin_ratio = max(0.0, float(safety_margin_pct) / 100.0)
    recommended_prep = int(round(predicted_demand * (1.0 + margin_ratio)))
    safety_units = max(0, recommended_prep - predicted_demand)
    
    # Calculate shortage deficit / additional units required
    additional_required = max(0, recommended_prep - current_inventory)
    
    # Determine Inventory Health Status
    if current_inventory < (predicted_demand * 0.5):
        status = 'CRITICAL LOW STOCK'
        status_color = 'red'
    elif current_inventory < predicted_demand:
        status = 'LOW STOCK'
        status_color = 'orange'
    elif current_inventory > (predicted_demand * 1.35):
        status = 'OVERSTOCK RISK'
        status_color = 'purple'
    else:
        status = 'OPTIMAL'
        status_color = 'green'
        
    # Generate intelligent recommendation advice
    if additional_required > 0:
        recommendation_text = (
            f"Prepare approximately {recommended_prep} units (predicted demand {predicted_demand} + "
            f"{safety_units} buffer units). Additional {additional_required} units required to cover expected demand."
        )
    else:
        surplus = current_inventory - recommended_prep
        recommendation_text = (
            f"Current inventory ({current_inventory} units) is sufficient for predicted demand ({predicted_demand} units). "
            f"Expected surplus of {surplus} units. No urgent preparation needed."
        )
        
    result = {
        'food_item': food_item,
        'category': category,
        'prediction_date': prediction_date,
        'students_present': students_present,
        'weather': weather,
        'temperature': temperature,
        'exam_day': exam_day,
        'holiday': holiday,
        'weekend': weekend,
        'food_price': food_price,
        'previous_day_sales': previous_day_sales,
        'previous_week_sales': previous_week_sales,
        'current_inventory': current_inventory,
        'predicted_demand': predicted_demand,
        'recommended_prep': recommended_prep,
        'safety_margin_units': safety_units,
        'safety_margin_pct': float(safety_margin_pct),
        'additional_required': additional_required,
        'status': status,
        'status_color': status_color,
        'recommendation_text': recommendation_text,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_used': payload.get('model_name', 'Random Forest Regressor')
    }
    
    # Persist prediction in SQLite log
    if save_to_db:
        save_prediction(result)
        
    return result

if __name__ == '__main__':
    sample_input = {
        'Food_Item': 'Samosa',
        'Category': 'Snacks',
        'Date': '2026-09-12',
        'Day': 'Saturday',
        'Students_Present': 850,
        'Weather': 'Sunny',
        'Temperature': 30.0,
        'Exam_Day': 'No',
        'Holiday': 'No',
        'Weekend': 'No',
        'Food_Price': 20.0,
        'Previous_Day_Sales': 165,
        'Previous_Week_Sales': 150,
        'Current_Inventory': 100
    }
    res = predict_canteen_demand(sample_input, safety_margin_pct=6.0)
    print('[PREDICTION RESULT]:')
    for k, v in res.items():
        print(f'  {k}: {v}')
