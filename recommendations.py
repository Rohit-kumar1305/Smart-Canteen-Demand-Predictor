# SMART CANTEEN DEMAND PREDICTOR
# Module: recommendations.py
# Purpose: Rule-based Smart Advisory Engine for inventory and preparation decisions.
# Technology: 100% Python.

import pandas as pd
import numpy as np

def get_smart_recommendations(result_dict, avg_daily_demand=None):
    """
    Generates actionable smart recommendations based on predicted demand,
    current inventory, and historical patterns. Returns a list of recommendation dicts.
    Each dict has: type (info/warning/error/success), title, message.
    """
    recommendations = []
    
    predicted = result_dict.get('predicted_demand', 0)
    recommended_prep = result_dict.get('recommended_prep', 0)
    current_inventory = result_dict.get('current_inventory', 0)
    weather = result_dict.get('weather', 'Sunny')
    exam_day = result_dict.get('exam_day', 'No')
    holiday = result_dict.get('holiday', 'No')
    weekend = result_dict.get('weekend', 'No')
    food_item = result_dict.get('food_item', 'Item')
    
    # 1. Stock Shortage Alert
    if current_inventory == 0:
        recommendations.append({
            'type': 'error',
            'title': '⛔ OUT OF STOCK',
            'message': f'{food_item} is completely out of stock! Prepare at least {recommended_prep} units immediately.'
        })
    elif current_inventory < (predicted * 0.5):
        recommendations.append({
            'type': 'error',
            'title': '🔴 CRITICAL LOW STOCK',
            'message': f'Only {current_inventory} units available vs predicted demand of {predicted} units. Urgently prepare {recommended_prep - current_inventory} additional units.'
        })
    elif current_inventory < predicted:
        recommendations.append({
            'type': 'warning',
            'title': '⚠️ LOW STOCK ALERT',
            'message': f'Current stock ({current_inventory} units) is below predicted demand ({predicted} units). Consider preparing {max(0, recommended_prep - current_inventory)} more units.'
        })
    elif current_inventory > (predicted * 1.5):
        recommendations.append({
            'type': 'warning',
            'title': '⚠️ OVERSTOCK RISK',
            'message': f'Current stock ({current_inventory} units) is 50%+ higher than predicted demand ({predicted} units). Risk of food wastage. Consider reducing stock by {current_inventory - recommended_prep} units.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'title': '✅ OPTIMAL INVENTORY',
            'message': f'Current inventory ({current_inventory} units) is well-balanced with predicted demand ({predicted} units). Proceed as planned.'
        })
    
    # 2. Weather-based Demand Advisories
    if weather == 'Cold':
        recommendations.append({
            'type': 'info',
            'title': '❄️ COLD WEATHER ADVISORY',
            'message': 'Cold weather detected. Expect higher demand for Tea, Coffee, Momos, and Samosa. Consider increasing hot beverage stock by 30-40%.'
        })
    elif weather == 'Rainy':
        recommendations.append({
            'type': 'info',
            'title': '🌧️ RAINY DAY ADVISORY',
            'message': 'Rainy weather detected. Expect higher demand for fried snacks (Samosa, Pakoda, French Fries) and hot beverages. Reduce cold drinks stock.'
        })
    elif weather == 'Sunny':
        recommendations.append({
            'type': 'info',
            'title': '☀️ HOT DAY ADVISORY',
            'message': 'Sunny and warm weather detected. Expect higher demand for Cold Drinks, Juice, and chilled items. Reduce hot beverages by 15-20%.'
        })
    
    # 3. Academic Calendar Advisories
    if exam_day == 'Yes':
        recommendations.append({
            'type': 'info',
            'title': '📚 EXAM DAY ADVISORY',
            'message': 'Exam day detected. Students prefer quick snacks: Tea, Coffee, Samosa, Maggi, and Sandwich. Reduce Thali/Lunch item preparation by 25-30%.'
        })
    
    if holiday == 'Yes':
        recommendations.append({
            'type': 'warning',
            'title': '🏖️ HOLIDAY DEMAND DROP',
            'message': 'Holiday detected. Student footfall expected to drop by 70-80%. Prepare minimal stock to reduce waste. Consider limiting items to 3-4 high-demand options only.'
        })
    
    if weekend == 'Yes' and holiday != 'Yes':
        recommendations.append({
            'type': 'info',
            'title': '📅 WEEKEND PATTERN',
            'message': 'Weekend detected. Student attendance typically 30-40% lower than weekdays. Reduce overall stock by 35% compared to regular weekday preparation.'
        })
    
    # 4. High/Low Demand Spike Alerts vs average
    if avg_daily_demand is not None and avg_daily_demand > 0:
        pct_diff = ((predicted - avg_daily_demand) / avg_daily_demand) * 100
        if pct_diff >= 30:
            recommendations.append({
                'type': 'warning',
                'title': '📈 HIGH DEMAND SPIKE DETECTED',
                'message': f'Predicted demand ({predicted} units) is {abs(pct_diff):.1f}% above average ({int(avg_daily_demand)} units). Ensure adequate ingredient sourcing and staff preparation.'
            })
        elif pct_diff <= -30:
            recommendations.append({
                'type': 'info',
                'title': '📉 BELOW-AVERAGE DEMAND',
                'message': f'Predicted demand ({predicted} units) is {abs(pct_diff):.1f}% below average ({int(avg_daily_demand)} units). Reduce preparation proportionally to minimize waste.'
            })
    
    return recommendations

def get_waste_reduction_recommendation(predicted_demand, actual_sales, recommended_prep, food_item):
    """Calculates food waste metrics and provides reduction recommendation."""
    estimated_waste = max(0, recommended_prep - actual_sales)
    waste_pct = (estimated_waste / recommended_prep * 100) if recommended_prep > 0 else 0
    
    if waste_pct > 20:
        reduction_advice = f"High waste rate ({waste_pct:.1f}%) detected for {food_item}. Reduce preparation by 10-15% on similar days."
        severity = 'HIGH'
    elif waste_pct > 10:
        reduction_advice = f"Moderate waste ({waste_pct:.1f}%) detected. Consider reducing preparation by 5-8% for {food_item}."
        severity = 'MODERATE'
    elif waste_pct > 5:
        reduction_advice = f"Low waste ({waste_pct:.1f}%) for {food_item}. Current preparation is near-optimal. Minor adjustment of 2-4% may improve efficiency."
        severity = 'LOW'
    else:
        reduction_advice = f"Minimal waste ({waste_pct:.1f}%) for {food_item}. Excellent demand forecasting and preparation balance!"
        severity = 'MINIMAL'
    
    return {
        'food_item': food_item,
        'predicted_demand': predicted_demand,
        'actual_sales': actual_sales,
        'recommended_prep': recommended_prep,
        'estimated_waste': estimated_waste,
        'waste_pct': round(waste_pct, 2),
        'severity': severity,
        'advice': reduction_advice
    }

def get_inventory_health_report(inventory_df, predictions_df=None):
    """
    Generates inventory health report for all items.
    Returns a DataFrame with stock status and recommended actions.
    """
    report_rows = []
    
    for _, row in inventory_df.iterrows():
        item = row['food_item']
        stock = int(row['current_stock'])
        min_threshold = int(row['min_threshold'])
        max_capacity = int(row['max_capacity'])
        
        if stock == 0:
            status = '⛔ OUT OF STOCK'
            action = 'Immediate restock required'
            priority = 1
        elif stock <= min_threshold:
            status = '🔴 LOW STOCK'
            action = f'Reorder urgently. Minimum threshold: {min_threshold}'
            priority = 2
        elif stock <= (min_threshold * 2):
            status = '🟡 MODERATE STOCK'
            action = 'Monitor closely. Consider reordering soon'
            priority = 3
        elif stock > (max_capacity * 0.85):
            status = '🟣 OVERSTOCK'
            action = 'High overstock risk. Hold purchasing until stock depletes'
            priority = 4
        else:
            status = '🟢 ADEQUATE STOCK'
            action = 'No action required'
            priority = 5
        
        report_rows.append({
            'Food Item': item,
            'Category': row['category'],
            'Current Stock': stock,
            'Min Threshold': min_threshold,
            'Status': status,
            'Recommended Action': action,
            'Priority': priority
        })
    
    report_df = pd.DataFrame(report_rows)
    if not report_df.empty:
        report_df = report_df.sort_values('Priority').reset_index(drop=True)
    return report_df

if __name__ == '__main__':
    sample_result = {
        'food_item': 'Samosa',
        'predicted_demand': 200,
        'recommended_prep': 212,
        'current_inventory': 80,
        'weather': 'Cold',
        'exam_day': 'No',
        'holiday': 'No',
        'weekend': 'No'
    }
    recs = get_smart_recommendations(sample_result, avg_daily_demand=175)
    for rec in recs:
        print(f"[{rec['type'].upper()}] {rec['title']}: {rec['message']}")
