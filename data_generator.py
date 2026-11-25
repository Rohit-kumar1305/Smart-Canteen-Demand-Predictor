# SMART CANTEEN DEMAND PREDICTOR
# Module: data_generator.py
# Purpose: Generate realistic historical canteen sales data (4,000+ records) and seed the database.
# Technology: 100% Python (pandas, numpy, random, datetime).

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from database import seed_database_from_csv

# Define 20 food items across 5 categories
FOOD_CATALOG = {
    'Samosa': {'category': 'Snacks', 'price': 20.0, 'base_demand': 175, 'shelf_life_days': 1},
    'Burger': {'category': 'Fast Food', 'price': 55.0, 'base_demand': 110, 'shelf_life_days': 1},
    'Pizza': {'category': 'Fast Food', 'price': 110.0, 'base_demand': 75, 'shelf_life_days': 1},
    'Sandwich': {'category': 'Snacks', 'price': 45.0, 'base_demand': 95, 'shelf_life_days': 1},
    'Chowmein': {'category': 'Fast Food', 'price': 65.0, 'base_demand': 85, 'shelf_life_days': 1},
    'Momos': {'category': 'Snacks', 'price': 60.0, 'base_demand': 120, 'shelf_life_days': 1},
    'Maggi': {'category': 'Snacks', 'price': 40.0, 'base_demand': 130, 'shelf_life_days': 2},
    'Paratha': {'category': 'Breakfast', 'price': 50.0, 'base_demand': 105, 'shelf_life_days': 1},
    'Dosa': {'category': 'Breakfast', 'price': 60.0, 'base_demand': 115, 'shelf_life_days': 1},
    'Idli': {'category': 'Breakfast', 'price': 40.0, 'base_demand': 90, 'shelf_life_days': 1},
    'Tea': {'category': 'Beverages', 'price': 15.0, 'base_demand': 260, 'shelf_life_days': 1},
    'Coffee': {'category': 'Beverages', 'price': 25.0, 'base_demand': 160, 'shelf_life_days': 1},
    'Cold Drink': {'category': 'Beverages', 'price': 35.0, 'base_demand': 140, 'shelf_life_days': 30},
    'Juice': {'category': 'Beverages', 'price': 40.0, 'base_demand': 80, 'shelf_life_days': 2},
    'Thali': {'category': 'Lunch', 'price': 90.0, 'base_demand': 145, 'shelf_life_days': 1},
    'Rice': {'category': 'Lunch', 'price': 40.0, 'base_demand': 85, 'shelf_life_days': 1},
    'Dal': {'category': 'Lunch', 'price': 40.0, 'base_demand': 85, 'shelf_life_days': 1},
    'Paneer': {'category': 'Lunch', 'price': 80.0, 'base_demand': 95, 'shelf_life_days': 1},
    'Noodles': {'category': 'Fast Food', 'price': 65.0, 'base_demand': 90, 'shelf_life_days': 1},
    'French Fries': {'category': 'Snacks', 'price': 60.0, 'base_demand': 100, 'shelf_life_days': 1}
}

WEATHER_CONDITIONS = ['Sunny', 'Rainy', 'Cold', 'Cloudy']

def generate_canteen_dataset(num_days=250, output_csv='data/canteen_sales.csv', seed=42):
    """
    Generates realistic daily canteen transaction and demand records.
    250 days * ~18-20 items per day = ~4,500 - 5,000 records.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    start_date = datetime(2025, 1, 1)
    records = []
    
    # Track historical memory per item for lag feature simulation
    history = {item: {'prev_day': catalog['base_demand'], 'prev_week': catalog['base_demand']} 
               for item, catalog in FOOD_CATALOG.items()}
    
    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        date_str = current_date.strftime('%Y-%m-%d')
        day_name = current_date.strftime('%A')
        month = current_date.month
        year = current_date.year
        
        is_weekend = 'Yes' if day_name in ['Saturday', 'Sunday'] else 'No'
        
        # Realistic holiday simulation (~6% of days)
        is_holiday = 'Yes' if random.random() < 0.06 and is_weekend == 'No' else ('Yes' if is_weekend == 'Yes' and random.random() < 0.2 else 'No')
        
        # Exam season simulation (e.g., April, May, November, December)
        is_exam_season = month in [4, 5, 11, 12]
        is_exam_day = 'Yes' if is_exam_season and is_weekend == 'No' and is_holiday == 'No' and random.random() < 0.35 else 'No'
        
        # Weather generation based on month
        if month in [12, 1, 2]:
            weather = random.choices(WEATHER_CONDITIONS, weights=[0.25, 0.1, 0.55, 0.1])[0]
            temp = round(random.uniform(11.0, 22.0), 1)
        elif month in [6, 7, 8, 9]:
            weather = random.choices(WEATHER_CONDITIONS, weights=[0.2, 0.55, 0.05, 0.2])[0]
            temp = round(random.uniform(24.0, 34.0), 1)
        else:
            weather = random.choices(WEATHER_CONDITIONS, weights=[0.55, 0.05, 0.15, 0.25])[0]
            temp = round(random.uniform(26.0, 42.0), 1)
            
        # Students present calculation
        if is_holiday == 'Yes':
            students_present = random.randint(40, 140)
        elif is_weekend == 'Yes':
            students_present = random.randint(120, 280)
        elif is_exam_day == 'Yes':
            students_present = random.randint(850, 1250)
        else:
            students_present = random.randint(650, 1100)
            
        attendance_ratio = students_present / 850.0  # Normalized to average 850 students
        
        for item_name, catalog in FOOD_CATALOG.items():
            category = catalog['category']
            price = catalog['price']
            base = catalog['base_demand']
            
            # 1. Base demand scaled with student attendance
            demand = base * attendance_ratio
            
            # 2. Weather influence
            if weather == 'Cold':
                if item_name in ['Tea', 'Coffee', 'Momos', 'Samosa', 'Maggi']:
                    demand *= 1.35
                elif item_name in ['Cold Drink', 'Juice', 'Idli']:
                    demand *= 0.60
            elif weather == 'Rainy':
                if item_name in ['Samosa', 'Tea', 'Maggi', 'French Fries', 'Coffee']:
                    demand *= 1.40
                elif item_name in ['Cold Drink', 'Juice']:
                    demand *= 0.70
            elif weather == 'Sunny' and temp > 30:
                if item_name in ['Cold Drink', 'Juice']:
                    demand *= 1.45
                elif item_name in ['Tea', 'Coffee']:
                    demand *= 0.75
            
            # 3. Exam Day influence (Students prefer fast/caffeinated bites over sit-down thali)
            if is_exam_day == 'Yes':
                if item_name in ['Tea', 'Coffee', 'Maggi', 'Samosa', 'Sandwich']:
                    demand *= 1.30
                elif item_name in ['Thali', 'Paneer', 'Dosa']:
                    demand *= 0.70
                    
            # 4. Weekend / Holiday severe drop
            if is_holiday == 'Yes':
                demand *= 0.18
            elif is_weekend == 'Yes':
                demand *= 0.32
                
            # 5. Day of week subtle seasonality
            if day_name == 'Monday':
                demand *= 1.08
            elif day_name == 'Friday':
                demand *= 1.12
                
            # 6. Add Gaussian noise (std dev = 7%)
            noise = np.random.normal(0, 0.07)
            demand = max(1, int(round(demand * (1 + noise))))
            
            # Lag values from historical record
            prev_day_sales = history[item_name]['prev_day']
            prev_week_sales = history[item_name]['prev_week']
            
            # Inventory & Waste modeling:
            # Canteen prepares stock based on estimated demand + 4% - 12% safety margin
            safety_buffer_pct = random.uniform(0.04, 0.10)
            stock_prepared = int(round(demand * (1 + safety_buffer_pct)))
            
            # Introduce occasional stock shortages (5% chance)
            if random.random() < 0.05:
                stock_prepared = int(round(demand * random.uniform(0.70, 0.92)))
                
            # Actual units sold is limited by current inventory
            quantity_sold = min(demand, stock_prepared)
            
            # Waste quantity is leftover unsold food
            waste_quantity = max(0, stock_prepared - quantity_sold)
            
            # If item is packaged/long shelf life (e.g. Cold Drink), waste is 0
            if catalog['shelf_life_days'] > 7:
                waste_quantity = 0
                
            revenue = round(quantity_sold * price, 2)
            
            # Update lag tracker
            history[item_name]['prev_day'] = quantity_sold
            if day_idx % 7 == 0:
                history[item_name]['prev_week'] = quantity_sold
                
            payment_methods = ['UPI', 'Cash', 'Canteen Card', 'Debit Card']
            pay_method = random.choices(payment_methods, weights=[0.68, 0.14, 0.14, 0.04])[0]
            
            records.append({
                'Date': date_str,
                'Day': day_name,
                'Month': month,
                'Year': year,
                'Food_Item': item_name,
                'Category': category,
                'Students_Present': students_present,
                'Weather': weather,
                'Temperature': temp,
                'Exam_Day': is_exam_day,
                'Holiday': is_holiday,
                'Weekend': is_weekend,
                'Food_Price': price,
                'Previous_Day_Sales': prev_day_sales,
                'Previous_Week_Sales': prev_week_sales,
                'Current_Inventory': stock_prepared,
                'Quantity_Sold': quantity_sold,
                'Revenue': revenue,
                'Waste_Quantity': waste_quantity,
                'Payment_Method': pay_method
            })
            
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f'[DATASET] Generated {len(df)} realistic canteen records saved to: {output_csv}')
    return df

if __name__ == '__main__':
    df = generate_canteen_dataset(num_days=230)
    seed_database_from_csv('data/canteen_sales.csv')
