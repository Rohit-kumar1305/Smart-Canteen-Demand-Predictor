# SMART CANTEEN DEMAND PREDICTOR
# Module: database.py
# Purpose: SQLite Database management layer for sales, inventory, predictions, and complaints.
# Technology: 100% Python with built-in sqlite3.

import sqlite3
import os
import pandas as pd
import random
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), 'database', 'canteen.db')

def get_db_connection(db_path=DEFAULT_DB_PATH):
    """Establishes and returns a SQLite database connection."""
    dburi = os.path.abspath(db_path)
    db_dir = os.path.dirname(dburi)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(dburi)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DEFAULT_DB_PATH):
    """Initializes all required database tables with proper schemas."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Sales / Transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            day TEXT NOT NULL,
            month INTEGER NOT NULL,
            year INTEGER NOT NULL,
            food_item TEXT NOT NULL,
            category TEXT NOT NULL,
            students_present INTEGER NOT NULL,
            weather TEXT NOT NULL,
            temperature REAL NOT NULL,
            exam_day TEXT NOT NULL,
            holiday TEXT NOT NULL,
            weekend TEXT NOT NULL,
            food_price REAL NOT NULL,
            previous_day_sales INTEGER NOT NULL,
            previous_week_sales INTEGER NOT NULL,
            current_inventory INTEGER NOT NULL,
            quantity_sold INTEGER NOT NULL,
            revenue REAL NOT NULL,
            waste_quantity INTEGER NOT NULL,
            payment_method TEXT DEFAULT 'UPI'
        )
    ''')

    # 2. Food Items Master table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_item TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            food_price REAL NOT NULL,
            cost_price REAL NOT NULL,
            min_threshold INTEGER DEFAULT 30,
            max_capacity INTEGER DEFAULT 300,
            description TEXT
        )
    ''')

    # 3. Inventory table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_item TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            current_stock INTEGER NOT NULL,
            min_threshold INTEGER NOT NULL DEFAULT 30,
            max_capacity INTEGER NOT NULL DEFAULT 300,
            last_updated TEXT NOT NULL
        )
    ''')

    # 4. Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            food_item TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            students_present INTEGER NOT NULL,
            weather TEXT NOT NULL,
            temperature REAL NOT NULL,
            exam_day TEXT NOT NULL,
            holiday TEXT NOT NULL,
            weekend TEXT NOT NULL,
            food_price REAL NOT NULL,
            previous_day_sales INTEGER NOT NULL,
            previous_week_sales INTEGER NOT NULL,
            current_inventory INTEGER NOT NULL,
            predicted_demand INTEGER NOT NULL,
            recommended_prep INTEGER NOT NULL,
            additional_required INTEGER NOT NULL,
            safety_margin_pct REAL NOT NULL,
            recommendation_text TEXT
        )
    ''')

    # 5. Complaints and Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            food_item TEXT NOT NULL,
            complaint_type TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT,
            date TEXT NOT NULL,
            status TEXT DEFAULT 'Pending'
        )
    ''')

    conn.commit()
    conn.close()
    print(f'[DATABASE] Initialized SQLite schema successfully at: {db_path}')

def seed_database_from_csv(csv_path='data/canteen_sales.csv', db_path=DEFAULT_DB_PATH):
    """Seeds sales, food_items, and inventory tables from the generated CSV dataset."""
    if not os.path.exists(csv_path):
        print(f'[DATABASE WARNING] CSV file {csv_path} not found. Skipping seeding.')
        return False

    init_db(db_path)
    df = pd.read_csv(csv_path)

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM sales')
    count = cursor.fetchone()[0]

    if count == 0:
        if 'Payment_Method' not in df.columns:
            payment_methods = ['UPI', 'Cash', 'Canteen Card', 'Debit Card']
            df['Payment_Method'] = [random.choices(payment_methods, weights=[0.65, 0.15, 0.15, 0.05])[0] for _ in range(len(df))]

        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO sales (
                    date, day, month, year, food_item, category, students_present,
                    weather, temperature, exam_day, holiday, weekend, food_price,
                    previous_day_sales, previous_week_sales, current_inventory,
                    quantity_sold, revenue, waste_quantity, payment_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(row['Date']), str(row['Day']), int(row['Month']), int(row['Year']),
                str(row['Food_Item']), str(row['Category']), int(row['Students_Present']),
                str(row['Weather']), float(row['Temperature']), str(row['Exam_Day']),
                str(row['Holiday']), str(row['Weekend']), float(row['Food_Price']),
                int(row['Previous_Day_Sales']), int(row['Previous_Week_Sales']),
                int(row['Current_Inventory']), int(row['Quantity_Sold']),
                float(row['Revenue']), int(row['Waste_Quantity']),
                str(row.get('Payment_Method', 'UPI'))
            ))

        unique_items = df[['Food_Item', 'Category', 'Food_Price']].drop_duplicates()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for _, row in unique_items.iterrows():
            item_name = str(row['Food_Item'])
            cat = str(row['Category'])
            price = float(row['Food_Price'])
            cost_price = round(price * 0.55, 2)

            cursor.execute('''
                INSERT OR REPLACE INTO food_items (food_item, category, food_price, cost_price, min_threshold, max_capacity, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (item_name, cat, price, cost_price, 30, 300, f'Freshly prepared {item_name}'))

            item_sales = df[df['Food_Item'] == item_name]
            latest_stock = int(item_sales['Current_Inventory'].iloc[-1]) if len(item_sales) > 0 else 80

            cursor.execute('''
                INSERT OR REPLACE INTO inventory (food_item, category, current_stock, min_threshold, max_capacity, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (item_name, cat, latest_stock, 35, 300, now_str))

        cursor.execute('SELECT COUNT(*) FROM complaints')
        if cursor.fetchone()[0] == 0:
            sample_complaints = [
                ('Aarav Sharma', 'Samosa', 'Food Quality', 4, 'Crispy and fresh, but was a bit salty.', '2026-09-10', 'Resolved'),
                ('Priya Patel', 'Tea', 'Availability', 2, 'Tea was finished at 4:30 PM peak break time.', '2026-09-11', 'In Progress'),
                ('Rohan Gupta', 'Burger', 'Late Service', 3, 'Took 15 minutes during lunch rush.', '2026-09-11', 'Pending'),
                ('Sneha Reddy', 'Cold Drink', 'Price', 4, 'Chilled bottle, fair pricing.', '2026-09-12', 'Resolved'),
                ('Vikram Singh', 'Dosa', 'Cleanliness', 5, 'Clean serving area and hot sambar!', '2026-09-12', 'Resolved'),
                ('Ananya Iyer', 'Thali', 'Food Quality', 3, 'Rice portion was small today.', '2026-09-12', 'Pending')
            ]
            for comp in sample_complaints:
                cursor.execute('''
                    INSERT INTO complaints (student_name, food_item, complaint_type, rating, comment, date, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', comp)

        conn.commit()
        print(f'[DATABASE] Successfully seeded {len(df)} sales rows and master tables into {db_path}')

    conn.close()
    return True

def get_all_sales_df(db_path=DEFAULT_DB_PATH):
    """Fetches all sales records as a Pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query('SELECT * FROM sales ORDER BY date DESC, transaction_id DESC', conn)
    conn.close()
    return df

def get_inventory_df(db_path=DEFAULT_DB_PATH):
    """Fetches live inventory records as a Pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query('SELECT * FROM inventory ORDER BY category, food_item', conn)
    conn.close()
    return df

def update_inventory_stock(food_item, new_stock, db_path=DEFAULT_DB_PATH):
    """Updates current stock level for a specific food item."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        UPDATE inventory 
        SET current_stock = ?, last_updated = ?
        WHERE food_item = ?
    ''', (int(new_stock), now_str, str(food_item)))
    conn.commit()
    conn.close()
    return True

def save_prediction(record_dict, db_path=DEFAULT_DB_PATH):
    """Saves a single prediction log into predictions table."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, food_item, prediction_date, students_present, weather,
            temperature, exam_day, holiday, weekend, food_price,
            previous_day_sales, previous_week_sales, current_inventory,
            predicted_demand, recommended_prep, additional_required,
            safety_margin_pct, recommendation_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record_dict['timestamp'], record_dict['food_item'], record_dict['prediction_date'],
        int(record_dict['students_present']), record_dict['weather'], float(record_dict['temperature']),
        record_dict['exam_day'], record_dict['holiday'], record_dict['weekend'],
        float(record_dict['food_price']), int(record_dict['previous_day_sales']),
        int(record_dict['previous_week_sales']), int(record_dict['current_inventory']),
        int(record_dict['predicted_demand']), int(record_dict['recommended_prep']),
        int(record_dict['additional_required']), float(record_dict.get('safety_margin_pct', 6.0)),
        record_dict.get('recommendation_text', '')
    ))
    conn.commit()
    conn.close()
    return True

def get_recent_predictions(limit=50, db_path=DEFAULT_DB_PATH):
    """Fetches recent predictions history as a Pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(f'SELECT * FROM predictions ORDER BY id DESC LIMIT {int(limit)}', conn)
    conn.close()
    return df

def save_complaint(record_dict, db_path=DEFAULT_DB_PATH):
    """Saves a new complaint or feedback record."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO complaints (student_name, food_item, complaint_type, rating, comment, date, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        record_dict['student_name'], record_dict['food_item'], record_dict['complaint_type'],
        int(record_dict['rating']), record_dict.get('comment', ''),
        record_dict['date'], record_dict.get('status', 'Pending')
    ))
    conn.commit()
    conn.close()
    return True

def get_complaints_df(db_path=DEFAULT_DB_PATH):
    """Fetches all complaints as a Pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query('SELECT * FROM complaints ORDER BY id DESC', conn)
    conn.close()
    return df

def update_complaint_status(complaint_id, new_status, db_path=DEFAULT_DB_PATH):
    """Updates the status of a complaint (Pending, In Progress, Resolved)."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE complaints SET status = ? WHERE id = ?
    ''', (str(new_status), int(complaint_id)))
    conn.commit()
    conn.close()
    return True

if __name__ == '__main__':
    init_db()
