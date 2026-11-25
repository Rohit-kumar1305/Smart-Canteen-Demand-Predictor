# SMART CANTEEN DEMAND PREDICTOR
# Module: analytics.py
# Purpose: Business intelligence and data visualization using Plotly.
# Technology: 100% Python (pandas, plotly).

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# SaaS color palette: Purple/Indigo theme
COLORS = {
    'primary': '#6366F1',
    'secondary': '#8B5CF6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#3B82F6',
    'purple': '#A855F7',
    'teal': '#14B8A6',
    'dark': '#1E293B',
    'light_bg': '#F8FAFC'
}

PLOTLY_THEME = dict(
    plot_bgcolor='rgba(248,250,252,1)',
    paper_bgcolor='rgba(255,255,255,1)',
    font=dict(family='Inter, Segoe UI, Arial', size=12, color='#334155'),
    margin=dict(l=10, r=10, t=40, b=10)
)

def get_daily_sales_trend(df, start_date=None, end_date=None):
    """Line chart of daily total sales revenue over time."""
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    if start_date:
        data = data[data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data['date'] <= pd.to_datetime(end_date)]
    
    daily = data.groupby('date').agg(
        Total_Revenue=('revenue', 'sum'),
        Total_Units=('quantity_sold', 'sum')
    ).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['Total_Revenue'],
        name='Daily Revenue (₹)',
        line=dict(color=COLORS['primary'], width=2.5),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.08)',
        mode='lines+markers', marker=dict(size=4)
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['Total_Units'],
        name='Units Sold',
        line=dict(color=COLORS['success'], width=1.8, dash='dot'),
        mode='lines', opacity=0.7
    ), secondary_y=True)
    
    fig.update_layout(
        title='Daily Sales & Demand Trend',
        yaxis_title='Revenue (₹)', yaxis2_title='Units Sold',
        legend=dict(x=0, y=1.05, orientation='h'),
        **PLOTLY_THEME
    )
    return fig

def get_food_item_bar_chart(df, top_n=10):
    """Bar chart comparing food item sales volumes."""
    item_stats = df.groupby('food_item').agg(
        Total_Units=('quantity_sold', 'sum'),
        Total_Revenue=('revenue', 'sum')
    ).reset_index().sort_values('Total_Units', ascending=False).head(top_n)
    
    fig = px.bar(
        item_stats, x='food_item', y='Total_Units',
        color='Total_Revenue',
        color_continuous_scale='Purples',
        labels={'food_item': 'Food Item', 'Total_Units': 'Total Units Sold'},
        title=f'Top {top_n} Food Items by Units Sold',
        text='Total_Units'
    )
    fig.update_traces(textposition='outside', textfont_size=10)
    fig.update_coloraxes(colorbar_title='Revenue (₹)')
    fig.update_layout(**PLOTLY_THEME)
    return fig

def get_category_pie_chart(df):
    """Pie chart of category-wise sales distribution."""
    cat_stats = df.groupby('category').agg(
        Total_Revenue=('revenue', 'sum')
    ).reset_index()
    
    cat_colors = [COLORS['primary'], COLORS['success'], COLORS['warning'],
                  COLORS['info'], COLORS['purple'], COLORS['teal']]
    
    fig = px.pie(
        cat_stats, names='category', values='Total_Revenue',
        title='Category-wise Revenue Distribution',
        color_discrete_sequence=cat_colors,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(**PLOTLY_THEME)
    return fig

def get_weekly_demand_chart(df):
    """Bar chart of average demand per day of the week."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    weekly = df.groupby('day').agg(
        Avg_Units=('quantity_sold', 'mean'),
        Avg_Revenue=('revenue', 'mean')
    ).reset_index()
    
    weekly['day'] = pd.Categorical(weekly['day'], categories=day_order, ordered=True)
    weekly = weekly.sort_values('day')
    
    colors = [COLORS['primary'] if d not in ['Saturday', 'Sunday'] else COLORS['warning'] 
              for d in weekly['day'].tolist()]
    
    fig = go.Figure(go.Bar(
        x=weekly['day'], y=weekly['Avg_Units'],
        marker_color=colors,
        text=weekly['Avg_Units'].round(1),
        textposition='outside',
        name='Avg Units/Day'
    ))
    
    fig.update_layout(
        title='Average Demand by Day of Week',
        xaxis_title='Day', yaxis_title='Average Units Sold',
        **PLOTLY_THEME
    )
    return fig

def get_weather_demand_chart(df):
    """Box plot of demand distribution across weather conditions."""
    fig = px.box(
        df, x='weather', y='quantity_sold',
        color='weather',
        color_discrete_map={
            'Sunny': COLORS['warning'],
            'Rainy': COLORS['info'],
            'Cold': COLORS['primary'],
            'Cloudy': COLORS['teal']
        },
        title='Demand Distribution by Weather Condition',
        labels={'quantity_sold': 'Units Sold', 'weather': 'Weather'}
    )
    fig.update_layout(**PLOTLY_THEME)
    return fig

def get_actual_vs_predicted_chart(df, pipeline, preprocessor_cols):
    """Line chart of actual vs predicted demand."""
    import joblib
    from preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    
    sample = df.sample(min(300, len(df)), random_state=42).copy()
    X_sample = sample[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_actual = sample['quantity_sold'].values
    y_pred = pipeline.predict(X_sample)
    
    sample_sorted = sample.sort_values('date').reset_index(drop=True)
    X_sorted = sample_sorted[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_actual_sorted = sample_sorted['quantity_sold'].values
    y_pred_sorted = pipeline.predict(X_sorted)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_actual_sorted[:100], mode='lines+markers',
        name='Actual Demand',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        y=y_pred_sorted[:100], mode='lines',
        name='Predicted Demand',
        line=dict(color=COLORS['danger'], width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Demand (Sample 100 Records)',
        xaxis_title='Sample Index', yaxis_title='Quantity',
        legend=dict(x=0, y=1.05, orientation='h'),
        **PLOTLY_THEME
    )
    return fig

def get_inventory_status_chart(inventory_df):
    """Bar chart showing current inventory status per food item."""
    df = inventory_df.copy()
    
    def classify(row):
        if row['current_stock'] == 0:
            return 'Out of Stock'
        elif row['current_stock'] <= row['min_threshold']:
            return 'Low Stock'
        elif row['current_stock'] > row['max_capacity'] * 0.85:
            return 'Overstock'
        else:
            return 'Adequate'
    
    df['Status'] = df.apply(classify, axis=1)
    
    status_colors = {
        'Out of Stock': COLORS['danger'],
        'Low Stock': COLORS['warning'],
        'Adequate': COLORS['success'],
        'Overstock': COLORS['purple']
    }
    df['color'] = df['Status'].map(status_colors)
    
    fig = px.bar(
        df, x='food_item', y='current_stock',
        color='Status',
        color_discrete_map=status_colors,
        title='Current Inventory Status by Food Item',
        labels={'food_item': 'Food Item', 'current_stock': 'Current Stock (Units)'}
    )
    fig.update_layout(**PLOTLY_THEME, xaxis_tickangle=-35)
    return fig

def get_exam_vs_normal_chart(df):
    """Bar chart comparing average demand on exam vs normal days."""
    exam_data = df.groupby('exam_day').agg(
        Avg_Units=('quantity_sold', 'mean'),
        Avg_Revenue=('revenue', 'mean')
    ).reset_index()
    exam_data['exam_day'] = exam_data['exam_day'].map({'Yes': 'Exam Day', 'No': 'Normal Day'})
    
    fig = px.bar(
        exam_data, x='exam_day', y='Avg_Units',
        color='exam_day',
        color_discrete_map={'Exam Day': COLORS['warning'], 'Normal Day': COLORS['primary']},
        title='Average Demand: Exam Day vs Normal Day',
        text='Avg_Units', labels={'exam_day': 'Day Type', 'Avg_Units': 'Avg Units Sold'}
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(**PLOTLY_THEME)
    return fig

def get_holiday_analysis_chart(df):
    """Bar chart showing demand on holidays vs working days."""
    hol_data = df.groupby('holiday').agg(
        Avg_Units=('quantity_sold', 'mean'),
        Avg_Revenue=('revenue', 'mean')
    ).reset_index()
    hol_data['holiday'] = hol_data['holiday'].map({'Yes': 'Holiday', 'No': 'Working Day'})
    
    fig = px.bar(
        hol_data, x='holiday', y='Avg_Units',
        color='holiday',
        color_discrete_map={'Holiday': COLORS['danger'], 'Working Day': COLORS['success']},
        title='Average Demand: Holiday vs Working Day',
        text='Avg_Units', labels={'holiday': 'Day Type', 'Avg_Units': 'Avg Units Sold'}
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(**PLOTLY_THEME)
    return fig

def get_waste_analysis_chart(df):
    """Line chart showing daily waste quantities over time."""
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    daily_waste = data.groupby('date').agg(
        Total_Waste=('waste_quantity', 'sum'),
        Total_Sold=('quantity_sold', 'sum')
    ).reset_index()
    daily_waste['Waste_Pct'] = (daily_waste['Total_Waste'] / 
                                 (daily_waste['Total_Waste'] + daily_waste['Total_Sold']) * 100).round(2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_waste['date'], y=daily_waste['Waste_Pct'],
        mode='lines', fill='tozeroy',
        line=dict(color=COLORS['danger'], width=2),
        fillcolor='rgba(239,68,68,0.1)',
        name='Waste %'
    ))
    fig.update_layout(
        title='Daily Food Waste Percentage Over Time',
        xaxis_title='Date', yaxis_title='Waste (%)',
        **PLOTLY_THEME
    )
    return fig

def get_revenue_trend_chart(df, freq='W'):
    """Area chart of weekly or monthly revenue trend."""
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    grouped = data.set_index('date').resample(freq)['revenue'].sum().reset_index()
    grouped.columns = ['Date', 'Revenue']
    
    freq_label = 'Weekly' if freq == 'W' else 'Monthly'
    
    fig = px.area(
        grouped, x='Date', y='Revenue',
        title=f'{freq_label} Revenue Trend',
        labels={'Revenue': 'Total Revenue (₹)', 'Date': 'Period'},
        color_discrete_sequence=[COLORS['primary']]
    )
    fig.update_traces(fillcolor='rgba(99,102,241,0.12)')
    fig.update_layout(**PLOTLY_THEME)
    return fig

def compute_kpi_metrics(df, inventory_df=None):
    """Computes key performance indicators for the Dashboard summary cards."""
    if df.empty:
        return {}
    
    total_revenue = df['revenue'].sum()
    total_units = df['quantity_sold'].sum()
    total_transactions = len(df)
    avg_daily_revenue = df.groupby('date')['revenue'].sum().mean()
    avg_daily_demand = df.groupby('date')['quantity_sold'].sum().mean()
    total_waste = df['waste_quantity'].sum()
    total_sold_plus_waste = total_units + total_waste
    waste_pct = (total_waste / total_sold_plus_waste * 100) if total_sold_plus_waste > 0 else 0
    active_items = df['food_item'].nunique()
    avg_price = df['food_price'].mean()
    
    kpis = {
        'total_revenue': round(total_revenue, 2),
        'total_transactions': total_transactions,
        'total_units_sold': int(total_units),
        'avg_daily_demand': round(avg_daily_demand, 1),
        'avg_daily_revenue': round(avg_daily_revenue, 2),
        'total_waste_units': int(total_waste),
        'waste_percentage': round(waste_pct, 2),
        'active_items': int(active_items),
        'avg_item_price': round(avg_price, 2)
    }
    
    if inventory_df is not None and not inventory_df.empty:
        inventory_value = (inventory_df['current_stock'] * 
                           inventory_df['food_item'].map(
                               dict(zip(df['food_item'], df['food_price']))
                           ).fillna(0)).sum()
        kpis['inventory_value'] = round(inventory_value, 2)
    
    return kpis

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/canteen_sales.csv')
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={'food_item': 'food_item', 'quantity_sold': 'quantity_sold'}, inplace=True)
    kpis = compute_kpi_metrics(df)
    print('[ANALYTICS] KPI Metrics:')
    for k, v in kpis.items():
        print(f'  {k}: {v}')
