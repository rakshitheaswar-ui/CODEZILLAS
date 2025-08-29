#!/usr/bin/env python3
"""
Graph generation tools for the chatbot
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import re

class GraphGenerator:
    """Tool for generating various types of graphs based on user requests"""
    
    def __init__(self, main_df=None, actuals_df=None, forecast_df=None):
        self.main_df = main_df
        self.actuals_df = actuals_df
        self.forecast_df = forecast_df
        
    def create_sales_trend_chart(self, item_ids=None, store_ids=None, date_range=None):
        """Create a sales trend chart"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Filter data
            df = self.main_df.copy()
            # Normalize inputs: accept string or list; case-insensitive for item_ids like 'foods_1_001'
            if isinstance(item_ids, str):
                item_ids = [item_ids]
            if isinstance(store_ids, str):
                store_ids = [store_ids]

            if item_ids:
                norm_items = [str(x).upper().replace('-', '_') for x in item_ids]
                df = df[df['item_id_display'].isin(norm_items)]
            if store_ids:
                norm_stores = []
                for s in store_ids:
                    s = str(s).upper()
                    # tolerate formats like CA1 or CA_1M
                    m = re.search(r"([A-Z]{2})[\s_]?([0-9]+)", s)
                    if m:
                        norm_stores.append(f"{m.group(1)}_{m.group(2)}")
                    else:
                        norm_stores.append(s)
                df = df[df['store_id_display'].isin(norm_stores)]
            if date_range:
                start, end = date_range
                df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

            if df.empty:
                return "No matching data for the given filters (items/stores/date range)."
            
            # Aggregate by date
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['sales'],
                mode='lines+markers',
                name='Daily Sales',
                line=dict(color='#007BFF', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Sales Trend Over Time",
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Sales trend chart generated successfully"
            
        except Exception as e:
            return f"Error creating sales trend chart: {str(e)}"
    
    def create_forecast_vs_actual_chart(self):
        """Create forecast vs actual comparison chart"""
        try:
            if self.forecast_df is None or self.actuals_df is None:
                return "No forecast or actual data available"
            
            # Get forecast data
            forecast_daily = self.forecast_df.groupby('date')['predicted_sales'].sum().reset_index()
            
            # Get actual data for same period
            encoded_items = self.forecast_df['item_id'].unique()
            encoded_stores = self.forecast_df['store_id'].unique()
            
            actual_data = self.actuals_df[
                (self.actuals_df['item_id'].isin(encoded_items)) &
                (self.actuals_df['store_id'].isin(encoded_stores)) &
                (self.actuals_df['date'].isin(self.forecast_df['date'].unique()))
            ]
            
            if actual_data.empty:
                return "No actual data available for comparison"
            
            actual_daily = actual_data.groupby('date')['sales'].sum().reset_index()
            
            # Merge for comparison
            comparison = pd.merge(forecast_daily, actual_daily, on='date', how='outer').fillna(0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison['date'],
                y=comparison['sales'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='#28A745', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=comparison['date'],
                y=comparison['predicted_sales'],
                mode='lines+markers',
                name='Predicted Sales',
                line=dict(color='#007BFF', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title="Forecast vs Actual Sales Comparison",
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Forecast vs actual comparison chart generated successfully"
            
        except Exception as e:
            return f"Error creating forecast vs actual chart: {str(e)}"
    
    def create_item_performance_chart(self, top_n=10):
        """Create item performance ranking chart"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Calculate item performance
            item_performance = self.main_df.groupby('item_id_display')['sales'].sum().reset_index()
            item_performance = item_performance.sort_values('sales', ascending=False).head(top_n)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=item_performance['item_id_display'],
                y=item_performance['sales'],
                marker_color='#007BFF'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} Items by Sales Performance",
                xaxis_title="Item ID",
                yaxis_title="Total Sales (Units)",
                template="plotly_white",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return f"Top {top_n} items performance chart generated successfully"
            
        except Exception as e:
            return f"Error creating item performance chart: {str(e)}"
    
    def create_store_performance_chart(self):
        """Create store performance comparison chart"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Calculate store performance
            store_performance = self.main_df.groupby('store_id_display')['sales'].sum().reset_index()
            store_performance = store_performance.sort_values('sales', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=store_performance['store_id_display'],
                y=store_performance['sales'],
                marker_color='#28A745'
            ))
            
            fig.update_layout(
                title="Store Performance Comparison",
                xaxis_title="Store ID",
                yaxis_title="Total Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Store performance chart generated successfully"
            
        except Exception as e:
            return f"Error creating store performance chart: {str(e)}"
    
    def create_weekly_pattern_chart(self):
        """Create weekly sales pattern chart"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Add day of week
            df = self.main_df.copy()
            df['day_of_week'] = df['date'].dt.day_name()
            
            # Calculate average sales by day of week
            weekly_pattern = df.groupby('day_of_week')['sales'].mean().reset_index()
            
            # Order days properly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern['day_of_week'] = pd.Categorical(weekly_pattern['day_of_week'], categories=day_order, ordered=True)
            weekly_pattern = weekly_pattern.sort_values('day_of_week')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekly_pattern['day_of_week'],
                y=weekly_pattern['sales'],
                marker_color='#FD7E14'
            ))
            
            fig.update_layout(
                title="Average Sales by Day of Week",
                xaxis_title="Day of Week",
                yaxis_title="Average Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Weekly pattern chart generated successfully"
            
        except Exception as e:
            return f"Error creating weekly pattern chart: {str(e)}"
    
    def create_monthly_trend_chart(self):
        """Create monthly sales trend chart"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Add month and year
            df = self.main_df.copy()
            df['month_year'] = df['date'].dt.to_period('M')
            
            # Calculate monthly sales
            monthly_sales = df.groupby('month_year')['sales'].sum().reset_index()
            monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_sales['month_year'],
                y=monthly_sales['sales'],
                mode='lines+markers',
                name='Monthly Sales',
                line=dict(color='#6F42C1', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Monthly Sales Trend",
                xaxis_title="Month-Year",
                yaxis_title="Total Sales (Units)",
                template="plotly_white",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Monthly trend chart generated successfully"
            
        except Exception as e:
            return f"Error creating monthly trend chart: {str(e)}"
    
    def create_category_performance_chart(self):
        """Create category performance chart"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Extract category from item_id_display
            df = self.main_df.copy()
            df['category'] = df['item_id_display'].str.split('_').str[0]
            
            # Calculate category performance
            category_performance = df.groupby('category')['sales'].sum().reset_index()
            category_performance = category_performance.sort_values('sales', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=category_performance['category'],
                values=category_performance['sales'],
                hole=0.3
            ))
            
            fig.update_layout(
                title="Sales Distribution by Category",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Category performance chart generated successfully"
            
        except Exception as e:
            return f"Error creating category performance chart: {str(e)}"
    
    def create_heatmap_chart(self, item_ids=None, store_ids=None):
        """Create sales heatmap by item and store"""
        try:
            if self.main_df is None:
                return "No data available for chart generation"
            
            # Filter data
            df = self.main_df.copy()
            if item_ids:
                df = df[df['item_id_display'].isin(item_ids)]
            if store_ids:
                df = df[df['store_id_display'].isin(store_ids)]
            
            # Calculate sales by item and store
            heatmap_data = df.groupby(['item_id_display', 'store_id_display'])['sales'].sum().reset_index()
            
            # Pivot for heatmap
            heatmap_pivot = heatmap_data.pivot(index='item_id_display', columns='store_id_display', values='sales').fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Sales Heatmap: Items vs Stores",
                xaxis_title="Store ID",
                yaxis_title="Item ID",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Sales heatmap generated successfully"
            
        except Exception as e:
            return f"Error creating heatmap chart: {str(e)}"
    
    def generate_graph_from_request(self, user_request: str) -> str:
        """Main function to generate appropriate graph based on user request"""
        try:
            request_lower = user_request.lower()
            
            # Sales trend
            if any(word in request_lower for word in ['trend', 'over time', 'sales trend', 'time series']):
                return self.create_sales_trend_chart()
            
            # Forecast comparison
            elif any(word in request_lower for word in ['forecast', 'prediction', 'actual vs', 'comparison']):
                return self.create_forecast_vs_actual_chart()
            
            # Item performance
            elif any(word in request_lower for word in ['item', 'product', 'top items', 'best items']):
                # Extract number if specified
                top_n = 10
                numbers = re.findall(r'\d+', user_request)
                if numbers:
                    top_n = min(int(numbers[0]), 20)  # Cap at 20
                return self.create_item_performance_chart(top_n)
            
            # Store performance
            elif any(word in request_lower for word in ['store', 'location', 'store performance']):
                return self.create_store_performance_chart()
            
            # Weekly pattern
            elif any(word in request_lower for word in ['week', 'daily', 'day of week', 'weekly']):
                return self.create_weekly_pattern_chart()
            
            # Monthly trend
            elif any(word in request_lower for word in ['month', 'monthly', 'year']):
                return self.create_monthly_trend_chart()
            
            # Category performance
            elif any(word in request_lower for word in ['category', 'categories', 'distribution']):
                return self.create_category_performance_chart()
            
            # Heatmap
            elif any(word in request_lower for word in ['heatmap', 'heat map', 'matrix']):
                return self.create_heatmap_chart()
            
            else:
                return "I can create various charts. Please specify what type of visualization you'd like:\n" + \
                       "- Sales trends over time\n" + \
                       "- Forecast vs actual comparisons\n" + \
                       "- Top performing items\n" + \
                       "- Store performance\n" + \
                       "- Weekly patterns\n" + \
                       "- Monthly trends\n" + \
                       "- Category distribution\n" + \
                       "- Sales heatmap"
            
        except Exception as e:
            return f"Error generating graph: {str(e)}"

# Initialize global graph generator
graph_generator = None

def init_graph_generator(main_df=None, actuals_df=None, forecast_df=None):
    """Initialize the global graph generator"""
    global graph_generator
    graph_generator = GraphGenerator(main_df, actuals_df, forecast_df)
    return graph_generator

def get_graph_generator():
    """Get the global graph generator instance"""
    return graph_generator
