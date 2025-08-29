#!/usr/bin/env python3
"""
Analytics agent for complex data analysis and forecasting tasks
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

class AnalyticsAgent:
    """Agent for complex analytics tasks including forecasting and ranking - Optimized for performance"""
    
    def __init__(self, main_df=None, actuals_df=None, forecast_df=None, models=None):
        self.main_df = main_df
        self.actuals_df = actuals_df
        self.forecast_df = forecast_df
        self.models = models
        
        # OPTIMIZATION: Pre-compute frequently used data structures
        self._cached_data = {}
        self._precompute_common_data()
        
        self.tools = {
            'analyze_top_items_by_period': self.analyze_top_items_by_period,
            'forecast_and_rank_items': self.forecast_and_rank_items,
            'compare_actual_vs_forecast_ranking': self.compare_actual_vs_forecast_ranking,
            'analyze_item_performance_trends': self.analyze_item_performance_trends,
            'generate_periodic_insights': self.generate_periodic_insights
        }
    
    def _precompute_common_data(self):
        """Pre-compute commonly used data structures for better performance"""
        if self.main_df is not None:
            # Cache item and store mappings
            self._cached_data['item_mapping'] = {val: idx for idx, val in enumerate(sorted(self.main_df['item_id'].dropna().unique()))}
            self._cached_data['store_mapping'] = {val: idx for idx, val in enumerate(sorted(self.main_df['store_id'].dropna().unique()))}
            
            # Cache historical averages
            self._cached_data['hist_avg'] = self.main_df.groupby(['item_id', 'store_id'])['sales'].mean().reset_index()
            self._cached_data['hist_avg'].columns = ['item_id', 'store_id', 'hist_avg_sales']
            
            # Cache item display mappings
            self._cached_data['item_display_mapping'] = self.main_df[['item_id', 'item_id_display']].drop_duplicates()
            
            # Cache top items for quick access
            self._cached_data['top_items'] = self.main_df.groupby('item_id_display')['sales'].sum().sort_values(ascending=False)
    
    def analyze_top_items_by_period(self, **kwargs):
        """Analyze top items by sales for a specific time period - Optimized for performance"""
        try:
            if self.main_df is None:
                return {"error": "No data available for analysis"}
            
            # Extract parameters
            top_n = kwargs.get('top_n', 15)
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            metric = kwargs.get('metric', 'sales')  # sales, revenue, etc.
            
            # Determine price column for revenue if needed
            price_col = None
            if metric == 'revenue':
                for cand in ['sell_price', 'price', 'unit_price']:
                    if cand in (self.main_df.columns if self.main_df is not None else []):
                        price_col = cand
                        break

            # OPTIMIZATION: Use cached data when possible
            if start_date is None and end_date is None:
                # Use cached top items if no date filter
                top_items = self._cached_data.get('top_items')
                if top_items is not None:
                    item_performance = top_items.head(top_n).reset_index()
                    item_performance.columns = ['item_id_display', 'sales']
                    if metric == 'revenue' and price_col is not None:
                        # Compute mean price per item and estimate revenue = sales * mean_price
                        price_map = self.main_df.groupby('item_id_display')[price_col].mean()
                        item_performance['revenue'] = item_performance['sales'] * item_performance['item_id_display'].map(price_map).fillna(0)
                else:
                    # Fallback to full calculation
                    df0 = self.main_df.copy()
                    if metric == 'revenue' and price_col is not None:
                        df0['revenue'] = df0['sales'] * df0[price_col]
                        item_performance = df0.groupby('item_id_display')['revenue'].sum().reset_index()
                    else:
                        item_performance = df0.groupby('item_id_display')['sales'].sum().reset_index()
                    sort_col = 'revenue' if (metric == 'revenue' and price_col is not None) else 'sales'
                    item_performance = item_performance.sort_values(sort_col, ascending=False).head(top_n)
            else:
                # Filter data by date range
                df = self.main_df.copy()
                if start_date and end_date:
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # Calculate metrics
                if metric == 'revenue' and price_col is not None:
                    df['revenue'] = df['sales'] * df[price_col]
                    item_performance = df.groupby('item_id_display')['revenue'].sum().reset_index()
                else:
                    item_performance = df.groupby('item_id_display')['sales'].sum().reset_index()
                
                # Sort and get top items
                sort_col = 'revenue' if (metric == 'revenue' and price_col is not None) else 'sales'
                item_performance = item_performance.sort_values(sort_col, ascending=False).head(top_n)
            
            # Return data for visualization in main app
            if start_date and end_date:
                period_text = f" ({pd.to_datetime(start_date).strftime('%b %Y')} to {pd.to_datetime(end_date).strftime('%b %Y')})"
            elif start_date:
                period_text = f" ({pd.to_datetime(start_date).strftime('%b %Y')})"
            else:
                period_text = ""
            
            return {
                "success": True,
                "chart_type": "bar",
                "title": f"Top {top_n} Items by {('Revenue ($)' if metric=='revenue' and price_col is not None else metric.title())}{period_text}",
                "data": item_performance,
                "x_col": "item_id_display",
                "y_col": ("revenue" if metric == 'revenue' and price_col is not None else 'sales'),
                "summary_table": item_performance.copy(),
                "message": f"Successfully analyzed top {top_n} items by {('revenue' if metric=='revenue' and price_col is not None else 'sales')} for the specified period"
            }
            
        except Exception as e:
            return {"error": f"Error analyzing top items: {str(e)}"}
    
    def forecast_and_rank_items(self, **kwargs):
        """Generate forecasts for items and rank them by predicted sales - Optimized for performance"""
        try:
            if self.main_df is None or self.models is None:
                return {"error": "No data or models available for forecasting"}
            
            # Extract parameters
            top_n = kwargs.get('top_n', 15)
            forecast_start = kwargs.get('forecast_start')
            forecast_end = kwargs.get('forecast_end')
            model_name = kwargs.get('model', 'LightGBM')
            
            if forecast_start is None or forecast_end is None:
                return {"error": "Forecast date range is required"}
            
            # Get the model
            if model_name not in self.models or self.models[model_name] is None:
                return {"error": f"Model {model_name} is not available"}
            
            model = self.models[model_name]
            
            # OPTIMIZATION: Use vectorized operations for better performance
            # Get unique items and stores
            unique_items = self.main_df['item_id'].unique()
            unique_stores = self.main_df['store_id'].unique()
            
            # Create forecast dataframe for the period using vectorized operations
            date_range = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
            
            # OPTIMIZATION: Use more efficient data structure creation
            # Create multi-index for all combinations
            item_store_combinations = pd.MultiIndex.from_product(
                [unique_items, unique_stores, date_range], 
                names=['item_id', 'store_id', 'date']
            )
            
            # Convert to DataFrame more efficiently
            prediction_df = item_store_combinations.to_frame(index=False)
            
            # OPTIMIZATION: Add features efficiently using vectorized operations
            prediction_df['day_of_week'] = prediction_df['date'].dt.dayofweek
            prediction_df['month'] = prediction_df['date'].dt.month
            prediction_df['year'] = prediction_df['date'].dt.year
            
            # OPTIMIZATION: Use cached historical averages
            hist_avg = self._cached_data.get('hist_avg')
            if hist_avg is None:
                hist_avg = self.main_df.groupby(['item_id', 'store_id'])['sales'].mean().reset_index()
                hist_avg.columns = ['item_id', 'store_id', 'hist_avg_sales']
            
            # OPTIMIZATION: Use efficient merge
            prediction_df = prediction_df.merge(hist_avg, on=['item_id', 'store_id'], how='left')
            prediction_df['hist_avg_sales'] = prediction_df['hist_avg_sales'].fillna(1.0)
            
            # OPTIMIZATION: Use cached categorical mappings
            item_mapping = self._cached_data.get('item_mapping')
            store_mapping = self._cached_data.get('store_mapping')
            
            if item_mapping is None or store_mapping is None:
                item_mapping = {val: idx for idx, val in enumerate(sorted(self.main_df['item_id'].dropna().unique()))}
                store_mapping = {val: idx for idx, val in enumerate(sorted(self.main_df['store_id'].dropna().unique()))}
            
            # Apply mappings efficiently
            prediction_df['item_id_encoded'] = prediction_df['item_id'].map(item_mapping).fillna(0)
            prediction_df['store_id_encoded'] = prediction_df['store_id'].map(store_mapping).fillna(0)
            
            # Prepare features for model
            feature_cols = ['day_of_week', 'month', 'year', 'hist_avg_sales', 'item_id_encoded', 'store_id_encoded']
            features = prediction_df[feature_cols].fillna(0)
            
            # Generate predictions
            try:
                predictions = model.predict(features)
                prediction_df['predicted_sales'] = np.maximum(0, predictions)
            except Exception as e:
                # Fallback to historical averages
                prediction_df['predicted_sales'] = prediction_df['hist_avg_sales']
            
            # OPTIMIZATION: Aggregate predictions by item efficiently
            item_forecasts = prediction_df.groupby('item_id')['predicted_sales'].sum().reset_index()
            
            # OPTIMIZATION: Use cached item display mapping
            item_display_mapping = self._cached_data.get('item_display_mapping')
            if item_display_mapping is None:
                item_display_mapping = self.main_df[['item_id', 'item_id_display']].drop_duplicates()
            
            item_forecasts = item_forecasts.merge(item_display_mapping, on='item_id', how='left')
            
            # Sort and get top items
            top_forecasted_items = item_forecasts.sort_values('predicted_sales', ascending=False).head(top_n)
            
            # Return data for visualization in main app
            period_text = f" ({forecast_start.strftime('%B %Y')} to {forecast_end.strftime('%B %Y')})"
            
            return {
                "success": True,
                "chart_type": "bar",
                "title": f"Top {top_n} Items by Forecasted Sales{period_text}",
                "data": top_forecasted_items,
                "x_col": "item_id_display",
                "y_col": "predicted_sales",
                "summary_table": top_forecasted_items[['item_id_display', 'predicted_sales']].copy(),
                "message": f"Successfully generated forecasts and ranked top {top_n} items"
            }
            
        except Exception as e:
            return {"error": f"Error in forecast and rank: {str(e)}"}
    
    def compare_actual_vs_forecast_ranking(self, **kwargs):
        """Compare actual vs forecasted rankings for items"""
        try:
            if self.main_df is None or self.forecast_df is None:
                return {"error": "No data available for comparison"}
            
            # Extract parameters
            top_n = kwargs.get('top_n', 15)
            
            # Get actual performance
            actual_performance = self.main_df.groupby('item_id_display')['sales'].sum().reset_index()
            actual_performance = actual_performance.sort_values('sales', ascending=False).head(top_n)
            actual_performance['actual_rank'] = range(1, len(actual_performance) + 1)
            
            # Get forecasted performance
            forecast_performance = self.forecast_df.groupby('item_id_display')['predicted_sales'].sum().reset_index()
            forecast_performance = forecast_performance.sort_values('predicted_sales', ascending=False).head(top_n)
            forecast_performance['forecast_rank'] = range(1, len(forecast_performance) + 1)
            
            # Merge for comparison
            comparison = actual_performance.merge(forecast_performance, on='item_id_display', how='outer').fillna(0)
            comparison['rank_difference'] = comparison['actual_rank'] - comparison['forecast_rank']
            
            # Return data for visualization in main app
            return {
                "success": True,
                "chart_type": "grouped_bar",
                "title": f"Actual vs Predicted Sales - Top {top_n} Items",
                "data": comparison,
                "x_col": "item_id_display",
                "y_cols": ["sales", "predicted_sales"],
                "summary_table": comparison[['item_id_display', 'actual_rank', 'forecast_rank', 'rank_difference', 'sales', 'predicted_sales']].copy(),
                "message": f"Successfully compared actual vs forecasted rankings for top {top_n} items"
            }
            
        except Exception as e:
            return {"error": f"Error in comparison: {str(e)}"}
    
    def analyze_item_performance_trends(self, **kwargs):
        """Analyze performance trends for top items over time"""
        try:
            if self.main_df is None:
                return {"error": "No data available for trend analysis"}
            
            # Extract parameters
            top_n = kwargs.get('top_n', 10)
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            # Get top items
            top_items = self.main_df.groupby('item_id_display')['sales'].sum().nlargest(top_n).index
            
            # Filter data
            df = self.main_df[self.main_df['item_id_display'].isin(top_items)].copy()
            if start_date and end_date:
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Create time series for each item
            item_trends = df.groupby(['date', 'item_id_display'])['sales'].sum().reset_index()
            
            # Return data for visualization in main app
            return {
                "success": True,
                "chart_type": "line",
                "title": f"Performance Trends - Top {top_n} Items",
                "data": item_trends,
                "x_col": "date",
                "y_col": "sales",
                "group_col": "item_id_display",
                "message": f"Successfully analyzed performance trends for top {top_n} items"
            }
            
        except Exception as e:
            return {"error": f"Error in trend analysis: {str(e)}"}
    
    def generate_periodic_insights(self, **kwargs):
        """Generate comprehensive insights for a specific period"""
        try:
            if self.main_df is None:
                return {"error": "No data available for insights"}
            
            # Extract parameters
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            top_n = kwargs.get('top_n', 15)
            
            if start_date is None or end_date is None:
                return {"error": "Date range is required for periodic insights"}
            
            # Filter data
            df = self.main_df[(self.main_df['date'] >= start_date) & (self.main_df['date'] <= end_date)].copy()
            
            if df.empty:
                return {"error": f"No data available for the period {start_date} to {end_date}"}
            
            # Generate comprehensive insights
            top_items = df.groupby('item_id_display')['sales'].sum().nlargest(top_n)
            store_performance = df.groupby('store_id_display')['sales'].sum().sort_values(ascending=False)
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            
            # Calculate summary statistics
            total_sales = df['sales'].sum()
            avg_daily_sales = df.groupby('date')['sales'].sum().mean()
            total_items = df['item_id_display'].nunique()
            total_stores = df['store_id_display'].nunique()
            
            # Return comprehensive data for visualization in main app
            return {
                "success": True,
                "chart_type": "multi_chart",
                "title": f"Insights for {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}",
                "charts": [
                    {
                        "type": "bar",
                        "title": f"Top {top_n} Items by Sales",
                        "data": top_items.reset_index(),
                        "x_col": "item_id_display",
                        "y_col": "sales"
                    },
                    {
                        "type": "bar",
                        "title": "Store Performance",
                        "data": store_performance.reset_index(),
                        "x_col": "store_id_display",
                        "y_col": "sales"
                    },
                    {
                        "type": "line",
                        "title": "Daily Sales Trends",
                        "data": daily_sales,
                        "x_col": "date",
                        "y_col": "sales"
                    }
                ],
                "summary_stats": {
                    "total_sales": total_sales,
                    "avg_daily_sales": avg_daily_sales,
                    "total_items": total_items,
                    "total_stores": total_stores
                },
                "message": f"Successfully generated comprehensive insights for the period"
            }
            
        except Exception as e:
            return {"error": f"Error generating insights: {str(e)}"}
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a specific analytics tool"""
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        else:
            return f"Tool '{tool_name}' not found"
    
    def analyze_request(self, user_request: str) -> Tuple[str, Dict]:
        """Analyze user request and determine appropriate analytics tool and parameters"""
        request_lower = user_request.lower()
        
        # Extract parameters
        params = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+', user_request)
        if numbers:
            params['top_n'] = min(int(numbers[0]), 50)
        
        # Extract date ranges
        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        dates = re.findall(date_pattern, user_request)
        if len(dates) >= 2:
            params['start_date'] = pd.to_datetime(dates[0])
            params['end_date'] = pd.to_datetime(dates[1])
        elif len(dates) == 1:
            # Single date - assume month
            date = pd.to_datetime(dates[0])
            params['start_date'] = date.replace(day=1)
            params['end_date'] = (date.replace(day=1) + pd.offsets.MonthEnd(1))
        
        # Extract month/year patterns
        month_patterns = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in month_patterns.items():
            if month_name in request_lower:
                year_match = re.search(r'20\d{2}', user_request)
                year = int(year_match.group()) if year_match else 2015
                params['start_date'] = pd.to_datetime(f"{year}-{month_num:02d}-01")
                params['end_date'] = pd.to_datetime(f"{year}-{month_num:02d}-01") + pd.offsets.MonthEnd(1)
                break
        
        # Revenue vs sales metric
        if any(k in request_lower for k in ['revenue', 'dollar', '$', 'usd']):
            params['metric'] = 'revenue'

        # Determine tool based on keywords
        if any(word in request_lower for word in ['forecast', 'predict', 'predicted']):
            if any(word in request_lower for word in ['rank', 'top', 'best']):
                return 'forecast_and_rank_items', params
        
        elif any(word in request_lower for word in ['compare', 'actual vs', 'forecast vs']):
            return 'compare_actual_vs_forecast_ranking', params
        
        elif any(word in request_lower for word in ['trend', 'trends', 'performance over time']):
            return 'analyze_item_performance_trends', params
        
        elif any(word in request_lower for word in ['insights', 'analysis', 'summary']):
            return 'generate_periodic_insights', params
        
        elif any(word in request_lower for word in ['top', 'best', 'rank']):
            return 'analyze_top_items_by_period', params
        
        else:
            return None, {}

# Global analytics agent
analytics_agent = None

def init_analytics_agent(main_df=None, actuals_df=None, forecast_df=None, models=None):
    """Initialize the global analytics agent"""
    global analytics_agent
    analytics_agent = AnalyticsAgent(main_df, actuals_df, forecast_df, models)
    return analytics_agent

def get_analytics_agent():
    """Get the global analytics agent instance"""
    return analytics_agent
