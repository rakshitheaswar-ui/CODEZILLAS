#!/usr/bin/env python3
"""
Graph generation agent with tools for sophisticated chart creation
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import re
from datetime import datetime, timedelta

class GraphAgent:
    """Agent for generating sophisticated graphs with multiple tools"""
    
    def __init__(self, main_df=None, actuals_df=None, forecast_df=None):
        self.main_df = main_df
        self.actuals_df = actuals_df
        self.forecast_df = forecast_df
        self.tools = {
            'create_sales_trend': self.create_sales_trend,
            'create_forecast_comparison': self.create_forecast_comparison,
            'create_item_ranking': self.create_item_ranking,
            'create_store_comparison': self.create_store_comparison,
            'create_weekly_pattern': self.create_weekly_pattern,
            'create_monthly_trend': self.create_monthly_trend,
            'create_category_distribution': self.create_category_distribution,
            'create_heatmap': self.create_heatmap,
            'create_correlation_analysis': self.create_correlation_analysis,
            'create_seasonal_decomposition': self.create_seasonal_decomposition
        }
    
    def create_sales_trend(self, **kwargs):
        """Create sales trend chart with advanced options"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters with normalization (case-insensitive items; tolerant stores)
            if 'item_ids' in kwargs and kwargs['item_ids']:
                item_ids = kwargs['item_ids']
                if isinstance(item_ids, str):
                    item_ids = [item_ids]
                norm_items = [str(x).upper().replace('-', '_') for x in item_ids]
                df = df[df['item_id_display'].isin(norm_items)]
            
            if 'store_ids' in kwargs and kwargs['store_ids']:
                store_ids = kwargs['store_ids']
                if isinstance(store_ids, str):
                    store_ids = [store_ids]
                norm_stores = []
                for s in store_ids:
                    s = str(s).upper()
                    m = re.search(r"([A-Z]{2})[\s_]?([0-9]+)", s)
                    if m:
                        norm_stores.append(f"{m.group(1)}_{m.group(2)}")
                    else:
                        norm_stores.append(s)
                df = df[df['store_id_display'].isin(norm_stores)]
            
            if 'date_range' in kwargs and kwargs['date_range']:
                start_date, end_date = kwargs['date_range']
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if df.empty:
                return "No matching data for the given filters (items/stores/date range)."

            # Aggregate by date
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            
            # Add moving average if requested
            if kwargs.get('moving_average', False):
                window = kwargs.get('window', 7)
                daily_sales[f'{window}_day_ma'] = daily_sales['sales'].rolling(window=window).mean()
            
            fig = go.Figure()
            
            # Main sales line
            fig.add_trace(go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['sales'],
                mode='lines+markers',
                name='Daily Sales',
                line=dict(color='#007BFF', width=3),
                marker=dict(size=6)
            ))
            
            # Moving average if requested
            if kwargs.get('moving_average', False) and f'{window}_day_ma' in daily_sales.columns:
                fig.add_trace(go.Scatter(
                    x=daily_sales['date'],
                    y=daily_sales[f'{window}_day_ma'],
                    mode='lines',
                    name=f'{window}-Day Moving Average',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=kwargs.get('title', 'Sales Trend Over Time'),
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Sales trend chart created successfully"
            
        except Exception as e:
            return f"Error creating sales trend: {str(e)}"
    
    def create_forecast_comparison(self, **kwargs):
        """Create forecast vs actual comparison with metrics"""
        try:
            if self.forecast_df is None or self.actuals_df is None:
                return "No forecast or actual data available"
            
            # Get forecast data
            forecast_daily = self.forecast_df.groupby('date')['predicted_sales'].sum().reset_index()
            
            # Get actual data
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
            comparison = pd.merge(forecast_daily, actual_daily, on='date', how='outer').fillna(0)
            
            # Calculate metrics
            mae = np.mean(np.abs(comparison['sales'] - comparison['predicted_sales']))
            mape = np.mean(np.abs((comparison['sales'] - comparison['predicted_sales']) / 
                                 np.where(comparison['sales'] != 0, comparison['sales'], 1))) * 100
            
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
                title=f"Forecast vs Actual Comparison (MAE: {mae:.2f}, MAPE: {mape:.2f}%)",
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return f"Forecast comparison created successfully (MAE: {mae:.2f}, MAPE: {mape:.2f}%)"
            
        except Exception as e:
            return f"Error creating forecast comparison: {str(e)}"
    
    def create_item_ranking(self, **kwargs):
        """Create item performance ranking with advanced options"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'store_ids' in kwargs and kwargs['store_ids']:
                df = df[df['store_id_display'].isin(kwargs['store_ids'])]
            if 'date_range' in kwargs and kwargs['date_range']:
                start_date, end_date = kwargs['date_range']
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Calculate metrics
            top_n = kwargs.get('top_n', 10)
            metric = kwargs.get('metric', 'sales')  # sales, revenue, etc.
            
            if metric == 'revenue' and 'sell_price' in df.columns:
                df['revenue'] = df['sales'] * df['sell_price']
                item_performance = df.groupby('item_id_display')['revenue'].sum().reset_index()
            else:
                item_performance = df.groupby('item_id_display')['sales'].sum().reset_index()
            
            item_performance = item_performance.sort_values(metric, ascending=False).head(top_n)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=item_performance['item_id_display'],
                y=item_performance[metric],
                marker_color='#007BFF'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} Items by {metric.title()}",
                xaxis_title="Item ID",
                yaxis_title=f"Total {metric.title()}",
                template="plotly_white",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return f"Item ranking chart created successfully"
            
        except Exception as e:
            return f"Error creating item ranking: {str(e)}"
    
    def create_store_comparison(self, **kwargs):
        """Create store performance comparison"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'item_ids' in kwargs and kwargs['item_ids']:
                df = df[df['item_id_display'].isin(kwargs['item_ids'])]
            if 'date_range' in kwargs and kwargs['date_range']:
                start_date, end_date = kwargs['date_range']
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            store_performance = df.groupby('store_id_display')['sales'].sum().reset_index()
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
            return "Store comparison chart created successfully"
            
        except Exception as e:
            return f"Error creating store comparison: {str(e)}"
    
    def create_weekly_pattern(self, **kwargs):
        """Create weekly sales pattern analysis"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'item_ids' in kwargs and kwargs['item_ids']:
                df = df[df['item_id_display'].isin(kwargs['item_ids'])]
            if 'store_ids' in kwargs and kwargs['store_ids']:
                df = df[df['store_id_display'].isin(kwargs['store_ids'])]
            
            df['day_of_week'] = df['date'].dt.day_name()
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
            return "Weekly pattern chart created successfully"
            
        except Exception as e:
            return f"Error creating weekly pattern: {str(e)}"
    
    def create_monthly_trend(self, **kwargs):
        """Create monthly sales trend analysis"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'item_ids' in kwargs and kwargs['item_ids']:
                df = df[df['item_id_display'].isin(kwargs['item_ids'])]
            if 'store_ids' in kwargs and kwargs['store_ids']:
                df = df[df['store_id_display'].isin(kwargs['store_ids'])]
            
            df['month_year'] = df['date'].dt.to_period('M')
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
            return "Monthly trend chart created successfully"
            
        except Exception as e:
            return f"Error creating monthly trend: {str(e)}"
    
    def create_category_distribution(self, **kwargs):
        """Create category performance distribution"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'store_ids' in kwargs and kwargs['store_ids']:
                df = df[df['store_id_display'].isin(kwargs['store_ids'])]
            if 'date_range' in kwargs and kwargs['date_range']:
                start_date, end_date = kwargs['date_range']
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Extract category
            df['category'] = df['item_id_display'].str.split('_').str[0]
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
            return "Category distribution chart created successfully"
            
        except Exception as e:
            return f"Error creating category distribution: {str(e)}"
    
    def create_heatmap(self, **kwargs):
        """Create sales heatmap"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'date_range' in kwargs and kwargs['date_range']:
                start_date, end_date = kwargs['date_range']
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Limit items and stores for better visualization
            top_items = df.groupby('item_id_display')['sales'].sum().nlargest(20).index
            df_filtered = df[df['item_id_display'].isin(top_items)]
            
            heatmap_data = df_filtered.groupby(['item_id_display', 'store_id_display'])['sales'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='item_id_display', columns='store_id_display', values='sales').fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Sales Heatmap: Top 20 Items vs Stores",
                xaxis_title="Store ID",
                yaxis_title="Item ID",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Sales heatmap created successfully"
            
        except Exception as e:
            return f"Error creating heatmap: {str(e)}"
    
    def create_correlation_analysis(self, **kwargs):
        """Create correlation analysis between variables"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_data = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.index,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Correlation analysis created successfully"
            
        except Exception as e:
            return f"Error creating correlation analysis: {str(e)}"
    
    def create_seasonal_decomposition(self, **kwargs):
        """Create seasonal decomposition analysis"""
        try:
            if self.main_df is None:
                return "No data available"
            
            df = self.main_df.copy()
            
            # Apply filters
            if 'item_ids' in kwargs and kwargs['item_ids']:
                df = df[df['item_id_display'].isin(kwargs['item_ids'])]
            if 'store_ids' in kwargs and kwargs['store_ids']:
                df = df[df['store_id_display'].isin(kwargs['store_ids'])]
            
            # Daily sales
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            daily_sales = daily_sales.set_index('date').sort_index()
            
            # Simple seasonal decomposition
            window = kwargs.get('window', 7)
            trend = daily_sales['sales'].rolling(window=window).mean()
            seasonal = daily_sales['sales'] - trend
            residual = daily_sales['sales'] - trend - seasonal
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_sales.index,
                y=daily_sales['sales'],
                mode='lines',
                name='Original',
                line=dict(color='#007BFF', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_sales.index,
                y=trend,
                mode='lines',
                name='Trend',
                line=dict(color='#28A745', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_sales.index,
                y=seasonal,
                mode='lines',
                name='Seasonal',
                line=dict(color='#FD7E14', width=2)
            ))
            
            fig.update_layout(
                title="Seasonal Decomposition",
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return "Seasonal decomposition created successfully"
            
        except Exception as e:
            return f"Error creating seasonal decomposition: {str(e)}"
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        else:
            return f"Tool '{tool_name}' not found"
    
    def analyze_request(self, user_request: str) -> Tuple[str, Dict]:
        """Analyze user request and determine appropriate tool and parameters"""
        request_lower = user_request.lower()
        
        # Extract parameters
        params = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+', user_request)
        if numbers:
            params['top_n'] = min(int(numbers[0]), 50)
        
        # Enhanced date range extraction - but don't override if already set in session state
        # The date_range will be set by the calling code from session state
        if 'date_range' not in params:
            params['date_range'] = self._extract_date_range(user_request)
        
        # Determine tool based on keywords (prioritize forecast-vs-actual over generic trend)
        if any(word in request_lower for word in ['predicted vs', 'vs actual', 'pred vs', 'pred v actual', 'actual vs', 'predicted', 'prediction', 'forecast vs']):
            return 'create_forecast_comparison', params
        if any(word in request_lower for word in ['forecast', 'prediction']):
            return 'create_forecast_comparison', params
        if any(word in request_lower for word in ['trend', 'over time', 'sales trend']):
            if 'moving average' in request_lower or 'ma' in request_lower:
                params['moving_average'] = True
                params['window'] = 7
            return 'create_sales_trend', params
        
        elif any(word in request_lower for word in ['item', 'product', 'top', 'best']):
            return 'create_item_ranking', params
        
        elif any(word in request_lower for word in ['store', 'location']):
            return 'create_store_comparison', params
        
        elif any(word in request_lower for word in ['week', 'daily', 'day of week']):
            return 'create_weekly_pattern', params
        
        elif any(word in request_lower for word in ['month', 'monthly', 'year']):
            return 'create_monthly_trend', params
        
        elif any(word in request_lower for word in ['category', 'distribution']):
            return 'create_category_distribution', params
        
        elif any(word in request_lower for word in ['heatmap', 'matrix']):
            return 'create_heatmap', params
        
        elif any(word in request_lower for word in ['correlation', 'correlate']):
            return 'create_correlation_analysis', params
        
        elif any(word in request_lower for word in ['seasonal', 'decomposition']):
            return 'create_seasonal_decomposition', params
        
        else:
            return None, {}
    
    def _extract_date_range(self, user_request: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract date range from user request with enhanced parsing"""
        try:
            # First try ISO date format: YYYY-MM-DD or YYYY/MM/DD
            date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
            dates = re.findall(date_pattern, user_request)
            if len(dates) >= 2:
                return (pd.to_datetime(dates[0]), pd.to_datetime(dates[1]))
            
            # Try month name patterns with more flexible matching
            month_names = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
            }
            
            # More flexible pattern for month ranges - allows text before and after
            month_range_pattern = r'(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})\s+(?:to|through|until|-)\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})'
            
            month_match = re.search(month_range_pattern, user_request.lower())
            if month_match:
                start_month, start_year, end_month, end_year = month_match.groups()
                start_month_num = month_names[start_month]
                end_month_num = month_names[end_month]
                start_year = int(start_year)
                end_year = int(end_year)
                
                # Create start date (first day of start month)
                start_date = datetime(start_year, start_month_num, 1)
                
                # Create end date (last day of end month)
                if end_month_num == 12:
                    end_date = datetime(end_year, 12, 31)
                else:
                    end_date = datetime(end_year, end_month_num + 1, 1) - timedelta(days=1)
                
                return (start_date, end_date)
            
            # Try single month pattern: "Jan 2015" or "January 2015" or "for Jan 2015"
            single_month_pattern = r'(?:for\s+)?(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})'
            single_month_match = re.search(single_month_pattern, user_request.lower())
            if single_month_match:
                month, year = single_month_match.groups()
                month_num = month_names[month]
                year = int(year)
                
                # Create start date (first day of month)
                start_date = datetime(year, month_num, 1)
                
                # Create end date (last day of month)
                if month_num == 12:
                    end_date = datetime(year, 12, 31)
                else:
                    end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                return (start_date, end_date)
            
            # Try "from X to Y" pattern with month names - more flexible
            from_to_pattern = r'from\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})\s+to\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})'
            from_to_match = re.search(from_to_pattern, user_request.lower())
            if from_to_match:
                start_month, start_year, end_month, end_year = from_to_match.groups()
                start_month_num = month_names[start_month]
                end_month_num = month_names[end_month]
                start_year = int(start_year)
                end_year = int(end_year)
                
                # Create start date (first day of start month)
                start_date = datetime(start_year, start_month_num, 1)
                
                # Create end date (last day of end month)
                if end_month_num == 12:
                    end_date = datetime(end_year, 12, 31)
                else:
                    end_date = datetime(end_year, end_month_num + 1, 1) - timedelta(days=1)
                
                return (start_date, end_date)
            
            return None
            
        except Exception as e:
            return None

# Global graph agent
graph_agent = None

def init_graph_agent(main_df=None, actuals_df=None, forecast_df=None):
    """Initialize the global graph agent"""
    global graph_agent
    graph_agent = GraphAgent(main_df, actuals_df, forecast_df)
    return graph_agent

def get_graph_agent():
    """Get the global graph agent instance"""
    return graph_agent
