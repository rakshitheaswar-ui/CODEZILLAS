from __future__ import annotations
from typing import Dict


class M5ForecastingAgent:
    """Forecasting specialist agent for M5 data analysis"""
    
    def __init__(self, ollama_manager: 'M5OllamaManager'):
        self.ollama = ollama_manager
        
    def analyze(self, state: 'M5ForecastState') -> str:
        system_prompt = """You are a retail forecasting specialist with expertise in the M5 Walmart dataset. 
        Analyze forecast patterns, assess model accuracy, identify seasonal trends, and provide technical insights. 
        Focus on M5-specific patterns like holiday effects, promotional impacts, and store-item relationships.
        Be concise and data-driven."""
        
        prompt = f"""
        M5 FORECAST ANALYSIS:
        - Model Used: {state.model_used}
        - Total Sales Forecast: {state.total_sales:,} units
        - Daily Average: {state.avg_daily:.0f} units/day
        - Peak Day: {state.peak_day.strftime('%A, %B %d') if state.peak_day else 'N/A'}
        - Peak Sales: {state.peak_sales:,} units
        - Items Analyzed: {state.item_count}
        - Stores Covered: {state.store_count}
        - Date Range: {state.date_range}
        
        Provide analysis on:
        1) Forecast pattern recognition and seasonality
        2) Model reliability for M5 retail data
        3) Holiday/promotional period impacts
        4) Store-item performance variations
        """
        
        return self.ollama.generate_response(prompt, system_prompt)


class M5InventoryAgent:
    """Inventory optimization specialist for M5 retail context"""
    
    def __init__(self, ollama_manager: 'M5OllamaManager'):
        self.ollama = ollama_manager
        
    def analyze(self, state: 'M5ForecastState') -> str:
        system_prompt = """You are a retail inventory optimization specialist focused on Walmart M5 data patterns.
        Calculate optimal stock levels, safety stock, and reorder strategies for retail environments.
        Consider M5-specific factors like store size, item categories, and seasonal variations."""
        
        # Calculate inventory metrics
        volatility = state.peak_sales / state.avg_daily if state.avg_daily > 0 else 1
        safety_multiplier = 1.5 if volatility > 2.5 else 1.3 if volatility > 2 else 1.2
        recommended_stock = state.avg_daily * 14 * safety_multiplier  # 2-week base stock
        
        prompt = f"""
        M5 INVENTORY OPTIMIZATION:
        - Daily Average Demand: {state.avg_daily:.0f} units/day
        - Peak Demand: {state.peak_sales:,} units
        - Demand Volatility: {volatility:.2f}x
        - Recommended Base Stock: {recommended_stock:.0f} units
        - Items: {state.item_count} | Stores: {state.store_count}
        
        Provide recommendations for:
        1) Optimal inventory levels per store-item combination
        2) Safety stock calculations for retail volatility
        3) Reorder points and replenishment timing
        4) Seasonal stock adjustments for M5 patterns
        """
        
        return self.ollama.generate_response(prompt, system_prompt)


class M5RiskAgent:
    """Risk assessment specialist for M5 retail forecasting"""
    
    def __init__(self, ollama_manager: 'M5OllamaManager'):
        self.ollama = ollama_manager
        
    def analyze(self, state: 'M5ForecastState') -> str:
        system_prompt = """You are a retail risk assessment specialist with M5 Walmart dataset expertise.
        Identify forecast risks, demand volatility, supply chain vulnerabilities, and stockout/overstock risks.
        Focus on retail-specific risks like promotional cannibalization and seasonal shifts."""
        
        volatility_ratio = state.peak_sales / state.avg_daily if state.avg_daily > 0 else 1
        risk_level = "HIGH" if volatility_ratio > 3 else "MEDIUM" if volatility_ratio > 2 else "LOW"
        
        prompt = f"""
        M5 RISK ASSESSMENT:
        - Demand Volatility: {volatility_ratio:.2f}x (Risk Level: {risk_level})
        - Peak vs Average Ratio: {volatility_ratio:.1f}:1
        - Model Used: {state.model_used}
        - Forecast Coverage: {state.item_count} items × {state.store_count} stores
        - Time Horizon: {state.date_range}
        
        Analyze risks for:
        1) Demand volatility and forecast accuracy
        2) Store-specific and item-specific risks
        3) Seasonal/promotional period exposures
        4) Supply chain and inventory risks
        5) Financial impact of forecast errors
        """
        
        return self.ollama.generate_response(prompt, system_prompt)


class M5StrategyAgent:
    """Strategic planning specialist synthesizing M5 insights"""
    
    def __init__(self, ollama_manager: 'M5OllamaManager'):
        self.ollama = ollama_manager
        
    def analyze(self, state: 'M5ForecastState', insights: Dict[str, str]) -> str:
        system_prompt = """You are a retail strategic planning specialist for M5 Walmart operations.
        Synthesize all agent insights into comprehensive business strategies and actionable recommendations.
        Focus on practical retail execution, resource allocation, and performance optimization."""
        
        insights_summary = "\n".join([f"• {agent.replace('_', ' ').title()}: {analysis[:120]}..." 
                                        for agent, analysis in insights.items()])
        
        prompt = f"""
        M5 STRATEGIC SYNTHESIS:
        
        FORECAST METRICS:
        - Total Forecast: {state.total_sales:,} units
        - Daily Average: {state.avg_daily:.0f} units/day
        - Peak Performance: {state.peak_sales:,} units
        - Model: {state.model_used}
        - Scope: {state.item_count} items × {state.store_count} stores
        
        AGENT INSIGHTS:
        {insights_summary}
        
        Create comprehensive strategy covering:
        1) Business priorities and focus areas
        2) Resource allocation recommendations
        3) Implementation timeline and milestones  
        4) Key performance indicators (KPIs)
        5) Risk mitigation actions
        """
        
        return self.ollama.generate_response(prompt, system_prompt)
