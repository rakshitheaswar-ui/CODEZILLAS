import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import requests
import json
import os
import importlib
import traceback
import ollama
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Annotated
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# --- LangGraph / LangChain (with graceful fallback) ---
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph import MessagesState  # typed state with messages
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

try:
    # Preferred modern import
    from langchain_ollama import ChatOllama  # type: ignore
    LANGCHAIN_OLLAMA_AVAILABLE = True
except Exception:
    try:
        # Legacy community import
        from langchain_community.chat_models import ChatOllama  # type: ignore
        LANGCHAIN_OLLAMA_AVAILABLE = True
    except Exception:
        LANGCHAIN_OLLAMA_AVAILABLE = False

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_CORE_MSGS_AVAILABLE = True
except Exception:
    LANGCHAIN_CORE_MSGS_AVAILABLE = False

# Orchestrator and RAG
# Orchestrator/RAG disabled by request; using direct LLM + agents

# (Removed LSTM model support by request)

# Import graph generation tools and specialist agents
try:
    from graph_tools import init_graph_generator, get_graph_generator
    from agents.graph_agent import init_graph_agent, get_graph_agent
    from agents.analytics_agent import init_analytics_agent, get_analytics_agent
    from agents.special_agents import (
        M5ForecastingAgent,
        M5InventoryAgent,
        M5RiskAgent,
        M5StrategyAgent,
    )
    GRAPH_TOOLS_AVAILABLE = True
    GRAPH_AGENT_AVAILABLE = True
    ANALYTICS_AGENT_AVAILABLE = True
    SPECIAL_AGENTS_AVAILABLE = True
except ImportError:
    GRAPH_TOOLS_AVAILABLE = False
    GRAPH_AGENT_AVAILABLE = False
    ANALYTICS_AGENT_AVAILABLE = False
    SPECIAL_AGENTS_AVAILABLE = False
    st.warning("Graph tools not available - chart generation will be disabled")

# Lightweight Ollama manager and forecast state for specialist agents
class M5OllamaManager:
    def generate_response(self, prompt: str, system_prompt: str) -> str:
        try:
            resp = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "num_predict": 300},
            )
            return resp.get("message", {}).get("content", "")
        except Exception as e:
            return f"Agent response unavailable: {e}"

class M5ForecastState:
    def __init__(
        self,
        model_used: str,
        total_sales: float,
        avg_daily: float,
        peak_day,
        peak_sales: float,
        item_count: int,
        store_count: int,
        date_range: str,
    ):
        self.model_used = model_used
        self.total_sales = total_sales
        self.avg_daily = avg_daily
        self.peak_day = peak_day
        self.peak_sales = peak_sales
        self.item_count = item_count
        self.store_count = store_count
        self.date_range = date_range

def _init_special_agents():
    if not SPECIAL_AGENTS_AVAILABLE:
        return None
    manager = M5OllamaManager()
    return {
        "forecast": M5ForecastingAgent(manager),
        "inventory": M5InventoryAgent(manager),
        "risk": M5RiskAgent(manager),
        "strategy": M5StrategyAgent(manager),
    }

SPECIAL_AGENTS = _init_special_agents()

def _build_m5_state() -> M5ForecastState:
    forecast_df = st.session_state.get("forecast_df")
    model_used = (
        st.session_state.get("model_used")
        or st.session_state.get("preferred_model_from_chat")
        or "Unknown"
    )
    start_date = st.session_state.get("sidebar_start_date")
    end_date = st.session_state.get("sidebar_end_date")
    date_range_str = f"{start_date} to {end_date}" if start_date and end_date else "N/A"
    item_count = len(st.session_state.get("sidebar_item_ids") or [])
    store_count = len(st.session_state.get("sidebar_store_ids") or [])

    total_sales = 0.0
    avg_daily = 0.0
    peak_day = None
    peak_sales = 0.0
    try:
        if forecast_df is not None and not forecast_df.empty:
            daily = forecast_df.groupby("date")["predicted_sales"].sum().reset_index()
            total_sales = float(daily["predicted_sales"].sum())
            days = max(len(daily), 1)
            avg_daily = float(total_sales / days)
            idx = int(daily["predicted_sales"].idxmax()) if not daily.empty else 0
            row = daily.iloc[idx] if not daily.empty else None
            peak_day = row["date"] if row is not None else None
            peak_sales = float(row["predicted_sales"]) if row is not None else 0.0
    except Exception:
        pass

    return M5ForecastState(
        model_used=model_used,
        total_sales=total_sales,
        avg_daily=avg_daily,
        peak_day=peak_day,
        peak_sales=peak_sales,
        item_count=item_count,
        store_count=store_count,
        date_range=date_range_str,
    )

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="InventoryGPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional look with horizontal navigation
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #F8F9FA;
    }
    
    /* Hide default sidebar navigation */
    .css-1d391kg {
        display: none;
    }
    
    /* Custom horizontal navigation */
    .nav-container {
        background-color: #FFFFFF;
        padding: 15px 0;
        border-bottom: 2px solid #E9ECEF;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 0;
        padding: 0;
        list-style: none;
    }
    
    .nav-tab {
        padding: 12px 24px;
        background-color: #F8F9FA;
        border: 2px solid #DEE2E6;
        border-radius: 25px;
        color: #495057;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-tab:hover {
        background-color: #E9ECEF;
        border-color: #007BFF;
        color: #007BFF;
        transform: translateY(-2px);
    }
    
    .nav-tab.active {
        background-color: #007BFF;
        border-color: #007BFF;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #FFFFFF;
        border-right: 2px solid #E9ECEF;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        text-align: center;
        border: 1px solid #E9ECEF;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #007BFF, #28A745);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 16px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #007BFF;
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(135deg, #007BFF 0%, #0056B3 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0056B3 0%, #003D82 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #6C757D;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #007BFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Alert styling */
    .success-alert {
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 8px;
        color: #155724;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-alert {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 8px;
        color: #856404;
        padding: 15px;
        margin: 10px 0;
    }
    
    .error-alert {
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        border-radius: 8px;
        color: #721C24;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        animation: fadeIn 0.3s ease;
    }
    
    .user-message {
        background-color: #007BFF;
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        margin-right: 20%;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. M5 DATASET ID DECODING FUNCTIONS ---

def create_m5_id_mappings():
    """Create mappings between encoded IDs and original M5 format IDs"""
    
    # M5 Store ID format: {state_id}_{store_id}
    # States: CA, TX, WI (California, Texas, Wisconsin)
    # Store numbers: 1-4 for each state
    store_mapping = {}
    reverse_store_mapping = {}
    store_counter = 0
    
    state_store_counts = {'CA': 4, 'TX': 3, 'WI': 3}
    for state in ['CA', 'TX', 'WI']:
        for store_num in range(1, state_store_counts[state] + 1):
            original_store_id = f"{state}_{store_num}"
            store_mapping[store_counter] = original_store_id
            reverse_store_mapping[original_store_id] = store_counter
            store_counter += 1
    
    # M5 Item ID format: {item_id} (HOBBIES_1_001, FOODS_3_090, etc.)
    # Categories: HOBBIES, HOUSEHOLD, FOODS
    # Departments: 1,2,3 for each category
    # Items: 001-999+ for each department
    
    # Since we don't have the exact item mapping from the original data,
    # we'll create a generic mapping that follows M5 conventions
    item_mapping = {}
    reverse_item_mapping = {}
    item_counter = 0
    
    categories = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
    
    # Generate realistic M5 item IDs
    for cat in categories:
        for dept in range(1, 4):  # 3 departments per category
            items_per_dept = 340 if cat == 'HOBBIES' else (330 if cat == 'HOUSEHOLD' else 380)  # Approximate M5 distribution
            for item_num in range(1, items_per_dept + 1):
                original_item_id = f"{cat}_{dept}_{item_num:03d}"
                item_mapping[item_counter] = original_item_id
                reverse_item_mapping[original_item_id] = item_counter
                item_counter += 1
    
    return item_mapping, reverse_item_mapping, store_mapping, reverse_store_mapping

# Initialize ID mappings globally
ITEM_MAPPING, REVERSE_ITEM_MAPPING, STORE_MAPPING, REVERSE_STORE_MAPPING = create_m5_id_mappings()

def decode_item_id(encoded_id):
    """Convert encoded item ID back to M5 format"""
    return ITEM_MAPPING.get(encoded_id, f"ITEM_{encoded_id}")

def decode_store_id(encoded_id):
    """Convert encoded store ID back to M5 format"""
    return STORE_MAPPING.get(encoded_id, f"STORE_{encoded_id}")

def encode_item_id(original_id):
    """Convert M5 format item ID to encoded format"""
    return REVERSE_ITEM_MAPPING.get(original_id, original_id)

def encode_store_id(original_id):
    """Convert M5 format store ID to encoded format"""
    return REVERSE_STORE_MAPPING.get(original_id, original_id)

# --- 3. DATA & MODEL LOADING (with Enhanced Caching and ID Decoding) ---

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Loads and caches the main dataset and actuals with error handling and ID decoding."""
    try:
        # Load the base dataset
        df = pd.read_parquet('featured_data_2014.parquet')
        df['date'] = pd.to_datetime(df['date'])
        
        # Decode IDs for display purposes
        df['item_id_display'] = df['item_id'].apply(decode_item_id)
        df['store_id_display'] = df['store_id'].apply(decode_store_id)
        
        # Load the actual sales data for 2015
        actuals_df = pd.read_parquet('featured_data_2015.parquet')  
        actuals_df['date'] = pd.to_datetime(actuals_df['date'])
        
        # Decode IDs for actuals too
        actuals_df['item_id_display'] = actuals_df['item_id'].apply(decode_item_id)
        actuals_df['store_id_display'] = actuals_df['store_id'].apply(decode_store_id)
        
        return df, actuals_df
    except FileNotFoundError as e:
        st.error(f"📂 Data file not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None

@st.cache_resource
def load_models(base_dir='.', model_info=None):
    """
    Load available models: LightGBM, XGBoost, Prophet.
    """
    if model_info is None:
        model_info = {
            "LightGBM": {"file": "lgbm_forecast_model_v3.joblib", "description": "Fast gradient boosting"},
            "XGBoost": {"file": "xgb_forecast_model_v2.joblib", "description": "Extreme gradient boosting"},
            "Prophet": {"file": "prophet_models.joblib", "description": "Time series forecasting"}
        }

    models = {}
    loaded_count = 0

    for name, info in model_info.items():
        filepath = os.path.join(base_dir, info["file"])
        try:
            if not os.path.exists(filepath):
                st.warning(f"📁 Model file not found: {filepath}")
                models[name] = None
                continue

            raw = joblib.load(filepath)
            loaded_count += 1
            models[name] = raw

        except Exception as e:
            st.warning(f"❌ Error loading {name}: {e}")
            models[name] = None

    if loaded_count > 0:
        st.success(f"✅ Successfully loaded {loaded_count} model files")

    return models

# Load data and models
with st.spinner("🔄 Loading data and models..."):
    main_df, actuals_df = load_data()
    models = load_models()
    
    # Initialize graph generator and agents
    if GRAPH_TOOLS_AVAILABLE:
        init_graph_generator(main_df, actuals_df, None)  # forecast_df will be updated when available
    if GRAPH_AGENT_AVAILABLE:
        init_graph_agent(main_df, actuals_df, None)  # forecast_df will be updated when available
    if ANALYTICS_AGENT_AVAILABLE:
        init_analytics_agent(main_df, actuals_df, None, models)  # forecast_df will be updated when available

# Orchestrator removed by request

# Get available options with decoded IDs
if main_df is not None:
    # Use display IDs for user interface
    ITEM_ID_OPTIONS = sorted(main_df['item_id_display'].unique())
    STORE_ID_OPTIONS = sorted(main_df['store_id_display'].unique())
else:
    ITEM_ID_OPTIONS = []
    STORE_ID_OPTIONS = []

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'selected_items' not in st.session_state:
    st.session_state.selected_items = []
if 'selected_stores' not in st.session_state:
    st.session_state.selected_stores = []
# Chat-synced filter state for sidebar controls
if 'sidebar_start_date' not in st.session_state:
    st.session_state.sidebar_start_date = datetime(2015, 1, 1)
if 'sidebar_end_date' not in st.session_state:
    st.session_state.sidebar_end_date = datetime(2015, 1, 31)
if 'sidebar_item_ids' not in st.session_state:
    st.session_state.sidebar_item_ids = []
if 'sidebar_store_ids' not in st.session_state:
    st.session_state.sidebar_store_ids = []
if 'stock_levels' not in st.session_state:
    st.session_state.stock_levels = {}
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'model_used' not in st.session_state:
    st.session_state.model_used = None

# --- 4. ENHANCED NAVIGATION ---
def render_navigation():
    """Renders horizontal navigation tabs"""
    pages = ["🏠 Home", "🔮 Forecast Dashboard", "📊 View Dataset", "💬 Chatbot", "📦 ABC Analysis"]
    
    st.markdown("""
    <div class="nav-container">
        <div style="text-align: center; margin-bottom: 10px;">
            <h2 style="color: #007BFF; margin: 0;">📈 InventoryGPT</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(pages))
    for i, page in enumerate(pages):
        with cols[i]:
            if st.button(page, key=f"nav_{i}", use_container_width=True):
                st.session_state.current_page = page.split(" ", 1)[1]  # Remove emoji

# --- 5. ENHANCED SIDEBAR CONTROLS ---
def render_sidebar():
    """Enhanced sidebar with better controls and decoded IDs"""
    with st.sidebar:
        st.markdown("## 🎛️ Forecasting Controls")
        st.markdown("Configure your forecast parameters")
        st.markdown("---")

        # Date range with validation
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "📅 Start Date",
                st.session_state.sidebar_start_date,
                min_value=datetime(2015, 1, 1),
                max_value=datetime(2016, 6, 28),
                key="sidebar_start_date_widget"
            )
        with col2:
            end_date = st.date_input(
                "📅 End Date",
                st.session_state.sidebar_end_date,
                min_value=start_date,
                max_value=datetime(2016, 6, 29),
                key="sidebar_end_date_widget"
            )

        # Validation and auto-correction to avoid transient errors during reruns
        if end_date <= start_date:
            # Automatically adjust end date to be at least one day after start date
            max_allowed_end = datetime(2016, 6, 29).date()
            adjusted_end = (start_date + timedelta(days=1))
            # Ensure both are date objects for comparison
            if hasattr(adjusted_end, 'date'):
                adjusted_end = adjusted_end if not isinstance(adjusted_end, datetime) else adjusted_end.date()
            if hasattr(start_date, 'date') and isinstance(start_date, datetime):
                start_date = start_date.date()
            if adjusted_end > max_allowed_end:
                # If exceeding max, pull start back instead to keep a valid 1-day range
                adjusted_start = max_allowed_end - timedelta(days=1)
                start_date = adjusted_start
                st.session_state.sidebar_start_date = adjusted_start
                adjusted_end = max_allowed_end
            end_date = adjusted_end
            st.session_state.sidebar_end_date = adjusted_end
            st.info("End date was adjusted to be after start date.")

        # Item and Store selection with decoded IDs
        st.markdown("### 🏪 Selection Criteria")
        item_ids = st.multiselect(
            "Select Items", 
            options=ITEM_ID_OPTIONS, 
            default=(st.session_state.get("sidebar_item_ids_widget") or (st.session_state.sidebar_item_ids or [])),
            help="Choose one or more items to forecast (showing M5 format IDs)",
            key="sidebar_item_ids_widget"
        )
        
        store_ids = st.multiselect(
            "Select Stores", 
            options=STORE_ID_OPTIONS, 
            default=(st.session_state.get("sidebar_store_ids_widget") or (st.session_state.sidebar_store_ids or [])),
            help="Choose one or more stores to forecast (showing M5 format IDs)",
            key="sidebar_store_ids_widget"
        )

        # If user clears items or stores, immediately clear any existing forecast and related state
        if not item_ids or not store_ids:
            st.session_state.forecast_df = None
            st.session_state.show_sidebar_forecast = False
            st.session_state.selected_items = []
            st.session_state.selected_stores = []
            st.session_state.model_used = None

        # Stock level management with decoded IDs
        st.markdown("### 📦 Current Stock Levels")
        stock_inputs = {}
        if item_ids and store_ids:
            with st.expander("📝 Edit Stock Levels", expanded=False):
                for item in item_ids:
                    for store in store_ids:
                        key = f"stock_{item}_{store}"
                        stock_inputs[(item, store)] = st.number_input(
                            f"{item} @ {store}", 
                            min_value=0, 
                            value=st.session_state.stock_levels.get((item, store), 100),
                            step=10,
                            key=key
                        )
        else:
            st.info("💡 Select items and stores to set stock levels")

        # Model selection
        st.markdown("### 🤖 Model Selection")
        available_models = [name for name, model in models.items() if model is not None]
        
        if not available_models:
            st.error("❌ No models available")
            return None
        
        # Add "(recommended)" to LightGBM option
        display_models = []
        for model_name in available_models:
            if model_name == "LightGBM":
                display_models.append(f"{model_name} (recommended)")
            else:
                display_models.append(model_name)
            
        default_model = st.session_state.get('preferred_model_from_chat')
        if default_model not in available_models:
            default_model = available_models[0]
        
        selected_display_model = st.selectbox(
            "Choose Model", 
            options=display_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            help="Select the forecasting model to use"
        )
        
        # Extract the actual model name from the display name
        selected_model = selected_display_model.replace(" (recommended)", "")

        # Generate forecast button
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            clear_clicked = st.button("🗑️ Clear All", use_container_width=True)
        with col_b:
            gen_clicked = st.button("🚀 Generate Forecast", use_container_width=True, type="primary")

        if clear_clicked:
            # Clear all sidebar parameters and any forecast
            st.session_state.sidebar_item_ids = []
            st.session_state.sidebar_store_ids = []
            # Avoid mutating widget keys directly; set underlying state and rerun
            st.session_state.stock_levels = {}
            st.session_state.forecast_df = None
            st.session_state.selected_items = []
            st.session_state.selected_stores = []
            st.session_state.model_used = None
            st.session_state.show_sidebar_forecast = False
            st.success("Cleared all sidebar selections.")
            st.rerun()

        if gen_clicked:
            # Persist current selections back to session_state to keep chatbot and UI in sync
            st.session_state.sidebar_start_date = start_date
            st.session_state.sidebar_end_date = end_date
            st.session_state.sidebar_item_ids = item_ids
            st.session_state.sidebar_store_ids = store_ids
            ok = generate_forecast(start_date, end_date, item_ids, store_ids, stock_inputs, selected_model)
            if ok:
                st.session_state.show_sidebar_forecast = True
                # Immediately show a compact Forecast Summary after generation
                try:
                    df = st.session_state.forecast_df
                    if df is not None and not df.empty and 'predicted_sales' in df.columns:
                        total_sales = float(df['predicted_sales'].sum())
                        daily = df.groupby('date')['predicted_sales'].sum().reset_index() if 'date' in df.columns else None
                        avg_daily = float(daily['predicted_sales'].mean()) if daily is not None and not daily.empty else 0.0
                        if daily is not None and not daily.empty:
                            peak_idx = daily['predicted_sales'].idxmax()
                            peak_day = pd.to_datetime(daily.loc[peak_idx, 'date']).date()
                            peak_val = float(daily.loc[peak_idx, 'predicted_sales'])
                        else:
                            peak_day, peak_val = None, 0.0

                        model_used = st.session_state.get('model_used') or selected_model
                        items = item_ids
                        stores = store_ids

                        st.markdown("### 📈 Forecast Summary")
                        st.markdown(
                            f"""
                            <div style="background:#f8fafb;border:1px solid #e9ecef;border-radius:10px;padding:16px;margin-top:8px;">
                                <div style="font-weight:600;color:#495057;margin-bottom:6px;">{model_used}</div>
                                <div style="color:#6c757d;font-size:12px;margin-bottom:8px;">{len(items)} item(s) × {len(stores)} store(s)</div>
                                <div style="display:flex;gap:16px;flex-wrap:wrap;">
                                    <div>
                                        <div style="font-size:22px;color:#007BFF;font-weight:700;">{total_sales:,.0f}</div>
                                        <div style="color:#6c757d;font-size:12px;">Total Predicted Sales</div>
                                    </div>
                                    <div>
                                        <div style="font-size:22px;color:#28A745;font-weight:700;">{avg_daily:,.0f}</div>
                                        <div style="color:#6c757d;font-size:12px;">Avg Daily</div>
                                    </div>
                                    <div>
                                        <div style="font-size:22px;color:#6f42c1;font-weight:700;">{peak_val:,.0f}</div>
                                        <div style="color:#6c757d;font-size:12px;">Peak{f" on {peak_day}" if peak_day else ""}</div>
                                    </div>
                                </div>
                                <div style="margin-top:8px;color:#6c757d;font-size:12px;">Go to <strong>Forecast Dashboard</strong> for full analysis and charts.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
            return ok
        
        # Compact forecast summary in sidebar (if a forecast exists)
        if (
            st.session_state.get('show_sidebar_forecast')
            and st.session_state.get('forecast_df') is not None
            and st.session_state.get('sidebar_item_ids')
            and st.session_state.get('sidebar_store_ids')
        ):
            try:
                df = st.session_state.forecast_df
                total_sales = float(df['predicted_sales'].sum()) if 'predicted_sales' in df.columns else 0.0
                daily = df.groupby('date')['predicted_sales'].sum().reset_index() if 'date' in df.columns else None
                avg_daily = float(daily['predicted_sales'].mean()) if daily is not None and not daily.empty else 0.0
                if daily is not None and not daily.empty:
                    peak_idx = daily['predicted_sales'].idxmax()
                    peak_day = pd.to_datetime(daily.loc[peak_idx, 'date']).date()
                    peak_val = float(daily.loc[peak_idx, 'predicted_sales'])
                else:
                    peak_day, peak_val = None, 0.0
                sd = st.session_state.sidebar_start_date
                ed = st.session_state.sidebar_end_date
                items = st.session_state.get('selected_items') or st.session_state.sidebar_item_ids
                stores = st.session_state.get('selected_stores') or st.session_state.sidebar_store_ids
                model_used = st.session_state.get('model_used') or st.session_state.get('preferred_model_from_chat') or 'Model'
                st.markdown("### 📈 Forecast (Sidebar)")
                st.markdown(
                    f"""
                    <div style="background:#f8fafb;border:1px solid #e9ecef;border-radius:10px;padding:12px;">
                        <div style="font-weight:600;color:#495057;margin-bottom:6px;">{model_used}</div>
                        <div style="font-size:24px;color:#007BFF;font-weight:700;">{total_sales:,.0f}</div>
                        <div style="color:#6c757d;font-size:12px;">Total Predicted Sales</div>
                        <div style="margin-top:8px;color:#6c757d;font-size:12px;">
                            {len(items)} item(s) · {len(stores)} store(s)<br/>
                            {pd.to_datetime(sd).date()} → {pd.to_datetime(ed).date()}
                            <br/>Avg Daily: {avg_daily:,.0f}{' · Peak: ' + str(peak_day) + ' (' + format(peak_val, ',.0f') + ')' if peak_day else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # Mini graph removed per request
            except Exception:
                pass
        
        return None

def generate_forecast(start_date, end_date, item_ids, store_ids, stock_inputs, selected_model):
    """
    Forecast generator with ID encoding/decoding support
    """
    # Convert display IDs back to encoded IDs for model processing
    encoded_item_ids = [encode_item_id(item_id) for item_id in item_ids]
    encoded_store_ids = [encode_store_id(store_id) for store_id in store_ids]
    
    # -- validations --
    if not encoded_item_ids or not encoded_store_ids:
        st.error("❌ Please select at least one item and one store")
        return False
    if main_df is None:
        st.error("❌ Training data (main_df) not available")
        return False
    if selected_model not in models:
        st.error(f"❌ Model '{selected_model}' not found in models dict")
        return False

    try:
        with st.spinner(f"🔄 Generating forecasts with {selected_model}..."):
            # --- date parsing & source selection ---
            forecast_start = pd.to_datetime(start_date)
            forecast_end = pd.to_datetime(end_date)
            data_end = pd.to_datetime(main_df['date'].max())

            st.write(f"Forecast period: {forecast_start} to {forecast_end}")
            

            if actuals_df is not None and forecast_start > data_end:
                # st.info("🔮 Forecasting future dates beyond training data using actuals_df")
                source_df = actuals_df.copy()
                is_future_forecast = True
            else:
                st.info("📊 Predicting within training period using main_df")
                source_df = main_df.copy()
                is_future_forecast = False

            source_df['date'] = pd.to_datetime(source_df['date'])
            prediction_df = source_df[
                (source_df['date'] >= forecast_start) & (source_df['date'] <= forecast_end)
            ].copy()

            if prediction_df.empty:
                st.error("❌ No data found for selected date range")
                return False

            # --- categorical mapping ---
            categorical_features = [
                'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
            ]
            mappings = {}
            reverse_mappings = {}

            for col in categorical_features:
                if col in main_df.columns:
                    unique_vals = sorted(main_df[col].dropna().unique())
                    mappings[col] = {str(val): idx for idx, val in enumerate(unique_vals)}
                    reverse_mappings[col] = {idx: str(val) for idx, val in enumerate(unique_vals)}
                    mappings[col]['__MISSING__'] = len(unique_vals)
                    reverse_mappings[col][len(unique_vals)] = '__MISSING__'

            def map_column_to_codes(df_col, col_name):
                if col_name not in mappings:
                    if pd.api.types.is_numeric_dtype(df_col):
                        return df_col.fillna(0).astype(int)
                    else:
                        return df_col.fillna('__MISSING__').astype(str).map(lambda _: 0).astype(int)
                map_dict = mappings[col_name]
                string_series = df_col.fillna('__MISSING__').astype(str)
                mapped = string_series.map(map_dict)
                unmapped_count = mapped.isna().sum()
                if unmapped_count > 0:
                    unmapped_values = string_series[mapped.isna()].unique()[:5]
                    # st.warning(f"⚠️ {unmapped_count} unmapped values in '{col_name}': {list(unmapped_values)}")
                    mapped = mapped.fillna(map_dict['__MISSING__'])
                return mapped.astype(int)

            # --- future / lag handling ---
            if is_future_forecast:
                if 'sales' in prediction_df.columns:
                    prediction_df = prediction_df.drop(columns=['sales'])

                lag_features = [c for c in main_df.columns if 'sales_lag' in c or 'sales_roll' in c]
                if any(col not in prediction_df.columns for col in lag_features) or (
                    lag_features and prediction_df[lag_features].isna().any().any()
                ):
                    prediction_df = calculate_lag_features_for_forecast(prediction_df, main_df)

            # Keep raw copy for Prophet usage
            raw_prediction_df = prediction_df.copy()
            for col in categorical_features:
                if col in prediction_df.columns:
                    prediction_df[col] = map_column_to_codes(prediction_df[col], col)

            # Use encoded IDs for filtering
            prediction_df = prediction_df[
                (prediction_df['item_id'].isin(encoded_item_ids)) & 
                (prediction_df['store_id'].isin(encoded_store_ids))
            ].copy()
            raw_prediction_df = raw_prediction_df.loc[prediction_df.index]

            if prediction_df.empty:
                st.error("❌ No data found for selected items/stores after filtering")
                return False

            # --- prepare training feature columns ---
            features_to_drop = ['id', 'sales', 'date', 'weekday', 'wday', 'year',
                                'price_lag_1', 'price_change_pct', 'item_id_display', 'store_id_display']
            available_features_to_drop = [c for c in features_to_drop if c in main_df.columns]
            training_feature_columns = main_df.drop(columns=available_features_to_drop).columns.tolist()

            # Add missing features where needed
            missing_features = [c for c in training_feature_columns if c not in prediction_df.columns]
            if missing_features:
                prediction_df = add_missing_features(prediction_df, missing_features, main_df)

            features = prediction_df.reindex(columns=training_feature_columns, fill_value=0).copy()

            for col in categorical_features:
                if col in features.columns:
                    if not pd.api.types.is_numeric_dtype(features[col]):
                        features[col] = map_column_to_codes(features[col], col)
                    else:
                        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0).astype(int)

            for col in features.columns:
                if col in main_df.columns and pd.api.types.is_numeric_dtype(main_df[col]):
                    features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

            
            features_array = features.to_numpy(dtype='float32')

            # --- RESOLVE model entry ---
            raw_model_entry = models[selected_model]
            resolved_model = raw_model_entry

            if isinstance(raw_model_entry, dict) and not any(
                isinstance(v, (list, tuple, dict)) for v in raw_model_entry.values()
            ):
                for k in ('model', 'estimator', 'predictor', 'sk_model'):
                    if k in raw_model_entry and hasattr(raw_model_entry[k], 'predict'):
                        resolved_model = raw_model_entry[k]
                        break
            else:
                if isinstance(raw_model_entry, dict):
                    if all(hasattr(v, 'predict') for v in raw_model_entry.values()):
                        resolved_model = raw_model_entry
                    else:
                        for v in raw_model_entry.values():
                            if hasattr(v, 'predict'):
                                resolved_model = v
                                break
                elif isinstance(resolved_model, (list, tuple)):
                    for cand in resolved_model:
                        if hasattr(cand, 'predict'):
                            resolved_model = cand
                            break

            # --- Prediction phase ---
            if isinstance(resolved_model, dict):
                st.write("Debug: Resolved model is a dict of per-series estimators.")
                per_series_dict = resolved_model
                preds_array = np.zeros(len(prediction_df), dtype=float)

                if 'dept_id' not in raw_prediction_df.columns or 'store_id' not in raw_prediction_df.columns:
                    st.error("❌ Per-series model dict requires 'dept_id' and 'store_id' present.")
                    return False

                for (dept, store), sub in raw_prediction_df.groupby(['dept_id', 'store_id']):
                    key = f"{dept}_{store}"
                    idx = sub.index.values

                    if key not in per_series_dict:
                        st.warning(f"⚠️ No model for series {key}; filling predictions with 0")
                        preds_array[np.searchsorted(prediction_df.index.values, idx)] = 0.0
                        continue

                    model_obj = per_series_dict[key]
                    if hasattr(model_obj, 'regressors') or (
                        hasattr(model_obj, 'model') and model_obj.__class__.__name__.lower().startswith('prophetwrapper')
                    ):
                        input_df = raw_prediction_df.loc[idx].copy()
                        if 'date' not in input_df.columns and 'ds' in input_df.columns:
                            input_df = input_df.rename(columns={'ds': 'date'})
                        try:
                            sub_preds = model_obj.predict(input_df)
                        except Exception as e:
                            st.warning(f"⚠️ Per-series Prophet predict failed for {key}: {e}. Filling 0s.")
                            sub_preds = np.zeros(len(idx), dtype=float)
                    else:
                        # Try array input first; if the estimator expects a DataFrame (has .columns), retry with DataFrame
                        try:
                            sub_features_array = features.loc[idx, :].to_numpy(dtype='float32')
                            sub_preds = model_obj.predict(sub_features_df)
                        except Exception as e_array:
                            try:
                                sub_features_df = features.loc[idx, :]
                                sub_preds = model_obj.predict(sub_features_df)
                            except Exception as e_df:
                                st.warning(
                                    f"⚠️ Per-series numeric estimator predict failed for {key}: {e_df}. Filling 0s."
                                )
                            sub_preds = np.zeros(len(idx), dtype=float)

                    pos = np.searchsorted(prediction_df.index.values, idx)
                    preds_array[pos] = np.asarray(sub_preds).squeeze()

                preds = preds_array

            else:
                is_prophet_wrapper = False
                if hasattr(resolved_model, 'regressors') or (
                    hasattr(resolved_model, 'model') and resolved_model.__class__.__name__.lower().startswith('prophetwrapper')
                ):
                    is_prophet_wrapper = True
                elif hasattr(resolved_model, 'model') and resolved_model.model.__class__.__name__.lower().startswith('prophet'):
                    is_prophet_wrapper = True

                if is_prophet_wrapper:
                    st.write("Debug: Resolved model appears to be a Prophet wrapper/estimator. Using DataFrame input for predict.")
                    input_df = raw_prediction_df.copy()
                    if 'date' not in input_df.columns and 'ds' in input_df.columns:
                        input_df = input_df.rename(columns={'ds': 'date'})
                    regs = getattr(resolved_model, 'regressors', None)
                    if regs is None and hasattr(resolved_model, 'model') and hasattr(resolved_model.model, 'extra_regressors'):
                        regs = list(getattr(resolved_model.model, 'extra_regressors', {}).keys())
                    if regs:
                        cols_to_keep = ['date'] + [r for r in regs if r in input_df.columns]
                        input_df = input_df[cols_to_keep].copy()
                    else:
                        input_df = input_df[['date']].copy()
                    try:
                        preds = resolved_model.predict(input_df)
                    except Exception as e:
                        st.error(f"❌ Prophet predict failed: {e}")
                        return False
                else:
                    model = resolved_model
                    try:
                        # LGBM FEATURE ALIGNMENT
                        if selected_model.lower().startswith("lgbm") or model.__class__.__name__.lower().startswith("lgbm"):
                            expected_features = getattr(model, "feature_name_", training_feature_columns)
                            # Add missing
                            for col in expected_features:
                                if col not in features.columns:
                                    features[col] = 0
                            # Drop extras
                            features = features[expected_features]
                            # st.write(f"Debug: LightGBM aligned features: {features.shape}")
                            preds = model.predict(features)
                        else:
                            # For estimators that accept arrays, try array first; if they require DataFrame (use .columns), fallback.
                            try:
                                preds = model.predict(features_array)
                            except Exception:
                                preds = model.predict(features)
                    except Exception as e:
                        st.warning(f"⚠️ Model.predict(features_array) failed with: {e}. Trying DataFrame fallback.")
                        try:
                            preds = model.predict(features)
                        except Exception as e2:
                            st.error(f"❌ Both array and DataFrame predict failed: {e2}")
                            return False

            # --- normalize predictions ---
            preds = np.asarray(preds).squeeze()
            if preds.ndim > 1 and preds.shape[1] == 1:
                preds = preds.ravel()
            preds = np.maximum(0, preds).astype(int)

            # Create forecast dataframe with both encoded and decoded IDs
            forecast_df = pd.DataFrame({
                'item_id': prediction_df['item_id'].values,  # Keep encoded for internal processing
                'store_id': prediction_df['store_id'].values,  # Keep encoded for internal processing
                'date': raw_prediction_df['date'].values,
                'predicted_sales': preds
            })
            
            # Add decoded IDs for display
            forecast_df['item_id_display'] = forecast_df['item_id'].apply(decode_item_id)
            forecast_df['store_id_display'] = forecast_df['store_id'].apply(decode_store_id)

            st.session_state.forecast_df = forecast_df
            st.session_state.selected_items = item_ids  # Store original display IDs
            st.session_state.selected_stores = store_ids  # Store original display IDs
            st.session_state.stock_levels = stock_inputs
            st.session_state.model_used = selected_model
            
            # Update graph generator and agents with new forecast data
            if GRAPH_TOOLS_AVAILABLE:
                init_graph_generator(main_df, actuals_df, forecast_df)
            if GRAPH_AGENT_AVAILABLE:
                init_graph_agent(main_df, actuals_df, forecast_df)
            if ANALYTICS_AGENT_AVAILABLE:
                init_analytics_agent(main_df, actuals_df, forecast_df, models)

            st.success(f"✅ Forecast generated successfully!")
            # Summary card is displayed on dashboard; avoid duplicate card here

            return True

    except Exception as e:
        st.error(f"❌ Forecast generation failed: {str(e)}")
        import traceback
        st.text("Full traceback:")
        st.text(traceback.format_exc())
        return False


# --- HELPER FUNCTIONS ---
def calculate_lag_features_for_forecast(prediction_df, historical_df):
    """ OPTIMIZED: Calculate lag features for future forecasting using vectorized operations """
    st.write("🔧 Calculating lag features (optimized)...")

    lag_windows = [7, 14, 28, 35, 56]
    roll_windows = [7, 14, 28, 56]

    historical_df = historical_df.copy()
    historical_df['date'] = pd.to_datetime(historical_df['date'])

    unique_combinations = prediction_df[['item_id', 'store_id']].drop_duplicates()
    st.write(f"Processing {len(unique_combinations)} unique item-store combinations...")

    hist_grouped = historical_df.groupby(['item_id', 'store_id'])['sales']
    hist_stats = hist_grouped.agg(['mean', 'std', 'count']).reset_index()
    hist_stats.columns = ['item_id', 'store_id', 'hist_mean', 'hist_std', 'hist_count']
    hist_stats['hist_std'] = hist_stats['hist_std'].fillna(0.5)

    prediction_df = prediction_df.merge(hist_stats, on=['item_id', 'store_id'], how='left')

    for lag in lag_windows:
        prediction_df[f'sales_lag_{lag}'] = prediction_df['hist_mean'].fillna(1.0)
    for window in roll_windows:
        prediction_df[f'sales_roll_mean_{window}'] = prediction_df['hist_mean'].fillna(1.0)
        prediction_df[f'sales_roll_std_{window}'] = prediction_df['hist_std'].fillna(0.5)

    sufficient_history = hist_stats[hist_stats['hist_count'] >= max(lag_windows)]
    if len(sufficient_history) > 0:
        st.write(f"Found {len(sufficient_history)} combinations with sufficient history")
        batch_size = 100
        for i in range(0, len(sufficient_history), batch_size):
            batch = sufficient_history.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                item_id, store_id = row['item_id'], row['store_id']
                hist_mask = (
                    (historical_df['item_id'] == item_id) &
                    (historical_df['store_id'] == store_id)
                )
                hist_sales = historical_df[hist_mask].sort_values('date')['sales'].values
                if len(hist_sales) >= max(lag_windows):
                    recent_sales = hist_sales[-max(lag_windows):]
                    pred_mask = (
                        (prediction_df['item_id'] == item_id) &
                        (prediction_df['store_id'] == store_id)
                    )
                    if pred_mask.any():
                        for lag in lag_windows:
                            if len(recent_sales) >= lag:
                                prediction_df.loc[pred_mask, f'sales_lag_{lag}'] = recent_sales[-lag]
                        for window in roll_windows:
                            if len(recent_sales) >= window:
                                window_data = recent_sales[-window:]
                                prediction_df.loc[pred_mask, f'sales_roll_mean_{window}'] = np.mean(window_data)
                                prediction_df.loc[pred_mask, f'sales_roll_std_{window}'] = np.std(window_data)
            if i % 500 == 0:
                st.write(f"Processed {min(i + batch_size, len(sufficient_history))} / {len(sufficient_history)} combinations")
    else:
        st.warning("⚠️ No combinations with sufficient historical data found. Using statistical fallbacks.")

    prediction_df = prediction_df.drop(columns=['hist_mean', 'hist_std', 'hist_count'], errors='ignore')
    st.write("✅ Lag feature calculation completed")
    return prediction_df


def add_missing_features(prediction_df, missing_features, historical_df):
    """ Add missing features to prediction dataframe with intelligent defaults """
    hist_stats = {}
    if 'sales' in historical_df.columns:
        hist_stats['sales_mean'] = historical_df['sales'].mean()
        hist_stats['sales_std'] = historical_df['sales'].std()
    if 'sell_price' in historical_df.columns:
        hist_stats['price_median'] = historical_df['sell_price'].median()

    for feat in missing_features:
        if feat in prediction_df.columns:
            continue
        if 'sales_lag' in feat:
            prediction_df[feat] = hist_stats.get('sales_mean', 1.0)
        elif 'sales_roll_mean' in feat:
            prediction_df[feat] = hist_stats.get('sales_mean', 1.0)
        elif 'sales_roll_std' in feat:
            prediction_df[feat] = hist_stats.get('sales_std', 0.5)
        elif feat == 'day_of_week':
            prediction_df[feat] = prediction_df['date'].dt.dayofweek
        elif feat == 'is_weekend':
            prediction_df[feat] = (prediction_df['date'].dt.dayofweek >= 5).astype(int)
        elif 'price' in feat.lower():
            prediction_df[feat] = prediction_df.get('sell_price', hist_stats.get('price_median', 2.5))
        elif feat == 'month':
            prediction_df[feat] = prediction_df['date'].dt.month
        else: # Default fallback 
            prediction_df[feat] = 0.0
        
    return prediction_df

# --- 6. LLM INTEGRATION ---
def get_llm_analysis(forecast_data, model_name):
    """Enhanced LLM analysis with better prompting and decoded IDs"""
    try:
        total_sales = forecast_data['predicted_sales'].sum()
        avg_daily = forecast_data.groupby('date')['predicted_sales'].sum().mean()
        peak_day = forecast_data.groupby('date')['predicted_sales'].sum().idxmax()
        peak_sales = forecast_data.groupby('date')['predicted_sales'].sum().max()
        
        # Use decoded IDs in analysis
        unique_items = forecast_data['item_id_display'].nunique() if 'item_id_display' in forecast_data.columns else forecast_data['item_id'].nunique()
        unique_stores = forecast_data['store_id_display'].nunique() if 'store_id_display' in forecast_data.columns else forecast_data['store_id'].nunique()
        
        analysis = f"""
## AI-Powered Forecast Analysis

**Model Used:** {model_name}

**Scope:** {unique_items} items across {unique_stores} stores

**Key Insights:**
- **Total Predicted Sales:** {total_sales:,} units
- **Daily Average:** {avg_daily:.0f} units/day  
- **Peak Day:** {peak_day.strftime('%A, %B %d, %Y')} ({peak_sales:,} units)

**Strategic Recommendations:**
1. **Inventory Planning:** Ensure adequate stock levels, especially approaching {peak_day.strftime('%A')}s
2. **Staffing:** Scale workforce for peak demand periods
3. **Marketing:** Focus promotional activities during high-demand forecasts

**Risk Assessment:** The {model_name} model shows {'high confidence' if total_sales > 1000 else 'moderate confidence'} 
in these predictions based on historical patterns and seasonal trends.
        """
        
        return analysis
        
    except Exception as e:
        return f"Analysis temporarily unavailable: {e}"

# --- 7. PAGE FUNCTIONS ---

def home_page():
    """Enhanced home page with better design"""
    st.markdown("## Welcome to InventoryGPT")
    st.markdown("**Advanced ML-powered retail sales forecasting and inventory optimization**")
    st.markdown("---")

    # Enhanced metrics display
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Records", "11.1M+", "#007BFF"),
        ("Unique Items", "3,049", "#28A745"),  
        ("Unique Stores", "10", "#FD7E14"),
        ("Time Range", "2011-2016", "#6F42C1")
    ]
    
    for i, (title, value, color) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value" style="color: {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features section
    st.markdown("### Application Features")
    
    features = [
        ("Dynamic Forecasting", "Select date ranges, items, stores, and ML models for real-time predictions"),
        ("AI-Powered Analysis", "Get intelligent insights and recommendations from advanced language models"),
        ("Smart Inventory", "Compare forecasts with current stock levels and get optimization suggestions"),
        ("Visual Analytics", "Interactive charts comparing actual vs predicted sales with performance metrics"),
        ("Intelligent Chatbot", "Ask questions about data, models, and get contextual assistance"),
        ("Multi-Model Support", "Choose from LightGBM, XGBoost, and Prophet models")
    ]
    
    for title, desc in features:
        st.markdown(f"""
        <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #007BFF;">
            <h4 style="color: #007BFF; margin: 0 0 8px 0;">{title}</h4>
            <p style="margin: 0; color: #6C757D;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick start guide
    # If there is no forecast and no selections, surface an error hint here too
    no_items = not st.session_state.get('sidebar_item_ids')
    no_stores = not st.session_state.get('sidebar_store_ids')
    no_forecast = st.session_state.get('forecast_df') is None
    if no_forecast and (no_items or no_stores):
        st.markdown("""
        <div class="warning-alert">
            <h4>Setup Required</h4>
            <p>Please select at least 1 item and 1 store in the sidebar, then click Generate Forecast.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Quick Start Guide")
    st.markdown("""
    1. **Configure Settings:** Use the sidebar to select date range, items, stores, and model
    2. **Generate Forecast:** Click the 'Generate Forecast' button to create predictions
    3. **Analyze Results:** Navigate to 'Forecast Dashboard' to view AI analysis and insights
    4. **Optimize Inventory:** Check stocking recommendations in the dashboard
    5. **Ask Questions:** Use the chatbot for additional support and insights
    """)

def forecast_dashboard_page():
    """Enhanced forecast dashboard with better analytics and decoded IDs"""
    st.markdown("## Forecast Dashboard")
    
    if st.session_state.forecast_df is None:
        st.markdown("""
        <div class="warning-alert">
            <h4>No Forecast Available</h4>
            <p>Please select at least 1 item and 1 store in the sidebar, then click Generate Forecast.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    forecast_df = st.session_state.forecast_df
    items = st.session_state.selected_items
    stores = st.session_state.selected_stores
    model = st.session_state.model_used
    
    # Dashboard header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #007BFF, #28A745); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h3 style="margin: 0;">Forecast Summary</h3>
        <p style="margin: 5px 0 0 0;">
            {len(items)} item(s) × {len(stores)} store(s) | Model: <strong>{model}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Forecast Analysis", "Inventory Analysis", "Performance Comparison"])

    with tab1:
        st.markdown("### AI-Powered Insights")
        
        # Get and display LLM analysis
        with st.spinner("AI analyzing forecast patterns..."):
            analysis = get_llm_analysis(forecast_df, model)
            st.markdown(analysis)
            
        # Additional visualizations
        daily_forecast = forecast_df.groupby('date')['predicted_sales'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_forecast['date'], 
            y=daily_forecast['predicted_sales'],
            mode='lines+markers',
            name='Daily Forecast',
            line=dict(color='#007BFF', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Daily Sales Forecast Trend",
            xaxis_title="Date",
            yaxis_title="Predicted Sales (Units)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Smart Inventory Management")
        
        if not st.session_state.stock_levels:
            st.warning("No stock levels configured. Set stock levels in sidebar to see analysis.")
            return
            
        # Calculate inventory metrics using decoded IDs for display
        demand_analysis = forecast_df.groupby(['item_id_display', 'store_id_display'])['predicted_sales'].sum().reset_index()
        demand_analysis.rename(columns={'predicted_sales': 'total_demand'}, inplace=True)
        
        # Add current stock levels using display IDs
        demand_analysis['current_stock'] = demand_analysis.apply(
            lambda row: st.session_state.stock_levels.get((row['item_id_display'], row['store_id_display']), 0),
            axis=1
        )
        
        demand_analysis['difference'] = demand_analysis['current_stock'] - demand_analysis['total_demand']
        # demand_analysis['coverage_days'] = demand_analysis['current_stock'] / (demand_analysis['total_demand'] / 30)
        demand_analysis['coverage_days'] = demand_analysis['current_stock'] / (demand_analysis['total_demand'] / max(1, (forecast_df['date'].max() - forecast_df['date'].min()).days + 1))
        # Status classification
        def classify_status(row):
            diff = row['difference']
            demand = row['total_demand']
            
            if diff < 0:
                return "Understocked"
            elif demand >= 0 and diff > (demand * 0.5):
                return "Overstocked"  
            else:
                return "Optimal"
                
        demand_analysis['status'] = demand_analysis.apply(classify_status, axis=1)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            understocked = len(demand_analysis[demand_analysis['status'] == 'Understocked'])
            st.markdown(
                f"""
                <div style="
                    background-color:#ffe6e6;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="color:#b30000; margin-bottom:8px;">Understocked Items</h4>
                    <h1 style="color:#ff0000; font-size:40px; margin:0;">{understocked}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            overstocked = len(demand_analysis[demand_analysis['status'] == 'Overstocked'])
            st.markdown(
                f"""
                <div style="
                    background-color:#fff8e1;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="color:#b36b00; margin-bottom:8px;">Overstocked Items</h4>
                    <h1 style="color:#ff9900; font-size:40px; margin:0;">{overstocked}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            optimal = len(demand_analysis[demand_analysis['status'] == 'Optimal'])
            st.markdown(
                f"""
                <div style="
                    background-color:#e6ffe6;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="color:#006600; margin-bottom:8px;">Optimal Items</h4>
                    <h1 style="color:#009933; font-size:40px; margin:0;">{optimal}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Inventory status table with renamed columns for clarity
        display_analysis = demand_analysis.copy()
        display_analysis = display_analysis.rename(columns={
            'item_id_display': 'Item ID',
            'store_id_display': 'Store ID',
            'total_demand': 'Total Demand',
            'current_stock': 'Current Stock',
            'difference': 'Stock Difference',
            'coverage_days': 'Coverage Days',
            'status': 'Status'
        })
        
        st.markdown("#### Detailed Inventory Status")
        def highlight_row(row):
            if row["Status"] == "Understocked":
                return ["background-color: #ffe6e6; color: #b30000; font-weight: bold;"] * len(row)
            elif row["Status"] == "Overstocked":
                return ["background-color: #fff8e1; color: #b36b00; font-weight: bold;"] * len(row)
            elif row["Status"] == "Optimal":
                return ["background-color: #e6ffe6; color: #006600; font-weight: bold;"] * len(row)
            else:
                return [""] * len(row)

        # Apply styling row-wise
        styled_df = display_analysis.style.apply(highlight_row, axis=1)

        # Show styled dataframe
        st.dataframe(styled_df, use_container_width=True)
        
        # Recommendations with decoded IDs
        critical_items = demand_analysis[demand_analysis['difference'] < 0].sort_values('difference')
        if not critical_items.empty:
            st.markdown("#### 🚨 Immediate Action Required")
            for _, item in critical_items.head(5).iterrows():
                shortage = abs(item['difference'])

                st.markdown(
                    f"""
                    <div style="
                        background-color:#ffe6e6;
                        padding:12px;
                        border-radius:10px;
                        margin-bottom:10px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.1);
                    ">
                        <span style="font-size:16px; font-weight:600; color:#b30000;">
                            {item['item_id_display']} @ {item['store_id_display']}
                        </span><br>
                        <span style="font-size:28px; font-weight:700; color:#ff0000;">
                            Short by {shortage:.0f} units
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
    with tab3:
        st.markdown("### Actual vs Predicted Performance")
        
        comparison_df = actuals_df if actuals_df is not None else main_df
        
        if comparison_df is not None:
            # Convert display IDs back to encoded for comparison
            encoded_items = [encode_item_id(item) for item in items]
            encoded_stores = [encode_store_id(store) for store in stores]
            
            # Filter actual data for comparison using encoded IDs
            actual_data = comparison_df[
                (comparison_df['item_id'].isin(encoded_items)) &
                (comparison_df['store_id'].isin(encoded_stores)) &
                (comparison_df['date'].isin(forecast_df['date'].unique()))
            ]
            
            if not actual_data.empty:
                # Aggregate data
                actual_daily = actual_data.groupby('date')['sales'].sum().reset_index()
                pred_daily = forecast_df.groupby('date')['predicted_sales'].sum().reset_index()
                
                # Merge for comparison
                comparison = pd.merge(pred_daily, actual_daily, on='date', how='outer').fillna(0)
                
                # Performance metrics
                if len(comparison) > 0 and comparison['sales'].sum() > 0:
                    
                    # Calculate basic metrics
                    mae = np.mean(np.abs(comparison['sales'] - comparison['predicted_sales']))
                    rmse = np.sqrt(np.mean((comparison['sales'] - comparison['predicted_sales']) ** 2))
                    
                    # Calculate MAPE with handling for zero actuals
                    non_zero_mask = comparison['sales'] != 0
                    if non_zero_mask.sum() > 0:
                        mape = np.mean(np.abs((comparison.loc[non_zero_mask, 'sales'] - comparison.loc[non_zero_mask, 'predicted_sales']) / 
                                             comparison.loc[non_zero_mask, 'sales'])) * 100
                    else:
                        mape = float('inf')
                    
                    # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
                    # SMAPE = 2 * |actual - predicted| / (|actual| + |predicted|)
                    smape_denominator = np.abs(comparison['sales']) + np.abs(comparison['predicted_sales'])
                    non_zero_smape_mask = smape_denominator != 0
                    if non_zero_smape_mask.sum() > 0:
                        smape = 2 * np.mean(np.abs(comparison.loc[non_zero_smape_mask, 'sales'] - comparison.loc[non_zero_smape_mask, 'predicted_sales']) / 
                                          smape_denominator[non_zero_smape_mask]) * 100
                    else:
                        smape = float('inf')
                    
                    # Calculate MAAPE (Mean Arctangent Absolute Percentage Error)
                    # MAAPE = mean(arctan(|actual - predicted| / |actual|))
                    if non_zero_mask.sum() > 0:
                        maape = np.mean(np.arctan(np.abs(comparison.loc[non_zero_mask, 'sales'] - comparison.loc[non_zero_mask, 'predicted_sales']) / 
                                                np.abs(comparison.loc[non_zero_mask, 'sales']))) * 180 / np.pi  # Convert to degrees
                    else:
                        maape = float('inf')
                    
                    # Calculate RMSLE (Root Mean Square Logarithmic Error)
                    # RMSLE = sqrt(mean((log(1 + actual) - log(1 + predicted))^2))
                    # Handle negative values by adding 1 to both actual and predicted
                    actual_plus_one = comparison['sales'] + 1
                    predicted_plus_one = comparison['predicted_sales'] + 1
                    # Ensure positive values for log
                    actual_plus_one = np.maximum(actual_plus_one, 1e-10)
                    predicted_plus_one = np.maximum(predicted_plus_one, 1e-10)
                    rmsle = np.sqrt(np.mean((np.log(actual_plus_one) - np.log(predicted_plus_one)) ** 2))
                    
                    # --- global CSS (do this once, anywhere before the cards) ---
                    st.markdown("""
                    <style>
                    .metric-card{
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px;
                    padding: 16px 18px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.20);
                    }
                    .metric-title{
                    font-size: 14px;
                    font-weight: 600;
                    letter-spacing: .3px;
                    color: #c7c9cc;
                    margin-bottom: 6px;
                    }
                    .metric-value{
                    font-size: 40px;
                    font-weight: 800;
                    line-height: 1;
                    margin: 0;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # First row of metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        mape_display = f"{mape:.2f}%" if mape != float('inf') else "N/A"
                        mape_color = "#28A745" if mape != float('inf') else "#6C757D"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">MAPE</div>
                                <div class="metric-value" style="color:{mape_color};">{mape_display}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">RMSE</div>
                                <div class="metric-value" style="color:#007BFF;">{rmse:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">MAE</div>
                                <div class="metric-value" style="color:#FD7E14;">{mae:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Second row of metrics
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        smape_display = f"{smape:.2f}%" if smape != float('inf') else "N/A"
                        smape_color = "#6F42C1" if smape != float('inf') else "#6C757D"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">SMAPE</div>
                                <div class="metric-value" style="color:{smape_color};">{smape_display}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col5:
                        maape_display = f"{maape:.2f}°" if maape != float('inf') else "N/A"
                        maape_color = "#E83E8C" if maape != float('inf') else "#6C757D"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">MAAPE</div>
                                <div class="metric-value" style="color:{maape_color};">{maape_display}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col6:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">RMSLE</div>
                                <div class="metric-value" style="color:#20C997;">{rmsle:.4f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Add metric explanations
                with st.expander("📊 Understanding the Metrics", expanded=False):
                    st.markdown("""
                    **Performance Metrics Explained:**
                    
                    - **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values
                    - **RMSE (Root Mean Square Error):** Square root of average squared differences, penalizes large errors more
                    - **MAPE (Mean Absolute Percentage Error):** Average percentage error, shows relative accuracy
                    - **SMAPE (Symmetric Mean Absolute Percentage Error):** Symmetric version of MAPE, handles zero values better
                    - **MAAPE (Mean Arctangent Absolute Percentage Error):** Uses arctangent function, more robust to outliers
                    - **RMSLE (Root Mean Square Logarithmic Error):** Logarithmic scale error, good for data with wide value ranges
                    
                    **Note:** N/A appears when actual values are zero, making percentage calculations impossible.
                    """)

                
                # Comparison visualization
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
                    title="Actual vs Predicted Sales Comparison",
                    xaxis_title="Date",
                    yaxis_title="Sales (Units)",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Detailed Comparison Data")

                # Add absolute difference column
                comparison["Difference"] = comparison["predicted_sales"] - comparison["sales"]

                # Styling function for Difference column
                def highlight_difference(val):
                    if val == 0:  # Perfect prediction
                        return "background-color: #e6ffe6; color: #006600; font-weight: bold;"  # green
                    elif val > 0:  # Over-predicted
                        return "background-color: #fff8e1; color: #b36b00; font-weight: bold;"  # yellow
                    else:  # Under-predicted
                        return "background-color: #ffe6e6; color: #b30000; font-weight: bold;"  # red

                # Apply styling only on Difference column
                styled_df = comparison.style.applymap(highlight_difference, subset=["Difference"])

                # Show styled dataframe
                st.dataframe(styled_df, use_container_width=True)

                # Add legend below the table
                st.markdown(
                    """
                    **Legend:**
                    - 🟩 Green → Actual = Predicted
                    - 🟨 Yellow → Forecast > Actual
                    - 🟥 Red → Forecast < Actual
                    """,
                    unsafe_allow_html=True
                )

                
            else:
                st.warning("No actual data available for selected items/stores in the forecast period")
        else:
            st.error("Actual sales data not available for comparison")

def view_dataset_page():
    """Enhanced dataset viewer with filtering capabilities and decoded IDs"""
    st.markdown("## Dataset Explorer")
    st.markdown("**Explore the feature-engineered M5 dataset used for model training**")
    st.markdown("---")
    
    if main_df is None:
        st.error("Dataset not available")
        return
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(main_df):,}")
    with col2:
        st.metric("Columns", len(main_df.columns))
    with col3:
        st.metric("Date Range", f"{main_df['date'].min().year}-{main_df['date'].max().year}")
    with col4:
        st.metric("Size", f"{main_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Filtering options with decoded IDs
    st.markdown("### Data Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_items_filter = st.multiselect(
            "Filter by Item ID", 
            options=ITEM_ID_OPTIONS[:20],  # Using decoded IDs
            default=[]
        )
    
    with col2:
        selected_stores_filter = st.multiselect(
            "Filter by Store ID",
            options=STORE_ID_OPTIONS,  # Using decoded IDs
            default=[]
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(main_df['date'].min(), main_df['date'].max()),
            min_value=main_df['date'].min(),
            max_value=main_df['date'].max()
        )
    
    # Apply filters using display columns
    filtered_df = main_df.copy()
    
    if selected_items_filter:
        filtered_df = filtered_df[filtered_df['item_id_display'].isin(selected_items_filter)]
    
    if selected_stores_filter:
        filtered_df = filtered_df[filtered_df['store_id_display'].isin(selected_stores_filter)]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['date'] <= pd.to_datetime(date_range[1]))
        ]
    
    st.info(f"Displaying {len(filtered_df):,} rows (limited to first 10,000 rows)")
    
    # Data quality indicators
    with st.expander("Data Quality Summary", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values:**")
            # Exclude display columns from missing value analysis
            analysis_df = filtered_df.drop(columns=['item_id_display', 'store_id_display'], errors='ignore')
            missing_data = analysis_df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_data) > 0:
                st.dataframe(missing_data.head(10))
            else:
                st.success("No missing values detected")
        
        with col2:
            st.markdown("**Data Types:**")
            dtype_counts = analysis_df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"**{dtype}:** {count} columns")
    
    # Display the dataset with both encoded and decoded IDs
    st.markdown("### Dataset Preview")
    display_df = filtered_df.head(10000)
    
    # Reorder columns to show display IDs prominently
    if 'item_id_display' in display_df.columns and 'store_id_display' in display_df.columns:
        cols = ['date', 'item_id_display', 'store_id_display', 'sales']
        remaining_cols = [col for col in display_df.columns if col not in cols]
        display_df = display_df[cols + remaining_cols]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download option
    if st.button("Download Filtered Data (CSV)", use_container_width=False):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"m5_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def chatbot_page():
    """Enhanced chatbot with simulated intelligent responses and graph generation"""
    st.markdown("## Intelligent Assistant")
    st.markdown("**Ask questions about the M5 dataset, forecasting models, or application features**")
    
    # Add information about graph generation and analytics capabilities
    if GRAPH_AGENT_AVAILABLE or GRAPH_TOOLS_AVAILABLE or ANALYTICS_AGENT_AVAILABLE:
        st.markdown("**All the tools are available**")
        pass
    
    st.markdown("---")
    
    # Initialize chat if empty
    if not st.session_state.chat_messages:
        st.session_state.chat_messages = [{
            "role": "assistant", 
            "content": "Hi — how can I help with your forecasting work today?"
        }]
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Clear conversation button (bottom control above input)
    st.markdown("---")
    if st.button("🧹 Clear Conversation", use_container_width=True, type="primary"):
        st.session_state.chat_messages = []
        st.session_state.chat_history = []
        # Also clear all forecast/session selections
        st.session_state.forecast_df = None
        st.session_state.selected_items = []
        st.session_state.selected_stores = []
        st.session_state.stock_levels = {}
        st.session_state.model_used = None
        st.session_state.show_sidebar_forecast = False
        st.session_state.sidebar_item_ids = []
        st.session_state.sidebar_store_ids = []
        # Delete persisted chat memory if present
        try:
            import os
            user_id = st.session_state.get("user_id") or "default_user"
            mem_path = f"/Users/dhineashkumar/Desktop/inventory-gpt/cache/chat_{user_id}.json"
            if os.path.exists(mem_path):
                os.remove(mem_path)
        except Exception:
            pass
        st.success("Conversation cleared. Starting a fresh chat.")
        st.rerun()

    # Sidebar controls (explain with code context removed)
    with st.sidebar:
        pass

    # Chat input
    if prompt := st.chat_input("Ask me anything about M5 forecasting..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chatbot_response(prompt)
                if isinstance(response, dict):
                    st.markdown(response.get("text", ""))
                    st.session_state.chat_messages.append({"role": "assistant", "content": response.get("text", "")})
                else:
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# Global memory for conversation history (persist per user across reloads)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    try:
        import json, os
        user_id = st.session_state.get("user_id") or "default_user"
        mem_path = f"/Users/dhineashkumar/Desktop/inventory-gpt/cache/chat_{user_id}.json"
        if os.path.exists(mem_path):
            with open(mem_path, "r", encoding="utf-8") as f:
                st.session_state.chat_history = json.load(f)
    except Exception:
        pass

def _normalize_item_tokens(text: str) -> str:
    """Convert variants like 'FOODS 1 001' to 'FOODS_1_001'."""
    try:
        # Upper-case and replace hyphens with underscores
        t = text.upper().replace("-", "_")
        # Replace multiple spaces with single underscore inside typical item specs
        t = re.sub(r"\b([A-Z]+)\s+(\d)\s+(\d{3})\b", r"\1_\2_\3", t)
        return t
    except Exception:
        return text

def _state_name_to_code(name: str) -> Optional[str]:
    """Map state words to M5 state codes."""
    mapping = {
        "CALIFORNIA": "CA",
        "TEXAS": "TX",
        "WISCONSIN": "WI",
        "CA": "CA",
        "TX": "TX",
        "WI": "WI",
    }
    return mapping.get(name.strip().upper())

def update_filters_from_prompt(prompt: str) -> Tuple[bool, Optional[str]]:
    """Parse prompt for dates, item_ids, store_ids, and preferred model; update session state.

    Returns (updated_any, desired_model).
    """
    updated_any = False
    desired_model = None

    try:
        normalized = _normalize_item_tokens(prompt)

        # Enhanced date extraction
        date_updated = _extract_dates_from_prompt(normalized)
        if date_updated:
            updated_any = True

        # Item IDs: support FOODS_1_001 and FOODS 1 001 after normalization
        if 'ITEM_ID_OPTIONS' in globals():
            item_candidates = re.findall(r"\b([A-Z]+_\d_\d{3})\b", normalized)
            valid_items = [itm for itm in item_candidates if itm in ITEM_ID_OPTIONS]
            if valid_items:
                new_items = list(dict.fromkeys(valid_items))
                if st.session_state.sidebar_item_ids != new_items:
                    st.session_state.sidebar_item_ids = new_items
                    st.session_state.sidebar_item_ids_widget = new_items
                    updated_any = True

        # Store IDs: support CA_1, 'CA 1', and 'California 1'
        store_ids = []
        # Matches like CA_1 or CA 1
        for m in re.findall(r"\b([A-Z]{2})[\s_]?([0-9]+)\b", normalized):
            state, num = m[0], m[1]
            sid = f"{state}_{num}"
            store_ids.append(sid)
        # Matches like 'California 1'
        for m in re.findall(r"\b([A-Za-z]{3,})\s+([0-9]+)\b", normalized):
            state_code = _state_name_to_code(m[0])
            if state_code:
                store_ids.append(f"{state_code}_{m[1]}")
        if 'STORE_ID_OPTIONS' in globals() and store_ids:
            valid_stores = [sid for sid in store_ids if sid in STORE_ID_OPTIONS]
            if valid_stores:
                new_stores = list(dict.fromkeys(valid_stores))
                if st.session_state.sidebar_store_ids != new_stores:
                    st.session_state.sidebar_store_ids = new_stores
                    st.session_state.sidebar_store_ids_widget = new_stores
                    updated_any = True

        # Model preference
        lowered = normalized.lower()
        if 'xgboost' in lowered or 'xgb' in lowered:
            desired_model = 'XGBoost'
        elif 'prophet' in lowered:
            desired_model = 'Prophet'
        # LSTM removed by request
        elif 'lightgbm' in lowered or 'lgbm' in lowered:
            desired_model = 'LightGBM'

    except Exception:
        pass

    return updated_any, desired_model

def _extract_dates_from_prompt(prompt: str) -> bool:
    """Extract dates from prompt and update session state. Returns True if dates were updated."""
    try:
        # Date extraction: explicit ISO-like dates
        date_matches = re.findall(r"(20\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01]))", prompt)
        if len(date_matches) >= 2:
            d1 = pd.to_datetime(date_matches[0]).date()
            d2 = pd.to_datetime(date_matches[1]).date()
            sd, ed = (d1, d2) if d1 <= d2 else (d2, d1)
            if st.session_state.sidebar_start_date != sd or st.session_state.sidebar_end_date != ed:
                st.session_state.sidebar_start_date = sd
                st.session_state.sidebar_end_date = ed
                return True
        
        # Enhanced month range handling: "Jan 2015 to Mar 2015" or "from Jan 2015 to Mar 2015"
        month_names = "january february march april may june july august september october november december".split()
        month_abbr = "jan feb mar apr may jun jul aug sep sept oct nov dec".split()
        
        # Pattern for month ranges: "Jan 2015 to Mar 2015" or "from Jan 2015 to Mar 2015" - more flexible
        month_range_pattern = r'(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})\s+(?:to|through|until|-)\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+(20\d{2})'
        
        month_range_match = re.search(month_range_pattern, prompt.lower())
        if month_range_match:
            start_month, start_year, end_month, end_year = month_range_match.groups()
            mon_idx_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }
            
            start_mon = start_month[:3].lower()
            end_mon = end_month[:3].lower()
            start_mon_idx = mon_idx_map["sept" if start_mon == "sep" else start_mon]
            end_mon_idx = mon_idx_map["sept" if end_mon == "sep" else end_mon]
            start_year = int(start_year)
            end_year = int(end_year)
            
            sd = datetime(start_year, start_mon_idx, 1).date()
            # MonthEnd equivalent
            if end_mon_idx == 12:
                ed = datetime(end_year, 12, 31).date()
            else:
                ed = (datetime(end_year, end_mon_idx + 1, 1) - timedelta(days=1)).date()
            
            if st.session_state.sidebar_start_date != sd or st.session_state.sidebar_end_date != ed:
                st.session_state.sidebar_start_date = sd
                st.session_state.sidebar_end_date = ed
                return True
        
        # Single month handling: e.g., "January 2015", "Jan 2015", "for Jan 2015"
        month_regex = r"(?:for\s+)?(" + "|".join([m[:3] for m in month_names] + month_abbr) + r"|" + "|".join(month_names) + r")\s+(20\d{2})\b"
        m = re.search(month_regex, prompt, flags=re.IGNORECASE)
        if m:
            mon = m.group(1)[:3].lower()
            mon_idx = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }["sept" if mon == "sep" else mon]
            year = int(m.group(2))
            sd = datetime(year, mon_idx, 1).date()
            # MonthEnd(1) equivalent
            if mon_idx == 12:
                ed = datetime(year, 12, 31).date()
            else:
                ed = datetime(year, mon_idx + 1, 1) - timedelta(days=1).date()
            if st.session_state.sidebar_start_date != sd or st.session_state.sidebar_end_date != ed:
                st.session_state.sidebar_start_date = sd
                st.session_state.sidebar_end_date = ed
                return True
        
        return False
        
    except Exception as e:
        return False

def generate_chatbot_response(prompt):
    """Chatbot response generator with multi-step reasoning and graph generation capabilities.

    Process:
    1) Check if user wants a graph/chart
    2) Generate appropriate visualization if requested
    3) Draft: produce a concise, structured answer using available tools/context.
    4) Review: self-critique and refine the draft for accuracy and clarity.
    Only the refined answer is shown to the user; intermediate thoughts are not displayed.
    """
    
    try:
        # Check if user wants a graph/chart or complex analytics
        graph_keywords = ['chart', 'graph', 'plot', 'visualization', 'visualize', 'show me', 'display', 'create']
        analytics_keywords = ['top', 'rank', 'forecast', 'predict', 'analyze', 'compare', 'insights', 'performance', 'trends']
        domain_keywords = graph_keywords + analytics_keywords + [
            'sales', 'item', 'items', 'store', 'stores', 'daily', 'weekly', 'monthly',
            'm5', 'inventory', 'prophet', 'xgboost', 'lgbm', 'lstm', 'revenue', 'units',
            'trend', 'plot', 'chart', 'visualize', 'rank', 'top', 'compare', 'prediction'
        ]
        small_talk_keywords = [
            'hi', 'hello', 'hey', 'how are you', 'what do you do', 'who are you',
            'help', 'what can you do', 'your name', 'about you', 'thanks', 'thank you',
            'good morning', 'good afternoon', 'good evening'
        ]
        
        prompt_lower = prompt.lower()
        wants_graph = any(keyword in prompt_lower for keyword in graph_keywords)
        wants_analytics = any(keyword in prompt_lower for keyword in analytics_keywords)

        # Small-talk and out-of-domain guard: respond conversationally, no analytics/graphs
        is_small_talk = any(kw in prompt_lower for kw in small_talk_keywords)
        is_domain_intent = any(kw in prompt_lower for kw in domain_keywords)
        if is_small_talk or not is_domain_intent:
            return (
                "I'm InventoryGPT, your inventory and forecasting assistant. "
                "I can analyze sales, rank top items, compare forecasts vs actuals, and create charts. "
                "Ask things like 'show top 15 items in January 2015' or 'compare actual vs forecast for March 2015'."
            )
        
        # If the prompt does NOT include any item_id or store_id, answer directly via Llama3
        try:
            normalized_for_match = _normalize_item_tokens(prompt)
            item_matches = re.findall(r"\b([A-Z]+_[0-9]+_[0-9]{3})\b", normalized_for_match)
            store_matches = re.findall(r"\b([A-Z]{2})[\s_]?([0-9]+)\b", normalized_for_match)
            if not item_matches and not store_matches and not wants_graph:
                fast = st.session_state.get("ai_mode", "Fast") == "Fast"
                if fast:
                    response = ollama.chat(
                        model="llama3",
                        messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, {"role": "user", "content": analysis_prompt}],
                        options={"temperature": 0.2, "num_predict": 300}
                    )
                    bot_reply = response["message"]["content"]
                else:
                    draft_system = {"role": "system", "content": "First, draft a concise, structured answer. Focus on correctness. Do not include your reasoning steps."}
                    response_draft = ollama.chat(
                        model="llama3",
                        messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, draft_system]
                                 + st.session_state.chat_history
                                 + [{"role": "user", "content": analysis_prompt}]
                    )
                    draft_text = response_draft["message"]["content"]
                    review_system = {"role": "system", "content": "Review and refine the previous draft for accuracy, clarity, and completeness. Output only the improved final answer."}
                    response_final = ollama.chat(
                        model="llama3",
                        messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, review_system]
                                 + st.session_state.chat_history
                                 + [
                                     {"role": "user", "content": analysis_prompt},
                                     {"role": "user", "content": f"Here is your draft answer to refine:\n\n{draft_text}"}
                                 ]
                    )
                    bot_reply = response_final["message"]["content"] or draft_text
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                try:
                    import json, os
                    user_id = st.session_state.get("user_id") or "default_user"
                    os.makedirs("/Users/dhineashkumar/Desktop/inventory-gpt/cache", exist_ok=True)
                    mem_path = f"/Users/dhineashkumar/Desktop/inventory-gpt/cache/chat_{user_id}.json"
                    with open(mem_path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.chat_history, f)
                except Exception:
                    pass
                return bot_reply
        except Exception:
            pass

        # Ensure any forecast-related parameters in the prompt are applied BEFORE answering,
        # so the summary (and UI) reflect the latest forecast.
        try:
            updated_any, model_from_prompt = update_filters_from_prompt(prompt)

            available_models = []
            if 'models' in globals():
                available_models = [name for name, model in models.items() if model is not None]

            desired_model = None
            if model_from_prompt and model_from_prompt in available_models:
                desired_model = model_from_prompt
            elif 'LightGBM' in available_models:
                desired_model = 'LightGBM'
            elif available_models:
                desired_model = available_models[0]

            if desired_model:
                st.session_state.preferred_model_from_chat = desired_model

            sd = st.session_state.sidebar_start_date
            ed = st.session_state.sidebar_end_date
            items_sel = st.session_state.sidebar_item_ids
            stores_sel = st.session_state.sidebar_store_ids
            if ed <= sd:
                ed = sd + timedelta(days=1)
                st.session_state.sidebar_end_date = ed

            # Accept multiple item_ids and store_ids parsed from prompt
            # update_filters_from_prompt already populates sidebar_item_ids/store_ids if it finds matches
            if items_sel and stores_sel and desired_model and 'generate_forecast' in globals():
                try:
                    generate_forecast(sd, ed, items_sel, stores_sel, st.session_state.get('stock_levels', {}), desired_model)
                except Exception:
                    pass
            else:
                if updated_any:
                    st.rerun()
        except Exception:
            pass
        
        # Specialist M5 agents: route analysis/inventory/risk/strategy intents
        if SPECIAL_AGENTS_AVAILABLE and SPECIAL_AGENTS:
            try:
                lower = prompt_lower
                agent_key = None
                if any(k in lower for k in ["inventory", "stock", "safety stock", "reorder"]):
                    agent_key = "inventory"
                elif any(k in lower for k in ["risk", "volatility", "exposure", "mitigation"]):
                    agent_key = "risk"
                elif any(k in lower for k in ["strategy", "strategic", "plan", "priorities"]):
                    agent_key = "strategy"
                elif any(k in lower for k in ["analysis", "analyze", "seasonality", "pattern"]):
                    agent_key = "forecast"

                if agent_key and agent_key in SPECIAL_AGENTS:
                    state = _build_m5_state()
                    if agent_key == "strategy":
                        # Gather inputs from other agents first
                        insights = {}
                        try:
                            insights["forecast"] = SPECIAL_AGENTS["forecast"].analyze(state)
                        except Exception:
                            pass
                        try:
                            insights["inventory"] = SPECIAL_AGENTS["inventory"].analyze(state)
                        except Exception:
                            pass
                        try:
                            insights["risk"] = SPECIAL_AGENTS["risk"].analyze(state)
                        except Exception:
                            pass
                        return SPECIAL_AGENTS["strategy"].analyze(state, insights)
                    else:
                        return SPECIAL_AGENTS[agent_key].analyze(state)
            except Exception:
                pass

        # Llama3-only mode: answer via LLM when not explicitly requesting a graph handled by graph agent
        try:
            if wants_graph and GRAPH_AGENT_AVAILABLE:
                raise RuntimeError("Skip LLM: graph agent will handle below")
            fast = st.session_state.get("ai_mode", "Fast") == "Fast"
            if fast:
                response = ollama.chat(
                    model="llama3",
                    messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, {"role": "user", "content": analysis_prompt}],
                    options={"temperature": 0.2, "num_predict": 300}
                )
                bot_reply = response["message"]["content"]
            else:
                draft_system = {"role": "system", "content": "First, draft a concise, structured answer. Focus on correctness. Do not include your reasoning steps."}
                response_draft = ollama.chat(
                    model="llama3",
                    messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, draft_system]
                             + st.session_state.chat_history
                             + [{"role": "user", "content": analysis_prompt}]
                )
                draft_text = response_draft["message"]["content"]
                review_system = {"role": "system", "content": "Review and refine the previous draft for accuracy, clarity, and completeness. Output only the improved final answer."}
                response_final = ollama.chat(
                    model="llama3",
                    messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, review_system]
                             + st.session_state.chat_history
                             + [
                                 {"role": "user", "content": analysis_prompt},
                                 {"role": "user", "content": f"Here is your draft answer to refine:\n\n{draft_text}"}
                             ]
                )
                bot_reply = response_final["message"]["content"] or draft_text
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            # Persist memory
            try:
                import json, os
                user_id = st.session_state.get("user_id") or "default_user"
                os.makedirs("/Users/dhineashkumar/Desktop/inventory-gpt/cache", exist_ok=True)
                mem_path = f"/Users/dhineashkumar/Desktop/inventory-gpt/cache/chat_{user_id}.json"
                with open(mem_path, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.chat_history, f)
            except Exception:
                pass
            return bot_reply
        except Exception:
            pass

        # Graph requests via GraphAgent
        if wants_graph and GRAPH_AGENT_AVAILABLE:
            try:
                graph_agent = get_graph_agent()
                if graph_agent:
                    if hasattr(st.session_state, "forecast_df") and st.session_state.forecast_df is not None:
                        graph_agent.forecast_df = st.session_state.forecast_df
                    analysis_result = graph_agent.analyze_request(prompt)
                    if analysis_result is not None and len(analysis_result) >= 2:
                        tool_name, params = analysis_result[0], analysis_result[1]
                    else:
                        tool_name, params = None, {}
                    # Enrich params with session filters
                    params.setdefault('item_ids', st.session_state.get('sidebar_item_ids') or [])
                    params.setdefault('store_ids', st.session_state.get('sidebar_store_ids') or [])
                    if st.session_state.get('sidebar_start_date') and st.session_state.get('sidebar_end_date'):
                        params.setdefault('date_range', (st.session_state.sidebar_start_date, st.session_state.sidebar_end_date))
                    result = graph_agent.execute_tool(tool_name, **params) if tool_name else None
                    return result or "Generated chart."
            except Exception as e:
                st.error(f"Error using graph agent: {e}")
                return "I couldn't create the chart right now."

        # Fallback: Handle complex analytics requests first (disabled for now)
        if False and wants_analytics and ANALYTICS_AGENT_AVAILABLE:
            try:
                analytics_agent = get_analytics_agent()
                if analytics_agent:
                    # Update forecast_df if available
                    if hasattr(st.session_state, "forecast_df") and st.session_state.forecast_df is not None:
                        analytics_agent.forecast_df = st.session_state.forecast_df
                    
                    # Analyze the request to determine the appropriate tool and parameters
                    try:
                        analysis_result = analytics_agent.analyze_request(prompt)
                        if analysis_result is not None and len(analysis_result) >= 2:
                            tool_name, params = analysis_result[0], analysis_result[1]
                        else:
                            tool_name, params = None, {}
                    except Exception as e:
                        st.error(f"Error analyzing analytics request: {str(e)}")
                        tool_name, params = None, {}
                    
                    if tool_name:
                        try:
                            # Execute the appropriate analytics tool
                            analytics_result = analytics_agent.execute_tool(tool_name, **params)
                            # Handle dict results from analytics agent safely
                            if isinstance(analytics_result, dict):
                                if analytics_result.get("error"):
                                    error_msg = analytics_result.get("error")
                                    st.error(error_msg)
                                    return f"📊 {error_msg}"
                                title = analytics_result.get("title") or "Analysis Results"
                                message = analytics_result.get("message") or "Analysis completed successfully."
                                return f"📊 {title}\n\n{message}"
                            else:
                                result_text = str(analytics_result)
                                if result_text and ("successfully" in result_text.lower()):
                                    return f"📊 {result_text}\n\nI've performed the analysis you requested above. The results show detailed insights based on your query. Is there anything specific about the findings you'd like me to explain?"
                                else:
                                    return f"📊 {result_text}\n\nI can perform various types of analytics including:\n" + \
                                           "• Top items by sales for specific periods\n" + \
                                           "• Forecast and rank items by predicted sales\n" + \
                                           "• Compare actual vs forecasted rankings\n" + \
                                           "• Analyze performance trends over time\n" + \
                                           "• Generate comprehensive periodic insights\n\n" + \
                                           "Try asking questions like 'show top 15 items in January 2015' or 'forecast sales for all items and rank them'!"
                        except Exception as e:
                            st.error(f"Error executing analytics tool: {str(e)}")
                            return f"📊 I encountered an error while performing the analysis. Please try rephrasing your request or check if the necessary data is available."
                    else:
                        return f"📊 I can perform various types of analytics including:\n" + \
                               "• Top items by sales for specific periods\n" + \
                               "• Forecast and rank items by predicted sales\n" + \
                               "• Compare actual vs forecasted rankings\n" + \
                               "• Analyze performance trends over time\n" + \
                               "• Generate comprehensive periodic insights\n\n" + \
                               "Try asking questions like 'show top 15 items in January 2015' or 'forecast sales for all items and rank them'!"
            except Exception as e:
                st.error(f"Error with analytics agent: {str(e)}")
                return f"📊 Analytics functionality is temporarily unavailable. Please try again later or ask a different question."
        
        # Generate graph if requested using the advanced graph agent
        if wants_graph and GRAPH_AGENT_AVAILABLE:
            try:
                graph_agent = get_graph_agent()
                if graph_agent:
                    # Update forecast_df if available
                    if hasattr(st.session_state, "forecast_df") and st.session_state.forecast_df is not None:
                        graph_agent.forecast_df = st.session_state.forecast_df
                    
                    # Analyze the request to determine the appropriate tool and parameters
                    try:
                        analysis_result = graph_agent.analyze_request(prompt)
                        if analysis_result is not None and len(analysis_result) >= 2:
                            tool_name, params = analysis_result[0], analysis_result[1]
                        else:
                            tool_name, params = None, {}
                    except Exception as e:
                        st.error(f"Error analyzing graph request: {str(e)}")
                        tool_name, params = None, {}
                    
                    if tool_name:
                        try:
                            # Execute the appropriate tool
                            graph_result = graph_agent.execute_tool(tool_name, **params)
                            if graph_result and ("successfully" in graph_result.lower() or "created successfully" in graph_result.lower()):
                                return f"📊 {graph_result}\n\nI've generated the chart you requested above. The visualization shows the data analysis based on your request. Is there anything specific about the patterns or insights you'd like me to explain?"
                            else:
                                return f"📊 {graph_result}\n\nI can create various types of charts including:\n" + \
                                       "• Sales trends over time (with moving averages)\n" + \
                                       "• Forecast vs actual comparisons with metrics\n" + \
                                       "• Top performing items ranking\n" + \
                                       "• Store performance comparison\n" + \
                                       "• Weekly sales patterns\n" + \
                                       "• Monthly trends analysis\n" + \
                                       "• Category distribution (pie chart)\n" + \
                                       "• Sales heatmap (items vs stores)\n" + \
                                       "• Correlation analysis\n" + \
                                       "• Seasonal decomposition\n\n" + \
                                       "Just ask me to create any of these visualizations! You can also specify parameters like 'top 15 items' or 'show me trends from 2015-01-01 to 2015-01-31'."
                        except Exception as e:
                            st.error(f"Error executing graph tool: {str(e)}")
                            return f"📊 I encountered an error while creating the visualization. Please try rephrasing your request or check if the necessary data is available."
                    else:
                        return f"📊 I can create various types of charts including:\n" + \
                               "• Sales trends over time (with moving averages)\n" + \
                               "• Forecast vs actual comparisons with metrics\n" + \
                               "• Top performing items ranking\n" + \
                               "• Store performance comparison\n" + \
                               "• Weekly sales patterns\n" + \
                               "• Monthly trends analysis\n" + \
                               "• Category distribution (pie chart)\n" + \
                               "• Sales heatmap (items vs stores)\n" + \
                               "• Correlation analysis\n" + \
                               "• Seasonal decomposition\n\n" + \
                               "Just ask me to create any of these visualizations! You can also specify parameters like 'top 15 items' or 'show me trends from 2015-01-01 to 2015-01-31'."
            except Exception as e:
                st.error(f"Error with graph agent: {str(e)}")
                return f"📊 Graph generation is temporarily unavailable. Please try again later or ask a different question."
        
        # Fallback to basic graph generator if agent is not available
        elif wants_graph and GRAPH_TOOLS_AVAILABLE:
            try:
                graph_generator = get_graph_generator()
                if graph_generator:
                    # Update forecast_df if available
                    if hasattr(st.session_state, "forecast_df") and st.session_state.forecast_df is not None:
                        graph_generator.forecast_df = st.session_state.forecast_df
                    
                    graph_result = graph_generator.generate_graph_from_request(prompt)
                    if graph_result and "successfully" in graph_result.lower():
                        return f"📊 {graph_result}\n\nI've generated the chart you requested above. Is there anything specific about the data you'd like me to explain?"
                    else:
                        return f"📊 {graph_result}\n\nI can create various types of charts including:\n" + \
                               "• Sales trends over time\n" + \
                               "• Forecast vs actual comparisons\n" + \
                               "• Top performing items\n" + \
                               "• Store performance\n" + \
                               "• Weekly patterns\n" + \
                               "• Monthly trends\n" + \
                               "• Category distribution\n" + \
                               "• Sales heatmap\n\n" + \
                               "Just ask me to create any of these visualizations!"
            except Exception as e:
                st.error(f"Error with graph generator: {str(e)}")
                return f"📊 Basic graph generation is temporarily unavailable. Please try again later."

        # -------- Context Gathering --------
        dataset_summary = None
        try:
            if 'main_df' in globals() and main_df is not None:
                dataset_summary = {
                    'rows': len(main_df),
                    'columns': len(main_df.columns),
                    'items': main_df['item_id'].nunique() if 'item_id' in main_df.columns else 'N/A',
                    'stores': main_df['store_id'].nunique() if 'store_id' in main_df.columns else 'N/A',
                    'date_range': f"{main_df['date'].min().year}-{main_df['date'].max().year}" if 'date' in main_df.columns else 'N/A'
                }
        except Exception:
            dataset_summary = "Dataset information unavailable"

        model_summary = None
        try:
            if 'models' in globals():
                available_models = [name for name, model in models.items() if model is not None]
                model_info = {
                    'LightGBM': 'Fast gradient boosting - excellent for tabular data with high performance',
                    'XGBoost': 'Extreme gradient boosting - robust and widely used in competitions',
                    'Prophet': 'Time series focused - great for seasonal patterns and holidays'
                }
                model_summary = {
                    model: model_info.get(model, "Advanced machine learning model")
                    for model in available_models
                }
        except Exception:
            model_summary = "Model information unavailable"

        forecast_summary = None
        try:
            if hasattr(st.session_state, "forecast_df") and st.session_state.forecast_df is not None:
                df = st.session_state.forecast_df
                total_sales = df['predicted_sales'].sum()
                avg_sales = df['predicted_sales'].mean()
                peak_day = df.loc[df['predicted_sales'].idxmax(), 'date']
                peak_value = df['predicted_sales'].max()
                low_day = df.loc[df['predicted_sales'].idxmin(), 'date']
                low_value = df['predicted_sales'].min()
                model_used = getattr(st.session_state, "model_used", "Unknown")

                forecast_summary = f"""
                Forecast Summary:
                - Model used: {model_used}
                - Total predicted sales: {total_sales:,.0f}
                - Average daily sales: {avg_sales:,.2f}
                - Peak sales: {peak_value:,.0f} on {peak_day.date()}
                - Lowest sales: {low_value:,.0f} on {low_day.date()}
                """
        except Exception:
            forecast_summary = "Forecast information unavailable"

        # -------- Parse user prompt for filters (date range, items, stores, model) using enhanced parser --------
        try:
            updated_any, model_from_prompt = update_filters_from_prompt(prompt)

            available_models = []
            if 'models' in globals():
                available_models = [name for name, model in models.items() if model is not None]
            
            desired_model = None
            if model_from_prompt and model_from_prompt in available_models:
                desired_model = model_from_prompt
            elif 'LightGBM' in available_models:
                desired_model = 'LightGBM'
            elif available_models:
                desired_model = available_models[0]

            if desired_model:
                st.session_state.preferred_model_from_chat = desired_model

            sd = st.session_state.sidebar_start_date
            ed = st.session_state.sidebar_end_date
            items_sel = st.session_state.sidebar_item_ids
            stores_sel = st.session_state.sidebar_store_ids
            if ed <= sd:
                ed = sd + timedelta(days=1)
                st.session_state.sidebar_end_date = ed

            if items_sel and stores_sel and desired_model and 'generate_forecast' in globals():
                try:
                    generate_forecast(sd, ed, items_sel, stores_sel, st.session_state.get('stock_levels', {}), desired_model)
                except Exception:
                    pass
            else:
                if updated_any:
                    st.rerun()
        except Exception as e:
            st.warning(f"Warning: Could not parse prompt parameters: {str(e)}")

        # -------- Prompt Construction --------
        analysis_prompt = f"""
        You are a retail forecasting assistant with memory of past conversation turns. 
        The user asked: "{prompt}"

        ## Context (Ground truth data)
        - Dataset: {dataset_summary if dataset_summary else "Dataset not available."}
        - Models: {model_summary if model_summary else "No models loaded."}
        - Forecast: {forecast_summary if forecast_summary else "No forecast results generated yet."}

        ## Conversation so far:
        {st.session_state.chat_history}

        ## Instructions:
        - Use ONLY the provided dataset/model/forecast info for facts.
        - If asked about forecasts, highlight concrete numbers (totals, peaks, averages).
        - If asked about models, explain strengths/weaknesses of available ones.
        - Keep memory of history: refer back to what the user asked earlier.
        - Be concise but structured: use bullet points or sections.
        - Avoid hallucinations: if something is not available, clearly say so.
        """

        # -------- Direct Ollama LLM path (llama3) --------
        fallback_error = None

        # -------- Fallback or Fast path: direct Ollama chat API --------
        try:
            fast = st.session_state.get("ai_mode", "Fast") == "Fast"
            if fast:
                # Single pass for speed
                response = ollama.chat(
                    model="llama3",
                    messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, {"role": "user", "content": analysis_prompt}],
                    options={"temperature": 0.2, "num_predict": 300}
                )
                bot_reply = response["message"]["content"]
            else:
                # Two-pass for quality
                draft_system = {"role": "system", "content": "First, draft a concise, structured answer. Focus on correctness. Do not include your reasoning steps."}
                response_draft = ollama.chat(
                    model="llama3",
                    messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, draft_system]
                             + st.session_state.chat_history
                             + [{"role": "user", "content": analysis_prompt}]
                )
                draft_text = response_draft["message"]["content"]
                review_system = {"role": "system", "content": "Review and refine the previous draft for accuracy, clarity, and completeness. Output only the improved final answer."}
                response_final = ollama.chat(
                    model="llama3",
                    messages=[{"role": "system", "content": "You are a helpful retail forecasting assistant."}, review_system]
                             + st.session_state.chat_history
                             + [
                                 {"role": "user", "content": analysis_prompt},
                                 {"role": "user", "content": f"Here is your draft answer to refine:\n\n{draft_text}"}
                             ]
                )
                bot_reply = response_final["message"]["content"] or draft_text
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            # Persist memory
            try:
                import json, os
                user_id = st.session_state.get("user_id") or "default_user"
                os.makedirs("/Users/dhineashkumar/Desktop/inventory-gpt/cache", exist_ok=True)
                mem_path = f"/Users/dhineashkumar/Desktop/inventory-gpt/cache/chat_{user_id}.json"
                with open(mem_path, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.chat_history, f)
            except Exception:
                pass
            return bot_reply
        except Exception as e:
            suffix = f" (LLM error: {fallback_error})" if fallback_error else ""
            return f"⚠️ Error generating response: {str(e)}{suffix}"
            
    except Exception as e:
        return f"⚠️ An unexpected error occurred while processing your request: {str(e)}. Please try again with a different question."

# 8.ABC ANALYSIS
def ABC_analysis_page():
    st.markdown("## ")
    st.markdown("****")
    st.markdown("---")
    st.title("📊 ABC Analysis")
    try:
        df = pd.read_parquet("/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2014.parquet")  # Same dataset as in View Dataset
        # Ensure original item IDs are available for the analysis and display
        if 'item_id_display' not in df.columns:
            df['item_id_display'] = df['item_id'].apply(decode_item_id)
        abc_result = perform_abc_analysis(df)

        # Show full result table with class colors
        st.subheader("ABC Classification Table")
        def color_rows_by_class(row):
            cls = row.get('abc_class')
            if cls == 'A':
                return ['background-color: #000000'] * len(row)
            if cls == 'B':
                return ['background-color: #000000'] * len(row)
            if cls == 'C':
                return ['background-color: #000000'] * len(row)
            return [''] * len(row)

        try:
            styled = abc_result.style.apply(color_rows_by_class, axis=1)
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(abc_result, use_container_width=True)

        # Class distribution with fixed colors
        st.subheader("Class Distribution")
        class_counts = abc_result['abc_class'].value_counts().reindex(['A','B','C']).fillna(0)
        class_colors = {'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545'}
        fig_dist = go.Figure()
        for cls in ['A','B','C']:
            fig_dist.add_trace(go.Bar(x=[cls], y=[class_counts.get(cls, 0)], marker_color=class_colors[cls], name=cls))
        fig_dist.update_layout(template='plotly_white', showlegend=False, yaxis_title='Count')
        st.plotly_chart(fig_dist, use_container_width=True)

        # Pareto chart by Subcategory (e.g., FOODS_1): aggregate revenue and cumulative percentage
        st.subheader("Pareto Analysis by Subcategory (80/20)")
        sub_df = abc_result[['item_id_display', 'revenue']].copy()
        # Subcategory is the first two tokens: CATEGORY_DEPT (e.g., FOODS_1)
        sub_df['subcategory'] = sub_df['item_id_display'].astype(str).str.extract(r'(^[A-Z]+_\d)')[0]
        sub_grouped = sub_df.groupby('subcategory', as_index=False)['revenue'].sum()
        sub_grouped = sub_grouped.sort_values('revenue', ascending=False)
        sub_grouped['cumulative_revenue'] = sub_grouped['revenue'].cumsum()
        total_rev = sub_grouped['revenue'].sum()
        sub_grouped['cumulative_percentage'] = (sub_grouped['cumulative_revenue'] / total_rev) * 100 if total_rev > 0 else 0

        # Highlight top 20% subcategories
        n_sub = len(sub_grouped)
        top_20_cutoff = int(np.ceil(0.2 * n_sub)) if n_sub > 0 else 0
        top_20_cutoff = max(top_20_cutoff, 1) if n_sub > 0 else 0
        sub_grouped['is_top_20pct'] = False
        if top_20_cutoff > 0:
            sub_grouped.iloc[:top_20_cutoff, sub_grouped.columns.get_loc('is_top_20pct')] = True

        x_labels = sub_grouped['subcategory'].fillna('UNKNOWN')

        bar_colors = np.where(sub_grouped['is_top_20pct'], '#007BFF', '#CED4DA')
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            x=x_labels,
            y=sub_grouped['revenue'],
            name='Revenue',
            marker_color=bar_colors,
            yaxis='y1'
        ))
        fig_pareto.add_trace(go.Scatter(
            x=x_labels,
            y=sub_grouped['cumulative_percentage'],
            name='Cumulative % Revenue',
            mode='lines+markers',
            marker=dict(color='#FF5733'),
            yaxis='y2'
        ))
        fig_pareto.update_layout(
            template='plotly_white',
            xaxis=dict(title='Item Subcategory', tickangle=-45, showticklabels=True),
            yaxis=dict(title='Revenue', side='left', rangemode='tozero'),
            yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0,100]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    except Exception as e:
        st.error(f"Error while performing ABC analysis: {e}")


def perform_abc_analysis(df):
    """Perform ABC analysis on products with decoded item IDs and class labels"""
    # Prefer display IDs when available
    group_key = 'item_id_display' if 'item_id_display' in df.columns else 'item_id'

    item_revenue = df.groupby(group_key).agg({
        'sales': 'sum',
        'sell_price': 'mean'
    }).reset_index()

    # Decode mapping fallback: if grouping by item_id but display exists elsewhere in df
    if group_key == 'item_id' and 'item_id_display' in df.columns:
        mapping = df[['item_id', 'item_id_display']].drop_duplicates()
        item_revenue = item_revenue.merge(mapping, on='item_id', how='left')
        item_revenue['item_id_display'] = item_revenue['item_id_display'].fillna(item_revenue['item_id'].astype(str))
    elif group_key == 'item_id_display':
        item_revenue['item_id_display'] = item_revenue['item_id_display']
    else:
        # last resort string cast
        item_revenue['item_id_display'] = item_revenue[group_key].astype(str)

    item_revenue['revenue'] = item_revenue['sales'] * item_revenue['sell_price']
    item_revenue = item_revenue.sort_values('revenue', ascending=False)

    total_revenue = item_revenue['revenue'].sum()
    item_revenue['cumulative_revenue'] = item_revenue['revenue'].cumsum()
    item_revenue['revenue_percentage'] = (item_revenue['revenue'] / total_revenue * 100) if total_revenue > 0 else 0
    item_revenue['cumulative_percentage'] = (item_revenue['cumulative_revenue'] / total_revenue * 100) if total_revenue > 0 else 0

    def classify_abc(cumulative_pct):
        if cumulative_pct <= 70:
            return 'A'
        elif cumulative_pct <= 90:
            return 'B'
        else:
            return 'C'

    item_revenue['abc_class'] = item_revenue['cumulative_percentage'].apply(classify_abc)

    # Reorder and rename columns for clarity
    cols = ['item_id_display', 'sales', 'sell_price', 'revenue', 'revenue_percentage', 'cumulative_revenue', 'cumulative_percentage', 'abc_class']
    item_revenue = item_revenue[cols]
    return item_revenue

# --- 8. MAIN APPLICATION LOGIC ---

def main():
    """Main application controller"""
    
    # Render navigation
    render_navigation()
    
    # Render sidebar controls
    forecast_generated = render_sidebar()
    
    # Route to appropriate page
    current_page = st.session_state.current_page
    
    if current_page == "Home":
        home_page()
    elif current_page == "Forecast Dashboard":
        forecast_dashboard_page()
    elif current_page == "View Dataset":
        view_dataset_page()
    elif current_page == "Chatbot":
        chatbot_page()
    elif current_page == "ABC Analysis":
        ABC_analysis_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6C757D; padding: 20px;">
        <p><strong>InventoryGPT v2.0</strong> | Built with Streamlit & Advanced ML Models</p>
        <p>Powered by LightGBM, XGBoost, Prophet | Enhanced with AI Analytics</p>
    </div>
    """, unsafe_allow_html=True)

def render_analytics_results():
    """Render analytics results stored in session state"""
    if hasattr(st.session_state, "analytics_result") and st.session_state.analytics_result:
        result = st.session_state.analytics_result
        
        if result.get("success"):
            st.markdown("## 📊 Analytics Results")
            
            # Handle different chart types
            chart_type = result.get("chart_type")
            
            if chart_type == "bar":
                render_bar_chart(result)
            elif chart_type == "grouped_bar":
                render_grouped_bar_chart(result)
            elif chart_type == "line":
                render_line_chart(result)
            elif chart_type == "multi_chart":
                render_multi_chart(result)
            
            # Clear the result after rendering
            st.session_state.analytics_result = None

def render_bar_chart(result):
    """Render a bar chart from analytics result"""
    try:
        data = result.get("data")
        title = result.get("title", "Chart")
        x_col = result.get("x_col")
        y_col = result.get("y_col")
        
        if data is not None and not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data[x_col],
                y=data[y_col],
                marker_color='#007BFF',
                text=data[y_col].round(0),
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                template="plotly_white",
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary table
            if "summary_table" in result:
                st.markdown("#### 📋 Detailed Summary")
                summary_df = result["summary_table"].copy()
                if 'Rank' not in summary_df.columns:
                    summary_df['Rank'] = range(1, len(summary_df) + 1)
                    summary_df = summary_df[['Rank'] + [col for col in summary_df.columns if col != 'Rank']]
                
                st.dataframe(summary_df, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error rendering bar chart: {str(e)}")

def render_grouped_bar_chart(result):
    """Render a grouped bar chart from analytics result"""
    try:
        data = result.get("data")
        title = result.get("title", "Chart")
        x_col = result.get("x_col")
        y_cols = result.get("y_cols", [])
        
        if data is not None and not data.empty and len(y_cols) >= 2:
            fig = go.Figure()
            
            # Add traces for each y column
            colors = ['#28A745', '#007BFF']
            names = ['Actual Sales', 'Predicted Sales']
            
            for i, (y_col, color, name) in enumerate(zip(y_cols, colors, names)):
                fig.add_trace(go.Bar(
                    x=data[x_col],
                    y=data[y_col],
                    name=name,
                    marker_color=color
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title="Sales (Units)",
                template="plotly_white",
                height=500,
                xaxis_tickangle=-45,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary table
            if "summary_table" in result:
                st.markdown("#### 📊 Ranking Comparison")
                summary_df = result["summary_table"].copy()
                st.dataframe(summary_df, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error rendering grouped bar chart: {str(e)}")

def render_line_chart(result):
    """Render a line chart from analytics result"""
    try:
        data = result.get("data")
        title = result.get("title", "Chart")
        x_col = result.get("x_col")
        y_col = result.get("y_col")
        group_col = result.get("group_col")
        
        if data is not None and not data.empty:
            fig = go.Figure()
            
            if group_col:
                # Multiple lines for different groups
                colors = px.colors.qualitative.Set3
                for i, group in enumerate(data[group_col].unique()):
                    group_data = data[data[group_col] == group]
                    if not group_data.empty:
                        fig.add_trace(go.Scatter(
                            x=group_data[x_col],
                            y=group_data[y_col],
                            mode='lines+markers',
                            name=group,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4)
                        ))
            else:
                # Single line
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name='Sales',
                    line=dict(color='#007BFF', width=3),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error rendering line chart: {str(e)}")

def render_multi_chart(result):
    """Render multiple charts from analytics result"""
    try:
        charts = result.get("charts", [])
        summary_stats = result.get("summary_stats", {})
        
        # Show summary statistics
        if summary_stats:
            st.markdown("#### 📋 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sales", f"{summary_stats.get('total_sales', 0):,.0f}")
            with col2:
                st.metric("Avg Daily Sales", f"{summary_stats.get('avg_daily_sales', 0):.0f}")
            with col3:
                st.metric("Unique Items", f"{summary_stats.get('total_items', 0)}")
            with col4:
                st.metric("Unique Stores", f"{summary_stats.get('total_stores', 0)}")
        
        # Render each chart
        for i, chart_config in enumerate(charts):
            chart_type = chart_config.get("type")
            title = chart_config.get("title", f"Chart {i+1}")
            
            st.markdown(f"### {title}")
            
            if chart_type == "bar":
                # Create a temporary result dict for the bar chart
                temp_result = {
                    "data": chart_config.get("data"),
                    "title": title,
                    "x_col": chart_config.get("x_col"),
                    "y_col": chart_config.get("y_col")
                }
                render_bar_chart(temp_result)
            elif chart_type == "line":
                # Create a temporary result dict for the line chart
                temp_result = {
                    "data": chart_config.get("data"),
                    "title": title,
                    "x_col": chart_config.get("x_col"),
                    "y_col": chart_config.get("y_col")
                }
                render_line_chart(temp_result)
                
    except Exception as e:
        st.error(f"Error rendering multi-chart: {str(e)}")

# Global memory for conversation history (persist per user across reloads)
if __name__ == "__main__":
    main()
