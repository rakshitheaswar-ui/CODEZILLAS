#!/usr/bin/env python3
"""
Chat Orchestrator: classifies user intent, extracts entities, routes to tools/agents or RAG.
Integrates with:
- AnalyticsAgent (tools for analysis)
- GraphAgent (charts)
- SimpleRAG (code/data Q&A)
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

from .rag_service import SimpleRAG
try:
    import ollama  # Optional for LLM answers
    _OLLAMA_OK = True
except Exception:
    _OLLAMA_OK = False
try:
    import requests  # HTTP fallback to Ollama server
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False

# Simple in-memory cache for RAG answers
from functools import lru_cache


class ChatOrchestrator:
    def __init__(self, project_root: str, analytics_agent=None, graph_agent=None, cache_dir: Optional[str] = None):
        self.analytics_agent = analytics_agent
        self.graph_agent = graph_agent
        self.rag = SimpleRAG(project_root)
        # Load cached index if available; otherwise build and save
        cached = False
        if cache_dir:
            cached = self.rag.load(cache_dir)
        if not cached:
            self.rag.build()
            if cache_dir:
                self.rag.save(cache_dir)

        # Domain guard keywords
        self.domain_keywords = [
            'inventory', 'stock', 'sales', 'forecast', 'forecasting', 'prediction', 'prophet', 'xgboost', 'lightgbm',
            'item', 'items', 'store', 'stores', 'm5', 'dataset', 'actual', 'trend', 'ranking', 'abc analysis'
        ]
        self.domain_disclaimer = (
            "I'm an inventory forecasting assistant. I can analyze sales, run forecasts (LightGBM/XGBoost/Prophet), "
            "compare forecasts vs actuals (2015), answer questions about the dataset/models, and generate charts."
        )

    # -------- Intent and entity extraction --------
    def extract_entities(self, text: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        # Dates
        dates = re.findall(r"(20\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01]))", text)
        if len(dates) >= 2:
            params['start_date'] = pd.to_datetime(dates[0])
            params['end_date'] = pd.to_datetime(dates[1])
        # Items
        text_up = text.upper().replace('-', '_')
        items = re.findall(r"\b([A-Z]+_\d_\d{3})\b", text_up)
        if items:
            params['item_ids'] = list(dict.fromkeys(items))
        # Stores
        # Stores (tolerant: accepts trailing chars like CA_1M)
        store_iter = re.finditer(r"([A-Z]{2})[\s_]?([0-9]+)", text_up)
        store_ids = []
        for m in store_iter:
            state, num = m.group(1), m.group(2)
            store_ids.append(f"{state}_{num}")
        if store_ids:
            params['store_ids'] = list(dict.fromkeys(store_ids))
        # Top N
        nums = re.findall(r"\b(\d{1,3})\b", text)
        if nums:
            params['top_n'] = min(int(nums[0]), 50)
        # Model
        low = text.lower()
        if 'xgboost' in low or 'xgb' in low:
            params['model'] = 'XGBoost'
        elif 'prophet' in low:
            params['model'] = 'Prophet'
        elif 'lightgbm' in low or 'lgbm' in low:
            params['model'] = 'LightGBM'
        return params

    def classify_intent(self, text: str) -> str:
        tl = text.lower()
        if any(k in tl for k in ['chart', 'graph', 'plot', 'visualize', 'show me']):
            return 'graph'
        if any(k in tl for k in ['forecast', 'forecasting', 'predict', 'top', 'rank', 'compare', 'insights', 'trend']):
            return 'analytics'
        return 'rag'

    def _is_domain(self, text: str) -> bool:
        tl = text.lower()
        return any(k in tl for k in self.domain_keywords)

    def _domain_shortcuts(self, text: str) -> Optional[Dict[str, Any]]:
        tl = text.lower()
        if 'what is forecasting' in tl or ('forecasting' in tl and 'what' in tl):
            answer = (
                "Forecasting predicts future sales using historical patterns and models. "
                "Here, we train on 2014 and predict 2015 using LightGBM/XGBoost/Prophet, "
                "then compare predictions with actual 2015 sales to measure accuracy."
            )
            return {"text": answer, "sources": [
                {"path": "/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2014.parquet", "score": 1.0},
                {"path": "/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2015.parquet", "score": 1.0}
            ]}
        if any(k in tl for k in [
            'what dataset', 'which dataset', 'parquet', '2014 dataset', '2015 dataset',
            'modeling dataset', 'training dataset', 'evaluation dataset', 'actual vs forecast'
        ]):
            answer = (
                "This project uses two parquet datasets:\n\n"
                "- 2014 modeling dataset: `/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2014.parquet` (used for training/models)\n"
                "- 2015 evaluation dataset: `/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2015.parquet` (used to compare forecasts vs actuals)\n\n"
                "Forecasts are generated over selected 2015 periods and compared against actuals from the 2015 dataset."
            )
            return {"text": answer, "sources": [
                {"path": "/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2014.parquet", "score": 1.0},
                {"path": "/Users/dhineashkumar/Desktop/inventory-gpt/data/m5_feature_engineered_2015.parquet", "score": 1.0}
            ]}
        return None

    # -------- Routing --------
    def route(self, text: str) -> Tuple[str, Dict[str, Any]]:
        intent = self.classify_intent(text)
        params = self.extract_entities(text)
        # Enforce dataset policy: forecasts and comparisons use 2015 evaluation window by default
        tl = text.lower()
        if intent == 'analytics' and any(k in tl for k in ['forecast', 'predict', 'compare']):
            if 'start_date' not in params or 'end_date' not in params:
                params['start_date'] = pd.to_datetime('2015-01-01')
                params['end_date'] = pd.to_datetime('2015-01-31')
            if 'model' not in params:
                params['model'] = 'LightGBM'
        return intent, params

    @lru_cache(maxsize=64)
    def _rag_augment_answer(self, question: str) -> Dict[str, Any]:
        """Generate an LLM answer augmented with RAG context. Returns dict with text and sources."""
        hits = self.rag.query(question, top_k=4)
        if not hits:
            return {"text": "I couldn't find relevant context in the project.", "sources": []}
        sources: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for path, chunk, score in hits:
            sources.append({"path": path, "score": score})
            context_blocks.append(f"Source: {path}\nContent: {chunk}")
        context_text = "\n\n---\n\n".join(context_blocks)
        system = (
            "You are a helpful assistant for an inventory forecasting app. "
            "Policy: 2014 parquet is the modeling dataset; 2015 parquet is used to evaluate forecasts vs actuals. "
            "Answer using only the provided context. Be concise and cite file paths inline when helpful. "
            "Explain your reasoning briefly in 2-3 bullet points when relevant."
        )
        user = f"Question:\n{question}\n\nContext:\n{context_text}\n\nInstructions:\n- Provide a direct answer.\n- If code or files are relevant, mention their paths.\n- Keep it under 200 words."
        if _OLLAMA_OK:
            try:
                resp = ollama.chat(model="llama3", messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], options={"temperature": 0.1, "num_predict": 300})
                text = resp.get("message", {}).get("content", "") or "Answer generated."
                return {"text": text, "sources": sources}
            except Exception:
                pass
        elif _REQUESTS_OK:
            try:
                payload = {
                    "model": "llama3",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    "options": {"temperature": 0.1, "num_predict": 300}
                }
                r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=20)
                if r.ok:
                    data = r.json()
                    # streaming vs non-streaming responses handling
                    text = data.get("message", {}).get("content") if isinstance(data, dict) else None
                    if not text and hasattr(r, 'iter_lines'):
                        # best-effort line concat if server streams
                        try:
                            text_chunks = []
                            for line in r.iter_lines():
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    chunk = obj.get("message", {}).get("content")
                                    if chunk:
                                        text_chunks.append(chunk)
                                except Exception:
                                    pass
                            text = "".join(text_chunks) if text_chunks else ""
                        except Exception:
                            text = ""
                    text = text or "Answer generated."
                    return {"text": text, "sources": sources}
            except Exception:
                pass
        # Fallback: non-LLM simple stitched response
        stitched = f"Based on project context, here are the most relevant pointers:\n\n" + "\n\n".join([f"- {s['path']}" for s in sources])
        return {"text": stitched, "sources": sources}

    def handle(self, text: str) -> Dict[str, Any] | str:
        # Domain guard
        if not self._is_domain(text):
            return {"text": self.domain_disclaimer, "sources": []}

        # Fast domain-specific answers
        shortcut = self._domain_shortcuts(text)
        if shortcut is not None:
            return shortcut
        intent, params = self.route(text)
        if intent == 'analytics' and self.analytics_agent is not None:
            tool, tool_params = self.analytics_agent.analyze_request(text)
            tool_params.update(params)
            result = self.analytics_agent.execute_tool(tool, **tool_params) if tool else None
            if isinstance(result, dict):
                title = result.get('title') or 'Analysis Results'
                msg = result.get('message') or 'Completed.'
                return {"text": f"{title}\n\n{msg}", "sources": []}
            return {"text": (str(result) if result else "I couldn't find a suitable analytics tool for that."), "sources": []}
        if intent == 'graph' and self.graph_agent is not None:
            tool, tool_params = self.graph_agent.analyze_request(text)
            tool_params.update(params)
            result = self.graph_agent.execute_tool(tool, **tool_params) if tool else None
            return {"text": (str(result) if result else "I couldn't create that chart."), "sources": []}
        # RAG fallback
        if not self.rag.is_ready():
            return {"text": "RAG index not ready. Please try again later.", "sources": []}
        return self._rag_augment_answer(text)


