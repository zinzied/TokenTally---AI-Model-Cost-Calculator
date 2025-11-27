#!/usr/bin/env python3
"""
Model Cost Calculator CLI

This script calculates and compares inference costs across major AI providers.
Update pricing by either:
1) Editing the `PRICING_DATA` mapping directly, or
2) Creating a `pricing_config.json` in the same directory with overrides:
   {
     "pricing": {
       "openai": {
         "gpt-4o": {"input": 5.0, "output": 15.0}
       }
     },
     "pricing_sources": {
       "openai": "https://example.com/openai-pricing.json"
     }
   }

Example:
python model_cost_calculator.py --model gpt-4o --input-tokens 5000 --output-tokens 2000 --compare
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception:
    requests = None

try:
    from tabulate import tabulate
except Exception:
    tabulate = None

try:
    from colorama import Fore, Style, init as colorama_init
except Exception:
    Fore = Style = None
    colorama_init = None

try:
    import tiktoken
except Exception:
    tiktoken = None

_BANNER_SHOWN = False

PRICING_DATA: Dict[str, Dict[str, Dict[str, float]]] = {
    "openai": {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt 5.1": {"input": 8.00, "output": 24.00},
        "gpt-5.1": {"input": 8.00, "output": 24.00},
    },
    "anthropic": {
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "opus 4.5": {"input": 12.00, "output": 60.00},
        "claude-4.5-opus": {"input": 12.00, "output": 60.00},
    },
    "xai": {
        "grok 4.1": {"input": 4.00, "output": 20.00},
        "grok-4.1": {"input": 4.00, "output": 20.00},
    },
    "google": {
        "gemini-1.5-pro": {"input": 4.00, "output": 16.00},
        "gemini 1.5 pro": {"input": 4.00, "output": 16.00},
        "gemini-1.5-flash": {"input": 1.00, "output": 3.00},
        "gemini 1.5 flash": {"input": 1.00, "output": 3.00},
    },
    "cohere": {
        "command-r-plus": {"input": 3.00, "output": 12.00},
        "command r plus": {"input": 3.00, "output": 12.00},
        "command-r": {"input": 1.50, "output": 6.00},
        "command r": {"input": 1.50, "output": 6.00},
    },
    "mistral": {
        "mistral-large": {"input": 2.00, "output": 6.00},
        "mistral large": {"input": 2.00, "output": 6.00},
        "mistral-small": {"input": 0.60, "output": 1.80},
        "mistral small": {"input": 0.60, "output": 1.80},
    },
    "meta": {
        "llama-3-70b-instruct": {"input": 0.80, "output": 0.80},
        "llama 3 70b instruct": {"input": 0.80, "output": 0.80},
        "llama-3-8b-instruct": {"input": 0.20, "output": 0.20},
        "llama 3 8b instruct": {"input": 0.20, "output": 0.20},
    },
}

CAPABILITY_TIERS: Dict[str, Dict[str, str]] = {
    "openai": {"gpt-4o": "tier-4", "gpt-4-turbo": "tier-4", "gpt 5.1": "tier-5", "gpt-5.1": "tier-5"},
    "anthropic": {"claude-3-opus": "tier-5", "claude-3-sonnet": "tier-4", "opus 4.5": "tier-5", "claude-4.5-opus": "tier-5"},
    "xai": {"grok 4.1": "tier-4", "grok-4.1": "tier-4"},
    "google": {"gemini-1.5-pro": "tier-5", "gemini 1.5 pro": "tier-5", "gemini-1.5-flash": "tier-3", "gemini 1.5 flash": "tier-3"},
    "cohere": {"command-r-plus": "tier-5", "command r plus": "tier-5", "command-r": "tier-4", "command r": "tier-4"},
    "mistral": {"mistral-large": "tier-4", "mistral large": "tier-4", "mistral-small": "tier-3", "mistral small": "tier-3"},
    "meta": {"llama-3-70b-instruct": "tier-4", "llama 3 70b instruct": "tier-4", "llama-3-8b-instruct": "tier-3", "llama 3 8b instruct": "tier-3"},
}

TOKEN_ENCODINGS: Dict[str, str] = {
    "openai": "cl100k_base",
    "anthropic": "cl100k_base",
    "google": "cl100k_base",
    "xai": "cl100k_base",
    "cohere": "cl100k_base",
    "mistral": "cl100k_base",
    "meta": "cl100k_base",
}

TEMPLATES: Dict[str, Tuple[int, int]] = {
    "qa": (500, 300),
    "summary": (2000, 600),
    "code": (3000, 1200),
    "report": (8000, 2000),
}


@dataclass
class ModelPrice:
    input_price: float
    output_price: float
    currency: str = "USD"


class PricingRegistry:
    def __init__(self, base: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        self._pricing: Dict[str, Dict[str, ModelPrice]] = {}
        for provider, models in base.items():
            self._pricing[provider] = {}
            for model, v in models.items():
                self._pricing[provider][model] = ModelPrice(
                    input_price=float(v["input"]),
                    output_price=float(v["output"]),
                    currency="USD",
                )

    def providers(self) -> List[str]:
        return list(self._pricing.keys())

    def models(self, provider: str) -> List[str]:
        return list(self._pricing.get(provider, {}).keys())

    def get(self, provider: str, model: str) -> Optional[ModelPrice]:
        return self._pricing.get(provider, {}).get(model)

    def update_from_dict(self, overrides: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        for provider, models in overrides.items():
            if provider not in self._pricing:
                self._pricing[provider] = {}
            for model, v in models.items():
                self._pricing[provider][model] = ModelPrice(
                    input_price=float(v["input"]),
                    output_price=float(v["output"]),
                    currency="USD",
                )

    def update_from_config_file(self, path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"loaded": False, "sources": {}}
        if not os.path.isfile(path):
            return result
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "pricing" in cfg and isinstance(cfg["pricing"], dict):
                self.update_from_dict(cfg["pricing"])
                result["loaded"] = True
            if "pricing_sources" in cfg and isinstance(cfg["pricing_sources"], dict):
                result["sources"] = cfg["pricing_sources"]
        except Exception as e:
            logging.error("Failed to load pricing_config.json: %s", e)
        return result

    def update_from_live_sources(self, sources: Dict[str, str], timeout: float = 5.0) -> None:
        if not requests:
            logging.info("requests not installed; skipping live pricing fetch")
            return
        for provider, url in sources.items():
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 429:
                    logging.warning("Rate limited fetching %s pricing; skipping", provider)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    cleaned: Dict[str, Dict[str, float]] = {}
                    for model, v in data.get("models", {}).items():
                        if {"input", "output"}.issubset(v.keys()):
                            cleaned[model] = {"input": float(v["input"]), "output": float(v["output"])}
                    if cleaned:
                        self.update_from_dict({provider: cleaned})
            except Exception as e:
                logging.warning("Failed fetching live pricing for %s: %s", provider, e)


def calculate_cost(input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
    return (input_tokens / 1e6 * input_price) + (output_tokens / 1e6 * output_price)


def cost_per_1k(input_price: float, output_price: float) -> float:
    return ((1000 / 1e6) * input_price) + ((1000 / 1e6) * output_price)

def estimate_tokens_from_text(text: str, model: Optional[str] = None) -> int:
    if text is None:
        return 0
    if 'tiktoken' in globals() and tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return int(round(len(text) / 4))

def convert_currency(amount_usd: float, target_currency: Optional[str], fx_rate: Optional[float]) -> Optional[float]:
    if not target_currency:
        return None
    if fx_rate is None or fx_rate <= 0:
        return None
    return amount_usd * fx_rate

def show_banner_animated() -> None:
    global _BANNER_SHOWN
    if _BANNER_SHOWN:
        return
    art = [
        "▄▄▄█████▓ ▒█████   ██ ▄█▀▓█████  ███▄    █ ▄▄▄█████▓ ▄▄▄       ██▓     ██▓   ▓██   ██▓ ",
        " ▓  ██▒ ▓▒▒██▒  ██▒ ██▄█▒ ▓█   ▀  ██ ▀█   █ ▓  ██▒ ▓▒▒████▄    ▓██▒    ▓██▒    ▒██  ██▒ ",
        " ▒ ▓██░ ▒░▒██░  ██▒▓███▄░ ▒███   ▓██  ▀█ ██▒▒ ▓██░ ▒░▒██  ▀█▄  ▒██░    ▒██░     ▒██ ██░ ",
        " ░ ▓██▓ ░ ▒██   ██░▓██ █▄ ▒▓█  ▄ ▓██▒  ▐▌██▒░ ▓██▓ ░ ░██▄▄▄▄██ ▒██░    ▒██░     ░ ▐██▓░ ",
        "   ▒██▒ ░ ░ ████▓▒░▒██▒ █▄░▒████▒▒██░   ▓██░  ▒██▒ ░  ▓█   ▓██▒░██████▒░██████▒ ░ ██▒▓░ ",
        "   ▒ ░░   ░ ▒░▒░▒░ ▒ ▒▒ ▓▒░░ ▒░ ░░ ▒░   ▒ ▒   ▒ ░░    ▒▒   ▓▒█░░ ▒░▓  ░░ ▒░▓  ░  ██▒▒▒ ",
        "     ░      ░ ▒ ▒░ ░ ░▒ ▒░ ░ ░  ░░ ░░   ░ ▒░    ░      ▒   ▒▒ ░░ ░ ▒  ░░ ░ ▒  ░▓██ ░▒░ ",
        "   ░      ░ ░ ░ ▒  ░ ░░ ░    ░      ░   ░ ░   ░        ░   ▒     ░ ░     ░ ░   ▒ ▒ ░░  ",
        "              ░ ░  ░  ░      ░  ░         ░                ░  ░    ░  ░    ░  ░░ ░     ",
        "                                                                               ░ ░",
    ]
    for line in art:
        if Fore and Style:
            print(Fore.MAGENTA + Style.BRIGHT + line + Style.RESET_ALL)
        else:
            print(line)
        time.sleep(0.05)
    _BANNER_SHOWN = True


def validate_tokens(value: str) -> int:
    s = value.strip().replace(",", "").replace("_", "")
    if not s:
        raise argparse.ArgumentTypeError("Tokens must be an integer or a number with k/m/b suffix")
    mult = 1
    last = s[-1].lower()
    if last in ("k", "m", "b"):
        mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[last]
        s = s[:-1]
    try:
        n_float = float(s)
    except Exception:
        raise argparse.ArgumentTypeError("Tokens must be an integer or a number with k/m/b suffix")
    n = int(n_float * mult)
    if n < 0:
        raise argparse.ArgumentTypeError("Tokens must be a non-negative integer")
    return n


def parse_budget(value: str) -> Optional[float]:
    s = value.strip().lower().replace(",", "").replace("_", "")
    if not s:
        return None
    if s in ("usd", "$", "none", "n/a"):
        return None
    mult = 1.0
    last = s[-1]
    if last in ("k", "m", "b"):
        mult = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[last]
        s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        return None


def validate_provider_model(registry: PricingRegistry, provider: Optional[str], model: str) -> Tuple[str, str]:
    if provider:
        if provider not in registry.providers():
            raise ValueError(f"Unknown provider '{provider}'")
        if model not in registry.models(provider):
            raise ValueError(f"Unknown model '{model}' for provider '{provider}'")
        return provider, model
    for p in registry.providers():
        if model in registry.models(p):
            return p, model
    raise ValueError(f"Unknown model '{model}' across providers")


def read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def colorize(text: str, severity: str) -> str:
    if not Fore or not Style:
        return text
    if severity == "low":
        return Fore.GREEN + text + Style.RESET_ALL
    if severity == "medium":
        return Fore.YELLOW + text + Style.RESET_ALL
    return Fore.RED + text + Style.RESET_ALL


def severity_from_cost(total_usd: float, budget_usd: Optional[float]) -> str:
    if budget_usd and budget_usd > 0:
        pct = (total_usd / budget_usd) * 100.0
        if pct < 1.0:
            return "low"
        if pct < 5.0:
            return "medium"
        return "high"
    if total_usd < 0.1:
        return "low"
    if total_usd < 1.0:
        return "medium"
    return "high"


def suggest_alternatives(
    registry: PricingRegistry,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    current = registry.get(provider, model)
    if not current:
        return []
    tier = CAPABILITY_TIERS.get(provider, {}).get(model)
    current_cost = calculate_cost(input_tokens, output_tokens, current.input_price, current.output_price)
    candidates: List[Dict[str, Any]] = []
    for p in registry.providers():
        for m in registry.models(p):
            mp = registry.get(p, m)
            if not mp:
                continue
            mtier = CAPABILITY_TIERS.get(p, {}).get(m)
            if mtier != tier:
                continue
            c = calculate_cost(input_tokens, output_tokens, mp.input_price, mp.output_price)
            if c < current_cost - 1e-12:
                candidates.append(
                    {"provider": p, "model": m, "usd_cost": c, "saving_usd": current_cost - c}
                )
    candidates.sort(key=lambda x: (x["usd_cost"], -x["saving_usd"]))
    return candidates[:top_k]


def make_table_rows(
    registry: PricingRegistry,
    jobs: List[Dict[str, Any]],
    budget_usd: Optional[float],
    currency: Optional[str] = None,
    fx_rate: Optional[float] = None,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    table: List[List[str]] = []
    csv_rows: List[Dict[str, Any]] = []
    for job in jobs:
        provider, model = validate_provider_model(registry, job.get("provider"), job["model"])
        mp = registry.get(provider, model)
        if not mp:
            raise ValueError(f"Missing pricing for {provider}:{model}")
        total_usd = calculate_cost(job["input_tokens"], job["output_tokens"], mp.input_price, mp.output_price)
        extra = float(job.get("extra_cost_usd", 0.0))
        total_usd = total_usd + extra
        per_1k = cost_per_1k(mp.input_price, mp.output_price)
        pct_budget = (total_usd / budget_usd * 100.0) if budget_usd else None
        sev = severity_from_cost(total_usd, budget_usd)
        total_str = colorize(f"${total_usd:,.6f}", sev)
        per_1k_str = f"${per_1k:.6f}"
        pct_str = f"{pct_budget:.3f}%" if pct_budget is not None else "-"
        base_currency = mp.currency
        base_cost_str = f"{total_usd:,.6f} {base_currency}"
        converted = convert_currency(total_usd, currency, fx_rate)
        converted_str = f"{converted:,.6f} {currency}" if converted is not None else "-"
        table.append(
            [
                provider,
                model,
                str(job["input_tokens"]),
                str(job["output_tokens"]),
                base_cost_str,
                f"${total_usd:,.6f}",
                converted_str,
                per_1k_str,
                pct_str,
                total_str,
            ]
        )
        csv_rows.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "provider": provider,
                "model": model,
                "input_tokens": job["input_tokens"],
                "output_tokens": job["output_tokens"],
                "currency": base_currency,
                "base_cost": round(total_usd, 6),
                "usd_cost": round(total_usd, 6),
                "cost_per_1k_tokens_usd": round(per_1k, 6),
                "budget_usd": round(budget_usd, 6) if budget_usd else "",
                "budget_pct": round(pct_budget, 6) if pct_budget is not None else "",
                "converted_currency": currency or "",
                "converted_cost": round(converted, 6) if converted is not None else "",
            }
        )
    return table, csv_rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Model Cost Calculator")
    parser.add_argument("--model", required=False, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--input-tokens", required=False, type=validate_tokens, help="Input token count (supports k/m/b suffix, e.g., 250k, 1m)")
    parser.add_argument("--output-tokens", required=False, type=validate_tokens, help="Output token count (supports k/m/b suffix)")
    parser.add_argument("--provider", required=False, help="Provider (e.g., openai, anthropic)")
    parser.add_argument("--batch-file", required=False, help="Path to JSON batch file")
    parser.add_argument("--compare", action="store_true", help="Compare costs across providers/models")
    parser.add_argument("--export-csv", required=False, nargs="?", const="", help="Export CSV to path or default")
    parser.add_argument("--budget", required=False, type=float, help="Monthly budget in USD for percentage display")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")
    parser.add_argument("--config", required=False, help="pricing_config.json path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--explain-tokens", action="store_true", help="Show a technical explanation of tokens and the pricing formula and exit")
    parser.add_argument("--estimate-input-words", required=False, type=int, help="Approx input words to estimate tokens (tokens ≈ words × 0.75)")
    parser.add_argument("--estimate-input-chars", required=False, type=int, help="Approx input characters to estimate tokens (tokens ≈ chars / 4)")
    parser.add_argument("--estimate-output-words", required=False, type=int, help="Approx output words to estimate tokens")
    parser.add_argument("--estimate-output-chars", required=False, type=int, help="Approx output characters to estimate tokens")
    parser.add_argument("--demo", action="store_true", help="Interactive demo to paste a prompt and estimate costs before choosing models")
    parser.add_argument("--init-pricing", action="store_true", help="Initialize pricing_data.json from hardcoded data")
    parser.add_argument("--simulate", action="store_true", help="Simulate cost optimization scenarios")
    parser.add_argument("--simulate-input-tokens", type=int, help="Input tokens for simulation")
    parser.add_argument("--simulate-output-tokens", type=int, help="Output tokens for simulation")
    parser.add_argument("--currency", required=False, help="Target currency code for conversion (e.g., EUR)")
    parser.add_argument("--fx-rate", required=False, type=float, help="USD to target currency rate")
    parser.add_argument("--per-request-fee", required=False, type=float, help="Additional per-request USD fee")
    parser.add_argument("--audio-minutes", required=False, type=float, help="Audio minutes for job")
    parser.add_argument("--audio-rate", required=False, type=float, help="USD per audio minute")
    parser.add_argument("--image-count", required=False, type=int, help="Image count for job")
    parser.add_argument("--image-rate", required=False, type=float, help="USD per image")
    parser.add_argument("--template", required=False, choices=["qa", "summary", "code", "report"], help="Usage template for typical tokens")
    parser.add_argument("--template-scale", required=False, type=float, help="Scale factor to adjust template tokens")
    parser.add_argument("--planner", action="store_true", help="Monthly planner: simulate monthly spend")
    parser.add_argument("--requests-per-day", required=False, type=int, help="Requests per day for planner")
    parser.add_argument("--days", required=False, type=int, help="Number of days for planner")
    parser.add_argument("--alarm-threshold-pct", required=False, type=float, help="Warn when job cost exceeds this percent of budget")
    return parser.parse_args(argv)


def interactive_prompt(registry: PricingRegistry) -> Dict[str, Any]:
    print("Interactive Mode")
    providers = registry.providers()
    print("Select Provider:")
    for i, p in enumerate(providers, 1):
        print(f"  {i}) {p}")
    print("  0) Skip (auto-detect by model)")
    sel = input("Enter number: ").strip()
    provider: Optional[str] = None
    if sel.isdigit():
        idx = int(sel)
        if 1 <= idx <= len(providers):
            provider = providers[idx - 1]
    model: str
    if provider:
        models = registry.models(provider)
        print(f"Select Model for {provider}:")
        for i, m in enumerate(models, 1):
            print(f"  {i}) {m}")
        msel = input("Enter number or type a model name: ").strip()
        if msel.isdigit():
            midx = int(msel)
            if 1 <= midx <= len(models):
                model = models[midx - 1]
            else:
                model = msel
        else:
            model = msel
    else:
        model = input("Model (e.g., gpt-4o): ").strip()
    print("Hint: tokens ≈ words × 0.75, or tokens ≈ characters / 4")
    raw_in = input("Input tokens (e.g., 5000, 1m, 250k) or type 'words'/'chars': ").strip().lower()
    if raw_in == "words":
        w = validate_tokens(input("Approx input words: ").strip())
        input_tokens = int(round(w * 0.75))
    elif raw_in == "chars":
        c = validate_tokens(input("Approx input characters: ").strip())
        input_tokens = int(round(c / 4))
    else:
        input_tokens = validate_tokens(raw_in)
    raw_out = input("Output tokens (e.g., 2000, 100k) or type 'words'/'chars': ").strip().lower()
    if raw_out == "words":
        w = validate_tokens(input("Approx output words: ").strip())
        output_tokens = int(round(w * 0.75))
    elif raw_out == "chars":
        c = validate_tokens(input("Approx output characters: ").strip())
        output_tokens = int(round(c / 4))
    else:
        output_tokens = validate_tokens(raw_out)
    budget_str = input("Budget USD (optional, supports k/m/b; e.g., 100, 1k, 10k): ").strip()
    budget = parse_budget(budget_str)
    compare = input("Compare across providers/models? [y/N]: ").strip().lower() == "y"
    export_ans = input("Export CSV? [y/N]: ").strip().lower() == "y"
    export_csv = "" if export_ans else None
    return {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "budget": budget,
        "compare": compare,
        "export_csv": export_csv,
    }


def simulate_cost_optimization(
    registry: PricingRegistry,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    budget: Optional[float] = None
) -> None:
    """Simulate cost optimization scenarios for the given model and tokens."""
    if Fore and Style:
        print(Fore.CYAN + Style.BRIGHT + "\n=== COST OPTIMIZATION SIMULATOR ===" + Style.RESET_ALL)
    else:
        print("\n=== COST OPTIMIZATION SIMULATOR ===")
    
    current_mp = registry.get(provider, model)
    if not current_mp:
        print(f"Error: Pricing not found for {provider}:{model}")
        return
    
    current_cost = calculate_cost(input_tokens, output_tokens, current_mp.input_price, current_mp.output_price)
    
    if Fore and Style:
        print(Fore.YELLOW + f"Current Configuration:" + Style.RESET_ALL)
        print(f"  Model: {provider}/{model}")
        print(f"  Input tokens: {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Current cost: ${current_cost:.6f}")
        print()
    else:
        print(f"Current Configuration:")
        print(f"  Model: {provider}/{model}")
        print(f"  Input tokens: {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Current cost: ${current_cost:.6f}")
        print()
    
    # Simulation scenarios
    scenarios = []
    
    # Scenario 1: Prompt optimization (reduce input tokens by 20-40%)
    input_reduction_scenarios = [0.2, 0.3, 0.4]
    for reduction in input_reduction_scenarios:
        new_input_tokens = int(input_tokens * (1 - reduction))
        new_cost = calculate_cost(new_input_tokens, output_tokens, current_mp.input_price, current_mp.output_price)
        savings = current_cost - new_cost
        savings_pct = (savings / current_cost) * 100
        scenarios.append({
            "name": f"Prompt Optimization ({int(reduction * 100)}% reduction)",
            "input_tokens": new_input_tokens,
            "output_tokens": output_tokens,
            "cost": new_cost,
            "savings": savings,
            "savings_pct": savings_pct,
            "description": "Optimize prompts to be more concise"
        })
    
    # Scenario 2: Output compression (reduce output tokens by 10-30%)
    for reduction in [0.1, 0.2, 0.3]:
        new_output_tokens = int(output_tokens * (1 - reduction))
        new_cost = calculate_cost(input_tokens, new_output_tokens, current_mp.input_price, current_mp.output_price)
        savings = current_cost - new_cost
        savings_pct = (savings / current_cost) * 100
        scenarios.append({
            "name": f"Output Compression ({int(reduction * 100)}% reduction)",
            "input_tokens": input_tokens,
            "output_tokens": new_output_tokens,
            "cost": new_cost,
            "savings": savings,
            "savings_pct": savings_pct,
            "description": "Use response compression/condensation"
        })
    
    # Scenario 3: Combined optimization
    scenarios.append({
        "name": "Combined Optimization",
        "input_tokens": int(input_tokens * 0.8),  # 20% reduction
        "output_tokens": int(output_tokens * 0.85),  # 15% reduction
        "cost": calculate_cost(int(input_tokens * 0.8), int(output_tokens * 0.85), current_mp.input_price, current_mp.output_price),
        "savings": 0,  # Will calculate below
        "savings_pct": 0,  # Will calculate below
        "description": "Optimize both prompt and output"
    })
    
    # Calculate combined optimization savings
    combined_scenario = scenarios[-1]
    combined_scenario["savings"] = current_cost - combined_scenario["cost"]
    combined_scenario["savings_pct"] = (combined_scenario["savings"] / current_cost) * 100
    
    # Scenario 4: Alternative model comparison
    tier = CAPABILITY_TIERS.get(provider, {}).get(model, "tier-3")
    for p in registry.providers():
        for m in registry.models(p):
            mp = registry.get(p, m)
            if not mp:
                continue
            mtier = CAPABILITY_TIERS.get(p, {}).get(m)
            if mtier == tier and (p, m) != (provider, model):
                new_cost = calculate_cost(input_tokens, output_tokens, mp.input_price, mp.output_price)
                savings = current_cost - new_cost
                savings_pct = (savings / current_cost) * 100
                if savings_pct > 5:  # Only show if significant savings
                    scenarios.append({
                        "name": f"Alternative Model: {p}/{m}",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost": new_cost,
                        "savings": savings,
                        "savings_pct": savings_pct,
                        "description": f"Same capability tier ({mtier})"
                    })
    
    # Sort scenarios by savings (descending)
    scenarios.sort(key=lambda x: x["savings"], reverse=True)
    
    # Display scenarios
    if Fore and Style:
        print(Fore.GREEN + "Optimization Scenarios (sorted by potential savings):" + Style.RESET_ALL)
    else:
        print("Optimization Scenarios (sorted by potential savings):")
    print()
    
    # Create table for scenarios
    table_headers = ["Scenario", "Input Tokens", "Output Tokens", "New Cost", "Savings", "Savings %", "Description"]
    table_data = []
    
    for i, scenario in enumerate(scenarios[:10], 1):  # Show top 10 scenarios
        table_data.append([
            scenario["name"],
            f"{scenario['input_tokens']:,}",
            f"{scenario['output_tokens']:,}",
            f"${scenario['cost']:.6f}",
            f"${scenario['savings']:.6f}",
            f"{scenario['savings_pct']:.1f}%",
            scenario["description"]
        ])
    
    if tabulate:
        print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    else:
        print(f"{'Scenario':<25} {'Input':<12} {'Output':<12} {'New Cost':<12} {'Savings':<12} {'Savings %':<12} {'Description'}")
        print("-" * 100)
        for row in table_data:
            print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12} {row[6]}")
    
    # Summary recommendations
    if scenarios:
        best_scenario = scenarios[0]
        if Fore and Style:
            print(Fore.MAGENTA + "\n" + "="*50)
            print(Fore.YELLOW + "RECOMMENDATION:" + Style.RESET_ALL)
            print(Fore.CYAN + f"Best optimization: {best_scenario['name']}" + Style.RESET_ALL)
            print(f"Potential savings: ${best_scenario['savings']:.6f} ({best_scenario['savings_pct']:.1f}%)")
            print(f"New estimated cost: ${best_scenario['cost']:.6f}")
            print(Fore.GREEN + f"Strategy: {best_scenario['description']}" + Style.RESET_ALL)
        else:
            print("\n" + "="*50)
            print("RECOMMENDATION:")
            print(f"Best optimization: {best_scenario['name']}")
            print(f"Potential savings: ${best_scenario['savings']:.6f} ({best_scenario['savings_pct']:.1f}%)")
            print(f"New estimated cost: ${best_scenario['cost']:.6f}")
            print(f"Strategy: {best_scenario['description']}")
    
    print()


def default_csv_path() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), f"cost_report_{ts}.csv")

def run_demo(registry: PricingRegistry) -> None:
    if colorama_init:
        try:
            colorama_init(autoreset=True)
        except Exception:
            pass
    providers = registry.providers()
    print("Select Provider:")
    for i, p in enumerate(providers, 1):
        print(f"  {i}) {p}")
    sel = input("Enter number: ").strip()
    provider: Optional[str] = None
    if sel.isdigit():
        idx = int(sel)
        if 1 <= idx <= len(providers):
            provider = providers[idx - 1]
    if not provider:
        provider = input("Provider (blank for auto by model): ").strip().lower() or None
    if provider:
        models = registry.models(provider)
        print(f"Select Model for {provider}:")
        for i, m in enumerate(models, 1):
            print(f"  {i}) {m}")
        msel = input("Enter number or type a model name: ").strip()
        if msel.isdigit():
            midx = int(msel)
            model = models[midx - 1] if 1 <= midx <= len(models) else msel
        else:
            model = msel
    else:
        model = input("Model name: ").strip()
    print("Paste your prompt. End with a blank line.")
    lines: List[str] = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    prompt_text = "\n".join(lines)
    input_tokens = estimate_tokens_from_text(prompt_text, model)
    print(f"Estimated input tokens: {input_tokens}")
    print("Select expected output length:")
    print("  1) Short (~200 tokens)")
    print("  2) Medium (~600 tokens)")
    print("  3) Long (~1500 tokens)")
    print("  4) Custom tokens")
    print("  5) Estimate from words/chars")
    choice = input("Enter number: ").strip()
    output_tokens = 200
    if choice == "2":
        output_tokens = 600
    elif choice == "3":
        output_tokens = 1500
    elif choice == "4":
        output_tokens = validate_tokens(input("Output tokens: ").strip())
    elif choice == "5":
        mode = input("Type 'words' or 'chars': ").strip().lower()
        if mode == "words":
            w = validate_tokens(input("Approx output words: ").strip())
            output_tokens = int(round(w * 0.75))
        elif mode == "chars":
            c = validate_tokens(input("Approx output characters: ").strip())
            output_tokens = int(round(c / 4))
    budget = parse_budget(input("Budget USD (optional): ").strip())
    compare = input("Compare across providers/models? [y/N]: ").strip().lower() == "y"
    export_ans = input("Export CSV? [y/N]: ").strip().lower() == "y"
    export_csv = "" if export_ans else None
    jobs: List[Dict[str, Any]] = []
    if compare:
        for p in registry.providers():
            for m in registry.models(p):
                jobs.append({"provider": p, "model": m, "input_tokens": input_tokens, "output_tokens": output_tokens})
    else:
        p, m = validate_provider_model(registry, provider, model)
        jobs.append({"provider": p, "model": m, "input_tokens": input_tokens, "output_tokens": output_tokens})
    table, csv_rows = make_table_rows(registry, jobs, budget, None, None)
    headers = [
        "Provider",
        "Model",
        "Input Tokens",
        "Output Tokens",
        "Base Cost",
        "USD Cost",
        "Cost / 1K",
        "Budget %",
        "Severity",
    ]
    fmt = "github" if tabulate else None
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt=fmt))
    else:
        print(headers)
        for row in table:
            print(row)
    if export_csv is not None:
        path = export_csv if export_csv else default_csv_path()
        write_csv(path, csv_rows)
        print(f"\nCSV exported to: {path}")

def run_planner(
    registry: PricingRegistry,
    provider: Optional[str],
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    requests_per_day: int,
    days: int,
    currency: Optional[str],
    fx_rate: Optional[float],
    per_request_fee: float,
    audio_minutes: float,
    audio_rate: float,
    image_count: int,
    image_rate: float,
) -> None:
    jobs: List[Dict[str, Any]] = []
    total_requests = max(0, requests_per_day) * max(0, days)
    extra_per_req = 0.0
    extra_per_req += float(per_request_fee or 0.0)
    extra_per_req += float(audio_minutes or 0.0) * float(audio_rate or 0.0)
    extra_per_req += int(image_count or 0) * float(image_rate or 0.0)
    if model and provider:
        p, m = validate_provider_model(registry, provider, model)
        jobs.append({"provider": p, "model": m, "input_tokens": input_tokens, "output_tokens": output_tokens, "extra_cost_usd": extra_per_req})
    else:
        for p in registry.providers():
            for m in registry.models(p):
                jobs.append({"provider": p, "model": m, "input_tokens": input_tokens, "output_tokens": output_tokens, "extra_cost_usd": extra_per_req})
    table, csv_rows = make_table_rows(registry, jobs, None, currency, fx_rate)
    headers = ["Provider", "Model", "Input Tokens", "Output Tokens", "Base Cost", "USD Cost", "Converted Cost", "Cost / 1K", "Budget %", "Severity"]
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        print(headers)
        for row in table:
            print(row)
    print()
    totals: List[List[str]] = []
    for r in csv_rows:
        usd_cost = float(r["usd_cost"]) * total_requests
        converted_cost = r.get("converted_cost")
        conv = float(converted_cost) * total_requests if converted_cost != "" else None
        totals.append([
            r["provider"],
            r["model"],
            f"${usd_cost:,.6f}",
            f"{conv:,.6f} {r.get('converted_currency')}" if conv is not None else "-",
        ])
    t_headers = ["Provider", "Model", "Monthly USD Cost", "Monthly Converted Cost"]
    if tabulate:
        print(tabulate(totals, headers=t_headers, tablefmt="github"))
    else:
        print(t_headers)
        for row in totals:
            print(row)


def run_single(
    registry: PricingRegistry,
    provider: Optional[str],
    model: str,
    input_tokens: int,
    output_tokens: int,
    budget: Optional[float],
    compare: bool,
    export_csv: Optional[str],
    currency: Optional[str] = None,
    fx_rate: Optional[float] = None,
    extra_cost_usd: float = 0.0,
    alarm_threshold_pct: Optional[float] = None,
) -> None:
    if colorama_init:
        try:
            colorama_init(autoreset=True)
        except Exception:
            pass

    jobs: List[Dict[str, Any]] = []
    if compare:
        for p in registry.providers():
            for m in registry.models(p):
                jobs.append(
                    {
                        "provider": p,
                        "model": m,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )
    else:
        p, m = validate_provider_model(registry, provider, model)
        jobs.append({"provider": p, "model": m, "input_tokens": input_tokens, "output_tokens": output_tokens})

    table, csv_rows = make_table_rows(registry, jobs, budget, currency, fx_rate)
    headers = [
        "Provider",
        "Model",
        "Input Tokens",
        "Output Tokens",
        "Base Cost",
        "USD Cost",
        "Converted Cost",
        "Cost / 1K",
        "Budget %",
        "Severity",
    ]
    fmt = "github" if tabulate else None
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt=fmt))
    else:
        print(headers)
        for row in table:
            print(row)

    if not compare:
        p, m = validate_provider_model(registry, provider, model)
        suggestions = suggest_alternatives(registry, p, m, input_tokens, output_tokens)
        if suggestions:
            s_table = []
            for s in suggestions:
                s_table.append(
                    [
                        s["provider"],
                        s["model"],
                        f"${s['usd_cost']:.6f}",
                        f"${s['saving_usd']:.6f}",
                    ]
                )
            s_headers = ["Alt Provider", "Alt Model", "USD Cost", "Saving vs Selected"]
            print("\nSuggested Cheaper Alternatives (similar tier)")
            if tabulate:
                print(tabulate(s_table, headers=s_headers, tablefmt=fmt))
            else:
                print(s_headers)
                for r in s_table:
                    print(r)

    if export_csv is not None:
        path = export_csv if export_csv else default_csv_path()
        write_csv(path, csv_rows)
        print(f"\nCSV exported to: {path}")
    if alarm_threshold_pct and budget and budget > 0:
        for r in csv_rows:
            pct = r.get("budget_pct")
            if pct and pct != "" and float(pct) >= alarm_threshold_pct:
                print("Warning: cost exceeds threshold percent of budget")


def run_batch(registry: PricingRegistry, batch_path: str, budget: Optional[float], export_csv: Optional[str]) -> None:
    data = read_json_file(batch_path)
    jobs: List[Dict[str, Any]] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "jobs" in data:
        items = data["jobs"]
    else:
        raise ValueError("Invalid batch file format; expected list or { 'jobs': [...] }")

    for item in items:
        provider = item.get("provider")
        model = item["model"]
        input_tokens = int(item["input_tokens"])
        output_tokens = int(item["output_tokens"])
        validate_provider_model(registry, provider, model)
        jobs.append(
            {
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    table, csv_rows = make_table_rows(registry, jobs, budget)
    headers = [
        "Provider",
        "Model",
        "Input Tokens",
        "Output Tokens",
        "Base Cost",
        "USD Cost",
        "Cost / 1K",
        "Budget %",
        "Severity",
    ]
    fmt = "github" if tabulate else None
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt=fmt))
    else:
        print(headers)
        for row in table:
            print(row)

    path = export_csv if export_csv is not None and export_csv != "" else default_csv_path()
    write_csv(path, csv_rows)
    print(f"\nBatch CSV exported to: {path}")


def setup_registry(config_path: Optional[str]) -> PricingRegistry:
    registry = PricingRegistry(PRICING_DATA)
    sources: Dict[str, str] = {}
    if config_path:
        cfg_result = registry.update_from_config_file(config_path)
        sources = cfg_result.get("sources", {})
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pricing_config.json")
    if not config_path and os.path.isfile(default_cfg):
        cfg_result = registry.update_from_config_file(default_cfg)
        sources = cfg_result.get("sources", sources)
    if sources:
        registry.update_from_live_sources(sources)
    return registry


def init_pricing() -> None:
    sources = {
        "openai": "https://openai.com/api/pricing/",
        "anthropic": "https://anthropic.com/pricing",
        "xai": "https://console.grok.com/pricing",
        "google": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "cohere": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "mistral": "https://mistral.ai/technology/#pricing",
        "meta": "https://ai.meta.com/llama/"
    }
    now = datetime.utcnow().isoformat() + "Z"
    data = {
        "pricing": {},
        "metadata": {
            "last_updated": now,
            "version": "1.0.0"
        },
        "sources": sources
    }
    for provider, models_dict in PRICING_DATA.items():
        data["pricing"][provider] = {}
        for model, prices in models_dict.items():
            data["pricing"][provider][model] = {
                "input_cost_per_1M_tokens": prices["input"],
                "output_cost_per_1M_tokens": prices["output"],
                "update_date": now,
                "source_url": sources.get(provider, "manual")
            }
    with open("pricing_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    
    # Initialize colorama for Windows color support
    if colorama_init:
        try:
            colorama_init(autoreset=True)
        except Exception:
            pass
    show_banner_animated()
    
    if Fore and Style:
        print(Fore.CYAN + Style.BRIGHT + "+=============================================================+" + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "|" + Style.RESET_ALL, end="")
        print(Fore.YELLOW + "  AI MODEL COST CALCULATOR  " + Style.RESET_ALL, end="")
        print(Fore.CYAN + Style.BRIGHT + "                               |" + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "+=============================================================+" + Style.RESET_ALL)
        print()
        print(Fore.GREEN + "Compare inference costs across major AI providers" + Style.RESET_ALL)
        print(Fore.MAGENTA + "OpenAI • Anthropic • Google • xAI • Cohere • Mistral • Meta" + Style.RESET_ALL)
    else:
        print("AI Model Cost Calculator")
        print("Compare inference costs across major AI providers")
        print("OpenAI • Anthropic • Google • xAI • Cohere • Mistral • Meta")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if args.run_tests:
        run_tests()
        return

    if getattr(args, "explain_tokens", False):
        print("Tokens are the units models use internally. Roughly: 1 token ≈ 4 characters ≈ 0.75 words.")
        print("Pricing formula: (input_tokens/1e6 * input_price) + (output_tokens/1e6 * output_price)")
        print("Estimate tokens from words: tokens ≈ words × 0.75; from characters: tokens ≈ characters / 4")
        print("Output tokens are what the model generates; input tokens are what you send (prompt + context).")
        return

    registry = setup_registry(args.config)

    if args.simulate:
        if not args.model or args.simulate_input_tokens is None or args.simulate_output_tokens is None:
            raise ValueError("Simulation requires: --model, --simulate-input-tokens, --simulate-output-tokens")
        
        provider, model = validate_provider_model(registry, args.provider, args.model)
        simulate_cost_optimization(
            registry=registry,
            provider=provider,
            model=model,
            input_tokens=args.simulate_input_tokens,
            output_tokens=args.simulate_output_tokens,
            budget=args.budget
        )
        sys.exit(0)

    if args.init_pricing:
        init_pricing()
        print("pricing_data.json created successfully.")
        sys.exit(0)
    if getattr(args, "demo", False):
        while True:
            run_demo(registry)
            again = input("Run another demo? [Y/n]: ").strip().lower()
            if again == "n":
                break
        input("Press Enter to exit...")
        return

    if len(sys.argv) == 1:
        while True:
            d = interactive_prompt(registry)
            run_single(
                registry=registry,
                provider=d["provider"],
                model=d["model"],
                input_tokens=d["input_tokens"],
                output_tokens=d["output_tokens"],
                budget=d["budget"],
                compare=d["compare"],
                export_csv=d["export_csv"],
            )
            again = input("Run another calculation? [Y/n]: ").strip().lower()
            if again == "n":
                break
        input("Press Enter to exit...")
        return

    if args.planner:
        ip = args.input_tokens or 0
        op = args.output_tokens or 0
        if args.template:
            base = TEMPLATES.get(args.template)
            if base:
                scale = float(args.template_scale or 1.0)
                ip = int(base[0] * scale)
                op = int(base[1] * scale)
        run_planner(
            registry=registry,
            provider=args.provider,
            model=args.model,
            input_tokens=ip,
            output_tokens=op,
            requests_per_day=int(args.requests_per_day or 0),
            days=int(args.days or 0),
            currency=args.currency,
            fx_rate=args.fx_rate,
            per_request_fee=float(args.per_request_fee or 0.0),
            audio_minutes=float(args.audio_minutes or 0.0),
            audio_rate=float(args.audio_rate or 0.0),
            image_count=int(args.image_count or 0),
            image_rate=float(args.image_rate or 0.0),
        )
        return

    if args.batch_file:
        if not os.path.isfile(args.batch_file):
            raise FileNotFoundError(f"Batch file not found: {args.batch_file}")
        run_batch(registry, args.batch_file, args.budget, args.export_csv, args.currency, args.fx_rate)
        return

    if args.input_tokens is None:
        if getattr(args, "estimate_input_chars", None) is not None:
            args.input_tokens = int(round(args.estimate_input_chars / 4))
        elif getattr(args, "estimate_input_words", None) is not None:
            args.input_tokens = int(round(args.estimate_input_words * 0.75))
    if args.output_tokens is None:
        if getattr(args, "estimate_output_chars", None) is not None:
            args.output_tokens = int(round(args.estimate_output_chars / 4))
        elif getattr(args, "estimate_output_words", None) is not None:
            args.output_tokens = int(round(args.estimate_output_words * 0.75))

    if not args.model or args.input_tokens is None or args.output_tokens is None:
        raise ValueError("Required: --model, --input-tokens, --output-tokens or provide estimation flags")

    extra_cost = 0.0
    if args.per_request_fee:
        extra_cost += float(args.per_request_fee)
    if args.audio_minutes and args.audio_rate:
        extra_cost += float(args.audio_minutes) * float(args.audio_rate)
    if args.image_count and args.image_rate:
        extra_cost += int(args.image_count) * float(args.image_rate)

    if args.template and (args.input_tokens is None or args.output_tokens is None):
        base = TEMPLATES.get(args.template)
        if base:
            scale = float(args.template_scale or 1.0)
            if args.input_tokens is None:
                args.input_tokens = int(base[0] * scale)
            if args.output_tokens is None:
                args.output_tokens = int(base[1] * scale)

    run_single(
        registry=registry,
        provider=args.provider,
        model=args.model,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        budget=args.budget,
        compare=bool(args.compare),
        export_csv=args.export_csv,
        currency=args.currency,
        fx_rate=args.fx_rate,
        extra_cost_usd=extra_cost,
        alarm_threshold_pct=args.alarm_threshold_pct,
    )


def run_tests() -> None:
    import tempfile
    import unittest

    class TestCostCalculation(unittest.TestCase):
        def setUp(self) -> None:
            self.registry = PricingRegistry(PRICING_DATA)

        def test_formula(self) -> None:
            mp = self.registry.get("openai", "gpt-4o")
            self.assertIsNotNone(mp)
            cost = calculate_cost(5000, 2000, mp.input_price, mp.output_price)
            expected = (5000 / 1e6 * mp.input_price) + (2000 / 1e6 * mp.output_price)
            self.assertAlmostEqual(cost, expected, places=12)

        def test_cost_per_1k(self) -> None:
            mp = self.registry.get("anthropic", "claude-3-sonnet")
            self.assertIsNotNone(mp)
            self.assertAlmostEqual(
                cost_per_1k(mp.input_price, mp.output_price),
                ((1000 / 1e6) * mp.input_price) + ((1000 / 1e6) * mp.output_price),
                places=12,
            )

        def test_large_tokens(self) -> None:
            mp = self.registry.get("anthropic", "claude-3-opus")
            self.assertIsNotNone(mp)
            cost = calculate_cost(1_000_000_000, 1_000_000_000, mp.input_price, mp.output_price)
            self.assertTrue(cost > 0)

        def test_csv_export_integrity(self) -> None:
            jobs = [
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "input_tokens": 5000,
                    "output_tokens": 2000,
                }
            ]
            table, rows = make_table_rows(self.registry, jobs, budget_usd=100.0)
            self.assertTrue(len(table) == 1)
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "out.csv")
                write_csv(path, rows)
                self.assertTrue(os.path.isfile(path))
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertIn("provider,model,input_tokens,output_tokens", content)

        def test_validation(self) -> None:
            with self.assertRaises(ValueError):
                validate_provider_model(self.registry, "unknown", "gpt-4o")
            with self.assertRaises(ValueError):
                validate_provider_model(self.registry, None, "nope-model")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCostCalculation)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    main()
