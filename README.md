# TokenTally — AI Model Cost Calculator

TokenTally helps you preview and plan inference costs before you subscribe to any provider. Paste a prompt, estimate tokens, compare models side‑by‑side, add non‑token API costs, convert to your currency, and even simulate monthly spend for typical workloads.

## What’s New 
- Currency conversion: show both USD and your currency with `--currency` and `--fx-rate`.
- Usage templates: one‑click token estimates for Q&A, summaries, code, and reports.
- Monthly planner: simulate spend (requests/day × days) across all models or a chosen model.
- Non‑token API costs: include per‑request fees, audio minutes, and image generation costs.
- Budget alarms: warn when a job exceeds a percentage of your budget.
- Animated banner: a friendly greeting on first run.
- Demo mode: paste a prompt to get realistic input token estimates before choosing models.

## Highlights
- Multi‑provider pricing registry with easy updates and JSON overrides
- Cost formula: `(input_tokens / 1e6 * input_price) + (output_tokens / 1e6 * output_price)`
- Interactive menus for provider/model selection and repeat runs
- Demo mode: paste your prompt, auto‑estimate tokens, preview costs, then compare providers/models
- Color‑coded severity based on budget thresholds
- CSV export with timestamped filenames
- Batch processing from JSON files
- Token guidance and estimators (words, characters); accurate when `tiktoken` is installed
- Optional live pricing fetch from configurable sources
- Alternative suggestions: shows cheaper models of similar capability tier

## Install
```bash
python -m pip install -r requirements.txt
# Optional (for more accurate token counts)
python -m pip install tiktoken
```

## Quick Start
```bash
# Single calculation
python tokentally.py --provider openai --model gpt-4o --input-tokens 5000 --output-tokens 2000

# Compare across all providers/models
python tokentally.py --model gpt-4o --input-tokens 5000 --output-tokens 2000 --compare

# Batch JSON and CSV export
python tokentally.py --batch-file jobs.json --export-csv

# Demo mode: paste a prompt, estimate tokens, preview costs, then compare
python tokentally.py --demo

# Explain tokens and pricing formula
python tokentally.py --explain-tokens
```

## Interactive Mode
- Start without arguments to open the guided flow.
- Choose provider and model from numbered lists.
- Enter token counts directly or type `words`/`chars` to estimate.
- Loop as needed; the app waits for confirmation before exit.

## Currency Conversion
- Show your currency alongside USD:
```bash
python tokentally.py --model gpt-4o --input-tokens 5k --output-tokens 2k --compare --currency EUR --fx-rate 0.92
```

## Usage Templates
- One‑click token profiles; scale as needed:
```bash
# Summary template scaled by 1.5×
python tokentally.py --template summary --template-scale 1.5 --model gpt-4o --compare
```

Templates:
- `qa` (500 in, 300 out)
- `summary` (2000 in, 600 out)
- `code` (3000 in, 1200 out)
- `report` (8000 in, 2000 out)

## Monthly Planner
- Simulate monthly spend with usage assumptions:
```bash
# All models
python tokentally.py --planner --requests-per-day 300 --days 30 --template qa --currency EUR --fx-rate 0.92

# Specific model
python tokentally.py --planner --provider anthropic --model claude-3-sonnet --requests-per-day 150 --days 30 --input-tokens 2k --output-tokens 800 --budget 250 --alarm-threshold-pct 10
```

## Non‑Token API Costs
- Add line items to reflect your real scenario:
```bash
python tokentally.py \
  --provider openai --model gpt-4o \
  --input-tokens 10k --output-tokens 3k \
  --per-request-fee 0.01 \
  --audio-minutes 5 --audio-rate 0.02 \
  --image-count 2 --image-rate 0.03
```

## Token Guidance
- Rough conversions:
  - `tokens ≈ words × 0.75`
  - `tokens ≈ characters / 4`
- In demo mode, TokenTally estimates tokens from your pasted prompt. If `tiktoken` is installed, it uses a tokenizer; otherwise it falls back to character length.

## Pricing Data
- Built‑in `PRICING_DATA` includes OpenAI, Anthropic, XAI, Google, Cohere, Mistral, Meta, plus common aliases.
- Each entry stores USD prices per 1M tokens for input and output.
- Capability tiers help suggest similar‑quality but cheaper alternatives.

### Update Pricing (Interactive)
- Choose “View/Edit Pricing” in the menu to update input/output prices or add new models.

### Update Pricing (Config File)
- Create a `pricing_config.json` next to the script:
```json
{
  "pricing": {
    "openai": {
      "gpt-4o": {"input": 5.0, "output": 15.0}
    }
  },
  "pricing_sources": {
    "openai": "https://example.com/openai-pricing.json",
    "anthropic": "https://example.com/anthropic-pricing.json"
  }
}
```
- Run with `--config pricing_config.json`. If sources are provided, the app will attempt fetching live pricing (gracefully handles rate limiting).

## Batch File Schema
```json
[
  {"provider": "openai", "model": "gpt-4o", "input_tokens": 5000, "output_tokens": 2000},
  {"provider": "anthropic", "model": "claude-3-sonnet", "input_tokens": 20000, "output_tokens": 5000}
]
```
Or:
```json
{
  "jobs": [
    {"provider": "xai", "model": "grok-4.1", "input_tokens": 100000, "output_tokens": 50000}
  ]
}
```

## Output Tables and CSV
- Displays provider, model, input/output tokens, base cost, USD cost, converted cost, cost per 1K, budget percentage, severity.
- Use `--export-csv` to write a timestamped report; include `--budget` to compute percentage of budget.

## Optimization Tips
- Trim prompts; reuse/cached context where possible.
- Cap output length; ask for concise formats (bullet lists, summaries).
- Choose models with sufficient capability tier at lower cost.

## Testing
```bash
python tokentally.py --run-tests
```
- Tests cover formula, 1K cost, large token counts, CSV integrity, and validation.

## Notes
- Some newly added model rates are placeholders—update them to match official provider pricing.
- Currency defaults to USD; pass `--currency` and `--fx-rate` to see your currency.
- Install `tiktoken` for more accurate token counts when previewing prompts.

## Naming Suggestions
- TokenTally (menu banner)
- PromptMeter
- CostSight
- ModelSpend
- LLMeter
- TokenScope

If you prefer a different name, tell us and we’ll update the menu banner and references accordingly.
