# TokenTally — AI Model Cost Calculator

TokenTally helps you estimate, compare, and optimize inference costs across major AI providers. It supports single and batch calculations, provider/model comparisons, CSV export, interactive menus, and a demo mode to paste prompts and preview costs before you choose a model.

## Highlights
- Multi‑provider pricing registry with easy updates and JSON overrides
- Cost formula: `(input_tokens / 1e6 * input_price) + (output_tokens / 1e6 * output_price)`
- Interactive menus for provider/model selection and repeat runs
- Demo mode: paste your prompt, auto‑estimate tokens, preview costs, then compare providers/models
- Color-coded severity based on budget thresholds
- CSV export with timestamped filenames
- Batch processing from JSON files
- Token guidance and estimators (words, characters)
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
python llm cost calculator.py --provider openai --model gpt-4o --input-tokens 5000 --output-tokens 2000

# Compare across all providers/models
python llm cost calculator.py --model gpt-4o --input-tokens 5000 --output-tokens 2000 --compare

# Batch JSON and CSV export
python llm cost calculator.py --batch-file jobs.json --export-csv

# Demo mode: paste a prompt, estimate tokens, preview costs, then compare
python llm cost calculator.py --demo

# Explain tokens and pricing formula
python llm cost calculator.py --explain-tokens
```

## Interactive Mode
- Start without arguments to open the main menu.
- Choose provider and model from numbered lists.
- Enter token counts directly or type `words`/`chars` to estimate.
- Loop as needed; the app waits for confirmation before exit.

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
- From the main menu, choose “View/Edit Pricing”.
- Update input/output prices or add new models under a provider.

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
A list of jobs or an object with `jobs` is accepted:
```json
[
  {"provider": "openai", "model": "gpt-4o", "input_tokens": 5000, "output_tokens": 2000},
  {"provider": "anthropic", "model": "claude-3-sonnet", "input_tokens": 20000, "output_tokens": 5000}
]
```
or
```json
{
  "jobs": [
    {"provider": "xai", "model": "grok-4.1", "input_tokens": 100000, "output_tokens": 50000}
  ]
}
```

## Output Tables and CSV
- Displays provider, model, input/output tokens, base cost, USD cost, cost per 1K tokens, budget percentage, severity.
- Use `--export-csv` to write a timestamped report; include `--budget` to compute percentage of budget.

## Optimization Tips
- Trim prompts; reuse/cached context where possible.
- Cap output length; ask for concise formats (bullet lists, summaries).
- Choose models with sufficient capability tier at lower cost.

## Testing
- Run built‑in tests:
```bash
python llm cost calculator.py --run-tests
```
- Tests cover formula, 1K cost, large token counts, CSV integrity, and validation.

## Notes
- Some newly added model rates are placeholders—update them to match official provider pricing.
- Currency defaults to USD. You can extend the registry to support other currencies if needed.

## Naming Suggestions
- TokenTally (used in menu banner)
- PromptMeter
- CostSight
- ModelSpend
- LLMeter
- TokenScope

If you prefer a different name, tell us and we’ll update the menu banner and references accordingly.

