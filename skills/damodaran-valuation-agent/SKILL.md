---
name: damodaran-valuation-agent
description: Build, audit, and extend equity valuation agents that estimate intrinsic value per ticker/symbol using Damodaran-style frameworks. Use when creating new valuation agents, improving existing DCF/multiples logic, selecting valuation method by asset type or lifecycle, calibrating discount rates/growth assumptions, adding margin-of-safety outputs, or standardizing valuation JSON schemas in Streamlit/LLM workflows.
---

# Damodaran Valuation Agent

Use this skill to produce valuation agents that are method-aware, assumption-explicit, and robust to asset differences.

## Workflow

1. Identify the valuation target and context.
- Determine `ticker/symbol`, currency, country, sector, and business lifecycle.
- Identify asset profile: `stable-cashflow`, `high-growth`, `financial`, `cyclical`, `commodity`, `no-earnings`.

2. Select valuation method before writing prompts or code.
- Run `scripts/select_valuation_method.py` to get a deterministic method recommendation and required assumption schema.
- If uncertainty remains, default to `fcff_dcf` and document why.

3. Build valuation logic with explicit assumptions.
- Store assumptions as structured JSON fields, not hidden prose.
- Separate:
  - operating assumptions (growth, margins, reinvestment),
  - financing assumptions (cost of equity/debt, target leverage),
  - terminal assumptions (terminal growth, terminal ROIC or payout policy).

4. Compute intrinsic value with method-specific output.
- Produce at minimum:
  - `intrinsic_value_per_share`
  - `current_price`
  - `upside_downside_pct`
  - `margin_of_safety_price` (user-defined MoS, default 25%).

5. Add uncertainty and controls.
- Run bull/base/bear scenario set.
- Validate range sanity (discount rate > terminal growth, no impossible payout, no division by zero).
- Return confidence diagnostics and data quality flags.

6. Render result for agent UI.
- Output machine-readable JSON first.
- Render executive summary second.
- Keep raw JSON available as a toggle in Streamlit.

## Required Output Contract

Return JSON with these top-level keys:

```json
{
  "ticker": "AAPL",
  "valuation_method": "fcff_dcf",
  "assumptions": {},
  "scenarios": {
    "bear": {},
    "base": {},
    "bull": {}
  },
  "result": {
    "intrinsic_value_per_share": 0.0,
    "current_price": 0.0,
    "upside_downside_pct": 0.0,
    "margin_of_safety_price": 0.0
  },
  "diagnostics": {
    "model_risk": "low|medium|high",
    "data_quality": "low|medium|high",
    "checks": []
  },
  "rationale": "short explanation"
}
```

## Repository Integration

When working in this repository:

1. Review valuation entrypoints first.
- `ai_agent/pages/03_valuation_agent.py`
- `ai_agent/pages/01_portfolio_agent.py` (valuation node)
- `ai_agent/tools.py` (financial metrics source)

2. Preserve existing interfaces unless explicitly changing them.
- If changing JSON schema, update all consumers in Streamlit tabs and parsers.

3. Keep valuation logic deterministic where possible.
- Put stable formulas in Python helpers.
- Use LLM for assumption proposal and narrative, not arithmetic execution.

## References

Read this file when choosing methods, assumptions, and checks:
- `references/damodaran-valuation-framework.md`

## Script

Use this script to choose a valuation method and required assumption template:
- `scripts/select_valuation_method.py`

Example:
```bash
python scripts/select_valuation_method.py \
  --ticker MELI \
  --asset-type high-growth \
  --sector Technology \
  --country Argentina
```
