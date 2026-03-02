# Damodaran Valuation Framework

## Table of Contents
- Method selection by asset type
- Core formulas
- Assumption standards
- Scenario design
- Validation checks

## Method Selection by Asset Type

Use this decision rule:

1. `financial` (banks/insurers):
- Prefer `dividend_discount` or `excess_return`.
- Avoid standard FCFF when regulatory capital is the main constraint.

2. `stable-cashflow` (mature, predictable operating cash flow):
- Prefer `fcff_dcf`.
- Use normalized margins and reinvestment.

3. `high-growth` (reinvestment-heavy, margins not yet steady):
- Prefer `fcff_dcf` in 2 stages (high growth -> stable phase).
- Add explicit convergence assumptions.

4. `cyclical` or `commodity`:
- Prefer `fcff_dcf` with cycle-normalized earnings/margins.
- Avoid anchoring only on last-year cash flow.

5. `no-earnings` early-stage:
- Use `revenue_to_margin` path + FCFF bridge when possible.
- If data is insufficient, use `relative_valuation` with strict discount for uncertainty.

## Core Formulas

1. Enterprise DCF (FCFF):
- `Value_firm = sum(FCFF_t / (1 + WACC)^t) + TV / (1 + WACC)^N`
- `Equity_value = Value_firm - Debt - Minority_interest + Cash + Non_operating_assets`
- `Intrinsic_per_share = Equity_value / Diluted_shares`

2. Terminal value (perpetuity growth):
- `TV = FCFF_(N+1) / (WACC - g)`
- Require `WACC > g`.

3. Cost of equity (CAPM):
- `Ke = Rf + Beta * ERP + Country_Risk_Premium`

4. WACC:
- `WACC = Ke * E/(D+E) + Kd * (1-T) * D/(D+E)`

5. Margin of safety:
- `MoS_price = Intrinsic_per_share * (1 - MoS_pct)`

## Assumption Standards

For each valuation, force explicit assumptions:

1. Growth:
- Revenue growth path by year.
- Terminal growth tied to long-run macro reality.

2. Profitability:
- Operating margin path with transition to steady-state.
- ROIC in steady-state should be economically plausible.

3. Reinvestment:
- Reinvestment rate linked to growth and returns.
- Do not assume high growth with near-zero reinvestment.

4. Risk/discount:
- Use country/sector-consistent risk premium.
- Document leverage target.

## Scenario Design

Require at least `bear/base/bull`:

1. Bear:
- Lower growth, lower margins, higher discount rate.

2. Base:
- Most probable assumptions.

3. Bull:
- Higher growth, faster margin improvement, slightly lower discount.

Return intrinsic value and upside/downside for each scenario and the base case.

## Validation Checks

Run checks before returning outputs:

1. Math constraints:
- `discount_rate > terminal_growth_rate`
- shares outstanding > 0
- no NaN/inf in valuation outputs

2. Economic constraints:
- terminal growth not above long-run nominal GDP proxy
- terminal margin not above feasible peer range without justification

3. Output quality:
- Include method used
- Include all key assumptions
- Include confidence and data quality flags

