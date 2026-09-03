# V2.1 09:25 Final Auction Backtest

Run: `33741447596`

## Scope

- Period: 2026-01-01..2026-08-31
- Exact 5m core data: 160/160 trading days, 100% complete for 6 PCB names + Shanghai Composite
- Valid auction-feature days: 159 (first day is excluded because previous close is required)
- Target: 002916
- PCB basket: 002916, 002463, 600183, 002938, 603228, 300476
- Development: 2026-01..06
- Validation: 2026-07..08

## Important data limitation

This run uses the first regular-session 5-minute bar `open` as the final 09:25 call-auction clearing price. It therefore tests the **final 09:25 auction price/gap + PCB opening breadth layer only**.

It does **not** contain historical 09:15/09:20/09:24 virtual-match price trajectory, matched volume, or unmatched volume. Those fields are required before calling this a full auction-process backtest.

## Win definition

- `PREOPEN_BULLISH` win: true daily high occurs after 10:00.
- `PREOPEN_BEARISH` win: true daily low occurs after 10:00.
- Secondary direction metric: 09:25 opening price to 10:00 price direction.

## Selected development rule

Mechanically selected on Jan-Jun only from a fixed grid with minimum per-period sample constraints:

- PCB mean opening gap >= +0.10% / <= -0.10%
- PCB positive/negative breadth >= 50%
- PCB relative opening gap vs Shanghai >= 0 / <= 0
- target opening gap >= +1.00% / <= -1.00%
- require all 4 votes, net margin >= 1

## Results

### Development: 2026 Jan-Jun

- 115 valid days
- Bullish: 28 signals, 16 wins, **57.14%** high-after-10 win rate
- Bearish: 28 signals, 14 wins, **50.00%** low-after-10 win rate
- Combined: 30/56 = **53.57%**
- Bullish 09:25->10:00 direction: 18/28 = **64.29%**
- Bearish 09:25->10:00 direction: 17/28 = **60.71%**

Unconditional development baselines:

- High after 10:00: **50.43%**
- Low after 10:00: **46.09%**
- 09:25->10:00 up: **49.57%**
- 09:25->10:00 down: **50.43%**

### Validation: 2026 Jul-Aug

- 44 valid days
- Bullish: 11 signals, 7 wins, **63.64%** high-after-10 win rate
- Bearish: 14 signals, 7 wins, **50.00%** low-after-10 win rate
- Combined: 14/25 = **56.00%**
- Bullish 09:25->10:00 direction: 5/11 = **45.45%**
- Bearish 09:25->10:00 direction: 7/14 = **50.00%**

Unconditional validation baselines:

- High after 10:00: **59.09%**
- Low after 10:00: **65.91%**
- 09:25->10:00 up: **45.45%**
- 09:25->10:00 down: **54.55%**

### Full 2026 Jan-Aug descriptive total

- Bullish: 23/39 = **58.97%**
- Bearish: 21/42 = **50.00%**
- Combined: 44/81 = **54.32%**

## Decision

**Do not promote the final-price-only auction layer into the frozen V1.8/V1.9 model.**

Reasons:

1. Bearish extreme-side signal fails validation: 50.00% versus a 65.91% unconditional low-after-10 baseline in Jul-Aug.
2. The apparently strong Jan-Jun 09:25->10:00 directional edge collapses in Jul-Aug to baseline or worse.
3. Bullish signal retains only a modest uplift and sample size is small.
4. The missing 09:15->09:25 auction path and matched/unmatched volume are likely the higher-information features and must be tested separately.

Next auction version should collect or acquire historical snapshots at minimum around 09:15, 09:20, 09:24:30-09:24:55, and 09:25, including virtual match price, matched volume, unmatched volume/direction, and their change rates.
