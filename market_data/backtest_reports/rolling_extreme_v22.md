# V2.2 Rolling Extreme Probability Validation

Run: `33742953424`

## Objective

The primary objective is **not** 10:00 direction prediction.

At every completed 5-minute bar, estimate:

- `P_TOP_LOCKED`: probability that no later high will materially exceed the current executable price;
- `P_BOTTOM_LOCKED`: probability that no later low will materially undercut the current executable price;
- whether current price is still close enough to the running extreme to make a T trade practical;
- remaining reversal room versus residual wrong-way room.

10:00 is only one rolling snapshot.

## Label

Diagnostic tolerance:

- top locked: no later high exceeds current close by more than `max(0.25%, 0.25 × signal-time ATR%)`;
- bottom locked: no later low undercuts current close by more than the same tolerance;
- future data is scoring-only.

A live-action candidate must also satisfy `near_top` or `near_bottom`, so a correct but already-late extreme diagnosis does not count as a useful T signal.

## Data split

- 160 trading days, exact 5-minute core data;
- Development/calibration: 2026-01-01..2026-06-30;
- Validation: 2026-07-01..2026-08-31;
- Validation decision rows: 1,804;
- Eligible times: approximately 09:50..14:40;
- Current run is a domestic price/PCB-context baseline. Full overseas priors, realtime-news impact and full auction-path data are not yet promoted into the probability calibration.

## Critical evaluation rule

Persistent signals across consecutive 5-minute bars are **not** independent wins.

Primary metrics use only episode starts: a new alert is counted when the condition changes from false to true within a trading day.

## Validation — TOP / reverse-T side

Overall top-lock base rate across all eligible validation bars: **20.95%**.

Probability discrimination:

- AUC: **0.639**
- Brier: **0.157**

### Episode-start signals with current price still near the running high

| Threshold | Episodes | Signal days | Actual top locked | Downside > residual upside | Median later downside | Median residual upside | Median reversal/residual ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| P>=50% | 37 | 18 | 43.24% | 64.86% | 1.99% | 0.41% | 1.66x |
| P>=60% | 19 | 12 | **63.16%** | **78.95%** | **2.74%** | **0.15%** | **5.22x** |
| P>=70% | 12 | 8 | **75.00%** | **75.00%** | **2.28%** | **0.07%** | **5.85x** |

Interpretation:

- The reverse-T side shows a usable concentration effect as probability threshold rises.
- At `P_TOP_LOCKED >= 70%` and `near_top`, validation precision is 75% on episode starts, with materially more later downside than residual upside.
- Sample size is still small.
- The >=70% validation episodes occur only in the afternoon buckets (13:05 onward); early-session top detection is not yet good enough to claim a full-day 70% signal.

Therefore **do not freeze or production-promote this rule yet**.

## Validation — BOTTOM / positive-T side

Overall bottom-lock base rate across all eligible validation bars: **16.46%**.

Probability discrimination:

- AUC: **0.626**
- Brier: **0.135**

The calibrated baseline did not produce any `near_bottom` validation episode at `P_BOTTOM_LOCKED >= 50%`; the maximum calibrated bottom probability in validation was below 0.50.

Interpretation:

- Current bottom features are not strong enough for a high-confidence positive-T signal.
- This is consistent with the historical observation that LOW requires a different confirmation mechanism from HIGH.
- Do not force symmetric top/bottom rules or lower the probability threshold merely to create signals.

## Decision

**V2.2 remains diagnostic, not frozen.**

Current evidence:

1. TOP / reverse-T side: promising, especially afternoon `P>=60/70% + near_top`, but sample count is insufficient and early-session behavior is weaker.
2. BOTTOM / positive-T side: not ready; no >=50% calibrated near-bottom validation signals.
3. Next work should improve bottom-specific features and then add external-news/overseas/auction information as separate incremental layers.
4. Future live evaluation must continue using episode-start/state-change notifications rather than counting persistent bars as separate wins.
