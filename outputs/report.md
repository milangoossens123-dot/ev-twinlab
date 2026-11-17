# EV personas from SDV: report

Concept: $45,000, 300 mi, 30 min fast charge. Personas: 1000 (gaussian_copula).

## 1. Does the synthetic data look like real people it never saw?

SDV quality report against a 20% hold-out of real respondents (0 to 1, higher is better).

| Comparison | Overall | Column shapes | Column pair trends |
|---|---|---|---|
| SDV synthetic vs hold-out | 0.86 | 0.94 | 0.78 |
| Independent draws vs hold-out (old demo) | 0.82 | 0.95 | 0.69 |
| Real train sample vs hold-out (ceiling) | 0.92 | 0.95 | 0.89 |

## 2. Cross-tabs that matter for EV demand

|                                    |   Real (all 2,058) |   SDV synthetic |   Independent draws |
|:-----------------------------------|-------------------:|----------------:|--------------------:|
| Democrat share, age 18-29          |              0.41  |           0.454 |               0.403 |
| Democrat share, age 65+            |              0.455 |           0.385 |               0.401 |
| Income $100k+, college degree      |              0.387 |           0.365 |               0.273 |
| Income $100k+, no degree           |              0.145 |           0.178 |               0.268 |
| Green values, Democrats            |              3.809 |           3.687 |               3.513 |
| Green values, Republicans          |              3.182 |           3.241 |               3.512 |
| Liberal or very liberal, age 18-29 |              0.482 |           0.479 |               0.454 |
| Liberal or very liberal, age 65+   |              0.451 |           0.396 |               0.419 |

## 3. EV benchmark for this concept (Pew level + 2019 survey trade-offs)

Would seriously consider: 45% of synthetic personas.

By age: {'18-29': 0.554, '30-49': 0.44, '50-64': 0.422, '65+': 0.402}

By party lean: {'Dem': 0.559, 'None': 0.475, 'Rep': 0.256}

Fit check at the reference EV (published targets: Dem-leaning 48%, Rep-leaning 18%; overall implied by this sample's party mix: 36%): {'lean=Dem': 0.48, 'lean=Rep': 0.18, 'lean=None': 0.33, 'age_band=18-29': 0.433, 'age_band=30-49': 0.368, 'age_band=50-64': 0.324, 'age_band=65+': 0.324, 'overall': 0.36}

## Files

- synthetic_personas.csv: one row per synthetic person
- prompts.jsonl: the step-2 prompt for each person, ready to send to an LLM
- report.json: everything above in machine-readable form
