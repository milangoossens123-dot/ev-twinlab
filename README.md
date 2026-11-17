# EV TwinLab: draft report package

Synthetic consumers for EV concept testing in the US: personas from real survey microdata (SDV),
an EV-demand benchmark that combines Pew's published shares with a choice model fitted on 5,898
real EV-vs-gasoline choices, and a reliability gate calibrated on real human-vs-AI data.
Marketing focus: price, target segments, message.

## Contents
| Path | What |
|---|---|
| `EV_TwinLab_report.ipynb` | **The report** (introduction, data exploration, solution, marketing component, summary). Executed, with outputs. |
| `EV_TwinLab_report.html` | Same report, readable in any browser. |
| `presentation/` | Short slide deck (link in `presentation/README.md`). |
| `scripts/ev_personas_sdv.py` | Step 1–3: SDV personas, EV benchmark, LLM prompts. |
| `scripts/ev_survey_model.py` | Choice model on the 2019 EV purchase survey: what real buyers trade off, with held-out validation. |
| `scripts/calibrate_weights.py` | Step 4: calibrates the reliability-score weights on the Twin-2K-500 Mega-Study. |
| `demo/ev-twinlab-us.html` | Interactive demo of the four steps (open in a browser). |
| `data/` | Input data (see below). |
| `outputs/` | Example output of `ev_personas_sdv.py`: 1,000 personas, prompts, quality report. |

## Data
| File | Source | Licence |
|---|---|---|
| `data/w13_label.csv` | Twin-2K-500, waves 1–3 responses (labels): demographics | CC-BY-4.0, huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500 |
| `data/scores1-3.csv` | Twin-2K-500 scale scores (green values, openness, risk aversion, ...) | CC-BY-4.0, same |
| `data/ev_survey_2019/` | 2019 EV infrastructure and purchase choice survey: 983 US car owners, 5,898 choices (see its README) | CC0, doi:10.7910/DVN/MOT6PN |
| `data/ev_survey_model.json` | Output of `ev_survey_model.py`: coefficients, willingness to pay, hold-out validation | Derived |
| `data/calibration_results.json` | Output of `calibrate_weights.py` on the Mega-Study `results.zip` (214 MB, not included) | Apache-2.0, huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study |

Published figures used in the EV benchmark: Pew Research Center (2025, 2026) and US Census ACS 2023.
Price, range, fast-charging and age effects in the scripts, the notebook and the demo come from the 2019 EV survey. Each is listed with its status (Published / Check / Assumption) in the script header and the demo.

## Reproduce
```bash
pip install -r requirements.txt
pip install --no-deps ./SDV-main            # SDV source from the provided zip
python scripts/ev_survey_model.py          # choice model on the 2019 EV survey
python scripts/ev_personas_sdv.py --data-dir data --n 1000 --match-acs-age --out outputs
jupyter nbconvert --to notebook --execute EV_TwinLab_report.ipynb
# optional, re-run the weight calibration (downloads 214 MB):
curl -L -o results.zip https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study/resolve/main/results.zip
python scripts/calibrate_weights.py results.zip data/w13_label.csv data/calibration_results.json
```

## Status
Draft. EV trade-offs come from a real (2019, hypothetical-choice) survey and the overall level from Pew;
confirm key numbers in a small, current human test. The LLM respondent (step 2) is represented by the
benchmark until real model calls are run.
