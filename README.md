# EV TwinLab: synthetic consumers for EV marketing

**Juan Esteban Londoño Gautibonza and Milan Goossens** · Quantitative Marketing

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

## Scope
The EV trade-offs come from a real (2019, hypothetical-choice) survey and the overall level of
interest from Pew. Running the generated prompts through an LLM (step 2) is outside this project's
scope: the benchmark stands in for those answers and is the yardstick they would be judged against.

## References

**Data**
- MacKenzie, D. (2020). *2019 EV infrastructure and purchase choice survey* [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/MOT6PN
- Toubia, O., Gui, G., Peng, T., Merlau, D. J., et al. (2025). *Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions*. arXiv:2505.17479. https://arxiv.org/abs/2505.17479 (data: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
- Peng, T., Gui, G., Brucks, M., Merlau, D. J., Fan, G. J., et al. (2026). *Digital twins as funhouse mirrors: Five key distortions*. arXiv:2509.19088. https://arxiv.org/abs/2509.19088 (Twin-2K-500 Mega-Study data: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study)
- U.S. Census Bureau. (2025). *Household income in states and metropolitan areas: 2024* (ACSBR-025). https://www.census.gov/library/publications/2025/acs/acsbr-025.html
- U.S. Census Bureau. (n.d.). *Urban and rural*. https://www.census.gov/programs-surveys/geography/guidance/geo-areas/urban-rural.html

**Published figures**
- Pew Research Center. (2023, July 13). *Americans' views of electric cars as gas alternative are mixed*. https://www.pewresearch.org/short-reads/2023/07/13/how-americans-view-electric-vehicles/
- Pew Research Center. (2024, June 27). *About 3 in 10 Americans would seriously consider buying an electric vehicle*. https://www.pewresearch.org/short-reads/2024/06/27/about-3-in-10-americans-would-seriously-consider-buying-an-electric-vehicle/
- Pew Research Center. (2025, June 5). *Americans' interest in purchasing electric and hybrid vehicles in 2025*. https://www.pewresearch.org/science/2025/06/05/americans-interest-in-purchasing-electric-and-hybrid-vehicles/
- Pew Research Center. (2026, April 3). *Are Americans interested in EVs and hybrids as their next car in 2026?* https://www.pewresearch.org/science/2026/04/03/how-appealing-are-electric-vehicles-and-hybrids-to-americans/

**Methods and literature**
- Hidrue, M. K., Parsons, G. R., Kempton, W., & Gardner, M. P. (2011). Willingness to pay for electric vehicles and their attributes. *Resource and Energy Economics, 33*(3), 686–705. https://doi.org/10.1016/j.reseneeco.2011.02.002
- Zou, T., Khaloei, M., & MacKenzie, D. (2020). Effects of charging infrastructure characteristics on electric vehicle preferences of new and used car buyers in the United States. *Transportation Research Record, 2674*(12), 165–175.
- Patki, N., Wedge, R., & Veeramachaneni, K. (2016). The Synthetic Data Vault. In *2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)* (pp. 399–410). https://doi.org/10.1109/DSAA.2016.49
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In *Proceedings of the 9th Python in Science Conference* (pp. 92–96).
