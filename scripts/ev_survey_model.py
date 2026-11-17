"""
EV TwinLab: what real US car buyers say. Choice model on the 2019 EV purchase survey.

Data (CC0, public domain):
  MacKenzie, Don (2020). "2019 EV infrastructure and purchase choice survey".
  Harvard Dataverse, https://doi.org/10.7910/DVN/MOT6PN
  983 US private car owners, 6 choice scenarios each (5,898 choices): buy an EV or a gasoline car.
  Paper: Zou, Khaloei & MacKenzie (2020), Transportation Research Record 2674(12).

Design of each scenario (from the data dictionary):
  - The EV costs 70%, 85% or 100% of the respondent's own budget for the next car (priceshow).
    We assume the gasoline alternative costs the full budget. Price enters as the EV's price gap
    as a SHARE of the budget (0, -0.15 or -0.30): this fits better than the gap in dollars
    (AIC 7,507 vs 7,573), so people judge price relative to what they planned to spend.
  - EV range 100-400 miles; fast charging 5-60 minutes (empty to full); walking minutes to the
    nearest slow charger at home and at work; fast charger in town (5/10/15 min drive or none);
    fast chargers on the highway (every 30/50/70 miles or none).

What the script does
  1. Cleans the data and prints a short exploration.
  2. Fits a binary logit of "chooses the EV" on the scenario attributes plus who the person is
     (age band, income, urban density, gender, EV owner, used-car buyer), with standard errors
     clustered by respondent (each person answers 6 times).
  3. Turns coefficients into willingness to pay (dollars per mile of range, per minute of fast
     charging, per minute of walking to a charger), with 90% intervals by bootstrap over people.
  4. Checks the model on 20% of respondents it never saw: overall share, calibration by
     predicted-probability band and by segment. This is the "held-out humans" test from the
     EV TwinLab reliability layer, on real EV choices.
  5. Writes data/ev_survey_model.json, used by scripts/ev_personas_sdv.py to replace the
     assumed values in the EV benchmark.

Usage
  python scripts/ev_survey_model.py [--data data/ev_survey_2019/ev_choice_survey_2019.tab] [--out data/ev_survey_model.json]
"""
import argparse, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
AGE = ["18-29", "30-49", "50-64", "65+"]


def load(path):
    d = pd.read_csv(path, sep="\t")
    d["ev"] = d.bichoice.astype(int)
    d["price_gap_share"] = d.priceshow / d.numprice - 1           # EV price vs budget, share (<= 0)
    d["price_gap_k"] = (d.priceshow - d.numprice) / 1000          # same in $k (comparison model only)
    d["range_100mi"] = d["range"] / 100
    d["fast_min"] = d.fasttime
    d["home_walk_min"] = d.home_chg
    d["work_walk_min"] = d.work_chg
    d["no_town_fast"] = (d.town == "no").astype(int)
    d["no_highway_fast"] = (d.highway == "no").astype(int)
    d["age_band"] = pd.cut(d.age, [0, 29, 49, 64, 200], labels=AGE).astype(str)
    d["urban"] = pd.cut(d.PopDensity, [-1, 500, 3000, 1e9], labels=["rural", "suburban", "urban"]).astype(str)
    d["income_100k_plus"] = (d.hsincome >= 5).astype(int)
    d["male"] = d.gender.astype(int)
    d["owns_ev"] = d.ev_owner.astype(int)
    d["used_buyer"] = d.used.astype(int)
    return d


ATTR = ["price_gap_share", "range_100mi", "fast_min", "home_walk_min", "work_walk_min", "no_town_fast", "no_highway_fast"]
PERSON = ["age_18_29", "age_50_64", "age_65_plus", "rural", "urban_dense", "income_100k_plus", "male", "owns_ev", "used_buyer"]


def design(d):
    X = d[ATTR].astype(float).copy()
    X["age_18_29"] = (d.age_band == "18-29").astype(float)
    X["age_50_64"] = (d.age_band == "50-64").astype(float)
    X["age_65_plus"] = (d.age_band == "65+").astype(float)
    X["rural"] = (d.urban == "rural").astype(float)
    X["urban_dense"] = (d.urban == "urban").astype(float)
    for c in ["income_100k_plus", "male", "owns_ev", "used_buyer"]:
        X[c] = d[c].astype(float)
    return sm.add_constant(X)


def fit(d):
    X = design(d)
    m = sm.GLM(d.ev, X, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": d.subject})


def wtp(params, budget):
    k = -budget / params["price_gap_share"]   # dollars per unit of utility for a buyer with this budget
    return {
        "usd_per_mile_range": params["range_100mi"] / 100 * k,
        "usd_per_minute_faster_charging": -params["fast_min"] * k,
        "usd_per_minute_shorter_walk_home_charger": -params["home_walk_min"] * k,
        "usd_per_minute_shorter_walk_work_charger": -params["work_walk_min"] * k,
        "usd_fast_charger_in_town": params["no_town_fast"] * -k,
        "usd_fast_chargers_on_highway": params["no_highway_fast"] * -k,
        "logit_scale_usd": k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ev_survey_2019/ev_choice_survey_2019.tab")
    ap.add_argument("--out", default="data/ev_survey_model.json")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = load(a.data)
    people = d.drop_duplicates("subject")
    explore = {
        "respondents": int(d.subject.nunique()), "choices": int(len(d)),
        "ev_choice_share": round(float(d.ev.mean()), 3),
        "age_mix": people.age_band.value_counts(normalize=True).reindex(AGE).round(3).to_dict(),
        "area_mix": people.urban.value_counts(normalize=True).round(3).to_dict(),
        "used_car_buyers": round(float(people.used_buyer.mean()), 3),
        "ev_owners": round(float(people.owns_ev.mean()), 3),
        "median_budget_usd": float(people.numprice.median()),
        "ev_share_by_price_vs_budget": d.groupby("priceprop").ev.mean().round(3).to_dict(),
        "ev_share_by_range": d.groupby("range").ev.mean().round(3).to_dict(),
        "ev_share_by_fast_charge_min": d.groupby("fasttime").ev.mean().round(3).to_dict(),
        "ev_share_by_age": d.groupby("age_band").ev.mean().reindex(AGE).round(3).to_dict(),
        "ev_share_by_area": d.groupby("urban").ev.mean().round(3).to_dict(),
    }

    # ---- full-sample model
    res = fit(d)
    params, ci = res.params, res.conf_int(alpha=0.10)
    coef = pd.DataFrame({"coef": params, "ci90_low": ci[0], "ci90_high": ci[1], "p_value": res.pvalues}).round(4)
    budgets = {"at_budget_20k_survey_median": 20000, "at_budget_50k_reference_ev": 50000}
    W = {b: wtp(params, v) for b, v in budgets.items()}
    # comparison: price gap in dollars instead of share of budget
    Xd = design(d).drop(columns=["price_gap_share"]); Xd["price_gap_k"] = d.price_gap_k
    aic = {"share_of_budget": round(float(res.aic), 1),
           "dollars": round(float(sm.GLM(d.ev, Xd, family=sm.families.Binomial()).fit().aic), 1)}

    # bootstrap over respondents for WTP intervals
    subj = d.subject.unique(); groups = {s: g for s, g in d.groupby("subject")}
    boots = []
    for _ in range(300):
        pick = rng.choice(subj, len(subj), replace=True)
        b = pd.concat([groups[s].assign(subject=i) for i, s in enumerate(pick)])
        try:
            p_ = sm.GLM(b.ev, design(b), family=sm.families.Binomial()).fit().params
            boots.append({f"{bn}|{k}": v for bn, bv in budgets.items() for k, v in wtp(p_, bv).items()})
        except Exception:
            pass
    B = pd.DataFrame(boots)
    wtp_tab = {bn: {k: dict(estimate=round(float(v), 1),
                            ci90=[round(float(B[f"{bn}|{k}"].quantile(.05)), 1), round(float(B[f"{bn}|{k}"].quantile(.95)), 1)])
                    for k, v in W[bn].items()} for bn in budgets}

    # ---- hold-out validation (20% of respondents)
    hold = set(rng.choice(subj, int(0.2 * len(subj)), replace=False))
    tr, te = d[~d.subject.isin(hold)], d[d.subject.isin(hold)]
    rt = fit(tr)
    te = te.assign(p=rt.predict(design(te)))
    base = tr.ev.mean()
    ll = lambda y, p: float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    seg_rows = []
    for name, mask in [("All", te.ev >= 0)] + [(f"Age {a}", te.age_band == a) for a in AGE] + \
                      [(f"Area {u}", te.urban == u) for u in ["rural", "suburban", "urban"]] + \
                      [("Income $100k+", te.income_100k_plus == 1), ("Used-car buyers", te.used_buyer == 1),
                       ("New-car buyers", te.used_buyer == 0), ("Owns an EV", te.owns_ev == 1)] + \
                      [(f"Range {r} mi", te["range"] == r) for r in [100, 200, 300, 400]] + \
                      [(f"EV at {int(round(p*100))}% of budget", te.priceprop == p) for p in [0.7, 0.85, 1.0]]:
        g = te[mask]
        if len(g) >= 30:
            seg_rows.append(dict(segment=name, choices=int(len(g)), actual=round(float(g.ev.mean()), 3),
                                 predicted=round(float(g.p.mean()), 3), gap_points=round(100 * float(g.p.mean() - g.ev.mean()), 1)))
    seg = pd.DataFrame(seg_rows)
    bands = pd.cut(te.p, [0, .3, .45, .6, .75, 1])
    calib = te.groupby(bands, observed=True).agg(n=("ev", "size"), predicted=("p", "mean"), actual=("ev", "mean")).round(3)
    validation = {
        "holdout_respondents": len(hold), "holdout_choices": int(len(te)),
        "log_loss_model": round(ll(te.ev, te.p), 4), "log_loss_constant": round(ll(te.ev, np.full(len(te), base)), 4),
        "accuracy_model": round(float(((te.p > .5) == te.ev).mean()), 3),
        "accuracy_constant": round(float(max(base, 1 - base)), 3),
        "mean_abs_segment_gap_points": round(float(seg.gap_points.abs().mean()), 2),
        "segments": seg.to_dict("records"),
        "calibration": calib.reset_index().astype({"p": str}).rename(columns={"p": "predicted_band"}).to_dict("records"),
    }

    # ---- relative odds by age and area, for the persona benchmark (at the sample's typical concept)
    age_logodds = {"18-29": params["age_18_29"], "30-49": 0.0, "50-64": params["age_50_64"], "65+": params["age_65_plus"]}
    area_logodds = {"rural": params["rural"], "suburban": 0.0, "urban": params["urban_dense"]}

    out = dict(
        source="MacKenzie (2020) 2019 EV infrastructure and purchase choice survey, Harvard Dataverse doi:10.7910/DVN/MOT6PN (CC0)",
        assumptions=["Gasoline alternative priced at the respondent's budget",
                     "Fast charging time in the survey is empty-to-full minutes",
                     "Stated choices: the EV share (55%) is higher than real-world intent; use for trade-offs, not levels"],
        exploration=explore, coefficients=coef.reset_index().rename(columns={"index": "term"}).to_dict("records"),
        n_obs=int(res.nobs), price_model_aic=aic, wtp=wtp_tab,
        price_gap_share_coef=float(params["price_gap_share"]), validation=validation,
        age_log_odds_vs_30_49=age_logodds, area_log_odds_vs_suburban=area_logodds,
    )
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    print("== Exploration"); print(json.dumps(explore, indent=1))
    print("\n== Coefficients (logit, cluster-robust 90% CI)"); print(coef.to_string())
    print("\n== Willingness to pay (90% bootstrap CI)")
    print("  price model AIC (lower is better):", aic)
    for bn, tab in wtp_tab.items():
        print(" ", bn)
        for k, v in tab.items():
            print(f"    {k:45s} {v['estimate']:>10,.1f}   [{v['ci90'][0]:,.1f}, {v['ci90'][1]:,.1f}]")
    print("\n== Hold-out validation (people the model never saw)")
    print({k: v for k, v in validation.items() if k not in ("segments", "calibration")})
    print(seg.to_string(index=False)); print(calib.to_string())


if __name__ == "__main__":
    main()
