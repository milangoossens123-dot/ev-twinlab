"""
EV TwinLab: step 1 (persona generator) rebuilt on real US microdata with SDV.

What it does
  1. Builds a training table from REAL respondents: Twin-2K-500 (2,058 US adults, CC-BY-4.0).
     Demographics plus a few traits that matter for EV adoption:
     green consumer values (GREEN scale), openness, risk aversion, spending style, financial literacy.
  2. Fits an SDV synthesizer (Gaussian copula by default, CTGAN if installed and requested)
     that learns the JOINT distribution: who is old and conservative, who is young, green and
     high-income, and so on. The live demo drew each variable independently; this keeps the
     correlations.
  3. Checks the synthetic people against real people it never saw (20% hold-out), using
     SDV's quality report and a few EV-relevant cross-tabs, and compares with independent draws.
  4. Adds the EV layer from published figures (same benchmark as the demo):
     Pew 2025 shares by party lean and age, Hidrue et al. (2011) dollar values for range and
     charging time, plus the person's own green-values score.
  5. Writes, for a chosen EV concept:
       synthetic_personas.csv   one row per synthetic person, with benchmark consideration probability
       prompts.jsonl            the step-2 LLM prompt for each person (persona + concept, JSON answer)
       report.md / report.json  quality checks and what came from where

Usage
  pip install pandas numpy scipy ./SDV-main        # SDV from the provided zip (CTGAN optional)
  python ev_personas_sdv.py --n 1000 --price 45000 --range 300 --charge 30 --out ev_personas_out

  Data are downloaded from Hugging Face on first run into --data-dir (needs access to
  huggingface.co). Pass --data-dir to a folder that already holds the three files to work offline.

Status of inputs (as on the demo page)
  Published : Pew 2025 overall 33%, Democrats 48%, Republicans 18%; Hidrue et al. $35-75/mile,
              $425-3,250 per hour of charging saved; Twin-2K-500 microdata.
  Check     : under-30 about 45% (Pew 2026 "split"); applying slow-charging values to fast charging.
  Assumption: age 30-49/50-64/65+ levels; price sensitivity; effect of green values (0.5 logit per SD);
              residual taste spread; reference EV ($50k, 270 mi, 30 min); urban/rural (not in this data).
"""
import argparse, json, os, sys, urllib.request, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HF = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/"
FILES = {
    "w13_label.csv": HF + "question_catalog_and_human_response_csv/wave1_3_response_label.csv",
    "scores1.csv": HF + "raw_data/wave%201%20scores.csv",
    "scores2.csv": HF + "raw_data/wave%202%20scores.csv",
    "scores3.csv": HF + "raw_data/wave%203%20scores.csv",
}
TABLE = "respondents"

AGE = ["18-29", "30-49", "50-64", "65+"]
EDU = ["Less than high school", "High school graduate", "Some college, no degree",
       "Associate's degree", "College graduate/some postgrad", "Postgraduate"]
INC = ["Less than $30,000", "$30,000-$50,000", "$50,000-$75,000", "$75,000-$100,000", "$100,000 or more"]
INC_MID = dict(zip(INC, [20000, 40000, 62500, 87500, 150000]))
IDEO = ["Very conservative", "Conservative", "Moderate", "Liberal", "Very liberal"]
HH = ["1", "2", "3", "4", "More than 4"]
# Party as a left-right order, so the copula can learn e.g. that Democrats score higher on green values.
PARTY = ["Republican", "Something else", "Independent", "Democrat"]

# Published-figures benchmark (see the demo's sources table)
REF = dict(price=50000, range=270, charge=30)
VAL = dict(range=55, charge=31, scale=6000, inc_el=0.4)
TARGET_TOTAL = 0.33
TARGET = {"lean": {"Dem": 0.48, "Rep": 0.18},
          "age_band": {"18-29": 0.45, "30-49": 0.38, "50-64": 0.26, "65+": 0.20}}
B_GREEN, TASTE_SD = 0.5, 0.8
ACS_AGE = {"18-29": 0.205, "30-49": 0.335, "50-64": 0.24, "65+": 0.22}   # approx. ACS 2023 (Check)

sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(p / (1 - p))


# ------------------------------------------------------------------ 1. real training table
def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...", file=sys.stderr)
            urllib.request.urlretrieve(url, path)


def build_real_table(data_dir):
    lab = pd.read_csv(os.path.join(data_dir, "w13_label.csv"), dtype=str,
                      usecols=["pid", "QID11", "QID12", "QID13", "QID14", "QID15", "QID20", "QID21", "QID22", "QID23", "QID24"])
    lab.columns = ["pid", "region", "sex", "age_band", "education", "race", "party", "income_band",
                   "ideology", "household_size", "employment"]
    lab["region"] = lab.region.str.split(" ").str[0]
    lab["pid"] = lab.pid.astype(int)
    s1 = pd.read_csv(os.path.join(data_dir, "scores1.csv"), encoding="utf-8-sig")
    s2 = pd.read_csv(os.path.join(data_dir, "scores2.csv"), encoding="utf-8-sig")
    s3 = pd.read_csv(os.path.join(data_dir, "scores3.csv"), encoding="utf-8-sig")
    sc = (s1[["TWIN_ID", "score_GREEN", "score_openness"]]
          .merge(s2[["TWIN_ID", "score_riskaversion", "score_finliteracy"]], on="TWIN_ID")
          .merge(s3[["TWIN_ID", "score_ST-TW"]], on="TWIN_ID"))
    sc.columns = ["pid", "green_values", "openness", "risk_aversion", "financial_literacy", "spending_style"]
    for c in sc.columns[1:]:
        sc[c] = pd.to_numeric(sc[c], errors="coerce")
    df = lab.merge(sc, on="pid").drop(columns="pid")
    valid = (df.age_band.isin(AGE) & df.education.isin(EDU) & df.income_band.isin(INC)
             & df.ideology.isin(IDEO) & df.household_size.isin(HH))
    # risk_aversion is missing for ~11% (inconsistent lottery choices); SDV models the gaps.
    df = df[valid].dropna(subset=[c for c in df.columns if c != "risk_aversion"]).reset_index(drop=True)
    return df


def make_metadata(df):
    from sdv.metadata import Metadata
    md = Metadata.detect_from_dataframes({TABLE: df})
    for col, order in [("age_band", AGE), ("education", EDU), ("income_band", INC),
                       ("ideology", IDEO), ("household_size", HH), ("party", PARTY)]:
        md.update_column(col, table_name=TABLE, sdtype="ordinal", range_values=order)
    for col in ["region", "sex", "race", "employment"]:
        md.update_column(col, table_name=TABLE, sdtype="categorical")
    for col in ["green_values", "openness", "risk_aversion", "financial_literacy", "spending_style"]:
        md.update_column(col, table_name=TABLE, sdtype="numerical")
    md.validate()
    return md


# ------------------------------------------------------------------ 2. synthesizer
def fit_synthesizer(train, md, kind, seed, distribution="gaussian_kde"):
    np.random.seed(seed)
    if kind == "ctgan":
        from sdv.single_table import CTGANSynthesizer
        syn = CTGANSynthesizer(md, epochs=300)
    else:
        from sdv.single_table import GaussianCopulaSynthesizer
        # gaussian_kde scored best on hold-out quality and on the EV cross-tabs
        # (compared with beta, truncnorm, norm, uniform).
        syn = GaussianCopulaSynthesizer(md, default_distribution=distribution)
    syn.fit({TABLE: train})
    return syn


def sample(syn, n, match_acs_age):
    if not match_acs_age:
        return syn.sample(n)[TABLE]
    from sdv.sampling import Condition
    counts = {a: int(round(n * s)) for a, s in ACS_AGE.items()}
    counts["30-49"] += n - sum(counts.values())
    conds = [Condition(column_values={"age_band": a}, num_rows=k) for a, k in counts.items()]
    out = syn.sample_from_conditions(conds)[TABLE]
    return out.sample(frac=1, random_state=0).reset_index(drop=True)


def independent_draws(train, n, seed):
    """What the live demo does: every column drawn on its own, no correlations."""
    r = np.random.default_rng(seed)
    return pd.DataFrame({c: train[c].values[r.integers(0, len(train), n)] for c in train.columns})


# ------------------------------------------------------------------ 3. checks
def quality(real, synth, md):
    from sdv.evaluation import evaluate_quality
    rep = evaluate_quality({TABLE: real}, {TABLE: synth}, md, verbose=False)
    props = rep.get_properties().set_index("Property")["Score"].to_dict()
    return dict(overall=float(rep.get_score()), **{k: float(v) for k, v in props.items()})


def crosstabs(d):
    dem = d.party.eq("Democrat")
    college = d.education.isin(EDU[4:])
    return {
        "Democrat share, age 18-29": dem[d.age_band == "18-29"].mean(),
        "Democrat share, age 65+": dem[d.age_band == "65+"].mean(),
        "Income $100k+, college degree": d.income_band.eq(INC[-1])[college].mean(),
        "Income $100k+, no degree": d.income_band.eq(INC[-1])[~college].mean(),
        "Green values, Democrats": d.green_values[dem].mean(),
        "Green values, Republicans": d.green_values[d.party.eq("Republican")].mean(),
        "Liberal or very liberal, age 18-29": d.ideology.isin(IDEO[3:])[d.age_band == "18-29"].mean(),
        "Liberal or very liberal, age 65+": d.ideology.isin(IDEO[3:])[d.age_band == "65+"].mean(),
    }


# ------------------------------------------------------------------ 4. EV layer
def lean(d):
    lib, con = d.ideology.isin(IDEO[3:]), d.ideology.isin(IDEO[:2])
    out = np.where(d.party.eq("Democrat") | (~d.party.isin(["Democrat", "Republican"]) & lib), "Dem",
          np.where(d.party.eq("Republican") | (~d.party.isin(["Democrat", "Republican"]) & con), "Rep", "None"))
    return pd.Series(out, index=d.index)


class EVBenchmark:
    """Consideration model fitted (raking) so that, at the reference EV, the real training
    respondents reproduce the published Pew shares."""

    def __init__(self, real, seed):
        self.g_mu, self.g_sd = real.green_values.mean(), real.green_values.std()
        r = np.random.default_rng(seed)
        d = real.assign(lean=lean(real), taste=r.normal(0, TASTE_SD, len(real)))
        self.off = {"lean": {"Dem": 0.0, "Rep": 0.0, "None": 0.0}, "age_band": {a: 0.0 for a in AGE}}
        # Party-lean targets are published: keep them exact (no-lean people get the overall 33%).
        # Age targets are partly assumptions: shift them so they imply the same total as party does.
        lean_tg = {**TARGET["lean"], "None": TARGET_TOTAL}
        total = float(sum((d.lean == k).mean() * t for k, t in lean_tg.items()))
        sh = np.array([(d.age_band == k).mean() for k in AGE])
        base = logit(np.array([TARGET["age_band"][k] for k in AGE]))
        lo, hi = -4.0, 4.0
        for _ in range(60):
            m = (lo + hi) / 2
            (lo, hi) = (m, hi) if (sh * sig(base + m)).sum() < total else (lo, m)
        self.adj = {"lean": lean_tg, "age_band": {k: float(sig(b + (lo + hi) / 2)) for k, b in zip(AGE, base)}}
        self.implied_total = total
        for _ in range(80):
            for v in ["lean", "age_band"]:
                for k in self.off[v]:
                    sel = d[v] == k
                    if not sel.any():
                        continue
                    target = self.adj[v][k]
                    m = sig(self.latent(d[sel])).mean()
                    self.off[v][k] += logit(target) - logit(m)
        self.fit_check = {f"{v}={k}": round(float(sig(self.latent(d[d[v] == k])).mean()), 3)
                          for v in self.off for k in self.off[v]}
        self.fit_check["overall"] = round(float(sig(self.latent(d)).mean()), 3)

    def latent(self, d):
        z = (d.green_values - self.g_mu) / self.g_sd
        return (d.lean.map(self.off["lean"]) + d.age_band.map(self.off["age_band"])
                + B_GREEN * z + d.taste).values

    def apply(self, d, concept, seed):
        r = np.random.default_rng(seed)
        d = d.assign(lean=lean(d), taste=r.normal(0, TASTE_SD, len(d)))
        lat = self.latent(d)
        inc = d.income_band.map(INC_MID).values
        scale = VAL["scale"] * (inc / 80000) ** VAL["inc_el"]
        attr = VAL["range"] * (concept["range"] - REF["range"]) + VAL["charge"] * (REF["charge"] - concept["charge"])
        p = sig(lat + (REF["price"] + attr - concept["price"]) / scale)
        p_ref = sig(lat)
        att = np.digitize(p_ref, [0.10, 0.25, 0.45, 0.65]) + 1
        d = d.drop(columns="taste")
        d["party_lean"] = d.pop("lean")
        d["ev_attitude_1to5"] = att
        d["max_price_usd"] = np.round((REF["price"] + attr + lat * scale) / 100) * 100
        d["p_consider_benchmark"] = np.round(p, 4)
        d["benchmark_answer"] = np.where(r.random(len(d)) < p, "A", "B")
        return d


# ------------------------------------------------------------------ 5. prompts
def describe(x, lo, hi, words):
    t = (x - lo) / (hi - lo)
    return words[0] if t < 0.4 else words[1] if t < 0.7 else words[2]


def prompt(row, concept, q):
    green = describe(row.green_values, q["green_values"][0], q["green_values"][1],
                     ["Protecting the environment is not a big factor in what you buy",
                      "You give some weight to the environment when you buy things",
                      "You care a lot about the environmental impact of what you buy"])
    spend = describe(row.spending_style, q["spending_style"][0], q["spending_style"][1],
                     ["You find it hard to spend money and avoid big purchases",
                      "You spend money carefully",
                      "You find it easy to spend money"])
    risk = ("" if pd.isna(row.risk_aversion) else
            " " + describe(row.risk_aversion, q["risk_aversion"][0], q["risk_aversion"][1],
                           ["You are comfortable taking risks.", "You are moderately cautious.", "You avoid risks."]))
    hh = "alone" if row.household_size == "1" else f"in a household of {row.household_size.lower()} people"
    return (
        f"You are a {row.sex.lower()} aged {row.age_band} living in "
        f"{'Hawaii or Alaska' if row.region == 'Pacific' else 'the ' + row.region + ' of the United States'}, {hh}. "
        f"Education: {row.education}. Employment: {row.employment.lower()}. Household income: {row.income_band} a year. "
        f"Politically you identify as {row.party if row.party != 'Something else' else 'neither Democrat, Republican nor Independent'}, "
        f"with {row.ideology.lower()} views. {green}. {spend}.{risk}\n\n"
        f"Option A: a new electric car, ${concept['price']:,}, {concept['range']} miles of range, "
        f"{concept['charge']}-minute fast charge (10-80%).\n"
        f"Option B: a comparable new gasoline car.\n\n"
        "Would you seriously consider Option A? Answer with JSON only:\n"
        '{"choice": "A" | "B", "max_price_usd": number, "reason": string}'
    )


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="number of synthetic personas")
    ap.add_argument("--price", type=int, default=45000)
    ap.add_argument("--range", type=int, default=300, help="miles")
    ap.add_argument("--charge", type=int, default=30, help="fast-charge minutes, 10-80%%")
    ap.add_argument("--synthesizer", choices=["gaussian_copula", "ctgan"], default="gaussian_copula")
    ap.add_argument("--distribution", default="gaussian_kde",
                    choices=["gaussian_kde", "beta", "truncnorm", "norm", "uniform"], help="Gaussian copula marginals")
    ap.add_argument("--match-acs-age", action="store_true", help="condition the age mix on ACS 2023 shares")
    ap.add_argument("--data-dir", default="twin2k_data")
    ap.add_argument("--out", default="ev_personas_out")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    concept = dict(price=a.price, range=a.range, charge=a.charge)

    download(a.data_dir)
    real = build_real_table(a.data_dir)
    md = make_metadata(real)
    r = np.random.default_rng(a.seed)
    hold = r.random(len(real)) < 0.2
    train, test = real[~hold].reset_index(drop=True), real[hold].reset_index(drop=True)
    print(f"real respondents: {len(real)} (train {len(train)}, hold-out {len(test)})", file=sys.stderr)

    syn = fit_synthesizer(train, md, a.synthesizer, a.seed, a.distribution)
    synth_eval = syn.sample(len(test))[TABLE]
    indep_eval = independent_draws(train, len(test), a.seed)
    q = {"SDV synthetic vs hold-out": quality(test, synth_eval, md),
         "Independent draws vs hold-out (old demo)": quality(test, indep_eval, md),
         "Real train sample vs hold-out (ceiling)": quality(test, train.sample(len(test), random_state=a.seed), md)}
    big_synth = syn.sample(5000)[TABLE]
    big_indep = independent_draws(train, 5000, a.seed + 1)
    ct = pd.DataFrame({"Real (all 2,058)": crosstabs(real), "SDV synthetic": crosstabs(big_synth),
                       "Independent draws": crosstabs(big_indep)}).round(3)

    personas = sample(syn, a.n, a.match_acs_age)
    bench = EVBenchmark(real, a.seed)
    personas = bench.apply(personas, concept, a.seed + 2)
    personas.insert(0, "persona_id", [f"P{i+1:05d}" for i in range(len(personas))])
    rng_q = {c: (float(real[c].dropna().quantile(0.05)), float(real[c].dropna().quantile(0.95))) for c in ["green_values", "spending_style", "risk_aversion"]}

    os.makedirs(a.out, exist_ok=True)
    personas.to_csv(os.path.join(a.out, "synthetic_personas.csv"), index=False)
    with open(os.path.join(a.out, "prompts.jsonl"), "w") as f:
        for row in personas.itertuples():
            f.write(json.dumps({"persona_id": row.persona_id, "concept": concept, "prompt": prompt(row, concept, rng_q)}) + "\n")

    seg = personas.assign(c=personas.benchmark_answer.eq("A"))
    summary = {
        "concept": concept, "n_personas": len(personas), "synthesizer": a.synthesizer,
        "benchmark_consider_share": round(float(personas.p_consider_benchmark.mean()), 3),
        "benchmark_by_age": seg.groupby("age_band").p_consider_benchmark.mean().round(3).to_dict(),
        "benchmark_by_lean": seg.groupby("party_lean").p_consider_benchmark.mean().round(3).to_dict(),
        "benchmark_fit_at_reference_ev": bench.fit_check,
        "quality": q, "crosstabs": ct.to_dict(),
        "age_mix": personas.age_band.value_counts(normalize=True).round(3).to_dict(),
    }
    json.dump(summary, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)

    fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
    lines = ["# EV personas from SDV: report", "",
             f"Concept: ${a.price:,}, {a.range} mi, {a.charge} min fast charge. Personas: {len(personas)} ({a.synthesizer}).", "",
             "## 1. Does the synthetic data look like real people it never saw?", "",
             "SDV quality report against a 20% hold-out of real respondents (0 to 1, higher is better).", "",
             "| Comparison | Overall | Column shapes | Column pair trends |", "|---|---|---|---|"]
    for k, v in q.items():
        lines.append(f"| {k} | {fmt(v['overall'])} | {fmt(v.get('Column Shapes', float('nan')))} | {fmt(v.get('Column Pair Trends', float('nan')))} |")
    lines += ["", "## 2. Cross-tabs that matter for EV demand", "", ct.to_markdown() if hasattr(ct, "to_markdown") else ct.to_string(), "",
              "## 3. EV benchmark for this concept (published figures)", "",
              f"Would seriously consider: {summary['benchmark_consider_share']:.0%} of synthetic personas.", "",
              f"By age: {summary['benchmark_by_age']}", "", f"By party lean: {summary['benchmark_by_lean']}", "",
              f"Fit check at the reference EV (published targets: Dem-leaning 48%, Rep-leaning 18%; overall implied by this sample's party mix: {bench.implied_total:.0%}): {bench.fit_check}", "",
              "## Files", "", "- synthetic_personas.csv: one row per synthetic person",
              "- prompts.jsonl: the step-2 prompt for each person, ready to send to an LLM",
              "- report.json: everything above in machine-readable form"]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
