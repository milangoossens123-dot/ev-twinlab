"""
EV TwinLab: calibrate the reliability-score weights on real human vs. AI-twin data.

Data (public):
  - Twin-2K-500 Mega-Study, results.zip (Apache-2.0)
    https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study
    19 US studies; for each, ~23 AI set-ups answered as "digital twins" of the
    same real participants. Per participant and question: human answer + twin answer.
  - Twin-2K-500, question_catalog_and_human_response_csv/wave1_3_response_label.csv (CC-BY-4.0)
    https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500
    Demographics per participant (age band, party, income, sex, region).

Design (mirrors how EV TwinLab will be used):
  One "validation exercise" = one study x one AI set-up.
  - Participants are split 50/50 (fixed hash of their ID): validation people / hold-out people.
  - The study's questions are split in two (alternating): validated questions / new questions.
    (Studies with one question reuse it; only the people differ.)
  - On validation people x validated questions we compute the demo's three tests:
      Accuracy  A = max(0, 100 - 5 * mean |twin - human| across segments)   [points of scale]
      Stability S = max(0, 100 - 12 * sd of the panel mean across sibling runs)
                   sibling run = the same model and persona with only sampling changed
                   (temperature 0 vs 0.7, or JSON-forced output). No true reruns exist in the data.
      Diversity D = 100 * min(r, 1/r), r = sd(segment means, twins) / sd(segment means, humans)
  - Ground truth: on hold-out people x new questions, the true segment error E
      (mean |twin - human| across segments, in points of scale), and quality Q = max(0, 100 - 5E).
  - Weights: the non-negative weights (summing to 1) for which T = wA*A + wS*S + wD*D best
    predicts Q (R^2 of a linear fit), searched on a 1% grid. Uncertainty: bootstrap over studies.
    Out-of-sample check: leave-one-study-out cross-validation.
  - Gate cut-offs: HIGH where >= 80% of exercises above the cut-off have E <= 5 points;
    LOW where >= 50% of exercises below the cut-off have E > 10 points.

Usage:
  python calibrate_weights.py RESULTS_ZIP DEMOGRAPHICS_CSV OUT_JSON
"""
import io, re, sys, json, zipfile, hashlib
import numpy as np
import pandas as pd

RESULTS_ZIP, DEMO_CSV, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
MIN_SEG_N = 10
rng = np.random.default_rng(42)

# Sibling runs: same model + same persona input, only sampling/format changed.
SIBLING = {
    "empty_persona": "empty_persona_temperature_zero",
    "demographics_only": "demographics_only_temperature_zero",
    "persona_summary": "persona_summary_temperature_zero",
    "persona_summary_include_reasoning": "persona_summary_include_reasoning_temperature_zero",
    "full_persona_without_reasoning": "temperature_zero",
    "with_reasoning": "include_reasoning_temperature_zero",
    "full_persona_fine_tuned_temperature_7": "full_persona_fine_tuned_temperature_0",
    "full_persona_fine_tuned_temperature_7_jsonforced": "full_persona_fine_tuned_temperature_7",
}
for a, b in list(SIBLING.items()):
    SIBLING.setdefault(b, a)

# ---------------------------------------------------------------- demographics
demo = pd.read_csv(DEMO_CSV, usecols=["pid", "QID11", "QID12", "QID13", "QID20", "QID21"], dtype=str)
demo.columns = ["pid", "region", "sex", "age", "party", "income"]
demo["pid"] = demo["pid"].astype(int)
SEGMENTS = {
    "Age 18-29": lambda d: d.age == "18-29",
    "Age 30-49": lambda d: d.age == "30-49",
    "Age 50-64": lambda d: d.age == "50-64",
    "Age 65+": lambda d: d.age == "65+",
    "Democrat": lambda d: d.party == "Democrat",
    "Republican": lambda d: d.party == "Republican",
    "Independent": lambda d: d.party == "Independent",
    "Income < $50k": lambda d: d.income.isin(["Less than $30,000", "$30,000-$50,000"]),
    "Income $100k+": lambda d: d.income == "$100,000 or more",
    "Male": lambda d: d.sex == "Male",
    "Female": lambda d: d.sex == "Female",
    "Northeast": lambda d: d.region.str.startswith("Northeast", na=False),
    "Midwest": lambda d: d.region.str.startswith("Midwest", na=False),
    "South": lambda d: d.region.str.startswith("South", na=False),
    "West": lambda d: d.region.str.startswith("West", na=False),
}
for name, fn in SEGMENTS.items():
    demo[name] = fn(demo).fillna(False).values
SEG_NAMES = list(SEGMENTS)
holdout_person = {p: int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 2 == 1 for p in demo.pid}

# ---------------------------------------------------------------- load results
z = zipfile.ZipFile(RESULTS_ZIP)
files = {}
for n in z.namelist():
    m = re.match(r"results/([^/]+)/(.+?)_(\d{4}-\d{2}-\d{2})/meta analysis individual level\.csv$", n)
    if m:
        study, spec, date = m.groups()
        if (study, spec) not in files or date > files[(study, spec)][0]:
            files[(study, spec)] = (date, n)

def load(study, spec):
    df = pd.read_csv(io.BytesIO(z.read(files[(study, spec)][1])))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    w = df.pivot_table(index=["TWIN_ID", "variable_name"], columns="respondent_type",
                       values="value", aggfunc="mean").reset_index()
    w = w.dropna(subset=["human", "twin"])
    w["TWIN_ID"] = w["TWIN_ID"].astype(int)
    return w

def spearman(x, y):
    return float(pd.Series(list(x)).rank().corr(pd.Series(list(y)).rank()))

cache = {}
def get(study, spec):
    if (study, spec) not in cache:
        cache[(study, spec)] = load(study, spec)
    return cache[(study, spec)]

# ---------------------------------------------------------------- metrics
def scale_points(w, ranges):
    w = w[w.variable_name.isin(list(ranges))].copy()
    lo = w.variable_name.map(lambda v: ranges[v][0]); hi = w.variable_name.map(lambda v: ranges[v][1])
    span = (hi - lo).replace(0, np.nan)
    w["h"] = 100 * (w.human - lo) / span
    w["t"] = 100 * (w.twin - lo) / span
    return w.dropna(subset=["h", "t"])

def seg_table(d):
    rows = []
    for s in SEG_NAMES:
        g = d[d[s]]
        if len(g) >= MIN_SEG_N:
            rows.append((g.h.mean(), g.t.mean()))
    return np.array(rows)

def acc_div(d):
    st = seg_table(d)
    if len(st) < 4:
        return None
    mae = np.mean(np.abs(st[:, 1] - st[:, 0]))
    sh, stw = st[:, 0].std(), st[:, 1].std()
    r = stw / sh if sh > 0 else 0
    D = 100 * min(r, 1 / r) if r > 0 else 0.0
    return mae, D

units = []
studies = sorted({s for s, _ in files})
for study in studies:
    specs = sorted(sp for s, sp in files if s == study)
    base = get(study, specs[0])
    variables = sorted(base.variable_name.unique())
    ranges = {}
    for v in variables:
        hv = base.loc[base.variable_name == v, "human"]
        ranges[v] = (hv.min(), hv.max())
    if len(variables) >= 2:
        v_val, v_new = variables[0::2], variables[1::2]
    else:
        v_val = v_new = variables
    for spec in specs:
        w = scale_points(get(study, spec).merge(demo, left_on="TWIN_ID", right_on="pid"), ranges)
        w["hold"] = w.TWIN_ID.map(holdout_person)
        val = w[~w.hold & w.variable_name.isin(v_val)]
        new = w[w.hold & w.variable_name.isin(v_new)]
        A_list, D_list, E_list, Etot_list, S_list, R_list = [], [], [], [], [], []
        means_h, means_t = [], []
        sib = SIBLING.get(spec)
        wsib = None
        if sib and (study, sib) in files:
            wsib = scale_points(get(study, sib).merge(demo, left_on="TWIN_ID", right_on="pid"), ranges)
            wsib = wsib[~wsib.TWIN_ID.map(holdout_person)]
        for v in v_val:
            d = val[val.variable_name == v]
            r = acc_div(d)
            if r is None:
                continue
            A_list.append(max(0.0, 100 - 5 * r[0])); D_list.append(r[1])
            if wsib is not None:
                ds = wsib[wsib.variable_name == v]
                if len(ds) >= MIN_SEG_N:
                    sdr = abs(d.t.mean() - ds.t.mean()) / 2
                    S_list.append(max(0.0, 100 - 12 * sdr))
        for v in v_new:
            d = new[new.variable_name == v]
            st = seg_table(d)
            if len(st) >= 4:
                E_list.append(np.mean(np.abs(st[:, 1] - st[:, 0])))
                Etot_list.append(abs(d.t.mean() - d.h.mean()))
                if st[:, 0].std() > 0 and st[:, 1].std() > 0:
                    R_list.append(spearman(st[:, 1], st[:, 0]))
                means_h.append(d.h.mean()); means_t.append(d.t.mean())
        if A_list and E_list:
            units.append(dict(study=study, spec=spec, A=np.mean(A_list), D=np.mean(D_list),
                              S=np.mean(S_list) if S_list else np.nan,
                              E=np.mean(E_list), E_total=np.mean(Etot_list),
                              seg_rank=np.mean(R_list) if R_list else np.nan,
                              concept_rank=spearman(means_t, means_h) if len(means_h) >= 4 else np.nan,
                              n_people=int(w.TWIN_ID.nunique())))
    print(f"{study}: {len(specs)} set-ups", file=sys.stderr)

U = pd.DataFrame(units)
U["Q"] = (100 - 5 * U.E).clip(lower=0)
U3 = U.dropna(subset=["S"]).reset_index(drop=True)

# ---------------------------------------------------------------- weight search
GRID = np.array([(a, s, 100 - a - s) for a in range(101) for s in range(101 - a)]) / 100.0

def r2_all(M, q):
    T = M @ GRID.T                                   # units x grid
    Tc = T - T.mean(0); qc = q - q.mean()
    cov = (Tc * qc[:, None]).sum(0)
    return cov ** 2 / ((Tc ** 2).sum(0) * (qc ** 2).sum() + 1e-12)

def best_weights(df, y="Q"):
    r2 = r2_all(df[["A", "S", "D"]].values, df[y].values)
    i = int(np.argmax(r2))
    return GRID[i], float(r2[i])


w_fit, r2_fit = best_weights(U3)
r2_grid = r2_all(U3[["A", "S", "D"]].values, U3.Q.values)
def r2_of(w):
    return float(r2_grid[np.argmin(np.abs(GRID - np.array(w)).sum(1))])

# bootstrap over studies
boot = []
st_list = U3.study.unique()
for _ in range(500):
    pick = rng.choice(st_list, len(st_list), replace=True)
    df = pd.concat([U3[U3.study == s] for s in pick])
    boot.append(best_weights(df)[0])
boot = np.array(boot)

# leave-one-study-out: fitted weights vs. proposed 50/30/20 vs. accuracy only
def loso(weight_fn, df=None, y="Q"):
    # Leave one study out: fit weights on the other studies, then check how well the score
    # ranks the AI set-ups *within* the held-out study (the real use: pick the best set-up).
    df = U3 if df is None else df
    rhos = []
    for s in df.study.unique():
        tr, te = df[df.study != s], df[df.study == s]
        if len(te) < 5 or te[y].std() == 0:
            continue
        w = weight_fn(tr)
        rhos.append(spearman(te[["A", "S", "D"]].values @ w, te[y].values))
    return dict(mean_within_study_spearman=float(np.mean(rhos)), n_studies=len(rhos),
                share_studies_positive=float(np.mean(np.array(rhos) > 0)))
cv = {
    "fitted": loso(lambda tr: best_weights(tr)[0]),
    "proposed_50_30_20": loso(lambda tr: np.array([.5, .3, .2])),
    "accuracy_only": loso(lambda tr: np.array([1.0, 0, 0])),
    "equal_thirds": loso(lambda tr: np.array([1/3, 1/3, 1/3])),
}

# single-metric correlations
single = {m: spearman(U3[m], U3.Q) for m in ["A", "S", "D"]}
single_all = {m: spearman(U[m], U.Q) for m in ["A", "D"]}

# ---------------------------------------------------------------- other jobs: ranking segments, ranking concepts
U3["Q_seg"] = 100 * U3.seg_rank.clip(lower=0)
U3["Q_concept"] = 100 * U3.concept_rank.clip(lower=0)
jobs = {}
for job, y in [("level_estimates", "Q"), ("segment_ranking", "Q_seg"), ("concept_ranking", "Q_concept")]:
    df = U3.dropna(subset=[y]).reset_index(drop=True)
    w, r2 = best_weights(df, y)
    sts = df.study.unique()
    bs = np.array([best_weights(pd.concat([df[df.study == s_] for s_ in rng.choice(sts, len(sts))]), y)[0] for _ in range(300)])
    def loso_y(fn):
        return loso(fn, df, y)
    jobs[job] = dict(n=int(len(df)), n_studies=int(len(sts)),
        weights=dict(accuracy=float(w[0]), stability=float(w[1]), diversity=float(w[2])),
        weights_ci90={k: [float(np.percentile(bs[:, i], 5)), float(np.percentile(bs[:, i], 95))] for i, k in enumerate(["accuracy", "stability", "diversity"])},
        share_bootstrap_accuracy_top=float((bs.argmax(1) == 0).mean()),
        single_metric_spearman={m: spearman(df[m], df[y]) for m in ["A", "S", "D"]},
        cv_spearman=dict(fitted=loso_y(lambda tr: best_weights(tr, y)[0]), proposed_50_30_20=loso_y(lambda tr: np.array([.5, .3, .2])),
                         accuracy_only=loso_y(lambda tr: np.array([1.0, 0, 0])), stability_only=loso_y(lambda tr: np.array([0, 1.0, 0])), diversity_only=loso_y(lambda tr: np.array([0, 0, 1.0]))))

# ---------------------------------------------------------------- gate cut-offs
U3["T"] = U3[["A", "S", "D"]].values @ w_fit
def cutoffs(df, col="T"):
    ts = np.sort(df[col].unique())
    high = None
    for t in ts:
        sub = df[df[col] >= t]
        if len(sub) >= 15 and (sub.E <= 5).mean() >= 0.8:
            high = float(t); break
    low = None
    for t in ts[::-1]:
        sub = df[df[col] < t]
        if len(sub) >= 15 and (sub.E > 10).mean() >= 0.5:
            low = float(t); break
    return high, low
t_high, t_low = cutoffs(U3)
def band(t):
    if t_high is not None and t >= t_high: return "HIGH"
    if t_low is not None and t < t_low: return "LOW"
    return "MEDIUM"
U3["gate"] = U3["T"].map(band)
gate_summary = {g: dict(n=int((U3.gate == g).sum()),
                        share_error_le_5=float((U3[U3.gate == g].E <= 5).mean()) if (U3.gate == g).any() else None,
                        median_error=float(U3[U3.gate == g].E.median()) if (U3.gate == g).any() else None)
                for g in ["HIGH", "MEDIUM", "LOW"]}
# how the proposed weights + 80/60 cut-offs perform on the same data
U3["T_prop"] = U3[["A", "S", "D"]].values @ np.array([.5, .3, .2])
prop_gate = {}
for g, cond in [("HIGH", U3.T_prop >= 80), ("MEDIUM", (U3.T_prop >= 60) & (U3.T_prop < 80)), ("LOW", U3.T_prop < 60)]:
    sub = U3[cond]
    prop_gate[g] = dict(n=int(len(sub)), share_error_le_5=float((sub.E <= 5).mean()) if len(sub) else None,
                        median_error=float(sub.E.median()) if len(sub) else None)

bins = [0, 40, 50, 60, 70, 80, 101]
calib = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = U3[(U3["T"] >= lo) & (U3["T"] < hi)]
    if len(sub):
        calib.append(dict(score=f"{lo}-{min(hi,100)}", n=int(len(sub)), share_error_le_5=round(float((sub.E <= 5).mean()), 2),
                          share_error_le_10=round(float((sub.E <= 10).mean()), 2), median_error=round(float(sub.E.median()), 1)))
out = dict(calibration_table=calib,
    n_exercises_all=int(len(U)), n_exercises_with_stability=int(len(U3)),
    n_studies=int(U3.study.nunique()), n_setups=int(U3.spec.nunique()),
    weights=dict(accuracy=float(w_fit[0]), stability=float(w_fit[1]), diversity=float(w_fit[2])),
    weights_ci90={k: [float(np.percentile(boot[:, i], 5)), float(np.percentile(boot[:, i], 95))]
                  for i, k in enumerate(["accuracy", "stability", "diversity"])},
    r2_fitted=r2_fit, r2_proposed=r2_of([.5, .3, .2]), r2_accuracy_only=r2_of([1, 0, 0]),
    cross_validation=cv, single_metric_spearman=single, single_metric_spearman_all_setups=single_all,
    cutoffs=dict(high=t_high, low=t_low), gate_summary=gate_summary, proposed_gate_summary=prop_gate,
    metric_means=U3[["A", "S", "D", "E", "Q"]].mean().round(2).to_dict(),
    jobs=jobs, share_bootstrap_accuracy_top=float((boot.argmax(1) == 0).mean()),
    by_setup=U3.groupby("spec")[["A", "S", "D", "T", "E"]].mean().round(1).reset_index().to_dict("records"),
    units=U3.round(3).to_dict("records"),
)
json.dump(out, open(OUT_JSON, "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["n_exercises_all", "n_exercises_with_stability", "n_studies", "n_setups",
      "weights", "weights_ci90", "r2_fitted", "r2_proposed", "r2_accuracy_only", "cross_validation",
      "single_metric_spearman", "single_metric_spearman_all_setups", "cutoffs", "gate_summary",
      "proposed_gate_summary", "metric_means", "jobs", "share_bootstrap_accuracy_top", "calibration_table", "by_setup"]}, indent=1))
