"""
twin_ui.py
Logic and charts for the digital-twin dashboard. No Streamlit imports here, so everything can be tested on its own.
Reads demo/demo_bundle.npz and demo/patients.json (made by src/make_demo_bundle.py).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BINS = 288
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
REFRACT = 24                       # 2-hour quiet period after a notification
C_CGM, C_ALERT, C_BAND, C_TRUTH = "#1f77b4", "#d62728", "#2ca02c", "#ff9896"


# ---------------------------------------------------------------- data
class Bundle:
    def __init__(self, folder=None):
        folder = Path(folder) if folder else ROOT / "demo"
        z = np.load(folder / "demo_bundle.npz", allow_pickle=True)
        self.a = {k: z[k] for k in z.files}
        meta = json.load(open(folder / "patients.json"))
        self.patients = meta["patients"]
        self.groups = meta["group_names"]
        self.T = self.a["cgm"].shape[1]
        self.warmup_days = meta["warmup_days"]

    def keys(self):
        return [p["key"] for p in self.patients]

    def index(self, key):
        return self.keys().index(key)

    def label(self, key):
        p = self.patients[self.index(key)]
        return f"{p['key']} - {p['archetype']}"


def time_label(meta, i):
    wd = (meta["window"]["weekday_of_first_day"] + i // BINS) % 7
    mins = (i % BINS) * 5
    return f"{DAYS[wd]} {mins // 60:02d}:{mins % 60:02d}"


def greedy_refractory(idx, gap=REFRACT):
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(int(i))
            last = i
    return np.array(out, dtype=int)


def notifications(risk, thr):
    return greedy_refractory(np.nonzero(np.nan_to_num(risk, nan=-1) >= thr)[0])


def status_for(cgm_now, risk, thr):
    if cgm_now is None or np.isnan(cgm_now):
        return "NO SENSOR DATA"
    if cgm_now > 180:
        return "ALREADY HIGH"
    if cgm_now < 70:
        return "LOW GLUCOSE"
    if np.isnan(risk):
        return "NO SENSOR DATA"
    return "ALERT" if risk >= thr else "OK"


ADVICE = {
    "ALERT": "Glucose is likely to rise above 180 mg/dL within about 2 hours. For clinician review: recent meal size "
             "and timing, a short walk after the next meal, and the timing of any medication.",
    "OK": "No high-glucose event is expected in the next 2 hours at the current sensitivity setting.",
    "ALREADY HIGH": "Glucose is above 180 mg/dL now. The 2-hour forecast only applies when glucose is in range.",
    "LOW GLUCOSE": "Glucose is below 70 mg/dL. Follow the low-glucose protocol; the forecast does not apply.",
    "NO SENSOR DATA": "The glucose sensor has no recent reading, so no forecast can be made.",
}


def ehr_summary(ehr):
    g = lambda k, d=np.nan: ehr.get(k, d)
    meds = [n for n, k in [("Metformin", "metformin_active"), ("Insulin (70/30)", "insulin_active")] if g(k, 0) == 1]
    conds = [n for n, k in [("Hypertension", "hypertension"), ("Kidney disease", "ckd"), ("Retinopathy", "retinopathy"),
                            ("Neuropathy", "neuropathy"), ("Obesity", "obesity"), ("Cardiovascular disease", "cvd")]
             if g(k, 0) == 1]
    rows = {
        "Age / sex": f"{g('age'):.0f} / {'Male' if g('male', 0) == 1 else 'Female'}",
        "BMI": f"{g('bmi'):.1f}",
        "HbA1c": f"{g('hba1c'):.1f} %",
        "Years since T2D diagnosis": f"{g('years_since_dx'):.0f}",
        "Blood pressure": f"{g('sbp'):.0f}/{g('dbp'):.0f} mmHg" if not np.isnan(g("sbp")) else "n/a",
        "Total cholesterol": f"{g('chol_total'):.0f} mg/dL" if not np.isnan(g("chol_total")) else "n/a",
        "Medications": ", ".join(meds) or "None recorded",
        "Comorbidities": ", ".join(conds) or "None recorded",
        "Genetic risk score (synthetic)": f"{g('prs', 0):+.2f} SD",
    }
    return rows


# ---------------------------------------------------------------- charts
def _xticks(ax, meta, lo, hi):
    step = 12 * 3 if hi - lo > 150 else 12
    ticks = np.arange((lo // step) * step, hi + 1, step)
    ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([time_label(meta, int(t)) for t in ticks], fontsize=7)


def fig_telemetry(B, k, t, thr, hours_back=12, reveal_future=False):
    p, meta = B.a, B.patients[B.index(k)]
    i = B.index(k)
    lo, hi = max(0, t - hours_back * 12), min(B.T - 1, t + (24 if reveal_future else 0))
    x = np.arange(lo, hi + 1)
    fig, axs = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.1, 1.1, 0.8]))
    ax = axs[0]
    ax.axhspan(70, 180, color=C_BAND, alpha=0.10)
    ax.axhline(180, color="grey", lw=0.6, ls="--"); ax.axhline(70, color="grey", lw=0.6, ls="--")
    cg = p["cgm"][i]
    seen = x <= t
    ax.plot(x[seen], cg[lo:t + 1], color=C_CGM, lw=1.6)
    if reveal_future and hi > t:
        ax.plot(x[~seen], cg[t + 1:hi + 1], color=C_CGM, lw=1.2, ls=":", alpha=0.7, label="what happened next")
        ep = np.nonzero(p["episode_start"][i][t:hi + 1])[0] + t
        ax.scatter(ep, np.full(len(ep), 185), marker="v", color=C_TRUTH, zorder=5, s=40, label="actual high episode")
        ax.legend(loc="upper left", fontsize=7)
    past = notifications(p["risk"][i][:t + 1], thr)
    past = past[past >= lo]
    ax.scatter(past, np.where(np.isnan(cg[past]), 150, cg[past]), marker="^", color=C_ALERT, s=60, zorder=6)
    ax.axvline(t, color="black", lw=0.8)
    ax.set_ylabel("CGM glucose (mg/dL)"); ax.set_ylim(40, max(260, np.nanmax(cg[lo:hi + 1]) + 15))
    ax.set_title("Live CGM (red triangles = alerts sent)", fontsize=10, loc="left")
    r = p["risk"][i]
    axs[1].plot(x[seen], r[lo:t + 1], color=C_ALERT, lw=1.2)
    axs[1].axhline(thr, color="black", lw=0.7, ls="--"); axs[1].set_ylim(0, 1); axs[1].set_ylabel("2-h risk")
    axs[1].set_title("Risk of glucose >180 mg/dL within 2 hours (dashed = alert threshold)", fontsize=9, loc="left")
    axs[2].plot(x[seen], p["hr"][i][lo:t + 1], color="#9467bd", lw=1, label="HR (bpm)")
    ax2 = axs[2].twinx()
    ax2.plot(x[seen], p["hrv"][i][lo:t + 1], color="#8c564b", lw=1, label="HRV (ms)")
    axs[2].set_ylabel("HR", fontsize=8); ax2.set_ylabel("HRV", fontsize=8)
    axs[2].set_title("Heart rate (purple) and HRV (brown)", fontsize=9, loc="left")
    axs[3].bar(x[seen], p["steps"][i][lo:t + 1], width=1, color="#17becf")
    stg = p["stage"][i][lo:t + 1]
    axs[3].fill_between(x[seen], 0, 1, where=(stg > 0), color="#7f7f7f", alpha=0.25, transform=axs[3].get_xaxis_transform())
    axs[3].set_ylabel("steps"); axs[3].set_title("Steps per 5 min (grey = asleep)", fontsize=9, loc="left")
    for a in axs:
        a.axvline(t, color="black", lw=0.8); a.spines[["top", "right"]].set_visible(False)
    _xticks(axs[3], meta, lo, hi)
    axs[3].set_xlim(lo, hi)
    fig.tight_layout()
    return fig


def fig_explain(B, k, t):
    i = B.index(k)
    c = B.a["contrib"][i][t]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if np.all(np.isnan(c)):
        ax.text(0.5, 0.5, "No forecast at this moment", ha="center", va="center"); ax.axis("off")
        return fig
    order = np.argsort(c)
    cols = [C_ALERT if v > 0 else C_CGM for v in c[order]]
    ax.barh(np.array(B.groups)[order], c[order], color=cols)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("effect on risk (probability points)\nred = pushes risk up, blue = pulls it down", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("What is driving this forecast", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_overview(B, k, t, thr):
    i, meta = B.index(k), B.patients[B.index(k)]
    r = B.a["risk"][i]
    x = np.arange(B.T)
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.plot(x, r, color=C_ALERT, lw=0.6)
    ax.axhline(thr, color="black", lw=0.7, ls="--")
    ep = np.nonzero(B.a["episode_start"][i])[0]
    ax.scatter(ep, np.full(len(ep), 0.97), marker="v", color=C_TRUTH, s=18, label="actual high episode")
    nt = notifications(r, thr)
    ax.scatter(nt, np.full(len(nt), 0.05), marker="^", color=C_ALERT, s=22, label="alert sent")
    ax.axvline(t, color="black", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("risk"); ax.legend(loc="upper right", fontsize=7, ncol=2)
    _xticks(ax, meta, 0, B.T - 1); ax.set_xlim(0, B.T - 1)
    ax.set_title("Seven-day overview", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- what-if simulator
def _simulate(B, k, t, carbs, steps):
    sys.path.insert(0, str(ROOT / "src" / "datagen"))
    import glucose_sim as GS
    i, meta = B.index(k), B.patients[B.index(k)]
    a = B.a
    s_t = t + B.warmup_days * BINS
    cur = s_t // BINS
    sd, ed = max(0, cur - 2), min(a["sim_carbs"].shape[1] // BINS, cur + 2)
    sl = slice(sd * BINS, ed * BINS)
    prm = pd.DataFrame([meta["sim_params"]])
    _, gi = GS.simulate(prm, carbs[None, sl].astype(np.float32), steps[None, sl].astype(np.float32),
                        a["sim_si_mult"][i][None, sd:ed], a["sim_stress_mult"][i][None, sd:ed],
                        a["sim_inj"][i][None, sl], np.array([a["sim_gb"][i]]), (ed - sd) * BINS,
                        record_from=(cur - sd) * BINS)
    return gi[0], s_t - cur * BINS            # series from the start of the current day, and "now" within it


def run_whatif(B, k, t, kind, offset_min, amount):
    """kind: 'meal' (amount = grams of carbs), 'walk' (amount = minutes), 'skip' (next meal skipped)."""
    i = B.index(k)
    a = B.a
    carbs0, steps0 = a["sim_carbs"][i].copy(), a["sim_steps"][i].astype(np.float32).copy()
    s_t = t + B.warmup_days * BINS
    carbs1, steps1 = carbs0.copy(), steps0.copy()
    at = min(s_t + offset_min // 5, len(carbs0) - 1)
    note = ""
    if kind == "meal":
        carbs1[at] += amount
    elif kind == "walk":
        steps1[at:at + max(1, amount // 5)] += 480
    elif kind == "skip":
        nxt = np.nonzero(carbs1[s_t:s_t + 72] > 0)[0]
        if len(nxt):
            note = f"Skipped the {carbs1[s_t + nxt[0]]:.0f} g meal at {nxt[0] * 5} minutes from now."
            carbs1[s_t + nxt[0]] = 0
        else:
            note = "No meal in the next 6 hours to skip."
    base, now = _simulate(B, k, t, carbs0, steps0)
    scen, _ = _simulate(B, k, t, carbs1, steps1)
    return base, scen, now, note


def whatif_stats(series, now, horizon=72):
    s = series[now:now + horizon]
    return dict(peak=float(np.max(s)), low=float(np.min(s)), minutes_above_180=int((s > 180).sum() * 5),
                minutes_below_70=int((s < 70).sum() * 5))


def fig_whatif(base, scen, now, kind_label):
    lo, hi = max(0, now - 24), min(len(base), now + 72)
    x = (np.arange(lo, hi) - now) * 5 / 60
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axhspan(70, 180, color=C_BAND, alpha=0.10)
    ax.plot(x, base[lo:hi], color="grey", lw=1.8, label="usual course (twin)")
    ax.plot(x, scen[lo:hi], color=C_ALERT, lw=1.8, label=kind_label)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(180, color="grey", lw=0.6, ls="--"); ax.axhline(70, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("hours from now"); ax.set_ylabel("twin-predicted CGM glucose (mg/dL)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
