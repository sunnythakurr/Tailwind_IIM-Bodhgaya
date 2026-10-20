"""
Digital Twin console (Streamlit), modern version.   Run from the project root:   streamlit run app/app.py
Pages: Care team | Patient monitor | What-if lab | Alert settings | About
Interactive charts use Plotly; if Plotly is missing or a chart fails, the app falls back to static charts.
All patients are synthetic.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin_ui as U
import twin_view as V

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

st.set_page_config(page_title="Digital Twin - Type 2 Diabetes", layout="wide", initial_sidebar_state="expanded")
st.markdown(V.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
def stretch(fn, *a, **k):
    """Full-width rendering that works across Streamlit versions."""
    try:
        return fn(*a, width="stretch", **k)
    except Exception:
        return fn(*a, use_container_width=True, **k)


def plot(spec, container=st, **kw):
    return stretch(container.plotly_chart, go.Figure(spec), config={"displaylogo": False}, **kw)


def rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


@st.cache_resource
def get_bundle():
    return U.Bundle()


B = get_bundle()
START_T, MIN_T = 400, 144
PAGES = ["Care team", "Patient monitor", "What-if lab", "Alert settings", "About"]
POLICIES = {"Global (same for everyone)": "global", "Personalised (first-week calibration)": "personal",
            "Blended": "blended"}
SENS = {"Strict": "fpr_2", "Balanced": "fpr_5", "Sensitive": "fpr_10"}

S = st.session_state
S.setdefault("t_idx", START_T)
S.setdefault("page", PAGES[0])
if "jump_to" in S:                                   # applied before the slider is created
    S["t_idx"] = int(min(max(S.pop("jump_to"), MIN_T), B.T - 1))


def reset_time():
    S["t_idx"] = START_T
    S.pop("last_click", None)


def open_patient(k):
    S["patient"] = k
    S["page"] = "Patient monitor"


def step(d):
    S["t_idx"] = int(min(max(S["t_idx"] + d, MIN_T), B.T - 1))


def jump(arr, lead):
    targets = np.maximum(arr - lead, MIN_T)
    later = targets[targets > S["t_idx"]]
    if len(later):
        S["t_idx"] = int(later[0])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Digital Twin Console")
key = st.sidebar.selectbox("Patient", B.keys(), format_func=B.label, key="patient", on_change=reset_time)
i = B.index(key)
meta = B.patients[i]
policy_label = st.sidebar.radio("Alert threshold", list(POLICIES), index=0)
policy = POLICIES[policy_label]
sens_label = st.sidebar.select_slider("Alert sensitivity", options=list(SENS), value="Balanced")
op = SENS[sens_label]
thr = meta["thresholds"][policy][op]
st.sidebar.caption(f"This patient's alert fires at {thr:.0%} risk. Stricter means fewer false alarms but more "
                   "missed episodes.")
t = st.sidebar.slider("Replay time", MIN_T, B.T - 1, key="t_idx")
st.sidebar.caption(f"Now showing: {U.time_label(meta, t)}")
reveal = st.sidebar.checkbox("Reveal what happened next (simulation ground truth)", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic patients (Synthea records + simulated wearables). Decision support only, "
                   "not a medical device.")

risk_series = B.a["risk"][i]
notifs = U.notifications(risk_series, thr)
episodes = np.nonzero(B.a["episode_start"][i])[0]

st.title("Digital Twin Console")
st.caption("Type 2 diabetes. A virtual patient built from a health record and wearable data, forecasting "
           "glucose above 180 mg/dL two hours ahead.")
page = st.radio("Navigate", PAGES, horizontal=True, key="page", label_visibility="collapsed")


def status_at(j, tt, pol=None, o=None):
    m = B.patients[j]
    th = m["thresholds"][pol or policy][o or op]
    cg, rk = float(B.a["cgm"][j][tt]), float(B.a["risk"][j][tt])
    return U.status_for(cg, rk, th), cg, rk, th


# ================================================================== Care team
if page == "Care team":
    rows = []
    for k in B.keys():
        j = B.index(k)
        stt, cg, rk, th = status_at(j, t)
        nt = U.notifications(B.a["risk"][j], th)
        recent = int(((nt <= t) & (nt > t - 288)).sum())
        rows.append((V.PRIORITY[stt], -np.nan_to_num(rk, nan=0.0), k, stt, cg, rk, recent))
    rows.sort()
    n_alert = sum(1 for r in rows if r[3] == "ALERT")
    st.markdown(
        '<div class="kpi-grid">'
        + V.kpi("Time", U.time_label(meta, t), "replay position")
        + V.kpi("Patients on alert now", f"{n_alert} / {len(rows)}", "at current settings")
        + V.kpi("Alerts, last 24 h", str(sum(r[6] for r in rows)), "whole cohort")
        + V.kpi("Threshold policy", policy.capitalize(), f"{sens_label} sensitivity") + "</div>",
        unsafe_allow_html=True)
    st.write("")
    cols = st.columns(4)
    for n, (_, _, k, stt, cg, rk, recent) in enumerate(rows):
        m = B.patients[B.index(k)]
        with cols[n % 4]:
            with st.container(border=True):
                st.markdown(V.pill(stt), unsafe_allow_html=True)
                st.markdown(f"**Patient {k}**  \n<span class='muted'>{m['archetype']}</span>", unsafe_allow_html=True)
                st.markdown(f"### {'n/a' if np.isnan(cg) else f'{cg:.0f}'} <span class='muted'>mg/dL</span>",
                            unsafe_allow_html=True)
                st.progress(float(min(max(np.nan_to_num(rk, nan=0.0), 0.0), 1.0)))
                st.caption(f"2-h risk {'n/a' if np.isnan(rk) else f'{rk:.0%}'} | {recent} alerts in last 24 h")
                st.button("Open", key=f"open_{k}", on_click=open_patient, args=(k,))
    st.caption("Sorted by urgency. Use the replay slider in the sidebar to move through the week.")

# ================================================================== Patient monitor
elif page == "Patient monitor":
    head1, head2 = st.columns([3, 2])
    head1.markdown(f"#### Patient {key} &nbsp; <span class='muted'>{meta['archetype']}</span>", unsafe_allow_html=True)
    ctl = head2.columns(5)
    ctl[0].button("-1 h", on_click=step, args=(-12,), key="b_m1h")
    ctl[1].button("-15 m", on_click=step, args=(-3,), key="b_m15")
    ctl[2].button("+15 m", on_click=step, args=(3,), key="b_p15")
    ctl[3].button("+1 h", on_click=step, args=(12,), key="b_p1h")
    play = ctl[4].button("Play", key="b_play", type="primary", disabled=not HAVE_PLOTLY)

    b1, b2, b3 = st.columns([1, 1, 1])
    b1.button("Next alert", on_click=jump, args=(notifs, 0), key="b_na")
    b2.button("Before next high episode", on_click=jump, args=(episodes, 18), key="b_ne")
    speed = b3.select_slider("Replay speed", options=["Slow", "Normal", "Fast"], value="Normal", label_visibility="collapsed")
    if not HAVE_PLOTLY:
        st.info("Install Plotly (pip install plotly) for interactive charts and live replay. Showing static charts.")

    ph_kpi, ph_banner = st.empty(), st.empty()
    left, right = st.columns([2.2, 1])
    ph_chart = left.empty()
    with right:
        st.markdown("**Why this forecast**")
        ph_explain = st.empty()
        st.markdown("**Key signals right now**")
        ph_signals = st.empty()

    def draw(tt):
        stt, cg, rk, th = status_at(i, tt)
        prev = float(B.a["cgm"][i][max(tt - 6, 0)])
        ph_kpi.markdown(V.kpi_row_html(cg, prev, rk, th, stt, U.time_label(meta, tt)), unsafe_allow_html=True)
        ph_banner.markdown(V.banner_html(stt, U.ADVICE[stt]), unsafe_allow_html=True)
        ph_signals.markdown(V.signals_html(B, i, tt), unsafe_allow_html=True)
        drawn = False
        if HAVE_PLOTLY:
            try:
                plot(V.telemetry_spec(B, key, tt, th, reveal=reveal), ph_chart)
                plot(V.explain_spec(B, key, tt), ph_explain)
                drawn = True
            except Exception as exc:
                st.warning(f"Interactive chart failed ({exc}); showing static charts instead.")
        if not drawn:
            f1 = U.fig_telemetry(B, key, tt, th, reveal_future=reveal)
            ph_chart.pyplot(f1); plt.close(f1)
            f2 = U.fig_explain(B, key, tt)
            ph_explain.pyplot(f2); plt.close(f2)

    draw(t)
    if play and HAVE_PLOTLY:
        frame, delay = {"Slow": (3, 0.55), "Normal": (3, 0.3), "Fast": (6, 0.2)}[speed]
        end, tt = min(B.T - 1, t + 288), t
        while tt < end:
            tt = min(tt + frame, end)
            draw(tt)
            time.sleep(delay)
        S["jump_to"] = tt
        rerun()

    st.markdown("**Seven-day overview** &nbsp; <span class='muted'>(click a point to jump there)</span>",
                unsafe_allow_html=True)
    if HAVE_PLOTLY:
        try:
            ev = st.plotly_chart(go.Figure(V.overview_spec(B, key, t, thr)), on_select="rerun",
                                 selection_mode="points", key=f"overview_{key}", config={"displaylogo": False},
                                 **{"width": "stretch"})
            pts = (ev.get("selection", {}) if hasattr(ev, "get") else {}).get("points", [])
            if pts and pts[0].get("curve_number", 0) == 0:
                idx = pts[0].get("point_index", pts[0].get("point_number"))
                if idx is not None and idx != S.get("last_click") and int(idx) != t:
                    S["last_click"] = int(idx)
                    S["jump_to"] = int(idx)
                    rerun()
        except TypeError:
            stretch(st.plotly_chart, go.Figure(V.overview_spec(B, key, t, thr)))
        except Exception:
            f3 = U.fig_overview(B, key, t, thr)
            st.pyplot(f3); plt.close(f3)
    else:
        f3 = U.fig_overview(B, key, t, thr)
        st.pyplot(f3); plt.close(f3)

    w = meta["window"]
    st.caption(f"Over these 7 days: {w['episodes']} high-glucose episodes, {w['time_in_range']:.0%} of time in "
               f"range, mean glucose {w['mean_cgm']:.0f} mg/dL; {len(notifs)} alerts sent at this setting.")
    with st.expander("Patient record"):
        st.table(pd.DataFrame(list(U.ehr_summary(meta["ehr"]).items()), columns=["Record", "Value"]).set_index("Record"))
    st.download_button("Download weekly summary", V.weekly_summary(B, key, thr, policy_label, sens_label),
                       file_name=f"patient_{key}_weekly_summary.txt")

# ================================================================== What-if lab
elif page == "What-if lab":
    st.subheader("What-if lab")
    st.write(f"Patient {key} at {U.time_label(meta, t)}. The twin re-runs its glucose model from this moment with a "
             "change you choose and compares it with the usual course. In this prototype the twin's parameters come "
             "from the simulator, which assumes the twin has been fitted perfectly to the patient.")
    scen = st.radio("Scenario", ["Add a meal", "Add a walk", "Skip the next meal"], horizontal=True)
    c1, c2 = st.columns(2)
    if scen == "Add a meal":
        grams = c1.slider("Carbohydrate (g)", 20, 150, 70, 5)
        off = c2.slider("Starts in (minutes)", 0, 180, 30, 5)
        args, label = ("meal", off, grams), f"{grams} g meal in {off} min"
    elif scen == "Add a walk":
        mins = c1.slider("Walk length (minutes)", 10, 60, 20, 5)
        off = c2.slider("Starts in (minutes)", 0, 120, 15, 5)
        args, label = ("walk", off, mins), f"{mins} min walk in {off} min"
    else:
        args, label = ("skip", 0, 0), "next meal skipped"
        if meta["ehr"].get("insulin_active", 0) == 1:
            st.warning("This patient takes insulin. Skipping a meal while the dose is unchanged raises the risk of low glucose.")
    if st.button("Run what-if", type="primary"):
        with st.spinner("Simulating..."):
            base, sc, now, note = U.run_whatif(B, key, t, *args)
        if note:
            st.info(note)
        if HAVE_PLOTLY:
            plot(V.whatif_spec(base, sc, now, label))
        else:
            f = U.fig_whatif(base, sc, now, label)
            st.pyplot(f); plt.close(f)
        s0, s1 = U.whatif_stats(base, now), U.whatif_stats(sc, now)
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak glucose, next 6 h", f"{s1['peak']:.0f} mg/dL", f"{s1['peak'] - s0['peak']:+.0f} vs usual", delta_color="inverse")
        m2.metric("Minutes above 180", f"{s1['minutes_above_180']}", f"{s1['minutes_above_180'] - s0['minutes_above_180']:+d} vs usual", delta_color="inverse")
        m3.metric("Minutes below 70", f"{s1['minutes_below_70']}", f"{s1['minutes_below_70'] - s0['minutes_below_70']:+d} vs usual", delta_color="inverse")
        st.caption("Simulated outcomes for decision support, not predictions for a real patient.")

# ================================================================== Alert settings
elif page == "Alert settings":
    st.subheader("Alert settings explorer")
    st.write("Every alert setting is a trade-off. A stricter setting sends fewer alerts and is right more often, but "
             "warns about fewer episodes. Measured on 150 held-out patients with continuous monitoring (days 8 to 30).")
    root = U.ROOT / "results"
    pers, glob = root / "personalised_event_level.csv", root / "event_level.csv"
    rows = None
    group = "All patients"
    if pers.exists():
        df = pd.read_csv(pers)
        group = st.radio("Patients", ["All patients", "On insulin", "Not on insulin"], horizontal=True)
        rows = df[df["group"] == group].to_dict("records")
    elif glob.exists():
        df = pd.read_csv(glob)
        df["policy"] = "global"
        rows = df.to_dict("records")
    if rows and HAVE_PLOTLY:
        cur = next((r for r in rows if r["policy"] == policy and r["operating_point"] == op), None)
        hl = (cur["alerts_per_patient_day"], 100 * cur["share_caught_with_30min_lead"]) if cur else None
        plot(V.tradeoff_spec(rows, hl))
        st.caption("The star is your current sidebar setting. Hover over a point for details.")
    elif rows:
        st.dataframe(pd.DataFrame(rows).round(3))
    else:
        st.info("Run src/final_eval.py and src/personalise.py to create the results files.")
    st.markdown("**How to read it.** With one global threshold, insulin users are almost never alerted (their "
                "glucose rises less often). Calibrating the threshold to each patient's first week restores their "
                "warnings, at the price of more false alerts. Pick the line that fits the clinical goal.")

# ================================================================== About
else:
    st.subheader("About this prototype")
    st.markdown(
        "**Idea.** A digital twin keeps a virtual copy of a patient, anchored to the electronic health record and "
        "updated by wearable data, and uses it to warn of adverse events before they happen.\n\n"
        "**Condition and event.** Type 2 diabetes. The model estimates the chance that glucose rises above "
        "180 mg/dL within 2 hours, for moments when glucose is currently in range.\n\n"
        "**Data.** No real patient data was used (DPDP Act and HIPAA safe). Records come from Synthea; the wearable "
        "streams (CGM, heart rate, HRV, steps, sleep) are simulated from each record with a glucose-insulin model "
        "and realistic sensor noise. Adverse events are read off the simulated glucose, never injected.\n\n"
        "**Model.** Gradient-boosted trees on sensor-history features plus the health record, evaluated on 150 "
        "held-out patients.")
    root = U.ROOT / "results"
    for title, fname in [("Model comparison (test patients)", "baseline_results.csv"),
                         ("Does the health record help? By data availability", "scenario_results.csv"),
                         ("Alerts and early warning at each sensitivity", "event_level.csv"),
                         ("Global vs personalised thresholds (days 8-30)", "personalised_event_level.csv"),
                         ("Performance by patient subgroup", "subgroups.csv")]:
        p = root / fname
        with st.expander(title):
            if p.exists():
                stretch(st.dataframe, pd.read_csv(p).round(3))
            else:
                st.caption(f"{fname} not found; copy the results folder next to the app to show it here.")
    st.markdown(
        "**Limitations.** Results describe the simulator, not real patients. Synthea's diabetic patients are mostly "
        "well controlled, unlike many Indian cohorts. The what-if twin uses simulator parameters instead of fitting "
        "them from CGM data. A single alert threshold under-alerts insulin users. This is decision support, not a "
        "medical device.")
