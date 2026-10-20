"""
twin_view.py
Presentation helpers for the modern dashboard: interactive Plotly chart specifications (plain dicts, so they
can be checked without Plotly installed), HTML snippets (status pills, KPI cards) and the weekly summary text.
"""
import numpy as np
import twin_ui as U

BINS = 288
BASE = np.datetime64("2026-10-05T00:00")          # a Monday; only used to give the time axis real weekday labels
C_CGM, C_RISK, C_HR, C_HRV, C_STEPS = "#2563eb", "#dc2626", "#7c3aed", "#b45309", "#0891b2"
C_BAND, C_TRUTH, C_GRID = "rgba(34,197,94,0.10)", "#f87171", "rgba(128,128,128,0.18)"
TRANSPARENT = "rgba(0,0,0,0)"

STATUS_CLASS = {"ALERT": "alert", "LOW GLUCOSE": "low", "ALREADY HIGH": "high", "OK": "ok", "NO SENSOR DATA": "none"}
PRIORITY = {"ALERT": 0, "LOW GLUCOSE": 1, "ALREADY HIGH": 2, "OK": 3, "NO SENSOR DATA": 4}

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
.kpi-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 8px;}
@media (max-width: 900px) {.kpi-grid {grid-template-columns: repeat(2, 1fr);}}
.kpi {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.20); border-radius: 14px; padding: 14px 18px;}
.kpi .label {font-size: .74rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}
.kpi .value {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.kpi .sub {font-size: .8rem; opacity: .7;}
.pill {display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: .78rem; letter-spacing: .03em;}
.pill-ok {background: #dcfce7; color: #166534;}
.pill-alert {background: #fee2e2; color: #991b1b;}
.pill-high {background: #fef3c7; color: #92400e;}
.pill-low {background: #fee2e2; color: #991b1b;}
.pill-none {background: #e5e7eb; color: #374151;}
.banner {border-radius: 14px; padding: 12px 18px; margin: 4px 0 10px 0; border-left: 6px solid;}
.banner-ok {background: #f0fdf4; border-color: #16a34a; color: #14532d;}
.banner-alert, .banner-low {background: #fef2f2; border-color: #dc2626; color: #7f1d1d;}
.banner-high {background: #fffbeb; border-color: #d97706; color: #78350f;}
.banner-none {background: #f3f4f6; border-color: #6b7280; color: #1f2937;}
.muted {opacity: .65; font-size: .85rem;}
table.signals {width: 100%; border-collapse: collapse; font-size: .9rem;}
table.signals td {padding: 5px 4px; border-bottom: 1px solid rgba(128,128,128,0.18);}
table.signals td:last-child {text-align: right; font-weight: 600;}
</style>
"""


def pill(status):
    return f'<span class="pill pill-{STATUS_CLASS[status]}">{status}</span>'


def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def kpi_row_html(cgm, prev, risk, thr, status, when):
    g = "n/a" if np.isnan(cgm) else f"{cgm:.0f}"
    d = "" if (np.isnan(cgm) or np.isnan(prev)) else f"{cgm - prev:+.0f} mg/dL vs 30 min ago"
    r = "n/a" if np.isnan(risk) else f"{risk:.0%}"
    return ('<div class="kpi-grid">' + kpi("Glucose (CGM)", g + (" mg/dL" if g != "n/a" else ""), d)
            + kpi("Risk of >180 in 2 h", r, "forecast, in-range moments only")
            + kpi("Alert threshold", f"{thr:.0%}", "alert fires at or above this")
            + kpi("Status", pill(status), when) + "</div>")


def banner_html(status, text):
    return f'<div class="banner banner-{STATUS_CLASS[status]}"><b>{status}</b> &nbsp; {text}</div>'


def signals_html(B, i, t):
    names = list(B.a["feature_names"])
    f = B.a["feats"][i][t]
    def val(n, fmt, scale=1.0):
        if n not in names or np.isnan(f[names.index(n)]):
            return "n/a"
        return fmt.format(f[names.index(n)] * scale)
    rows = [("Glucose trend (last 30 min)", val("cgm_slope_30m", "{:+.1f} mg/dL per 10 min", 10)),
            ("Highest glucose, last 2 h", val("cgm_max_2h", "{:.0f} mg/dL")),
            ("Time above 180, last 24 h", val("cgm_tar_24h", "{:.0%}")),
            ("Heart rate (1 h mean)", val("hr_mean_1h", "{:.0f} bpm")),
            ("HRV (1 h mean)", val("hrv_mean_1h", "{:.0f} ms")),
            ("Steps, last hour", val("steps_1h", "{:.0f}")),
            ("Last night's sleep", val("sleep_tst_h", "{:.1f} h"))]
    return "<table class='signals'>" + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows) + "</table>"


# ---------------------------------------------------------------- chart helpers
def iso_times(meta, lo, hi):
    wd = meta["window"]["weekday_of_first_day"]
    start = BASE + np.timedelta64(wd * BINS * 5, "m")
    idx = np.arange(lo, hi + 1)
    return (start + (idx * 5).astype("timedelta64[m]")).astype("datetime64[m]").astype(str).tolist()


def clean(a):
    return [None if (v is None or np.isnan(v)) else round(float(v), 4) for v in np.asarray(a, dtype=float)]


def _domains(heights, gap=0.035):
    scale = (1 - gap * (len(heights) - 1)) / sum(heights)
    top, out = 1.0, []
    for h in heights:
        out.append([round(top - h * scale, 4), round(top, 4)])
        top = top - h * scale - gap
    return out


def _runs(mask):
    runs, start = [], None
    for j, m in enumerate(mask):
        if m and start is None:
            start = j
        if not m and start is not None:
            runs.append((start, j - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _axis(**kw):
    base = dict(gridcolor=C_GRID, zeroline=False)
    base.update(kw)
    return base


def telemetry_spec(B, key, t, thr, hours_back=12, reveal=False):
    i = B.index(key)
    meta, a = B.patients[i], B.a
    lo, hi = max(0, t - hours_back * 12), min(B.T - 1, t + (24 if reveal else 0))
    x = iso_times(meta, lo, hi)
    n_seen = t - lo + 1
    xs = x[:n_seen]
    cg, risk = a["cgm"][i], a["risk"][i]
    data = [
        dict(type="scatter", mode="lines", x=xs, y=clean(cg[lo:t + 1]), name="CGM glucose", yaxis="y",
             line=dict(color=C_CGM, width=2.6), hovertemplate="%{y:.0f} mg/dL<extra>Glucose</extra>"),
    ]
    if reveal and hi > t:
        data.append(dict(type="scatter", mode="lines", x=x[n_seen - 1:], y=clean(cg[t:hi + 1]), yaxis="y",
                         name="what happened next", line=dict(color=C_CGM, width=1.6, dash="dot"),
                         hovertemplate="%{y:.0f} mg/dL<extra>Next</extra>"))
        ep = np.nonzero(a["episode_start"][i][t:hi + 1])[0] + t
        if len(ep):
            data.append(dict(type="scatter", mode="markers", x=[x[j - lo] for j in ep], y=[190] * len(ep), yaxis="y",
                             name="actual high episode", marker=dict(symbol="triangle-down", size=12, color=C_TRUTH),
                             hovertemplate="High episode starts<extra></extra>"))
    sent = U.notifications(risk[:t + 1], thr)
    sent = sent[sent >= lo]
    if len(sent):
        yy = [150 if np.isnan(cg[j]) else float(cg[j]) for j in sent]
        data.append(dict(type="scatter", mode="markers", x=[x[j - lo] for j in sent], y=yy, yaxis="y",
                         name="alert sent", marker=dict(symbol="triangle-up", size=13, color=C_RISK,
                                                        line=dict(width=1, color="white")),
                         hovertemplate="Alert sent<extra></extra>"))
    data += [
        dict(type="scatter", mode="lines", x=xs, y=clean(risk[lo:t + 1]), name="2-h risk", yaxis="y2",
             line=dict(color=C_RISK, width=1.8), fill="tozeroy", fillcolor="rgba(220,38,38,0.12)",
             hovertemplate="%{y:.0%}<extra>Risk</extra>"),
        dict(type="scatter", mode="lines", x=xs, y=clean(a["hr"][i][lo:t + 1]), name="Heart rate", yaxis="y3",
             line=dict(color=C_HR, width=1.4), hovertemplate="%{y:.0f} bpm<extra>HR</extra>"),
        dict(type="scatter", mode="lines", x=xs, y=clean(a["hrv"][i][lo:t + 1]), name="HRV", yaxis="y5",
             line=dict(color=C_HRV, width=1.4), hovertemplate="%{y:.0f} ms<extra>HRV</extra>"),
        dict(type="bar", x=xs, y=[float(v) for v in a["steps"][i][lo:t + 1]], name="Steps / 5 min", yaxis="y4",
             marker=dict(color=C_STEPS), width=270000, hovertemplate="%{y:.0f} steps<extra></extra>"),
    ]
    d = _domains([0.46, 0.20, 0.20, 0.10])
    ymax = float(max(240, np.nanmax(cg[lo:hi + 1]) + 20)) if not np.all(np.isnan(cg[lo:hi + 1])) else 240
    shapes = [
        dict(type="rect", xref="x", yref="y", x0=x[0], x1=x[-1], y0=70, y1=180, fillcolor=C_BAND, line=dict(width=0), layer="below"),
        dict(type="line", xref="x", yref="y2", x0=x[0], x1=x[-1], y0=thr, y1=thr, line=dict(color="#111827", width=1, dash="dash")),
        dict(type="line", xref="x", yref="paper", x0=x[n_seen - 1], x1=x[n_seen - 1], y0=0, y1=1, line=dict(color="#111827", width=1)),
    ]
    for s, e in _runs((a["stage"][i][lo:hi + 1] > 0).tolist()):          # shade the night across all panels
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=x[s], x1=x[e], y0=0, y1=1,
                           fillcolor="rgba(100,116,139,0.14)", line=dict(width=0), layer="below"))
    layout = dict(
        height=660, margin=dict(l=60, r=60, t=30, b=40), hovermode="x unified", dragmode="pan",
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT, showlegend=True,
        legend=dict(orientation="h", x=0, y=1.07),
        xaxis=_axis(type="date", anchor="y4", range=[x[0], x[-1]], tickformat="%a %H:%M", showgrid=False,
                    showspikes=True, spikemode="across", spikethickness=1),
        yaxis=_axis(domain=d[0], title=dict(text="mg/dL"), range=[40, ymax]),
        yaxis2=_axis(domain=d[1], title=dict(text="risk"), range=[0, 1], tickformat=".0%"),
        yaxis3=_axis(domain=d[2], title=dict(text="HR")),
        yaxis4=_axis(domain=d[3], title=dict(text="steps")),
        yaxis5=_axis(overlaying="y3", side="right", title=dict(text="HRV"), showgrid=False),
        shapes=shapes)
    return dict(data=data, layout=layout)


def explain_spec(B, key, t):
    i = B.index(key)
    c = B.a["contrib"][i][t]
    if np.all(np.isnan(c)):
        return dict(data=[], layout=dict(height=300, paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                                         annotations=[dict(text="No forecast at this moment", showarrow=False, x=0.5, y=0.5,
                                                           xref="paper", yref="paper")],
                                         xaxis=dict(visible=False), yaxis=dict(visible=False)))
    order = np.argsort(c)
    vals = [float(c[j]) for j in order]
    names = [B.groups[j] for j in order]
    return dict(
        data=[dict(type="bar", orientation="h", x=vals, y=names,
                   marker=dict(color=[C_RISK if v > 0 else C_CGM for v in vals]),
                   hovertemplate="%{y}: %{x:+.1%}<extra></extra>")],
        layout=dict(height=320, margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                    xaxis=_axis(tickformat="+.0%", title=dict(text="effect on risk (red raises it, blue lowers it)")),
                    yaxis=dict(automargin=True)))


def overview_spec(B, key, t, thr):
    i = B.index(key)
    meta = B.patients[i]
    x = iso_times(meta, 0, B.T - 1)
    r = B.a["risk"][i]
    ep = np.nonzero(B.a["episode_start"][i])[0]
    nt = U.notifications(r, thr)
    data = [dict(type="scatter", mode="lines+markers", x=x, y=clean(r), name="2-h risk",
                 line=dict(color=C_RISK, width=1.2), marker=dict(size=5, color=C_RISK, opacity=0.35),
                 hovertemplate="%{x}<br>risk %{y:.0%}<extra></extra>"),
            dict(type="scatter", mode="markers", x=[x[j] for j in ep], y=[0.97] * len(ep), name="actual high episode",
                 marker=dict(symbol="triangle-down", size=9, color=C_TRUTH), hoverinfo="skip"),
            dict(type="scatter", mode="markers", x=[x[j] for j in nt], y=[0.04] * len(nt), name="alert sent",
                 marker=dict(symbol="triangle-up", size=9, color=C_RISK), hoverinfo="skip")]
    shapes = [dict(type="line", xref="x", yref="y", x0=x[0], x1=x[-1], y0=thr, y1=thr, line=dict(color="#111827", width=1, dash="dash")),
              dict(type="line", xref="x", yref="paper", x0=x[t], x1=x[t], y0=0, y1=1, line=dict(color="#111827", width=2))]
    layout = dict(height=250, margin=dict(l=50, r=20, t=30, b=40), paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                  hovermode="closest", dragmode="select", clickmode="event+select",
                  legend=dict(orientation="h", x=0, y=1.2),
                  xaxis=_axis(type="date", tickformat="%a %H:%M", showgrid=False, range=[x[0], x[-1]]),
                  yaxis=_axis(range=[0, 1], tickformat=".0%", title=dict(text="risk")), shapes=shapes)
    return dict(data=data, layout=layout)


def whatif_spec(base, scen, now, label):
    lo, hi = max(0, now - 24), min(len(base), now + 72)
    x = [round((j - now) * 5 / 60, 3) for j in range(lo, hi)]
    data = [dict(type="scatter", mode="lines", x=x, y=clean(base[lo:hi]), name="usual course (twin)",
                 line=dict(color="#6b7280", width=2.4), hovertemplate="%{y:.0f} mg/dL<extra>usual</extra>"),
            dict(type="scatter", mode="lines", x=x, y=clean(scen[lo:hi]), name=label,
                 line=dict(color=C_RISK, width=2.6), hovertemplate="%{y:.0f} mg/dL<extra>scenario</extra>")]
    shapes = [dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=70, y1=180, fillcolor=C_BAND, line=dict(width=0), layer="below"),
              dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="#111827", width=1))]
    layout = dict(height=400, margin=dict(l=60, r=20, t=30, b=50), hovermode="x unified", paper_bgcolor=TRANSPARENT,
                  plot_bgcolor=TRANSPARENT, legend=dict(orientation="h", x=0, y=1.1),
                  xaxis=_axis(title=dict(text="hours from now")), yaxis=_axis(title=dict(text="twin-predicted CGM glucose (mg/dL)")),
                  shapes=shapes)
    return dict(data=data, layout=layout)


def tradeoff_spec(rows, highlight=None):
    """rows: list of dicts with policy, operating_point, alerts_per_patient_day, share_caught_with_30min_lead."""
    colors = {"global": "#6b7280", "personal": "#dc2626", "blended": "#2563eb"}
    names = {"fpr_2": "Strict", "fpr_5": "Balanced", "fpr_10": "Sensitive"}
    data = []
    for pol in ("global", "personal", "blended"):
        rr = sorted([r for r in rows if r["policy"] == pol], key=lambda r: r["alerts_per_patient_day"])
        if not rr:
            continue
        data.append(dict(type="scatter", mode="lines+markers+text", name=pol.capitalize(),
                         x=[r["alerts_per_patient_day"] for r in rr], y=[100 * r["share_caught_with_30min_lead"] for r in rr],
                         text=[names.get(r["operating_point"], r["operating_point"]) for r in rr], textposition="top center",
                         line=dict(color=colors[pol], width=2.4), marker=dict(size=10, color=colors[pol]),
                         customdata=[[100 * r["alert_precision"], 100 * r["share_caught_any_lead"]] for r in rr],
                         hovertemplate="%{text}<br>%{x:.1f} alerts/day<br>%{y:.0f}% caught 30+ min early"
                                       "<br>%{customdata[0]:.0f}% of alerts correct<br>%{customdata[1]:.0f}% caught at any lead<extra>" + pol + "</extra>"))
    if highlight:
        data.append(dict(type="scatter", mode="markers", name="your setting", x=[highlight[0]], y=[highlight[1]],
                         marker=dict(symbol="star", size=20, color="#f59e0b", line=dict(width=1.5, color="#111827")),
                         hovertemplate="your current setting<extra></extra>"))
    layout = dict(height=430, margin=dict(l=60, r=20, t=30, b=60), paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                  legend=dict(orientation="h", x=0, y=1.1), hovermode="closest",
                  xaxis=_axis(title=dict(text="alerts per patient per day (lower is calmer)")),
                  yaxis=_axis(title=dict(text="high-glucose episodes warned 30+ minutes ahead (%)")))
    return dict(data=data, layout=layout)


# ---------------------------------------------------------------- report text
def weekly_summary(B, key, thr, policy_label, sens_label):
    i = B.index(key)
    m = B.patients[i]
    a = B.a
    lines = [f"DIGITAL TWIN WEEKLY SUMMARY (synthetic patient {key})", "=" * 60,
             f"Patient type: {m['archetype']}", "", "Health record"]
    for k, v in U.ehr_summary(m["ehr"]).items():
        lines.append(f"  {k}: {v}")
    w = m["window"]
    notifs = U.notifications(a["risk"][i], thr)
    ep = np.nonzero(a["episode_start"][i])[0]
    lines += ["", "Seven-day overview",
              f"  Time in range (70-180 mg/dL): {w['time_in_range']:.0%}",
              f"  Mean glucose: {w['mean_cgm']:.0f} mg/dL",
              f"  High-glucose episodes: {len(ep)}",
              f"  Alerts sent: {len(notifs)}  (threshold {thr:.0%}; {policy_label}; sensitivity {sens_label})", "",
              "Alerts"]
    for j in notifs:
        hit = bool(np.any((ep > j) & (ep <= j + 24)))
        cg = a["cgm"][i][j]
        lines.append(f"  {U.time_label(m, int(j))}  glucose {'n/a' if np.isnan(cg) else f'{cg:.0f}'} mg/dL  "
                     f"risk {a['risk'][i][j]:.0%}  -> {'high episode followed within 2 h' if hit else 'no high episode followed'}")
    lines += ["", "Prototype on synthetic data. Decision support only, not a medical device."]
    return "\n".join(lines)
