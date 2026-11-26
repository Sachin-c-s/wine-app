import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

BASE = Path(__file__).parent
FILES = ("wine_model.pkl", "wine_scaler.pkl", "wine_stats.pkl")

# ----------------------------------------------------------------------------
# Constants: verdict thresholds, feature metadata, presets
# ----------------------------------------------------------------------------
GOOD_AT, BAD_AT = 0.60, 0.40
MAIN = ["alcohol", "volatile acidity", "sulphates", "total sulfur dioxide"]
GROUPS = {
    "Acidity": ["fixed acidity", "citric acid", "pH"],
    "Sweetness & body": ["residual sugar", "density", "chlorides"],
    "Preservatives": ["free sulfur dioxide"],
}
# dec = natural measurement precision (used as the smallest slider step and display format)
FEATURES = {
    "alcohol": dict(label="Alcohol strength", short="alcohol", unit="% vol", dec=1,
                    help="How strong the wine is. In this dataset, stronger reds tend to be rated higher."),
    "volatile acidity": dict(label="Vinegar-like sharpness", short="vinegar-like sharpness", unit="g/L", dec=2,
                             help="Acetic acid. A little is normal; too much smells like vinegar and usually hurts ratings."),
    "sulphates": dict(label="Preservative boost", short="sulphates", unit="g/L", dec=2,
                      help="Sulphates protect wine from spoiling. Moderate-to-high levels tend to go with better ratings."),
    "total sulfur dioxide": dict(label="Total preservative level", short="total sulfur dioxide", unit="mg/L", dec=0,
                                 help="All the sulfur dioxide in the wine (free + bound). Very high levels can taste harsh."),
    "fixed acidity": dict(label="Tartness", short="fixed acidity", unit="g/L", dec=1,
                          help="The wine's main non-volatile acids (mostly tartaric). Gives a fresh, tart backbone."),
    "citric acid": dict(label="Citrus freshness", short="citric acid", unit="g/L", dec=2,
                        help="A small amount adds freshness and a lemony lift."),
    "pH": dict(label="Acidity scale (pH)", short="pH", unit="pH", dec=2,
               help="Lower pH means a more acidic, sharper-tasting wine. Most wines sit between 3 and 4."),
    "residual sugar": dict(label="Sweetness", short="residual sugar", unit="g/L", dec=1,
                           help="Sugar left after fermentation. Most reds here are dry (low sugar)."),
    "density": dict(label="Body (density)", short="density", unit="g/cm³", dec=4,
                    help="How heavy the wine is compared with water. Linked to alcohol and sugar."),
    "chlorides": dict(label="Saltiness", short="chlorides", unit="g/L", dec=3,
                      help="Salt content. Lower is usually better."),
    "free sulfur dioxide": dict(label="Active preservative (free)", short="free sulfur dioxide", unit="mg/L", dec=0,
                                help="The part of the sulfur dioxide that is still actively protecting the wine."),
}
PRESETS = {
    "Typical wine": None,  # all medians
    "Light everyday red": {"alcohol": 9.5, "volatile acidity": 0.55, "sulphates": 0.58, "total sulfur dioxide": 45},
    "Bold high-alcohol red": {"alcohol": 13.5, "volatile acidity": 0.40, "sulphates": 0.85, "total sulfur dioxide": 30},
    "Flawed or spoiled": {"alcohol": 9.0, "volatile acidity": 1.20, "sulphates": 0.45, "total sulfur dioxide": 150},
}


# ----------------------------------------------------------------------------
# Pure logic (no Streamlit calls): easy to unit test
# ----------------------------------------------------------------------------
def predict_probs(model, scaler, cols, rows):
    # rows: list of dicts or a DataFrame; always scale with a named, ordered DataFrame
    X = pd.DataFrame(rows, columns=cols)
    pos = list(model.classes_).index(1)
    return model.predict_proba(scaler.transform(X))[:, pos]


def predict_prob(model, scaler, cols, values):
    return float(predict_probs(model, scaler, cols, [values])[0])


def verdict_for(p):
    if p >= GOOD_AT:
        return "good", "Likely a good wine", "✓"
    if p <= BAD_AT:
        return "bad", "Likely not a good wine", "✕"
    return "close", "Too close to call", "≈"


def slider_spec(lo, hi, dec, target_steps=150):
    # nice step (1, 2 or 5 x 10^k) near range/150, never finer than the natural precision
    target = max((hi - lo) / target_steps, 10.0 ** -dec)
    exp = math.floor(math.log10(target))
    step = 10.0 ** exp
    for m in (1, 2, 5, 10):
        step = m * 10.0 ** exp
        if step >= target - 1e-12:
            break
    return round(step, dec + 2), f"%.{dec}f"


def fmt_val(v, dec):
    return f"{v:.{dec}f}"


def snap(v, lo, hi, step, dec):
    # clip to [lo, hi] and align to the slider's step grid
    v = min(max(v, lo), hi)
    return round(round(lo + round((v - lo) / step) * step, dec + 2), dec + 2)


def preset_values(name, stats, cols):
    # main 4 from the preset (clipped + snapped), everything else at median
    out = {c: float(stats["median"][c]) for c in cols}
    for c, v in (PRESETS.get(name) or {}).items():
        lo, hi = stats["min"][c], stats["max"][c]
        step, _ = slider_spec(lo, hi, FEATURES[c]["dec"])
        out[c] = float(min(max(snap(v, lo, hi, step, FEATURES[c]["dec"]), lo), hi))
    return out


def compute_drivers(model, scaler, cols, stats, values, min_effect=0.005, top=3):
    # effect of each feature = P(good) now minus P(good) with that feature set to its median
    base = predict_prob(model, scaler, cols, values)
    rows = []
    for c in cols:
        r = dict(values)
        r[c] = float(stats["median"][c])
        rows.append(r)
    alt = predict_probs(model, scaler, cols, rows)
    effects = [(c, base - float(a)) for c, a in zip(cols, alt)]
    pos = sorted([e for e in effects if e[1] >= min_effect], key=lambda e: -e[1])[:top]
    neg = sorted([e for e in effects if e[1] <= -min_effect], key=lambda e: e[1])[:top]
    return pos, neg


def find_nudge(model, scaler, cols, stats, values, features=MAIN, frac=0.10, min_gain=0.02):
    # try +/- 10% of range on each main feature; return the single best move if it gains enough
    base = predict_prob(model, scaler, cols, values)
    cands = []
    for f in features:
        lo, hi = float(stats["min"][f]), float(stats["max"][f])
        for sign in (1, -1):
            new = min(max(values[f] + sign * frac * (hi - lo), lo), hi)
            new = round(new, FEATURES[f]["dec"])
            if new == round(values[f], FEATURES[f]["dec"]):
                continue
            r = dict(values)
            r[f] = new
            cands.append((f, new, sign, r))
    if not cands:
        return None
    probs = predict_probs(model, scaler, cols, [c[3] for c in cands])
    i = int(np.argmax(probs))
    gain = float(probs[i]) - base
    if gain < min_gain:
        return None
    f, new, sign, _ = cands[i]
    return dict(feature=f, new_value=new, direction="up" if sign > 0 else "down",
                new_prob=float(probs[i]), gain=gain)


def sweep_feature(model, scaler, cols, stats, feat, n=60):
    # P(good) as one feature runs min -> max with everything else at median
    xs = np.linspace(float(stats["min"][feat]), float(stats["max"][feat]), n)
    rows = []
    for x in xs:
        r = {c: float(stats["median"][c]) for c in cols}
        r[feat] = float(x)
        rows.append(r)
    probs = predict_probs(model, scaler, cols, rows) * 100
    meta = FEATURES[feat]
    df = pd.DataFrame({"Chance of good (%)": probs}, index=pd.Index(xs, name=f'{meta["label"]} ({meta["unit"]})'))
    return df


def score_batch(model, scaler, cols, raw):
    # raw: uploaded DataFrame. Returns (result df, missing cols, n_invalid)
    lookup = {str(c).strip().lower(): c for c in raw.columns}
    missing = [c for c in cols if c.lower() not in lookup]
    if missing:
        return None, missing, 0
    X = pd.DataFrame({c: pd.to_numeric(raw[lookup[c.lower()]], errors="coerce") for c in cols})
    ok = X.notna().all(axis=1)
    p = pd.Series(np.nan, index=raw.index)
    if ok.any():
        p.loc[ok] = predict_probs(model, scaler, cols, X.loc[ok])
    out = raw.copy()
    out["P(good) %"] = (p * 100).round(1)
    out["Verdict"] = [verdict_for(v)[1] if pd.notna(v) else "Invalid input" for v in p]
    return out, [], int((~ok).sum())


# ---- small HTML builders (single-line strings so Markdown never treats them as code) ----
def result_html(p):
    key, title, icon = verdict_for(p)
    pct = f"{p * 100:.0f}"
    return (
        f'<div class="wq-card wq-{key}">'
        f'<div class="wq-vrow"><div class="wq-icon" aria-hidden="true">{icon}</div>'
        f'<div class="wq-vtext"><div class="wq-vtitle">{title}</div>'
        f'<div class="wq-vsub">Chance this is a good wine (rated 6 or higher)</div></div>'
        f'<div class="wq-pct">{pct}<span>%</span></div></div>'
        f'<div class="wq-gauge" role="img" aria-label="{pct} percent chance of a good wine. {title}.">'
        f'<div class="wq-track"><div class="wq-fill" style="width:{pct}%"></div>'
        f'<span class="wq-mark" style="left:40%"></span><span class="wq-mark" style="left:60%"></span></div>'
        f'<div class="wq-zones"><span style="width:40%">Unlikely</span>'
        f'<span style="width:20%">Close call</span><span style="width:40%">Likely</span></div></div></div>'
    )


def nudge_html(n):
    meta = FEATURES[n["feature"]]
    verb = "Raising" if n["direction"] == "up" else "Lowering"
    val = f'{fmt_val(n["new_value"], meta["dec"])} {meta["unit"]}'
    return (
        f'<div class="wq-nudge"><b>Biggest lever:</b> {verb} {meta["short"]} to {val} '
        f'would lift this to <b>{n["new_prob"] * 100:.0f}%</b> (+{n["gain"] * 100:.0f} pts).</div>'
    )


def drivers_html(items, kind, values, stats, scale):
    if not items:
        return '<div class="wq-muted">Nothing notable here.</div>'
    rows = []
    for f, d in items:
        meta = FEATURES[f]
        word = "higher" if values[f] > stats["median"][f] else "lower"
        sub = (f'{word} than typical ({fmt_val(values[f], meta["dec"])} vs '
               f'{fmt_val(stats["median"][f], meta["dec"])} {meta["unit"]})')
        width = max(4, round(abs(d) / scale * 100))
        pts = abs(d) * 100
        sign = "+" if d > 0 else "−"
        txt = f"{sign}{pts:.0f} pts" if pts >= 0.5 else f"{sign}<1 pt"
        rows.append(
            f'<div class="wq-drv"><div class="wq-drv-name">{meta["label"]}<small>{sub}</small></div>'
            f'<div class="wq-drv-bar"><div class="wq-drv-fill {kind}" style="width:{width}%"></div></div>'
            f'<div class="wq-drv-val">{txt}</div></div>'
        )
    return "".join(rows)


CSS = """
<style>
:root{
  --wq-accent:#8B1E3F;
  --wq-accent-text:#8B1E3F;
  --wq-accent-text:color-mix(in srgb,#8B1E3F 62%,currentColor);
  --wq-line:rgba(128,128,128,.28);
  --wq-soft:rgba(128,128,128,.07);
}
[data-testid="stMainBlockContainer"],.block-container{max-width:1100px;padding-top:2rem;}
html,body,.stApp{overflow-x:hidden;}
.wq-hero h1{font-family:Georgia,'Times New Roman',serif;font-weight:600;letter-spacing:.2px;
  color:var(--wq-accent-text);margin:0 0 .25rem 0;padding:0;font-size:2.2rem;line-height:1.15;}
.wq-hero p{margin:0 0 1rem 0;opacity:.75;font-size:1.02rem;}
h4,h5{font-family:Georgia,'Times New Roman',serif;font-weight:600;}
.wq-card{border:1px solid var(--wq-line);border-radius:16px;padding:1.2rem 1.3rem;margin:.25rem 0 1rem 0;
  background:var(--wq-soft);}
.wq-good{--tone:46,125,80;}
.wq-bad{--tone:139,30,63;}
.wq-close{--tone:168,108,6;}
.wq-card[class*="wq-good"],.wq-card[class*="wq-bad"],.wq-card[class*="wq-close"]{
  background:rgba(var(--tone),.10);border-color:rgba(var(--tone),.50);}
.wq-vrow{display:flex;align-items:center;gap:.9rem;flex-wrap:nowrap;}
.wq-icon{flex:0 0 auto;width:2.8rem;height:2.8rem;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:1.5rem;font-weight:700;color:#fff;background:rgb(var(--tone));}
.wq-vtext{flex:1 1 auto;min-width:0;}
.wq-vtitle{font-family:Georgia,'Times New Roman',serif;font-size:1.55rem;font-weight:600;line-height:1.2;}
.wq-vsub{opacity:.75;font-size:.9rem;margin-top:.15rem;}
.wq-pct{flex:0 0 auto;font-family:Georgia,'Times New Roman',serif;font-size:2.8rem;font-weight:600;line-height:1;}
.wq-pct span{font-size:1.4rem;opacity:.7;}
.wq-gauge{margin-top:1.1rem;}
.wq-track{position:relative;height:16px;border-radius:999px;background:rgba(128,128,128,.25);overflow:hidden;}
.wq-fill{height:100%;border-radius:999px;background:rgb(var(--tone));}
.wq-mark{position:absolute;top:0;bottom:0;width:2px;background:rgba(128,128,128,.9);}
.wq-zones{display:flex;margin-top:.3rem;font-size:.75rem;opacity:.7;}
.wq-zones span{text-align:center;}
.wq-nudge{border:1px dashed var(--wq-line);border-radius:12px;padding:.75rem 1rem;margin:0 0 1rem 0;font-size:.95rem;}
.wq-muted{opacity:.65;font-size:.9rem;padding:.2rem 0 .6rem 0;}
.wq-sec{font-weight:600;margin:.6rem 0 .2rem 0;}
.wq-drv{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr) 4.2rem;gap:.7rem;align-items:center;margin:.35rem 0;}
.wq-drv-name{font-size:.95rem;line-height:1.2;}
.wq-drv-name small{display:block;opacity:.65;font-size:.76rem;margin-top:.1rem;}
.wq-drv-bar{height:10px;border-radius:999px;background:rgba(128,128,128,.2);overflow:hidden;}
.wq-drv-fill{height:100%;border-radius:999px;}
.wq-drv-fill.pos{background:rgb(46,125,80);}
.wq-drv-fill.neg{background:rgb(168,40,78);}
.wq-drv-val{text-align:right;font-variant-numeric:tabular-nums;font-size:.9rem;}
.wq-foot{opacity:.65;font-size:.82rem;margin-top:1.2rem;border-top:1px solid var(--wq-line);padding-top:.8rem;}
.stButton button{border-radius:999px;border:1px solid rgba(139,30,63,.55);min-height:2.6rem;}
.stButton button:hover{border-color:#8B1E3F;color:var(--wq-accent-text);}
[data-testid="stExpander"]{border:1px solid var(--wq-line);border-radius:14px;}
@media (max-width:640px){
  .wq-hero h1{font-size:1.7rem;}
  .wq-vtitle{font-size:1.25rem;}
  .wq-pct{font-size:2.1rem;}
  .wq-drv{grid-template-columns:minmax(0,1fr) 4rem;}
  .wq-drv-bar{grid-column:1 / span 2;grid-row:2;}
}
</style>
"""

# ===================== UI below (Streamlit) =====================
st.markdown(CSS, unsafe_allow_html=True)

# ---- load artifacts (clear error if anything is missing or incompatible) ----
missing_files = [f for f in FILES if not (BASE / f).exists()]
if missing_files:
    st.error(f"Missing file(s) next to app.py: {', '.join(missing_files)}. Upload them to the same folder in the repo.")
    st.stop()


@st.cache_resource
def load_artifacts():
    return (joblib.load(BASE / "wine_model.pkl"),
            joblib.load(BASE / "wine_scaler.pkl"),
            joblib.load(BASE / "wine_stats.pkl"))


try:
    model, scaler, stats = load_artifacts()
    cols = list(stats["columns"])
    assert all(k in stats for k in ("median", "min", "max")) and set(cols) == set(FEATURES)
except Exception as e:  # corrupt file, version mismatch, wrong keys...
    st.error(f"Could not load the model files: {e}")
    st.stop()

KEY = {c: "in_" + c.replace(" ", "_") for c in cols}


@st.cache_data(show_spinner=False)
def cached_sweeps():
    return {c: sweep_feature(model, scaler, cols, stats, c) for c in cols}


# ---- session state + callbacks ----
for c in cols:
    st.session_state.setdefault(KEY[c], float(stats["median"][c]))


def apply_preset(name):
    for c, v in preset_values(name, stats, cols).items():
        st.session_state[KEY[c]] = v


def feature_slider(c):
    meta = FEATURES[c]
    lo, hi, med = float(stats["min"][c]), float(stats["max"][c]), float(stats["median"][c])
    step, fmt = slider_spec(lo, hi, meta["dec"])
    st.slider(f'{meta["label"]} ({meta["unit"]})', min_value=lo, max_value=hi, step=step,
              format=fmt, key=KEY[c], help=meta["help"])
    st.caption(f'{c} · typical: {fmt_val(med, meta["dec"])} {meta["unit"]}')


st.markdown(
    '<div class="wq-hero"><h1>🍷 Wine Quality Predictor</h1>'
    '<p>Move a slider or pick a style. The verdict updates instantly.</p></div>',
    unsafe_allow_html=True,
)

tab_predict, tab_how, tab_explore, tab_batch = st.tabs(["Predict", "How it works", "Explore", "Batch"])

# ================= Predict =================
with tab_predict:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### Quick styles")
        names = list(PRESETS)
        r1 = st.columns(2)
        r2 = st.columns(2)
        for slot, name in zip(r1 + r2, names):
            slot.button(name, key="btn_" + name, on_click=apply_preset, args=(name,))
        st.button("Reset everything", key="btn_reset", on_click=apply_preset, args=("Typical wine",))

        st.markdown("##### Your wine")
        for c in MAIN:
            feature_slider(c)

        with st.expander("Advanced chemistry"):
            for group, feats in GROUPS.items():
                st.markdown(f"**{group}**")
                for c in feats:
                    feature_slider(c)

    values = {c: float(st.session_state[KEY[c]]) for c in cols}

    with right:
        p = predict_prob(model, scaler, cols, values)
        st.markdown(result_html(p), unsafe_allow_html=True)

        nudge = find_nudge(model, scaler, cols, stats, values)
        if nudge:
            st.markdown(nudge_html(nudge), unsafe_allow_html=True)

        st.markdown("##### Why this result")
        st.caption("Compared with a typical wine: how much each setting adds to or takes from the chance of a good wine.")
        pos, neg = compute_drivers(model, scaler, cols, stats, values)
        if not pos and not neg:
            st.markdown('<div class="wq-muted">Nothing stands out. This looks like a typical wine.</div>',
                        unsafe_allow_html=True)
        else:
            scale = max(abs(d) for _, d in pos + neg)
            st.markdown('<div class="wq-sec">Working in its favour</div>', unsafe_allow_html=True)
            st.markdown(drivers_html(pos, "pos", values, stats, scale), unsafe_allow_html=True)
            st.markdown('<div class="wq-sec">Working against it</div>', unsafe_allow_html=True)
            st.markdown(drivers_html(neg, "neg", values, stats, scale), unsafe_allow_html=True)

# ================= How it works =================
with tab_how:
    st.markdown("#### How it works")
    st.markdown(
        "The model looks at 11 lab measurements of a red wine (alcohol, acidity, preservatives and so on) "
        "and estimates how likely it is that professional tasters rated it **6 out of 10 or higher**, "
        "which we call a *good* wine."
    )
    st.markdown(
        f"- **Likely a good wine:** chance of {GOOD_AT:.0%} or more\n"
        f"- **Too close to call:** between {BAD_AT:.0%} and {GOOD_AT:.0%}\n"
        f"- **Likely not a good wine:** chance of {BAD_AT:.0%} or less"
    )
    st.markdown("**\"Why this result\"** swaps each setting, one at a time, for the typical (median) value and "
                "sees how the chance changes. **\"Biggest lever\"** nudges each of the four main sliders by about "
                "10% of its range and reports the single change that helps most.")
    st.markdown("#### How accurate is it?")
    st.markdown(
        "On wines it had never seen, it gets roughly three out of four right (F1 score about 0.75). "
        "That is decent for a fun demo, and far from perfect: one in four calls will be wrong."
    )
    st.markdown("#### Limits worth knowing")
    st.markdown(
        "- Trained on about 1,000 **red** wines only. It knows nothing about white, rosé or sparkling.\n"
        "- Ratings are human opinions. Chemistry cannot capture aroma, grape, age or how a wine is stored.\n"
        "- Sliders stay inside the ranges seen in training. Unusual combinations may give odd answers.\n"
        "- The percentage is a model estimate, not a guarantee."
    )
    with st.expander("For the curious: technical details"):
        st.markdown(
            "- Model: tuned gradient boosting classifier (scikit-learn), trained on standardised features.\n"
            "- Data: WineQT red wine dataset, duplicates removed; target is quality ≥ 6.\n"
            "- Every input goes through the saved `StandardScaler` before prediction, with columns in training order.\n"
            "- Feature order: " + ", ".join(f"`{c}`" for c in cols)
        )

# ================= Explore =================
with tab_explore:
    st.markdown("#### Explore one setting at a time")
    st.caption("Each chart sweeps one measurement from its lowest to highest value in the training data, "
               "with every other setting held at a typical (median) value.")
    sweeps = cached_sweeps()
    grid = st.columns(2, gap="large")
    for i, c in enumerate(cols):
        meta = FEATURES[c]
        with grid[i % 2]:
            st.markdown(f'**{meta["label"]}**  \n<small>{c} ({meta["unit"]})</small>', unsafe_allow_html=True)
            st.line_chart(sweeps[c], color="#8B1E3F", height=200)

# ================= Batch =================
with tab_batch:
    st.markdown("#### Score many wines at once")
    st.markdown("Upload a CSV with these 11 columns (extra columns are kept and passed through):")
    st.code(", ".join(cols), language=None)
    template = pd.DataFrame([{c: float(stats["median"][c]) for c in cols}])
    st.download_button("Download a template CSV", template.to_csv(index=False).encode("utf-8"),
                       file_name="wine_template.csv", mime="text/csv")

    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read that file as a CSV: {e}")
            raw = None
        if raw is not None:
            if raw.empty:
                st.error("The file has no rows.")
            elif len(raw) > 20000:
                st.error("Please upload 20,000 rows or fewer.")
            else:
                result, missing, n_bad = score_batch(model, scaler, cols, raw)
                if missing:
                    st.error("Missing required column(s): " + ", ".join(missing))
                else:
                    if n_bad:
                        st.warning(f"{n_bad} row(s) have blank or non-numeric values and were not scored.")
                    counts = result["Verdict"].value_counts()
                    st.markdown("  \n".join(f"**{k}:** {v}" for k, v in counts.items()))
                    st.dataframe(result)
                    st.download_button("Download results as CSV", result.to_csv(index=False).encode("utf-8"),
                                       file_name="wine_predictions.csv", mime="text/csv")

st.markdown(
    '<div class="wq-foot">A fun demo trained on about 1,000 red wines. It is not a sommelier and '
    'shouldn\'t be used to decide what to buy, serve or sell.</div>',
    unsafe_allow_html=True,
)
