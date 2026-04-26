import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GoldSense · Price Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400;500&family=Outfit:wght@200;300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0800 !important;
    color: #e8dfc0 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(180,140,20,0.05) 0%, transparent 55%),
        repeating-linear-gradient(0deg, transparent, transparent 60px, rgba(212,175,55,0.015) 60px, rgba(212,175,55,0.015) 61px),
        repeating-linear-gradient(90deg, transparent, transparent 60px, rgba(212,175,55,0.015) 60px, rgba(212,175,55,0.015) 61px);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: #0e0b00 !important;
    border-right: 1px solid rgba(212,175,55,0.15) !important;
}
[data-testid="stSidebar"] * { color: #e8dfc0 !important; }
[data-testid="stSidebarContent"] { padding: 2rem 1.5rem !important; }

h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; }

[data-testid="stNumberInput"] input {
    background: rgba(212,175,55,0.04) !important;
    border: 1px solid rgba(212,175,55,0.2) !important;
    border-radius: 4px !important;
    color: #e8dfc0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(212,175,55,0.6) !important;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #b8960c 0%, #d4af37 50%, #f0d060 100%) !important;
    color: #0a0800 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(212,175,55,0.25) !important;
}

[data-testid="stMetric"] {
    background: rgba(212,175,55,0.04) !important;
    border: 1px solid rgba(212,175,55,0.12) !important;
    border-radius: 6px !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(232,223,192,0.5) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
    font-weight: 300 !important;
    color: #d4af37 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(212,175,55,0.15) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(232,223,192,0.4) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.5rem !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #d4af37 !important;
    border-bottom-color: #d4af37 !important;
    background: transparent !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(212,175,55,0.15) !important;
    border-radius: 4px !important;
}

[data-testid="stAlert"] {
    background: rgba(212,175,55,0.06) !important;
    border: 1px solid rgba(212,175,55,0.2) !important;
    border-radius: 4px !important;
    color: #e8dfc0 !important;
}

hr { border-color: rgba(212,175,55,0.12) !important; margin: 2rem 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0800; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1.5rem; position: relative;">
    <div style="font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.25em;
                text-transform:uppercase; color:rgba(212,175,55,0.5); margin-bottom:0.6rem;">
        ✦ Commodity Intelligence Platform
    </div>
    <h1 style="font-family:'Cormorant Garamond',serif; font-size:clamp(2.8rem,6vw,4.5rem);
               font-weight:300; line-height:1.05; color:#e8dfc0; margin:0 0 0.5rem;">
        Gold Price<br><span style="color:#d4af37; font-style:italic;">Predictor</span>
    </h1>
    <p style="font-family:'Outfit',sans-serif; font-size:1rem; font-weight:200;
              color:rgba(232,223,192,0.45); margin:0; max-width:480px; line-height:1.7;">
        Random Forest model trained on SPX, USO, SLV &amp; EUR/USD signals to forecast GLD price.
    </p>
</div>
<div style="height:1px; background:linear-gradient(90deg,rgba(212,175,55,0.4) 0%,rgba(212,175,55,0.05) 60%,transparent 100%); margin-bottom:2.5rem;"></div>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("RandomForestRegressor.pkl"), None
    except FileNotFoundError:
        return None, "Model file `RandomForestRegressor.pkl` not found. Place it in the same directory as this app."

model, model_error = load_model()


# ── Sidebar — Market Inputs only ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
                    text-transform:uppercase; color:rgba(212,175,55,0.4); margin-bottom:0.4rem;">
            ✦ Market Signals
        </div>
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.6rem; font-weight:300;
                    color:#e8dfc0; line-height:1.2;">
            Input Panel
        </div>
    </div>
    """, unsafe_allow_html=True)

    spx = st.number_input("SPX — S&P 500 Index", value=1500.0, step=10.0,  format="%.2f")
    uso = st.number_input("USO — Oil ETF Price",  value=35.0,   step=0.5,   format="%.2f")
    slv = st.number_input("SLV — Silver ETF",     value=15.0,   step=0.1,   format="%.2f")
    eur = st.number_input("EUR/USD — FX Rate",    value=1.10,   step=0.01,  format="%.4f")

    st.markdown("<div style='height:1px; background:rgba(212,175,55,0.1); margin:1.5rem 0;'></div>",
                unsafe_allow_html=True)

    predict_btn = st.button("✦  Run Prediction", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem; padding:1rem; border:1px solid rgba(212,175,55,0.1);
                border-radius:4px; background:rgba(212,175,55,0.03);">
        <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
                    text-transform:uppercase; color:rgba(212,175,55,0.35); margin-bottom:0.5rem;">
            Key Driver
        </div>
        <div style="font-family:'Outfit',sans-serif; font-size:0.8rem;
                    color:rgba(232,223,192,0.5); line-height:1.6;">
            GLD tracks <span style="color:#d4af37;">SLV</span> most closely,
            with inverse sensitivity to <span style="color:#d4af37;">USO</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  Prediction  ", "  Batch Forecast  "])


# ══════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════
with tab1:
    if model_error:
        st.warning(model_error)
    else:
        prediction = model.predict(np.array([[spx, uso, slv, eur]]))[0]

        st.markdown(f"""
        <div style="margin:1.5rem 0 2rem; padding:3rem; border:1px solid rgba(212,175,55,0.25);
                    border-radius:6px; background:linear-gradient(135deg,
                    rgba(212,175,55,0.06) 0%, rgba(212,175,55,0.02) 50%, transparent 100%);
                    position:relative; overflow:hidden;">
            <div style="position:absolute; top:-40px; right:-40px; width:200px; height:200px;
                        border-radius:50%; background:radial-gradient(circle,
                        rgba(212,175,55,0.08) 0%, transparent 70%);"></div>
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.25em;
                        text-transform:uppercase; color:rgba(212,175,55,0.5); margin-bottom:0.8rem;">
                ✦ Predicted GLD Price
            </div>
            <div style="font-family:'Cormorant Garamond',serif; font-size:5rem; font-weight:300;
                        color:#d4af37; line-height:1; margin-bottom:0.5rem;">
                ${prediction:.2f}
            </div>
            <div style="font-family:'Outfit',sans-serif; font-size:0.85rem;
                        color:rgba(232,223,192,0.35); font-weight:200;">
                Gold ETF (GLD) · USD per Share
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Input Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("SPX",     f"{spx:,.2f}")
        with c2: st.metric("USO",     f"${uso:.2f}")
        with c3: st.metric("SLV",     f"${slv:.2f}")
        with c4: st.metric("EUR/USD", f"{eur:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
                    text-transform:uppercase; color:rgba(212,175,55,0.4); margin-bottom:1.2rem;">
            Typical Feature importance's
        </div>
        """, unsafe_allow_html=True)

        for feat, imp in [("SLV", 0.61), ("SPX", 0.18), ("EUR/USD", 0.13), ("USO", 0.08)]:
            pct = imp * 100
            st.markdown(f"""
            <div style="margin-bottom:0.9rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                    <span style="font-family:'DM Mono',monospace; font-size:0.75rem;
                                 color:rgba(232,223,192,0.7);">{feat}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.75rem;
                                 color:#d4af37;">{pct:.0f}%</span>
                </div>
                <div style="height:4px; background:rgba(212,175,55,0.1); border-radius:2px; overflow:hidden;">
                    <div style="height:100%; width:{pct}%;
                                background:linear-gradient(90deg,#b8960c,#d4af37); border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 2 — Batch Forecast
# ══════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="font-family:'Outfit',sans-serif; font-size:0.9rem;
                color:rgba(232,223,192,0.45); margin-bottom:1.5rem; line-height:1.7;">
        Upload a CSV with columns
        <code style="color:#d4af37; background:rgba(212,175,55,0.1);
                     padding:0.1rem 0.4rem; border-radius:3px;">SPX, USO, SLV, EUR/USD</code>
        to generate batch predictions.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        if model_error:
            st.warning(model_error)
        else:
            df_up    = pd.read_csv(uploaded)
            required = ["SPX", "USO", "SLV", "EUR/USD"]
            missing  = [c for c in required if c not in df_up.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                preds = model.predict(df_up[required])
                df_up["Predicted GLD"] = np.round(preds, 2)

                st.markdown(f"""
                <div style="display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap;">
                    <div style="padding:0.8rem 1.5rem; border:1px solid rgba(212,175,55,0.2);
                                border-radius:4px; background:rgba(212,175,55,0.04);">
                        <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
                                    text-transform:uppercase; color:rgba(212,175,55,0.4);">Rows</div>
                        <div style="font-family:'Cormorant Garamond',serif; font-size:2rem;
                                    font-weight:300; color:#d4af37;">{len(df_up)}</div>
                    </div>
                    <div style="padding:0.8rem 1.5rem; border:1px solid rgba(212,175,55,0.2);
                                border-radius:4px; background:rgba(212,175,55,0.04);">
                        <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
                                    text-transform:uppercase; color:rgba(212,175,55,0.4);">Avg GLD</div>
                        <div style="font-family:'Cormorant Garamond',serif; font-size:2rem;
                                    font-weight:300; color:#d4af37;">${preds.mean():.2f}</div>
                    </div>
                    <div style="padding:0.8rem 1.5rem; border:1px solid rgba(212,175,55,0.2);
                                border-radius:4px; background:rgba(212,175,55,0.04);">
                        <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
                                    text-transform:uppercase; color:rgba(212,175,55,0.4);">Range</div>
                        <div style="font-family:'Cormorant Garamond',serif; font-size:2rem;
                                    font-weight:300; color:#d4af37;">${preds.min():.0f} – ${preds.max():.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(df_up, use_container_width=True, height=320)

                st.download_button(
                    label="✦  Download Predictions",
                    data=df_up.to_csv(index=False).encode("utf-8"),
                    file_name="gld_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.markdown("""
        <div style="border:1px dashed rgba(212,175,55,0.2); border-radius:6px;
                    padding:3rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.8rem; opacity:0.3;">⬆</div>
            <div style="font-family:'Outfit',sans-serif; font-size:0.85rem;
                        color:rgba(232,223,192,0.3); font-weight:200;">
                Drop your CSV here · SPX · USO · SLV · EUR/USD
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem; padding-top:1.5rem;
            border-top:1px solid rgba(212,175,55,0.08);
            display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
                text-transform:uppercase; color:rgba(212,175,55,0.2);">
        ✦ GoldSense · Powered by Random Forest
    </div>
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.1em;
                color:rgba(232,223,192,0.15);">
        For research purposes only · Not financial advice
    </div>
</div>
""", unsafe_allow_html=True)