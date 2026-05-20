"""
Medical Billing Intelligence Dashboard
=======================================
World-class Streamlit dashboard with data storytelling.
Reads directly from Delta Lake Parquet files (no Spark needed).
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedBilling Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
GOLD   = BASE / "delta" / "gold"
SILVER = BASE / "delta" / "silver"

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "primary":   "#1E3A5F",
    "accent":    "#2196F3",
    "success":   "#00C853",
    "warning":   "#FF9800",
    "danger":    "#F44336",
    "purple":    "#7C4DFF",
    "teal":      "#00BCD4",
    "gold":      "#FFC107",
    "bg":        "#0E1117",
    "card":      "#1A1F2E",
    "text":      "#FAFAFA",
    "muted":     "#9E9E9E",
}

SEQ_BLUE  = px.colors.sequential.Blues
SEQ_TEAL  = px.colors.sequential.Teal
DIV_RdBu  = px.colors.diverging.RdBu
QUAL      = px.colors.qualitative.Set2

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0E1117; }
  .block-container { padding: 1.5rem 2rem; }

  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #1E2640 100%);
    border: 1px solid #2D3561;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
  }
  .kpi-card:hover { transform: translateY(-2px); }
  .kpi-value { font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
  .kpi-label { font-size: 0.78rem; color: #9E9E9E; text-transform: uppercase; letter-spacing: 0.08em; }
  .kpi-delta { font-size: 0.82rem; margin-top: 0.3rem; }
  .delta-up   { color: #00C853; }
  .delta-down { color: #F44336; }
  .delta-neu  { color: #9E9E9E; }

  /* Section headers */
  .section-header {
    font-size: 1.1rem; font-weight: 600; color: #FAFAFA;
    border-left: 4px solid #2196F3;
    padding-left: 0.8rem; margin: 1.5rem 0 0.8rem 0;
  }
  .story-box {
    background: linear-gradient(135deg, #1A2744 0%, #1E2A50 100%);
    border-left: 4px solid #2196F3;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.88rem; color: #CFD8DC; line-height: 1.6;
  }
  .insight-box {
    background: linear-gradient(135deg, #1A2A1A 0%, #1E3020 100%);
    border-left: 4px solid #00C853;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.88rem; color: #C8E6C9; line-height: 1.6;
  }
  .alert-box {
    background: linear-gradient(135deg, #2A1A1A 0%, #301E1E 100%);
    border-left: 4px solid #F44336;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.88rem; color: #FFCDD2; line-height: 1.6;
  }
  /* Sidebar */
  [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1F2937; }
  [data-testid="stSidebar"] .stSelectbox label { color: #9CA3AF; font-size: 0.8rem; }
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1A1F2E; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 0.4rem 1rem; font-size: 0.85rem; }
  .stTabs [aria-selected="true"] { background: #2196F3 !important; color: white !important; }
  /* Metric */
  [data-testid="metric-container"] { background: #1A1F2E; border-radius: 8px; padding: 0.8rem; border: 1px solid #2D3561; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS  (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_gold(table: str) -> pd.DataFrame:
    path = GOLD / table
    if not path.exists():
        return pd.DataFrame()
    files = list(path.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_silver(table: str, cols=None) -> pd.DataFrame:
    path = SILVER / table
    if not path.exists():
        return pd.DataFrame()
    files = list(path.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=cols) if cols else pd.read_parquet(f)
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fmt_currency(v):
    if v >= 1_000_000: return f"${v/1_000_000:.1f}M"
    if v >= 1_000:     return f"${v/1_000:.0f}K"
    return f"${v:.0f}"

def fmt_num(v):
    if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if v >= 1_000:     return f"{v/1_000:.0f}K"
    return f"{v:,.0f}"

def kpi_card(label, value, delta=None, color=None, prefix="", suffix=""):
    color = color or C["accent"]
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta >= 0 else "delta-down"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {abs(delta):.1f}%</div>'
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{prefix}{value}{suffix}</div>
      {delta_html}
    </div>"""

def story(text):
    st.markdown(f'<div class="story-box">💡 {text}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">✅ {text}</div>', unsafe_allow_html=True)

def alert(text):
    st.markdown(f'<div class="alert-box">⚠️ {text}</div>', unsafe_allow_html=True)

def section(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 MedBilling Intelligence")
    st.markdown("*Healthcare Data Analytics Platform*")
    st.markdown("---")

    st.markdown("### 📊 Navigation")
    page = st.radio("", [
        "🏠 Executive Overview",
        "💰 Revenue & Claims",
        "🏦 Medical Billing Deep-Dive",
        "👥 Patient Demographics",
        "🔬 Clinical Operations",
        "💊 Procedures & Labs",
        "📅 Appointment Analytics",
        "⚠️ Prior Authorizations",
        "🏥 Facility Intelligence",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🔧 Filters")

    # Year filter for appointment ops
    appt_df_raw = load_gold("kpi_appointment_ops")
    if not appt_df_raw.empty:
        years = sorted(appt_df_raw["year"].dropna().unique().astype(int))
        year_range = st.select_slider(
            "Year Range",
            options=years,
            value=(max(years[-5], years[0]), years[-1]) if len(years) >= 5 else (years[0], years[-1])
        )
    else:
        year_range = (2022, 2025)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#6B7280; text-align:center;'>
    📁 Data: Delta Lake Medallion<br>
    🔄 Refreshed: Live Parquet<br>
    ��️ Stack: PySpark + Delta + Airflow
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Executive Overview":
    st.markdown("# 🏥 Medical Billing Intelligence Platform")
    st.markdown("*Real-time analytics across the full revenue cycle — from patient intake to payment collection*")
    st.markdown("---")

    # Load data
    patients   = load_silver("patients",   ["patient_id","age","gender","state","race","smoking_status","bmi_category"])
    claim_lines = load_silver("claim_lines", ["billed_amount","paid_amount","allowed_amount","service_date","cpt_code"])
    vitals     = load_silver("vitals",      ["systolic_bp","diastolic_bp","heart_rate","bmi","temperature_f"])
    prior_auth = load_silver("prior_authorizations", ["status","service_type","request_date"])
    appt_ops   = load_gold("kpi_appointment_ops")
    proc_rev   = load_gold("kpi_procedure_revenue")

    # Compute headline KPIs
    total_patients = len(patients) if not patients.empty else 0
    total_revenue  = claim_lines["paid_amount"].sum() if not claim_lines.empty else 0
    total_billed   = claim_lines["billed_amount"].sum() if not claim_lines.empty else 0
    collection_rate = (total_revenue / total_billed * 100) if total_billed > 0 else 0
    total_claims   = len(claim_lines) if not claim_lines.empty else 0

    auth_approved = 0
    auth_total = 0
    if not prior_auth.empty:
        auth_total = len(prior_auth)
        auth_approved = (prior_auth["status"] == "APPROVED").sum()
    auth_rate = (auth_approved / auth_total * 100) if auth_total > 0 else 0

    completion_rate = 0
    if not appt_ops.empty:
        total_sched = appt_ops["total_scheduled"].sum()
        total_comp  = appt_ops["completed"].sum()
        completion_rate = (total_comp / total_sched * 100) if total_sched > 0 else 0

    story("This executive overview consolidates the entire medical billing operation into a single command centre. "
          "Every metric below is derived from the live Delta Lake medallion pipeline — Bronze ingestion → Silver cleaning → Gold aggregation.")

    # KPI row 1
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Total Patients", fmt_num(total_patients), color=C["accent"]), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Total Revenue", fmt_currency(total_revenue), color=C["success"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Collection Rate", f"{collection_rate:.1f}", suffix="%", color=C["teal"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Auth Approval Rate", f"{auth_rate:.1f}", suffix="%", color=C["gold"]), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Appt Completion", f"{completion_rate:.1f}", suffix="%", color=C["purple"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row 2
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Claim Lines", fmt_num(total_claims), color=C["accent"]), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Total Billed", fmt_currency(total_billed), color=C["warning"]), unsafe_allow_html=True)
    avg_paid = claim_lines["paid_amount"].mean() if not claim_lines.empty else 0
    with c3: st.markdown(kpi_card("Avg Paid/Claim", fmt_currency(avg_paid), color=C["teal"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Prior Auths", fmt_num(auth_total), color=C["gold"]), unsafe_allow_html=True)
    no_show_rate = 0
    if not appt_ops.empty:
        no_show_rate = appt_ops["no_show"].sum() / appt_ops["total_scheduled"].sum() * 100
    with c5: st.markdown(kpi_card("No-Show Rate", f"{no_show_rate:.1f}", suffix="%", color=C["danger"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Charts row
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section("📈 Monthly Revenue Trend")
        if not claim_lines.empty:
            cl = claim_lines.copy()
            cl["service_date"] = pd.to_datetime(cl["service_date"], errors="coerce")
            cl = cl.dropna(subset=["service_date"])
            cl["ym"] = cl["service_date"].dt.to_period("M").astype(str)
            monthly = cl.groupby("ym").agg(
                billed=("billed_amount","sum"),
                paid=("paid_amount","sum"),
                allowed=("allowed_amount","sum")
            ).reset_index().sort_values("ym").tail(36)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["billed"], name="Billed",
                line=dict(color=C["warning"], width=2), fill="tozeroy",
                fillcolor="rgba(255,152,0,0.08)"))
            fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["allowed"], name="Allowed",
                line=dict(color=C["accent"], width=2)))
            fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["paid"], name="Paid",
                line=dict(color=C["success"], width=2.5), fill="tozeroy",
                fillcolor="rgba(0,200,83,0.1)"))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=300,
                margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(showgrid=False, tickangle=-45, nticks=12),
                yaxis=dict(showgrid=True, gridcolor="#1F2937"),
            )
            st.plotly_chart(fig, use_container_width=True)
            story("Revenue trend shows the gap between billed and paid amounts — the difference represents contractual adjustments, "
                  "denials, and patient responsibility. A narrowing gap indicates improving collection efficiency.")

    with col_right:
        section("🥧 Revenue by Procedure Category")
        if not proc_rev.empty:
            top = proc_rev.nlargest(8, "total_paid").copy()
            top["label"] = top["cpt_code"] + " - " + top["procedure_description"].str[:20].fillna("")
            fig = px.pie(top, values="total_paid", names="label",
                         color_discrete_sequence=QUAL,
                         hole=0.55)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_size=9)
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=0,r=0,t=10,b=0),
                showlegend=False,
            )
            fig.add_annotation(text=f"<b>{fmt_currency(proc_rev['total_paid'].sum())}</b><br>Total",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=13, color="white"))
            st.plotly_chart(fig, use_container_width=True)

    # Bottom row
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        section("👥 Patient Age Distribution")
        if not patients.empty:
            fig = px.histogram(patients, x="age", nbins=30,
                               color_discrete_sequence=[C["accent"]],
                               labels={"age":"Age","count":"Patients"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=220,
                              margin=dict(l=0,r=0,t=10,b=0),
                              bargap=0.05)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("📋 Prior Auth Status")
        if not prior_auth.empty:
            status_counts = prior_auth["status"].value_counts().reset_index()
            status_counts.columns = ["status","count"]
            colors_map = {"APPROVED":C["success"],"DENIED":C["danger"],
                          "PENDING":C["warning"],"EXPIRED":C["muted"],"APPEALED":C["purple"]}
            fig = px.bar(status_counts, x="status", y="count",
                         color="status",
                         color_discrete_map=colors_map,
                         labels={"count":"Count","status":"Status"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=220,
                              margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_c:
        section("📅 Appointment Outcomes")
        if not appt_ops.empty:
            outcomes = {
                "Completed": int(appt_ops["completed"].sum()),
                "Cancelled": int(appt_ops["cancelled"].sum()),
                "No Show":   int(appt_ops["no_show"].sum()),
            }
            fig = go.Figure(go.Pie(
                labels=list(outcomes.keys()),
                values=list(outcomes.values()),
                hole=0.6,
                marker_colors=[C["success"], C["warning"], C["danger"]],
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=220, margin=dict(l=0,r=0,t=10,b=0),
                              showlegend=True,
                              legend=dict(orientation="h", y=-0.1, font_size=10))
            st.plotly_chart(fig, use_container_width=True)

    if collection_rate < 70:
        alert(f"Collection rate of {collection_rate:.1f}% is below the industry benchmark of 75-85%. "
              "Review denial patterns and patient responsibility follow-up processes.")
    elif collection_rate >= 80:
        insight(f"Collection rate of {collection_rate:.1f}% is above the 80% benchmark — strong revenue cycle performance.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — REVENUE & CLAIMS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💰 Revenue & Claims":
    st.markdown("# 💰 Revenue & Claims Analytics")
    st.markdown("*Deep-dive into billing performance, collection efficiency, and claim-line economics*")
    st.markdown("---")

    claim_lines = load_silver("claim_lines",
        ["billed_amount","paid_amount","allowed_amount","patient_responsibility",
         "service_date","cpt_code","line_status","adjustment_amount","deductible_amount"])

    if claim_lines.empty:
        st.warning("No claim lines data available.")
        st.stop()

    cl = claim_lines.copy()
    cl["service_date"] = pd.to_datetime(cl["service_date"], errors="coerce")
    cl = cl.dropna(subset=["service_date"])
    cl["year"]  = cl["service_date"].dt.year
    cl["month"] = cl["service_date"].dt.month
    cl["ym"]    = cl["service_date"].dt.to_period("M").astype(str)
    cl["quarter"] = cl["service_date"].dt.to_period("Q").astype(str)

    # Filter by year range
    cl = cl[(cl["year"] >= year_range[0]) & (cl["year"] <= year_range[1])]

    total_billed  = cl["billed_amount"].sum()
    total_allowed = cl["allowed_amount"].sum()
    total_paid    = cl["paid_amount"].sum()
    total_pt_resp = cl["patient_responsibility"].sum()
    total_adj     = cl["adjustment_amount"].sum()
    collection_rate = total_paid / total_billed * 100 if total_billed > 0 else 0
    allowed_rate    = total_allowed / total_billed * 100 if total_billed > 0 else 0

    story("The revenue waterfall tells the story of every dollar billed: from gross charges, through contractual adjustments "
          "and allowed amounts, down to what was actually collected. The gap between billed and paid is not waste — "
          "it reflects payer contracts, patient cost-sharing, and write-offs.")

    # KPIs
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Gross Billed",    fmt_currency(total_billed),  color=C["warning"]), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Allowed",         fmt_currency(total_allowed), color=C["accent"]),  unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Collected",       fmt_currency(total_paid),    color=C["success"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Patient Resp.",   fmt_currency(total_pt_resp), color=C["purple"]),  unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Collection Rate", f"{collection_rate:.1f}",    suffix="%", color=C["teal"]), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Allowed Rate",    f"{allowed_rate:.1f}",       suffix="%", color=C["gold"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue waterfall
    col1, col2 = st.columns([2,1])
    with col1:
        section("💧 Revenue Waterfall — Where Does the Money Go?")
        waterfall_vals = [
            total_billed,
            -(total_billed - total_allowed),
            -(total_allowed - total_paid - total_pt_resp),
            -total_pt_resp,
            total_paid,
        ]
        waterfall_labels = ["Gross Billed","Contractual Adj.","Denials & Write-offs","Patient Resp.","Net Collected"]
        colors_wf = [C["warning"], C["danger"], C["danger"], C["purple"], C["success"]]
        fig = go.Figure(go.Waterfall(
            name="Revenue",
            orientation="v",
            measure=["absolute","relative","relative","relative","total"],
            x=waterfall_labels,
            y=waterfall_vals,
            connector=dict(line=dict(color="#374151", width=1)),
            decreasing=dict(marker_color=C["danger"]),
            increasing=dict(marker_color=C["success"]),
            totals=dict(marker_color=C["success"]),
            text=[fmt_currency(abs(v)) for v in waterfall_vals],
            textposition="outside",
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320,
                          margin=dict(l=0,r=0,t=20,b=0),
                          yaxis=dict(showgrid=True, gridcolor="#1F2937"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📊 Revenue Composition")
        labels = ["Net Collected","Patient Resp.","Contractual Adj.","Write-offs"]
        values = [
            total_paid,
            total_pt_resp,
            total_billed - total_allowed,
            max(0, total_allowed - total_paid - total_pt_resp),
        ]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.6,
            marker_colors=[C["success"], C["purple"], C["warning"], C["danger"]],
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=320, margin=dict(l=0,r=0,t=20,b=0),
                          legend=dict(orientation="v", font_size=10))
        fig.add_annotation(text=f"<b>{fmt_currency(total_billed)}</b><br>Gross",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=12, color="white"))
        st.plotly_chart(fig, use_container_width=True)

    # Monthly trend
    section("📈 Monthly Revenue Trend (Billed vs Allowed vs Collected)")
    monthly = cl.groupby("ym").agg(
        billed=("billed_amount","sum"),
        allowed=("allowed_amount","sum"),
        paid=("paid_amount","sum"),
        claims=("cpt_code","count"),
    ).reset_index().sort_values("ym")
    monthly["collection_rate"] = monthly["paid"] / monthly["billed"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["ym"], y=monthly["billed"], name="Billed",
                         marker_color=C["warning"], opacity=0.6), secondary_y=False)
    fig.add_trace(go.Bar(x=monthly["ym"], y=monthly["paid"], name="Collected",
                         marker_color=C["success"], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["collection_rate"],
                             name="Collection %", line=dict(color=C["teal"], width=2.5),
                             mode="lines+markers", marker_size=4), secondary_y=True)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=300,
                      margin=dict(l=0,r=0,t=10,b=0), barmode="overlay",
                      legend=dict(orientation="h", y=1.1),
                      xaxis=dict(showgrid=False, tickangle=-45, nticks=18))
    fig.update_yaxes(title_text="Amount ($)", secondary_y=False, showgrid=True, gridcolor="#1F2937")
    fig.update_yaxes(title_text="Collection Rate (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Top CPT codes
    col_a, col_b = st.columns(2)
    with col_a:
        section("🏆 Top 15 CPT Codes by Revenue")
        top_cpt = cl.groupby("cpt_code").agg(
            total_paid=("paid_amount","sum"),
            total_billed=("billed_amount","sum"),
            count=("cpt_code","count"),
        ).reset_index().nlargest(15, "total_paid")
        top_cpt["reimbursement_rate"] = top_cpt["total_paid"] / top_cpt["total_billed"] * 100
        fig = px.bar(top_cpt, x="total_paid", y="cpt_code", orientation="h",
                     color="reimbursement_rate",
                     color_continuous_scale="RdYlGn",
                     labels={"total_paid":"Total Paid","cpt_code":"CPT Code","reimbursement_rate":"Reimb. Rate %"},
                     text=top_cpt["total_paid"].apply(fmt_currency))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=380,
                          margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(categoryorder="total ascending"),
                          coloraxis_showscale=True)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("📉 Line Status Distribution")
        if "line_status" in cl.columns:
            status = cl["line_status"].value_counts().head(10).reset_index()
            status.columns = ["status","count"]
            fig = px.bar(status, x="count", y="status", orientation="h",
                         color="count", color_continuous_scale="Blues",
                         labels={"count":"Claim Lines","status":"Status"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=380,
                              margin=dict(l=0,r=0,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Insights
    best_month = monthly.loc[monthly["paid"].idxmax(), "ym"] if not monthly.empty else "N/A"
    insight(f"Best revenue month: **{best_month}** with {fmt_currency(monthly['paid'].max() if not monthly.empty else 0)} collected. "
            f"Average monthly collection: {fmt_currency(monthly['paid'].mean() if not monthly.empty else 0)}.")

    if collection_rate < 70:
        alert(f"Collection rate of {collection_rate:.1f}% is below the 75% industry floor. "
              "Investigate top denial reasons and accelerate patient balance follow-up.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PATIENT DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Patient Demographics":
    st.markdown("# 👥 Patient Demographics & Population Health")
    st.markdown("*Understanding who your patients are — age, gender, geography, risk factors, and social determinants*")
    st.markdown("---")

    patients = load_silver("patients", [
        "patient_id","age","gender","race","ethnicity","state","zip_code",
        "smoking_status","bmi_category","income_bracket","employment_status",
        "marital_status","preferred_language"
    ])

    if patients.empty:
        st.warning("No patient data available.")
        st.stop()

    pt = patients.copy()
    pt["age"] = pd.to_numeric(pt["age"], errors="coerce")
    pt = pt.dropna(subset=["age"])
    pt["age_group"] = pd.cut(pt["age"],
        bins=[0,17,34,49,64,79,120],
        labels=["0-17","18-34","35-49","50-64","65-79","80+"])

    story("Population health starts with knowing your patients. This page reveals the demographic composition, "
          "geographic spread, and social determinants of health across your patient panel — "
          "critical inputs for care management, resource planning, and value-based care contracts.")

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Total Patients", fmt_num(len(pt)), color=C["accent"]), unsafe_allow_html=True)
    avg_age = pt["age"].mean()
    with c2: st.markdown(kpi_card("Avg Age", f"{avg_age:.0f}", suffix=" yrs", color=C["teal"]), unsafe_allow_html=True)
    pct_65 = (pt["age"] >= 65).mean() * 100
    with c3: st.markdown(kpi_card("Age 65+", f"{pct_65:.1f}", suffix="%", color=C["gold"]), unsafe_allow_html=True)
    pct_female = (pt["gender"].str.upper() == "FEMALE").mean() * 100
    with c4: st.markdown(kpi_card("Female", f"{pct_female:.1f}", suffix="%", color=C["purple"]), unsafe_allow_html=True)
    pct_smoker = (pt["smoking_status"].str.upper() == "CURRENT").mean() * 100
    with c5: st.markdown(kpi_card("Current Smokers", f"{pct_smoker:.1f}", suffix="%", color=C["danger"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        section("🎂 Age Group Distribution")
        age_grp = pt["age_group"].value_counts().sort_index().reset_index()
        age_grp.columns = ["age_group","count"]
        fig = px.bar(age_grp, x="age_group", y="count",
                     color="count", color_continuous_scale="Blues",
                     labels={"age_group":"Age Group","count":"Patients"},
                     text="count")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚧ Gender Distribution")
        gender = pt["gender"].str.title().value_counts().reset_index()
        gender.columns = ["gender","count"]
        fig = px.pie(gender, values="count", names="gender", hole=0.55,
                     color_discrete_sequence=[C["accent"], C["purple"], C["teal"]])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        section("🌍 Race / Ethnicity")
        race = pt["race"].str.title().value_counts().head(8).reset_index()
        race.columns = ["race","count"]
        fig = px.bar(race, x="count", y="race", orientation="h",
                     color="count", color_continuous_scale="Teal",
                     labels={"count":"Patients","race":"Race"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(categoryorder="total ascending"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        section("🗺️ Patients by State (Top 20)")
        state_ct = pt["state"].value_counts().head(20).reset_index()
        state_ct.columns = ["state","count"]
        fig = px.bar(state_ct, x="state", y="count",
                     color="count", color_continuous_scale="Blues",
                     labels={"state":"State","count":"Patients"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("💰 Income Bracket Distribution")
        income = pt["income_bracket"].value_counts().reset_index()
        income.columns = ["bracket","count"]
        income_order = ["<$25k","$25k-$50k","$50k-$75k","$75k-$100k",">$100k"]
        income["bracket"] = pd.Categorical(income["bracket"], categories=income_order, ordered=True)
        income = income.sort_values("bracket")
        fig = px.bar(income, x="bracket", y="count",
                     color="count", color_continuous_scale="Greens",
                     labels={"bracket":"Income Bracket","count":"Patients"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_c, col_d, col_e = st.columns(3)

    with col_c:
        section("🚬 Smoking Status")
        smoke = pt["smoking_status"].str.title().value_counts().reset_index()
        smoke.columns = ["status","count"]
        fig = px.pie(smoke, values="count", names="status", hole=0.5,
                     color_discrete_sequence=[C["danger"], C["warning"], C["success"], C["muted"]])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=250, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        section("⚖️ BMI Category")
        bmi = pt["bmi_category"].str.title().value_counts().reset_index()
        bmi.columns = ["category","count"]
        bmi_colors = {"Normal":C["success"],"Overweight":C["warning"],
                      "Obese":C["danger"],"Morbidly Obese":"#B71C1C","Underweight":C["teal"]}
        fig = px.bar(bmi, x="category", y="count",
                     color="category", color_discrete_map=bmi_colors,
                     labels={"category":"BMI Category","count":"Patients"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=250,
                          margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_e:
        section("💼 Employment Status")
        emp = pt["employment_status"].str.title().value_counts().reset_index()
        emp.columns = ["status","count"]
        fig = px.pie(emp, values="count", names="status", hole=0.5,
                     color_discrete_sequence=QUAL)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=250, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    pct_obese = pt["bmi_category"].str.upper().isin(["OBESE","MORBIDLY OBESE"]).mean() * 100
    if pct_obese > 35:
        alert(f"{pct_obese:.1f}% of patients are obese or morbidly obese — significantly above the national average of 30%. "
              "Consider targeted chronic disease management programs.")
    if pct_65 > 40:
        insight(f"{pct_65:.1f}% of patients are 65+ — a Medicare-heavy panel. "
                "Ensure HCC coding accuracy and chronic condition documentation for risk adjustment.")


elif page == "🔬 Clinical Operations":
    st.markdown("# 🔬 Clinical Operations")
    st.markdown("*Vitals monitoring, encounter patterns, and clinical quality indicators*")
    st.markdown("---")

    vitals = load_silver("vitals", ["patient_id","encounter_id","recorded_at",
                                     "systolic_bp","diastolic_bp","heart_rate",
                                     "bmi","temperature_f","oxygen_saturation",
                                     "height_inches","weight_lbs","pain_scale"])
    procs  = load_silver("procedures", ["procedure_id","patient_id","cpt_code",
                                         "procedure_description","procedure_charge",
                                         "units","created_at"])

    story("Clinical operations data reveals the health status of your patient population and the efficiency of care delivery. "
          "Vital sign distributions help identify population-level health trends and flag patients needing intervention.")

    if not vitals.empty:
        # KPIs
        avg_sbp  = vitals["systolic_bp"].dropna().mean()
        avg_dbp  = vitals["diastolic_bp"].dropna().mean()
        avg_hr   = vitals["heart_rate"].dropna().mean()
        avg_bmi  = vitals["bmi"].dropna().mean()
        avg_temp = vitals["temperature_f"].dropna().mean()
        avg_o2   = vitals["oxygen_saturation"].dropna().mean() if "oxygen_saturation" in vitals.columns else 0

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: st.markdown(kpi_card("Avg Systolic BP", f"{avg_sbp:.0f}", suffix=" mmHg",
                                       color=C["danger"] if avg_sbp > 130 else C["success"]), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Avg Diastolic BP", f"{avg_dbp:.0f}", suffix=" mmHg",
                                       color=C["warning"] if avg_dbp > 80 else C["success"]), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Avg Heart Rate", f"{avg_hr:.0f}", suffix=" bpm",
                                       color=C["teal"]), unsafe_allow_html=True)
        with c4: st.markdown(kpi_card("Avg BMI", f"{avg_bmi:.1f}",
                                       color=C["warning"] if avg_bmi > 25 else C["success"]), unsafe_allow_html=True)
        with c5: st.markdown(kpi_card("Avg Temperature", f"{avg_temp:.1f}", suffix=" °F",
                                       color=C["accent"]), unsafe_allow_html=True)
        with c6: st.markdown(kpi_card("Vital Records", fmt_num(len(vitals)),
                                       color=C["purple"]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Vital distributions
        section("📊 Vital Sign Distributions")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(vitals.dropna(subset=["systolic_bp"]),
                               x="systolic_bp", nbins=40,
                               color_discrete_sequence=[C["danger"]],
                               labels={"systolic_bp":"Systolic BP (mmHg)"})
            fig.add_vline(x=120, line_dash="dash", line_color=C["success"],
                          annotation_text="Normal 120", annotation_position="top right")
            fig.add_vline(x=130, line_dash="dash", line_color=C["warning"],
                          annotation_text="High 130")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=240,
                              margin=dict(l=0,r=0,t=10,b=0), bargap=0.02,
                              title=dict(text="Systolic BP", font_size=12))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(vitals.dropna(subset=["heart_rate"]),
                               x="heart_rate", nbins=40,
                               color_discrete_sequence=[C["teal"]],
                               labels={"heart_rate":"Heart Rate (bpm)"})
            fig.add_vline(x=60, line_dash="dash", line_color=C["success"])
            fig.add_vline(x=100, line_dash="dash", line_color=C["warning"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=240,
                              margin=dict(l=0,r=0,t=10,b=0), bargap=0.02,
                              title=dict(text="Heart Rate", font_size=12))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(vitals.dropna(subset=["bmi"]),
                               x="bmi", nbins=40,
                               color_discrete_sequence=[C["purple"]],
                               labels={"bmi":"BMI"})
            fig.add_vline(x=18.5, line_dash="dash", line_color=C["teal"])
            fig.add_vline(x=25, line_dash="dash", line_color=C["success"])
            fig.add_vline(x=30, line_dash="dash", line_color=C["warning"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=240,
                              margin=dict(l=0,r=0,t=10,b=0), bargap=0.02,
                              title=dict(text="BMI Distribution", font_size=12))
            st.plotly_chart(fig, use_container_width=True)

        # BP scatter
        section("🫀 Blood Pressure Scatter — Systolic vs Diastolic")
        sample_v = vitals.dropna(subset=["systolic_bp","diastolic_bp"]).sample(
            min(5000, len(vitals)), random_state=42)
        sample_v["bp_category"] = "Normal"
        sample_v.loc[(sample_v["systolic_bp"] >= 130) | (sample_v["diastolic_bp"] >= 80), "bp_category"] = "Stage 1 HTN"
        sample_v.loc[(sample_v["systolic_bp"] >= 140) | (sample_v["diastolic_bp"] >= 90), "bp_category"] = "Stage 2 HTN"
        sample_v.loc[sample_v["systolic_bp"] >= 180, "bp_category"] = "Crisis"

        bp_colors = {"Normal":C["success"],"Stage 1 HTN":C["warning"],
                     "Stage 2 HTN":C["danger"],"Crisis":"#B71C1C"}
        fig = px.scatter(sample_v, x="systolic_bp", y="diastolic_bp",
                         color="bp_category", color_discrete_map=bp_colors,
                         opacity=0.4,
                         labels={"systolic_bp":"Systolic (mmHg)","diastolic_bp":"Diastolic (mmHg)"})
        fig.add_hline(y=80, line_dash="dash", line_color="#555")
        fig.add_vline(x=130, line_dash="dash", line_color="#555")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320,
                          margin=dict(l=0,r=0,t=10,b=0))
        fig.update_traces(marker_size=3)
        st.plotly_chart(fig, use_container_width=True)

        htn_pct = (sample_v["bp_category"] != "Normal").mean() * 100
        if htn_pct > 30:
            alert(f"{htn_pct:.1f}% of patients show hypertensive readings. "
                  "Consider population-level hypertension management programme.")
        else:
            insight(f"{100-htn_pct:.1f}% of patients have normal blood pressure readings.")

    # Procedures
    if not procs.empty:
        section("🔧 Top Procedures by Volume & Revenue")
        top_procs = procs.groupby(["cpt_code","procedure_description"]).agg(
            count=("procedure_id","count"),
            total_charge=("procedure_charge","sum"),
            avg_charge=("procedure_charge","mean"),
        ).reset_index().nlargest(15, "count")
        top_procs["label"] = top_procs["cpt_code"] + " " + top_procs["procedure_description"].str[:25].fillna("")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig = px.bar(top_procs.nlargest(12,"count"), x="count", y="label",
                         orientation="h", color="count",
                         color_continuous_scale="Blues",
                         labels={"count":"Volume","label":"Procedure"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=340,
                              margin=dict(l=0,r=0,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"),
                              coloraxis_showscale=False,
                              title=dict(text="By Volume", font_size=12))
            st.plotly_chart(fig, use_container_width=True)

        with col_p2:
            fig = px.bar(top_procs.nlargest(12,"total_charge"), x="total_charge", y="label",
                         orientation="h", color="avg_charge",
                         color_continuous_scale="RdYlGn",
                         labels={"total_charge":"Total Charge","label":"Procedure","avg_charge":"Avg Charge"},
                         text=top_procs.nlargest(12,"total_charge")["total_charge"].apply(fmt_currency))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=340,
                              margin=dict(l=0,r=0,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"),
                              title=dict(text="By Revenue", font_size=12))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PROCEDURES & LABS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💊 Procedures & Labs":
    st.markdown("# 💊 Procedures & Lab Results")
    st.markdown("*Procedure utilisation, reimbursement rates, and laboratory abnormality analysis*")
    st.markdown("---")

    proc_rev   = load_gold("kpi_procedure_revenue")
    lab_abn    = load_gold("kpi_lab_abnormals")

    story("Procedure revenue analysis identifies your highest-value services and their reimbursement efficiency. "
          "Lab abnormality rates reveal population health trends and can guide preventive care investments.")

    # Procedure Revenue
    if not proc_rev.empty:
        section("💰 Procedure Revenue Intelligence")

        total_proc_billed = proc_rev["total_billed"].sum()
        total_proc_paid   = proc_rev["total_paid"].sum()
        avg_reimb_rate    = (total_proc_paid / total_proc_billed * 100) if total_proc_billed > 0 else 0
        top_proc          = proc_rev.nlargest(1,"total_paid").iloc[0]

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi_card("Total Proc. Billed", fmt_currency(total_proc_billed), color=C["warning"]), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Total Proc. Paid",   fmt_currency(total_proc_paid),   color=C["success"]), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Avg Reimb. Rate",    f"{avg_reimb_rate:.1f}", suffix="%", color=C["teal"]), unsafe_allow_html=True)
        with c4: st.markdown(kpi_card("Unique Procedures",  str(len(proc_rev)), color=C["accent"]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([3,2])
        with col1:
            section("🏆 Top 20 Procedures — Billed vs Paid")
            top20 = proc_rev.nlargest(20,"total_paid").copy()
            top20["label"] = top20["cpt_code"] + " " + top20["procedure_description"].str[:22].fillna("")
            top20 = top20.sort_values("total_paid")

            fig = go.Figure()
            fig.add_trace(go.Bar(y=top20["label"], x=top20["total_billed"],
                                 name="Billed", orientation="h",
                                 marker_color=C["warning"], opacity=0.6))
            fig.add_trace(go.Bar(y=top20["label"], x=top20["total_paid"],
                                 name="Paid", orientation="h",
                                 marker_color=C["success"], opacity=0.9))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=480,
                              margin=dict(l=0,r=0,t=10,b=0), barmode="overlay",
                              legend=dict(orientation="h", y=1.05),
                              xaxis=dict(showgrid=True, gridcolor="#1F2937"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section("📊 Reimbursement Rate by Procedure")
            top20_r = proc_rev.nlargest(15,"utilization_count").copy()
            top20_r["label"] = top20_r["cpt_code"]
            top20_r = top20_r.sort_values("avg_reimbursement_rate_pct")
            fig = px.bar(top20_r, x="avg_reimbursement_rate_pct", y="label",
                         orientation="h",
                         color="avg_reimbursement_rate_pct",
                         color_continuous_scale="RdYlGn",
                         range_color=[0,100],
                         labels={"avg_reimbursement_rate_pct":"Reimb. Rate %","label":"CPT"})
            fig.add_vline(x=75, line_dash="dash", line_color=C["warning"],
                          annotation_text="75% target")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=480,
                              margin=dict(l=0,r=0,t=10,b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: utilization vs avg paid
        section("🔵 Procedure Utilisation vs Average Payment")
        fig = px.scatter(proc_rev, x="utilization_count", y="avg_paid",
                         size="total_paid", color="avg_reimbursement_rate_pct",
                         color_continuous_scale="RdYlGn",
                         hover_data=["cpt_code","procedure_description"],
                         labels={"utilization_count":"Utilisation Count",
                                 "avg_paid":"Avg Paid ($)",
                                 "avg_reimbursement_rate_pct":"Reimb. Rate %"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
        story("Bubble size = total revenue. High utilisation + high avg paid = your most valuable procedures. "
              "Low reimbursement rate (red) procedures may need contract renegotiation with payers.")

    # Lab Abnormals
    if not lab_abn.empty:
        st.markdown("---")
        section("🧪 Laboratory Abnormality Analysis")

        total_tests    = lab_abn["total_tests"].sum()
        total_abnormal = lab_abn["abnormal_count"].sum()
        overall_abn_rate = total_abnormal / total_tests * 100 if total_tests > 0 else 0

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi_card("Total Lab Tests",    fmt_num(total_tests),    color=C["accent"]),  unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Abnormal Results",   fmt_num(total_abnormal), color=C["danger"]),  unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Overall Abn. Rate",  f"{overall_abn_rate:.1f}", suffix="%", color=C["warning"]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            section("🔴 Top Tests by Abnormal Count")
            top_abn = lab_abn.groupby("test_name").agg(
                abnormal_count=("abnormal_count","sum"),
                total_tests=("total_tests","sum"),
            ).reset_index()
            top_abn["abn_rate"] = top_abn["abnormal_count"] / top_abn["total_tests"] * 100
            top_abn = top_abn.nlargest(15,"abnormal_count")

            fig = px.bar(top_abn, x="abnormal_count", y="test_name",
                         orientation="h", color="abn_rate",
                         color_continuous_scale="RdYlGn_r",
                         labels={"abnormal_count":"Abnormal Count","test_name":"Test","abn_rate":"Abn. Rate %"},
                         text=top_abn["abn_rate"].apply(lambda x: f"{x:.1f}%"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=380,
                              margin=dict(l=0,r=0,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col_l2:
            section("📈 Abnormality Flag Distribution")
            flag_dist = lab_abn.groupby("abnormal_flag")["abnormal_count"].sum().reset_index()
            flag_dist.columns = ["flag","count"]
            flag_colors = {
                "HIGH":C["danger"],"Low":C["accent"],"CRITICAL HIGH":"#B71C1C",
                "Critical Low":"#0D47A1","NORMAL":C["success"],
            }
            fig = px.pie(flag_dist, values="count", names="flag", hole=0.55,
                         color="flag", color_discrete_map=flag_colors)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=380, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Avg result values for top tests
        section("📊 Average Result Values — Top Abnormal Tests")
        top_tests = lab_abn.groupby("test_name")["abnormal_count"].sum().nlargest(10).index
        top_lab = lab_abn[lab_abn["test_name"].isin(top_tests)].copy()
        if "avg_result_value" in top_lab.columns:
            fig = px.box(top_lab, x="test_name", y="avg_result_value",
                         color="test_name",
                         color_discrete_sequence=QUAL,
                         labels={"test_name":"Test","avg_result_value":"Avg Result Value"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=280,
                              margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                              xaxis=dict(tickangle=-30))
            st.plotly_chart(fig, use_container_width=True)

        if overall_abn_rate > 25:
            alert(f"Overall lab abnormality rate of {overall_abn_rate:.1f}% is elevated. "
                  "Review chronic disease management protocols and preventive screening programmes.")
        else:
            insight(f"Lab abnormality rate of {overall_abn_rate:.1f}% is within acceptable range. "
                    "Continue monitoring high-frequency tests for trend changes.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — APPOINTMENT ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📅 Appointment Analytics":
    st.markdown("# 📅 Appointment Analytics")
    st.markdown("*Scheduling efficiency, no-show patterns, completion rates, and capacity utilisation*")
    st.markdown("---")

    appt = load_gold("kpi_appointment_ops")

    if appt.empty:
        st.warning("No appointment data available.")
        st.stop()

    # Filter by year range
    appt = appt[(appt["year"] >= year_range[0]) & (appt["year"] <= year_range[1])]

    total_sched  = int(appt["total_scheduled"].sum())
    total_comp   = int(appt["completed"].sum())
    total_cancel = int(appt["cancelled"].sum())
    total_noshow = int(appt["no_show"].sum())
    comp_rate    = total_comp   / total_sched * 100 if total_sched > 0 else 0
    cancel_rate  = total_cancel / total_sched * 100 if total_sched > 0 else 0
    noshow_rate  = total_noshow / total_sched * 100 if total_sched > 0 else 0
    avg_duration = appt["avg_duration_minutes"].mean()

    story("Appointment analytics is the heartbeat of operational efficiency. Every no-show is lost revenue and wasted capacity. "
          "Every cancellation is an opportunity to backfill. Understanding patterns by type and time enables smarter scheduling.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Total Scheduled", fmt_num(total_sched),  color=C["accent"]),  unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Completed",       fmt_num(total_comp),   color=C["success"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Completion Rate", f"{comp_rate:.1f}", suffix="%",
                                   color=C["success"] if comp_rate >= 70 else C["danger"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Cancellations",   fmt_num(total_cancel), color=C["warning"]), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("No-Shows",        fmt_num(total_noshow), color=C["danger"]),  unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("No-Show Rate",    f"{noshow_rate:.1f}", suffix="%",
                                   color=C["danger"] if noshow_rate > 10 else C["success"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trend
    section("📈 Monthly Appointment Trends")
    monthly_a = appt.groupby(["year","month"]).agg(
        scheduled=("total_scheduled","sum"),
        completed=("completed","sum"),
        cancelled=("cancelled","sum"),
        no_show=("no_show","sum"),
    ).reset_index()
    monthly_a["ym"] = monthly_a["year"].astype(str) + "-" + monthly_a["month"].astype(str).str.zfill(2)
    monthly_a = monthly_a.sort_values("ym")
    monthly_a["comp_rate"] = monthly_a["completed"] / monthly_a["scheduled"] * 100
    monthly_a["noshow_rate"] = monthly_a["no_show"] / monthly_a["scheduled"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly_a["ym"], y=monthly_a["scheduled"], name="Scheduled",
                         marker_color=C["accent"], opacity=0.5), secondary_y=False)
    fig.add_trace(go.Bar(x=monthly_a["ym"], y=monthly_a["completed"], name="Completed",
                         marker_color=C["success"], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly_a["ym"], y=monthly_a["comp_rate"],
                             name="Completion %", line=dict(color=C["gold"], width=2),
                             mode="lines"), secondary_y=True)
    fig.add_trace(go.Scatter(x=monthly_a["ym"], y=monthly_a["noshow_rate"],
                             name="No-Show %", line=dict(color=C["danger"], width=2, dash="dot"),
                             mode="lines"), secondary_y=True)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=320,
                      margin=dict(l=0,r=0,t=10,b=0), barmode="overlay",
                      legend=dict(orientation="h", y=1.1),
                      xaxis=dict(showgrid=False, tickangle=-45, nticks=18))
    fig.update_yaxes(title_text="Appointments", secondary_y=False, showgrid=True, gridcolor="#1F2937")
    fig.update_yaxes(title_text="Rate (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # By appointment type
    col1, col2 = st.columns(2)
    with col1:
        section("🏷️ Performance by Appointment Type")
        by_type = appt.groupby("appointment_type").agg(
            scheduled=("total_scheduled","sum"),
            completed=("completed","sum"),
            no_show=("no_show","sum"),
            cancelled=("cancelled","sum"),
        ).reset_index()
        by_type["comp_rate"]   = by_type["completed"]  / by_type["scheduled"] * 100
        by_type["noshow_rate"] = by_type["no_show"]     / by_type["scheduled"] * 100
        by_type = by_type.sort_values("scheduled", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Completed",   x=by_type["appointment_type"], y=by_type["completed"],
                             marker_color=C["success"]))
        fig.add_trace(go.Bar(name="Cancelled",   x=by_type["appointment_type"], y=by_type["cancelled"],
                             marker_color=C["warning"]))
        fig.add_trace(go.Bar(name="No Show",     x=by_type["appointment_type"], y=by_type["no_show"],
                             marker_color=C["danger"]))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320,
                          margin=dict(l=0,r=0,t=10,b=0), barmode="stack",
                          legend=dict(orientation="h", y=1.1),
                          xaxis=dict(tickangle=-30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📊 No-Show Rate by Appointment Type")
        by_type_s = by_type.sort_values("noshow_rate", ascending=True)
        fig = px.bar(by_type_s, x="noshow_rate", y="appointment_type",
                     orientation="h",
                     color="noshow_rate",
                     color_continuous_scale="RdYlGn_r",
                     range_color=[0, 20],
                     labels={"noshow_rate":"No-Show Rate %","appointment_type":"Type"},
                     text=by_type_s["noshow_rate"].apply(lambda x: f"{x:.1f}%"))
        fig.add_vline(x=10, line_dash="dash", line_color=C["warning"],
                      annotation_text="10% threshold")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320,
                          margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: month vs appointment type
    section("🗓️ Appointment Volume Heatmap — Month × Type")
    pivot = appt.pivot_table(values="total_scheduled", index="appointment_type",
                              columns="month", aggfunc="sum", fill_value=0)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    labels={"color":"Appointments"},
                    aspect="auto")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      height=300, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Duration analysis
    section("⏱️ Average Duration by Appointment Type")
    dur = appt.groupby("appointment_type")["avg_duration_minutes"].mean().reset_index()
    dur.columns = ["type","avg_duration"]
    dur = dur.sort_values("avg_duration", ascending=False)
    fig = px.bar(dur, x="type", y="avg_duration",
                 color="avg_duration", color_continuous_scale="Teal",
                 labels={"type":"Appointment Type","avg_duration":"Avg Duration (min)"},
                 text=dur["avg_duration"].apply(lambda x: f"{x:.0f} min"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=260,
                      margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False,
                      xaxis=dict(tickangle=-30))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    if noshow_rate > 10:
        alert(f"No-show rate of {noshow_rate:.1f}% exceeds the 10% industry benchmark. "
              f"That is {fmt_num(total_noshow)} missed appointments. "
              "Implement automated reminder calls/texts 24-48 hours before appointments.")
    else:
        insight(f"No-show rate of {noshow_rate:.1f}% is within acceptable range. "
                f"Completion rate of {comp_rate:.1f}% reflects strong patient engagement.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — PRIOR AUTHORIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ Prior Authorizations":
    st.markdown("# ⚠️ Prior Authorizations")
    st.markdown("*Approval rates, denial patterns, turnaround times, and payer behaviour*")
    st.markdown("---")

    pa = load_silver("prior_authorizations",
                     ["auth_id","patient_id","provider_id","payer_id","service_type",
                      "cpt_code","status","request_date","decision_date","approved_units"])

    if pa.empty:
        st.warning("No prior authorization data available.")
        st.stop()

    pa["request_date"]  = pd.to_datetime(pa["request_date"],  errors="coerce")
    pa["decision_date"] = pd.to_datetime(pa["decision_date"], errors="coerce")
    pa["turnaround_days"] = (pa["decision_date"] - pa["request_date"]).dt.days
    pa["year"]  = pa["request_date"].dt.year
    pa["month"] = pa["request_date"].dt.month
    pa["ym"]    = pa["request_date"].dt.to_period("M").astype(str)

    pa = pa[(pa["year"] >= year_range[0]) & (pa["year"] <= year_range[1])]

    total_auth   = len(pa)
    approved     = (pa["status"] == "APPROVED").sum()
    denied       = (pa["status"] == "DENIED").sum()
    pending      = (pa["status"] == "PENDING").sum()
    expired      = (pa["status"] == "EXPIRED").sum()
    appealed     = (pa["status"] == "APPEALED").sum()
    approval_rate = approved / total_auth * 100 if total_auth > 0 else 0
    denial_rate   = denied   / total_auth * 100 if total_auth > 0 else 0
    avg_tat       = pa["turnaround_days"].dropna().mean()

    story("Prior authorizations are a critical bottleneck in the revenue cycle. Delays and denials directly impact patient care "
          "and cash flow. Tracking approval rates by payer and service type reveals where to focus contract negotiations.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Total Auths",    fmt_num(total_auth), color=C["accent"]),  unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Approved",       fmt_num(approved),   color=C["success"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Approval Rate",  f"{approval_rate:.1f}", suffix="%",
                                   color=C["success"] if approval_rate >= 80 else C["warning"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Denied",         fmt_num(denied),     color=C["danger"]),  unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Denial Rate",    f"{denial_rate:.1f}", suffix="%",
                                   color=C["danger"] if denial_rate > 15 else C["success"]), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Avg Turnaround", f"{avg_tat:.1f}", suffix=" days",
                                   color=C["warning"] if avg_tat > 5 else C["success"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        section("📊 Status Distribution")
        status_c = pa["status"].value_counts().reset_index()
        status_c.columns = ["status","count"]
        sc_map = {"APPROVED":C["success"],"DENIED":C["danger"],"PENDING":C["warning"],
                  "EXPIRED":C["muted"],"APPEALED":C["purple"]}
        fig = px.pie(status_c, values="count", names="status", hole=0.55,
                     color="status", color_discrete_map=sc_map)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🏷️ Top Service Types")
        svc = pa.groupby("service_type").agg(
            total=("auth_id","count"),
            approved=("status", lambda x: (x=="APPROVED").sum()),
        ).reset_index()
        svc["approval_rate"] = svc["approved"] / svc["total"] * 100
        svc = svc.nlargest(10,"total")
        fig = px.bar(svc, x="total", y="service_type", orientation="h",
                     color="approval_rate", color_continuous_scale="RdYlGn",
                     range_color=[50,100],
                     labels={"total":"Total Auths","service_type":"Service Type",
                             "approval_rate":"Approval Rate %"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        section("⏱️ Turnaround Time Distribution")
        tat = pa["turnaround_days"].dropna()
        tat = tat[(tat >= 0) & (tat <= 60)]
        fig = px.histogram(tat, nbins=30,
                           color_discrete_sequence=[C["teal"]],
                           labels={"value":"Days","count":"Auths"})
        fig.add_vline(x=avg_tat, line_dash="dash", line_color=C["warning"],
                      annotation_text=f"Avg {avg_tat:.1f}d")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          margin=dict(l=0,r=0,t=10,b=0), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly trend
    section("📈 Monthly Authorization Trend")
    monthly_pa = pa.groupby("ym").agg(
        total=("auth_id","count"),
        approved=("status", lambda x: (x=="APPROVED").sum()),
        denied=("status",   lambda x: (x=="DENIED").sum()),
        pending=("status",  lambda x: (x=="PENDING").sum()),
    ).reset_index().sort_values("ym")
    monthly_pa["approval_rate"] = monthly_pa["approved"] / monthly_pa["total"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly_pa["ym"], y=monthly_pa["approved"], name="Approved",
                         marker_color=C["success"], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Bar(x=monthly_pa["ym"], y=monthly_pa["denied"], name="Denied",
                         marker_color=C["danger"], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Bar(x=monthly_pa["ym"], y=monthly_pa["pending"], name="Pending",
                         marker_color=C["warning"], opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly_pa["ym"], y=monthly_pa["approval_rate"],
                             name="Approval %", line=dict(color=C["gold"], width=2.5),
                             mode="lines+markers", marker_size=4), secondary_y=True)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=300,
                      margin=dict(l=0,r=0,t=10,b=0), barmode="stack",
                      legend=dict(orientation="h", y=1.1),
                      xaxis=dict(showgrid=False, tickangle=-45, nticks=18))
    fig.update_yaxes(title_text="Authorizations", secondary_y=False, showgrid=True, gridcolor="#1F2937")
    fig.update_yaxes(title_text="Approval Rate (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Payer analysis
    if "payer_id" in pa.columns:
        section("🏦 Approval Rate by Payer")
        payer_pa = pa.groupby("payer_id").agg(
            total=("auth_id","count"),
            approved=("status", lambda x: (x=="APPROVED").sum()),
            avg_tat=("turnaround_days","mean"),
        ).reset_index()
        payer_pa["approval_rate"] = payer_pa["approved"] / payer_pa["total"] * 100
        payer_pa = payer_pa[payer_pa["total"] >= 50].nlargest(15,"total")

        fig = px.scatter(payer_pa, x="avg_tat", y="approval_rate",
                         size="total", color="approval_rate",
                         color_continuous_scale="RdYlGn",
                         range_color=[50,100],
                         hover_data=["payer_id","total"],
                         labels={"avg_tat":"Avg Turnaround (days)",
                                 "approval_rate":"Approval Rate %",
                                 "total":"Total Auths"})
        fig.add_hline(y=80, line_dash="dash", line_color=C["warning"],
                      annotation_text="80% target")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
        story("Each bubble is a payer. Size = volume. Ideal payers are top-right: high approval rate, fast turnaround. "
              "Bottom-right payers (slow + low approval) need contract review.")

    if denial_rate > 15:
        alert(f"Prior auth denial rate of {denial_rate:.1f}% is above the 15% threshold. "
              f"{fmt_num(denied)} authorizations were denied. Review top denied service types and appeal outcomes.")
    else:
        insight(f"Prior auth approval rate of {approval_rate:.1f}% is strong. "
                f"Average turnaround of {avg_tat:.1f} days — target is under 5 business days.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — FACILITY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏥 Facility Intelligence":
    st.markdown("# 🏥 Facility Intelligence")
    st.markdown("*Clinic performance, network status, geographic footprint, and capacity analysis*")
    st.markdown("---")

    clinics  = load_silver("clinics", ["clinic_id","clinic_name","clinic_type","state","city",
                                        "network_status","bed_count","accreditation","emr_system"])
    dim_cl   = load_gold("dim_clinics")

    df_cl = dim_cl if not dim_cl.empty else clinics
    if df_cl.empty:
        st.warning("No clinic data available.")
        st.stop()

    total_clinics   = len(df_cl)
    in_network      = (df_cl["network_status"].str.upper() == "IN-NETWORK").sum() if "network_status" in df_cl.columns else 0
    in_network_pct  = in_network / total_clinics * 100 if total_clinics > 0 else 0
    states_covered  = df_cl["state"].nunique() if "state" in df_cl.columns else 0
    avg_beds        = df_cl["bed_count"].dropna().mean() if "bed_count" in df_cl.columns else 0

    story("Facility intelligence drives network strategy. Understanding which clinics are in-network, their geographic spread, "
          "and their capacity helps optimise patient routing, contract negotiations, and expansion planning.")

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Total Facilities",  str(total_clinics),       color=C["accent"]),  unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("In-Network",        str(int(in_network)),     color=C["success"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("In-Network Rate",   f"{in_network_pct:.1f}", suffix="%",
                                   color=C["success"] if in_network_pct >= 80 else C["warning"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("States Covered",    str(states_covered),      color=C["teal"]),    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        section("🏷️ Clinic Type Distribution")
        if "clinic_type" in df_cl.columns:
            ct = df_cl["clinic_type"].value_counts().reset_index()
            ct.columns = ["type","count"]
            fig = px.pie(ct, values="count", names="type", hole=0.5,
                         color_discrete_sequence=QUAL)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🌐 Network Status")
        if "network_status" in df_cl.columns:
            ns = df_cl["network_status"].str.title().value_counts().reset_index()
            ns.columns = ["status","count"]
            ns_colors = {"In-Network":C["success"],"Out-Of-Network":C["danger"]}
            fig = px.pie(ns, values="count", names="status", hole=0.55,
                         color="status", color_discrete_map=ns_colors)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        section("🏆 Accreditation Status")
        if "accreditation" in df_cl.columns:
            acc = df_cl["accreditation"].fillna("None").value_counts().reset_index()
            acc.columns = ["accreditation","count"]
            fig = px.bar(acc, x="count", y="accreditation", orientation="h",
                         color="count", color_continuous_scale="Greens",
                         labels={"count":"Clinics","accreditation":""})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=260,
                              margin=dict(l=0,r=0,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Geographic distribution
    section("🗺️ Facility Distribution by State")
    if "state" in df_cl.columns:
        state_cl = df_cl["state"].value_counts().reset_index()
        state_cl.columns = ["state","count"]
        fig = px.choropleth(state_cl, locations="state", locationmode="USA-states",
                            color="count", scope="usa",
                            color_continuous_scale="Blues",
                            labels={"count":"Facilities"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          geo_bgcolor="rgba(0,0,0,0)", height=340,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Clinic type vs network status
    section("📊 Clinic Type × Network Status")
    if "clinic_type" in df_cl.columns and "network_status" in df_cl.columns:
        cross = df_cl.groupby(["clinic_type","network_status"]).size().reset_index(name="count")
        fig = px.bar(cross, x="clinic_type", y="count", color="network_status",
                     color_discrete_map={"In-Network":C["success"],"Out-Of-Network":C["danger"],
                                         "In-network":C["success"],"Out-of-network":C["danger"]},
                     barmode="stack",
                     labels={"count":"Facilities","clinic_type":"Clinic Type","network_status":"Network"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          margin=dict(l=0,r=0,t=10,b=0),
                          xaxis=dict(tickangle=-20),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    # EMR system
    if "emr_system" in df_cl.columns:
        section("💻 EMR System Adoption")
        emr = df_cl["emr_system"].value_counts().reset_index()
        emr.columns = ["emr","count"]
        fig = px.bar(emr, x="emr", y="count",
                     color="count", color_continuous_scale="Purples",
                     labels={"count":"Facilities","emr":"EMR System"},
                     text=emr["count"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=240,
                          margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Bed capacity for hospitals
    if "bed_count" in df_cl.columns and "clinic_type" in df_cl.columns:
        hospitals = df_cl[df_cl["clinic_type"].str.upper() == "HOSPITAL"].dropna(subset=["bed_count"])
        if not hospitals.empty:
            section("🛏️ Hospital Bed Capacity")
            hospitals_s = hospitals.sort_values("bed_count", ascending=False).head(20)
            fig = px.bar(hospitals_s, x="clinic_name" if "clinic_name" in hospitals_s.columns else hospitals_s.index,
                         y="bed_count",
                         color="bed_count", color_continuous_scale="Blues",
                         labels={"bed_count":"Beds","clinic_name":"Hospital"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=260,
                              margin=dict(l=0,r=0,t=10,b=0),
                              xaxis=dict(tickangle=-30), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    if in_network_pct < 80:
        alert(f"Only {in_network_pct:.1f}% of facilities are in-network. "
              "Out-of-network facilities generate higher patient cost-sharing and lower reimbursement rates.")
    else:
        insight(f"{in_network_pct:.1f}% in-network rate across {states_covered} states. "
                "Strong network coverage supports patient access and competitive reimbursement.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 9 — MEDICAL BILLING DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏦 Medical Billing Deep-Dive":
    st.markdown("# 🏦 Medical Billing Deep-Dive")
    st.markdown("*Complete revenue cycle analysis: AR aging, denial management, adjustment codes, payer performance, and collection efficiency*")
    st.markdown("---")

    # ── Load data ────────────────────────────────────────────────────────────
    cl = load_silver("claim_lines", [
        "claim_line_id","claim_id","cpt_code","service_date","units",
        "billed_amount","allowed_amount","paid_amount","patient_responsibility",
        "coinsurance_amount","copay_amount","deductible_amount","adjustment_amount",
        "adjustment_reason_code","line_status","remark_code",
    ])
    pa = load_silver("prior_authorizations", [
        "auth_id","payer_id","service_type","status","denial_reason",
        "urgency","request_date","decision_date","approved_units",
    ])
    proc_rev = load_gold("kpi_procedure_revenue")

    if cl.empty:
        st.warning("No claim lines data available.")
        st.stop()

    # ── Pre-process ───────────────────────────────────────────────────────────
    cl = cl.copy()
    cl["service_date"] = pd.to_datetime(cl["service_date"], errors="coerce")
    cl = cl.dropna(subset=["service_date"])
    cl["year"]  = cl["service_date"].dt.year
    cl["month"] = cl["service_date"].dt.month
    cl["ym"]    = cl["service_date"].dt.to_period("M").astype(str)
    cl["days_since"] = (pd.Timestamp.now() - cl["service_date"]).dt.days

    # Normalise line_status (DQ issues injected mixed case)
    cl["status_clean"] = cl["line_status"].str.strip().str.title()
    cl.loc[cl["status_clean"].str.contains("Paid", na=False), "status_clean"] = "Paid"
    cl.loc[cl["status_clean"].str.contains("Denied", na=False), "status_clean"] = "Denied"
    cl.loc[cl["status_clean"].str.contains("Pending", na=False), "status_clean"] = "Pending"
    cl.loc[cl["status_clean"].str.contains("Rejected", na=False), "status_clean"] = "Rejected"
    cl.loc[cl["status_clean"].str.contains("Appealed", na=False), "status_clean"] = "Appealed"
    cl.loc[cl["status_clean"].str.contains("Void", na=False), "status_clean"] = "Void"
    cl.loc[cl["status_clean"].str.contains("Adjusted", na=False), "status_clean"] = "Adjusted"

    # AR aging buckets (outstanding = not paid/void)
    outstanding = cl[~cl["status_clean"].isin(["Paid","Void","Adjusted"])].copy()
    def aging_bucket(d):
        if d <= 30:  return "0-30 days"
        elif d <= 60: return "31-60 days"
        elif d <= 90: return "61-90 days"
        elif d <= 120: return "91-120 days"
        else:         return "120+ days"
    outstanding["aging_bucket"] = outstanding["days_since"].apply(aging_bucket)
    BUCKET_ORDER = ["0-30 days","31-60 days","61-90 days","91-120 days","120+ days"]

    # ── Headline KPIs ─────────────────────────────────────────────────────────
    total_billed   = cl["billed_amount"].sum()
    total_allowed  = cl["allowed_amount"].sum()
    total_paid     = cl["paid_amount"].sum()
    total_pt_resp  = cl["patient_responsibility"].sum()
    total_adj      = cl["adjustment_amount"].sum()
    total_ar       = outstanding["billed_amount"].sum()
    denied_amt     = cl[cl["status_clean"]=="Denied"]["billed_amount"].sum()
    collection_rt  = total_paid / total_billed * 100 if total_billed > 0 else 0
    denial_rt      = cl[cl["status_clean"]=="Denied"]["billed_amount"].sum() / total_billed * 100 if total_billed > 0 else 0
    adj_rt         = total_adj / total_billed * 100 if total_billed > 0 else 0
    ar_days        = outstanding["days_since"].mean() if not outstanding.empty else 0

    story("The medical billing deep-dive exposes every dollar in your revenue cycle. "
          "From the moment a claim line is created to final payment posting, this page tracks "
          "AR aging, denial root causes, adjustment reason codes, payer-level performance, "
          "and the true cost of write-offs. Use this to drive denial prevention and accelerate cash flow.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Gross Billed",    fmt_currency(total_billed),  color=C["warning"]), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Net Collected",   fmt_currency(total_paid),    color=C["success"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Collection Rate", f"{collection_rt:.1f}", suffix="%",
                                   color=C["success"] if collection_rt>=75 else C["danger"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Total AR",        fmt_currency(total_ar),      color=C["danger"]),  unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Denial Rate",     f"{denial_rt:.1f}", suffix="%",
                                   color=C["danger"] if denial_rt>10 else C["success"]), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Avg AR Days",     f"{ar_days:.0f}", suffix=" d",
                                   color=C["danger"] if ar_days>45 else C["success"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TAB LAYOUT ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Revenue Cycle Funnel",
        "⏳ AR Aging",
        "❌ Denial Analysis",
        "🔧 Adjustment Codes",
        "🏦 Payer Performance",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — REVENUE CYCLE FUNNEL
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 💧 Revenue Cycle Waterfall")
        story("Every dollar billed flows through a series of reductions before reaching your bank account. "
              "Contractual adjustments are expected — they reflect payer contracts. "
              "Denials and write-offs are preventable losses. Patient responsibility is collectible with the right follow-up.")

        col_wf, col_comp = st.columns([3, 2])
        with col_wf:
            contractual_adj = total_billed - total_allowed
            write_offs      = max(0, total_allowed - total_paid - total_pt_resp)
            wf_vals   = [total_billed, -contractual_adj, -write_offs, -total_pt_resp, total_paid]
            wf_labels = ["Gross Billed", "Contractual Adj.", "Denials/Write-offs", "Patient Resp.", "Net Collected"]
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute","relative","relative","relative","total"],
                x=wf_labels, y=wf_vals,
                connector=dict(line=dict(color="#374151", width=1)),
                decreasing=dict(marker_color=C["danger"]),
                increasing=dict(marker_color=C["success"]),
                totals=dict(marker_color=C["success"]),
                text=[fmt_currency(abs(v)) for v in wf_vals],
                textposition="outside",
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=340,
                              margin=dict(l=0,r=0,t=20,b=0),
                              yaxis=dict(showgrid=True, gridcolor="#1F2937"))
            st.plotly_chart(fig, use_container_width=True)

        with col_comp:
            st.markdown("#### Revenue Composition")
            labels = ["Net Collected","Patient Resp.","Contractual Adj.","Write-offs"]
            values = [total_paid, total_pt_resp, contractual_adj, write_offs]
            colors = [C["success"], C["purple"], C["warning"], C["danger"]]
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.6,
                                   marker_colors=colors))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=200, margin=dict(l=0,r=0,t=10,b=0))
            fig.add_annotation(text=f"<b>{fmt_currency(total_billed)}</b><br>Gross",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=12, color="white"))
            st.plotly_chart(fig, use_container_width=True)

            # Metrics table
            metrics = {
                "Gross Billed":       fmt_currency(total_billed),
                "Contractual Adj.":   fmt_currency(contractual_adj),
                "Allowed Amount":     fmt_currency(total_allowed),
                "Net Collected":      fmt_currency(total_paid),
                "Patient Resp.":      fmt_currency(total_pt_resp),
                "Write-offs":         fmt_currency(write_offs),
                "Collection Rate":    f"{collection_rt:.1f}%",
                "Allowed Rate":       f"{total_allowed/total_billed*100:.1f}%" if total_billed>0 else "N/A",
            }
            for k,v in metrics.items():
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1F2937;font-size:0.82rem'>"
                            f"<span style='color:#9E9E9E'>{k}</span><span style='color:#FAFAFA;font-weight:600'>{v}</span></div>",
                            unsafe_allow_html=True)

        # Monthly trend
        st.markdown("#### Monthly Revenue Trend")
        monthly = cl.groupby("ym").agg(
            billed=("billed_amount","sum"),
            allowed=("allowed_amount","sum"),
            paid=("paid_amount","sum"),
            pt_resp=("patient_responsibility","sum"),
        ).reset_index().sort_values("ym")
        monthly["collection_rate"] = monthly["paid"] / monthly["billed"] * 100
        monthly["denial_rate"]     = (monthly["billed"] - monthly["allowed"]) / monthly["billed"] * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["ym"], y=monthly["billed"], name="Billed",
                             marker_color=C["warning"], opacity=0.5), secondary_y=False)
        fig.add_trace(go.Bar(x=monthly["ym"], y=monthly["paid"], name="Collected",
                             marker_color=C["success"], opacity=0.85), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["collection_rate"],
                                 name="Collection %", line=dict(color=C["teal"], width=2.5),
                                 mode="lines+markers", marker_size=4), secondary_y=True)
        fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["denial_rate"],
                                 name="Adj. Rate %", line=dict(color=C["danger"], width=1.5, dash="dot"),
                                 mode="lines"), secondary_y=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(l=0,r=0,t=10,b=0), barmode="overlay",
                          legend=dict(orientation="h", y=1.1),
                          xaxis=dict(showgrid=False, tickangle=-45, nticks=18))
        fig.update_yaxes(title_text="Amount ($)", secondary_y=False, showgrid=True, gridcolor="#1F2937")
        fig.update_yaxes(title_text="Rate (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # Claim line status breakdown
        st.markdown("#### Claim Line Status — Volume & Revenue")
        status_rev = cl.groupby("status_clean").agg(
            count=("claim_line_id","count"),
            billed=("billed_amount","sum"),
            paid=("paid_amount","sum"),
        ).reset_index().sort_values("billed", ascending=False)
        status_rev["collection_rate"] = status_rev["paid"] / status_rev["billed"] * 100

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sc_colors = {"Paid":C["success"],"Denied":C["danger"],"Pending":C["warning"],
                         "Rejected":"#FF5722","Appealed":C["purple"],"Void":C["muted"],"Adjusted":C["teal"]}
            fig = px.bar(status_rev, x="status_clean", y="billed",
                         color="status_clean", color_discrete_map=sc_colors,
                         text=status_rev["billed"].apply(fmt_currency),
                         labels={"billed":"Billed Amount","status_clean":"Status"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=280,
                              margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col_s2:
            fig = px.bar(status_rev, x="status_clean", y="count",
                         color="collection_rate", color_continuous_scale="RdYlGn",
                         range_color=[0,100],
                         text=status_rev["count"].apply(fmt_num),
                         labels={"count":"Claim Lines","status_clean":"Status","collection_rate":"Coll. Rate %"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=280,
                              margin=dict(l=0,r=0,t=10,b=0))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — AR AGING
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### ⏳ Accounts Receivable Aging")
        story("AR aging shows how long outstanding claims have been unpaid. "
              "Claims over 90 days are at high risk of write-off. "
              "The 120+ bucket is your most urgent collection priority.")

        aging = outstanding.groupby("aging_bucket").agg(
            claim_count=("claim_line_id","count"),
            total_outstanding=("billed_amount","sum"),
            avg_days=("days_since","mean"),
        ).reindex(BUCKET_ORDER).reset_index()
        aging.columns = ["bucket","claim_count","total_outstanding","avg_days"]

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi_card("Total AR",       fmt_currency(total_ar),                color=C["danger"]),  unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("AR Claims",      fmt_num(len(outstanding)),             color=C["warning"]), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Avg AR Days",    f"{ar_days:.0f}", suffix=" days",      color=C["danger"] if ar_days>45 else C["warning"]), unsafe_allow_html=True)
        with c4:
            over90 = outstanding[outstanding["days_since"]>90]["billed_amount"].sum()
            st.markdown(kpi_card("90+ Day AR",  fmt_currency(over90), color=C["danger"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_a1, col_a2 = st.columns([3,2])
        with col_a1:
            bucket_colors = {
                "0-30 days":C["success"],"31-60 days":C["teal"],
                "61-90 days":C["warning"],"91-120 days":"#FF5722","120+ days":C["danger"]
            }
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=aging["bucket"], y=aging["total_outstanding"],
                marker_color=[bucket_colors.get(b, C["muted"]) for b in aging["bucket"]],
                text=aging["total_outstanding"].apply(fmt_currency),
                textposition="outside",
                name="Outstanding",
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=320,
                              margin=dict(l=0,r=0,t=20,b=0),
                              yaxis=dict(showgrid=True, gridcolor="#1F2937"),
                              title=dict(text="Outstanding Amount by Aging Bucket", font_size=13))
            st.plotly_chart(fig, use_container_width=True)

        with col_a2:
            fig = go.Figure(go.Pie(
                labels=aging["bucket"], values=aging["total_outstanding"],
                hole=0.55,
                marker_colors=[bucket_colors.get(b, C["muted"]) for b in aging["bucket"]],
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=320, margin=dict(l=0,r=0,t=20,b=0),
                              title=dict(text="AR Distribution", font_size=13))
            fig.add_annotation(text=f"<b>{fmt_currency(total_ar)}</b><br>Total AR",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=11, color="white"))
            st.plotly_chart(fig, use_container_width=True)

        # AR aging table
        st.markdown("#### AR Aging Summary Table")
        aging_display = aging.copy()
        aging_display["total_outstanding"] = aging_display["total_outstanding"].apply(fmt_currency)
        aging_display["claim_count"]       = aging_display["claim_count"].apply(fmt_num)
        aging_display["avg_days"]          = aging_display["avg_days"].apply(lambda x: f"{x:.0f} days")
        aging_display["pct_of_ar"]         = (outstanding.groupby("aging_bucket")["billed_amount"].sum()
                                               .reindex(BUCKET_ORDER) / total_ar * 100).apply(lambda x: f"{x:.1f}%").values
        aging_display.columns = ["Aging Bucket","Claim Lines","Outstanding Amount","Avg Days","% of Total AR"]
        st.dataframe(aging_display, use_container_width=True, hide_index=True)

        # Monthly AR trend
        st.markdown("#### Monthly Outstanding AR Trend")
        monthly_ar = outstanding.groupby("ym").agg(
            outstanding=("billed_amount","sum"),
            count=("claim_line_id","count"),
        ).reset_index().sort_values("ym").tail(24)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly_ar["ym"], y=monthly_ar["outstanding"],
                             name="Outstanding", marker_color=C["danger"], opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly_ar["ym"], y=monthly_ar["count"],
                                 name="Claim Count", line=dict(color=C["warning"], width=2),
                                 mode="lines+markers", marker_size=4), secondary_y=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=260,
                          margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation="h", y=1.1),
                          xaxis=dict(showgrid=False, tickangle=-45))
        fig.update_yaxes(title_text="Outstanding ($)", secondary_y=False, showgrid=True, gridcolor="#1F2937")
        fig.update_yaxes(title_text="Claim Lines", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        over90_pct = over90 / total_ar * 100 if total_ar > 0 else 0
        if over90_pct > 30:
            alert(f"{over90_pct:.1f}% of AR ({fmt_currency(over90)}) is over 90 days old. "
                  "Immediate action required: assign dedicated collectors, escalate to secondary payers, "
                  "and review timely filing limits before claims expire.")
        else:
            insight(f"AR aging profile is healthy — only {over90_pct:.1f}% over 90 days. "
                    f"Average AR days of {ar_days:.0f} is within acceptable range.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — DENIAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### ❌ Denial Analysis")
        story("Denials are the single biggest controllable revenue leakage in medical billing. "
              "Every denial has a root cause — administrative, clinical, or contractual. "
              "Fixing the top 3 denial reasons typically recovers 60-70% of denied revenue.")

        denied_cl = cl[cl["status_clean"] == "Denied"].copy()
        total_denied_amt   = denied_cl["billed_amount"].sum()
        total_denied_count = len(denied_cl)
        appealed_cl        = cl[cl["status_clean"] == "Appealed"]
        appeal_rate        = len(appealed_cl) / total_denied_count * 100 if total_denied_count > 0 else 0

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi_card("Denied Amount",  fmt_currency(total_denied_amt),   color=C["danger"]),  unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Denied Claims",  fmt_num(total_denied_count),       color=C["danger"]),  unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Denial Rate",    f"{denial_rt:.1f}", suffix="%",   color=C["danger"] if denial_rt>10 else C["success"]), unsafe_allow_html=True)
        with c4: st.markdown(kpi_card("Appeal Rate",    f"{appeal_rate:.1f}", suffix="%", color=C["warning"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Denial by CPT code
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### Top 15 Denied CPT Codes by Amount")
            denied_cpt = denied_cl.groupby("cpt_code").agg(
                denied_amount=("billed_amount","sum"),
                denied_count=("claim_line_id","count"),
            ).reset_index().nlargest(15,"denied_amount")
            denied_cpt["denial_rate_cpt"] = denied_cpt["denied_count"] / \
                cl.groupby("cpt_code")["claim_line_id"].count().reindex(denied_cpt["cpt_code"]).values * 100
            fig = px.bar(denied_cpt.sort_values("denied_amount"),
                         x="denied_amount", y="cpt_code", orientation="h",
                         color="denial_rate_cpt", color_continuous_scale="RdYlGn_r",
                         range_color=[0,50],
                         text=denied_cpt.sort_values("denied_amount")["denied_amount"].apply(fmt_currency),
                         labels={"denied_amount":"Denied Amount","cpt_code":"CPT Code","denial_rate_cpt":"Denial Rate %"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=380,
                              margin=dict(l=0,r=0,t=10,b=0))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col_d2:
            st.markdown("#### Denial Trend — Monthly")
            monthly_denial = cl.groupby("ym").agg(
                total=("billed_amount","sum"),
                denied=("billed_amount", lambda x: x[cl.loc[x.index,"status_clean"]=="Denied"].sum()),
            ).reset_index().sort_values("ym")
            monthly_denial["denial_rate"] = monthly_denial["denied"] / monthly_denial["total"] * 100
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly_denial["ym"], y=monthly_denial["denied"],
                                 name="Denied $", marker_color=C["danger"], opacity=0.8), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly_denial["ym"], y=monthly_denial["denial_rate"],
                                     name="Denial Rate %", line=dict(color=C["warning"], width=2.5),
                                     mode="lines+markers", marker_size=4), secondary_y=True)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=380,
                              margin=dict(l=0,r=0,t=10,b=0),
                              legend=dict(orientation="h", y=1.1),
                              xaxis=dict(showgrid=False, tickangle=-45, nticks=12))
            fig.update_yaxes(title_text="Denied Amount ($)", secondary_y=False, showgrid=True, gridcolor="#1F2937")
            fig.update_yaxes(title_text="Denial Rate (%)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        # Prior auth denial reasons
        if not pa.empty and "denial_reason" in pa.columns:
            st.markdown("#### Prior Auth Denial Reasons")
            pa_denied = pa[pa["status"]=="DENIED"].copy()
            if not pa_denied.empty:
                dr = pa_denied["denial_reason"].value_counts().reset_index()
                dr.columns = ["reason","count"]
                dr["pct"] = dr["count"] / dr["count"].sum() * 100

                col_dr1, col_dr2 = st.columns([2,1])
                with col_dr1:
                    fig = px.bar(dr, x="count", y="reason", orientation="h",
                                 color="pct", color_continuous_scale="Reds",
                                 text=dr["pct"].apply(lambda x: f"{x:.1f}%"),
                                 labels={"count":"Denials","reason":"Denial Reason","pct":"% of Denials"})
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)", height=260,
                                      margin=dict(l=0,r=0,t=10,b=0),
                                      yaxis=dict(categoryorder="total ascending"),
                                      coloraxis_showscale=False)
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

                with col_dr2:
                    fig = px.pie(dr, values="count", names="reason", hole=0.5,
                                 color_discrete_sequence=px.colors.sequential.Reds_r[:len(dr)])
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                      height=260, margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig, use_container_width=True)

        # Urgency vs denial rate
        if not pa.empty and "urgency" in pa.columns:
            st.markdown("#### Auth Denial Rate by Urgency Level")
            pa_clean = pa.copy()
            pa_clean["urgency_clean"] = pa_clean["urgency"].str.strip().str.title()
            pa_clean.loc[pa_clean["urgency_clean"].str.contains("Routine", na=False), "urgency_clean"] = "Routine"
            urg = pa_clean.groupby("urgency_clean").agg(
                total=("auth_id","count"),
                denied=("status", lambda x: (x=="DENIED").sum()),
            ).reset_index()
            urg["denial_rate"] = urg["denied"] / urg["total"] * 100
            urg = urg[urg["total"] >= 100]
            fig = px.bar(urg.sort_values("denial_rate", ascending=False),
                         x="urgency_clean", y="denial_rate",
                         color="denial_rate", color_continuous_scale="RdYlGn_r",
                         range_color=[0,30],
                         text=urg.sort_values("denial_rate",ascending=False)["denial_rate"].apply(lambda x: f"{x:.1f}%"),
                         labels={"urgency_clean":"Urgency","denial_rate":"Denial Rate %"})
            fig.add_hline(y=15, line_dash="dash", line_color=C["warning"],
                          annotation_text="15% threshold")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=240,
                              margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        top_denied_cpt = denied_cpt.iloc[0]["cpt_code"] if not denied_cpt.empty else "N/A"
        if denial_rt > 10:
            alert(f"Denial rate of {denial_rt:.1f}% ({fmt_currency(total_denied_amt)} denied). "
                  f"Top denied CPT: {top_denied_cpt}. "
                  "Implement pre-submission eligibility checks and prior auth verification to reduce front-end denials.")
        else:
            insight(f"Denial rate of {denial_rt:.1f}% is within the 10% industry benchmark. "
                    f"Focus on the {fmt_currency(total_denied_amt)} in denied claims for appeal opportunities.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — ADJUSTMENT CODES
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔧 Adjustment Reason Code Analysis")
        story("Adjustment reason codes (CARC) explain why a claim was paid differently than billed. "
              "CO codes = contractual obligations (payer responsibility). "
              "PR codes = patient responsibility. OA codes = other adjustments. "
              "High CO-45 volume means charges exceed fee schedule — review your chargemaster.")

        # CARC descriptions
        carc_desc = {
            "CO-4":  "Inconsistent Modifier",
            "CO-11": "Diagnosis Inconsistent with Procedure",
            "CO-16": "Claim Lacks Information",
            "CO-22": "Coordination of Benefits",
            "CO-45": "Charge Exceeds Fee Schedule",
            "CO-97": "Service Included in Another Service",
            "CO-50": "Non-Covered Service",
            "CO-29": "Timely Filing Exceeded",
            "OA-23": "Payment Adjusted — Timely Filing",
            "PR-1":  "Deductible Amount",
            "PR-2":  "Coinsurance Amount",
            "PR-3":  "Copay Amount",
        }

        adj_cl = cl[cl["adjustment_reason_code"].notna() & (cl["adjustment_reason_code"] != "")].copy()
        adj_cl["adj_code_clean"] = adj_cl["adjustment_reason_code"].str.strip().str.upper()
        adj_cl["adj_category"] = adj_cl["adj_code_clean"].apply(
            lambda x: "Contractual (CO)" if str(x).startswith("CO") else
                      "Patient Resp. (PR)" if str(x).startswith("PR") else
                      "Other (OA)" if str(x).startswith("OA") else "Unknown"
        )
        adj_cl["adj_description"] = adj_cl["adj_code_clean"].map(carc_desc).fillna("Other Adjustment")

        total_adj_amt = adj_cl["adjustment_amount"].sum()
        co_amt = adj_cl[adj_cl["adj_category"]=="Contractual (CO)"]["adjustment_amount"].sum()
        pr_amt = adj_cl[adj_cl["adj_category"]=="Patient Resp. (PR)"]["adjustment_amount"].sum()
        oa_amt = adj_cl[adj_cl["adj_category"]=="Other (OA)"]["adjustment_amount"].sum()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi_card("Total Adjustments", fmt_currency(total_adj_amt), color=C["warning"]), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Contractual (CO)",  fmt_currency(co_amt),        color=C["accent"]),  unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Patient Resp (PR)", fmt_currency(pr_amt),        color=C["purple"]),  unsafe_allow_html=True)
        with c4: st.markdown(kpi_card("Other (OA)",        fmt_currency(oa_amt),        color=C["teal"]),    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_adj1, col_adj2 = st.columns([2,1])
        with col_adj1:
            st.markdown("#### Top Adjustment Reason Codes — Volume & Amount")
            top_adj = adj_cl.groupby(["adj_code_clean","adj_description","adj_category"]).agg(
                count=("claim_line_id","count"),
                total_adj=("adjustment_amount","sum"),
                avg_adj=("adjustment_amount","mean"),
            ).reset_index().nlargest(12,"count")
            top_adj["label"] = top_adj["adj_code_clean"] + " — " + top_adj["adj_description"].str[:30]

            cat_colors = {"Contractual (CO)":C["accent"],"Patient Resp. (PR)":C["purple"],
                          "Other (OA)":C["teal"],"Unknown":C["muted"]}
            fig = go.Figure()
            for cat, grp in top_adj.groupby("adj_category"):
                fig.add_trace(go.Bar(
                    y=grp["label"], x=grp["count"],
                    name=cat, orientation="h",
                    marker_color=cat_colors.get(cat, C["muted"]),
                    text=grp["count"].apply(fmt_num),
                    textposition="outside",
                ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=400,
                              margin=dict(l=0,r=0,t=10,b=0), barmode="stack",
                              legend=dict(orientation="h", y=1.05),
                              yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)

        with col_adj2:
            st.markdown("#### Adjustment Category Split")
            cat_summary = adj_cl.groupby("adj_category").agg(
                count=("claim_line_id","count"),
                total=("adjustment_amount","sum"),
            ).reset_index()
            fig = go.Figure(go.Pie(
                labels=cat_summary["adj_category"],
                values=cat_summary["total"],
                hole=0.55,
                marker_colors=[cat_colors.get(c, C["muted"]) for c in cat_summary["adj_category"]],
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=220, margin=dict(l=0,r=0,t=10,b=0))
            fig.add_annotation(text=f"<b>{fmt_currency(total_adj_amt)}</b><br>Total Adj.",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=11, color="white"))
            st.plotly_chart(fig, use_container_width=True)

            # Adj amount by category table
            for _, row in cat_summary.iterrows():
                pct = row["total"] / total_adj_amt * 100 if total_adj_amt > 0 else 0
                cat_name  = row["adj_category"]
                cat_color = cat_colors.get(cat_name, C["muted"])
                cat_amt   = fmt_currency(row["total"])
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                    f"border-bottom:1px solid #1F2937;font-size:0.82rem'>"
                    f"<span style='color:{cat_color}'>{cat_name}</span>"
                    f"<span style='color:#FAFAFA'>{cat_amt} ({pct:.1f}%)</span></div>",
                    unsafe_allow_html=True
                )

        # Remark codes
        st.markdown("#### Top Remark Codes (RARC)")
        remark_desc = {
            "MA04":"Secondary payment cannot be considered without identity of primary payer",
            "N95": "This provider type/specialty cannot bill this service",
            "N30": "Patient ineligible for this service",
            "M20": "Missing/incomplete/invalid HCPCS modifier",
            "N362":"The number of Days or Units of Service exceeds our acceptable maximum",
            "N115":"This decision was based on a Local Coverage Determination",
        }
        if "remark_code" in cl.columns:
            rc = cl[cl["remark_code"].notna() & (cl["remark_code"]!="")].groupby("remark_code").agg(
                count=("claim_line_id","count"),
                total_billed=("billed_amount","sum"),
            ).reset_index().nlargest(10,"count")
            rc["description"] = rc["remark_code"].map(remark_desc).fillna("See RARC lookup")
            rc["label"] = rc["remark_code"] + " — " + rc["description"].str[:40]
            fig = px.bar(rc.sort_values("count"), x="count", y="label", orientation="h",
                         color="total_billed", color_continuous_scale="Blues",
                         text=rc.sort_values("count")["count"].apply(fmt_num),
                         labels={"count":"Occurrences","label":"Remark Code","total_billed":"Billed Amount"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=320,
                              margin=dict(l=0,r=0,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        top_co = top_adj[top_adj["adj_category"]=="Contractual (CO)"].iloc[0]["adj_code_clean"] if not top_adj[top_adj["adj_category"]=="Contractual (CO)"].empty else "N/A"
        insight(f"Top contractual adjustment code: {top_co}. "
                f"Contractual adjustments ({fmt_currency(co_amt)}) are expected and non-recoverable. "
                f"Focus recovery efforts on PR codes ({fmt_currency(pr_amt)}) — these are patient balances.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — PAYER PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 🏦 Payer Performance Analysis")
        story("Not all payers are equal. Some pay quickly and at high rates; others deny frequently and pay slowly. "
              "This analysis ranks payers by collection efficiency, denial rate, and payment speed "
              "to guide contract negotiations and network strategy.")

        # Payer performance from prior auth (has payer_id)
        if not pa.empty and "payer_id" in pa.columns:
            pa_perf = pa.groupby("payer_id").agg(
                total_auths=("auth_id","count"),
                approved=("status", lambda x: (x=="APPROVED").sum()),
                denied=("status",   lambda x: (x=="DENIED").sum()),
                pending=("status",  lambda x: (x=="PENDING").sum()),
            ).reset_index()
            pa_perf["approval_rate"] = pa_perf["approved"] / pa_perf["total_auths"] * 100
            pa_perf["denial_rate"]   = pa_perf["denied"]   / pa_perf["total_auths"] * 100
            pa_perf = pa_perf[pa_perf["total_auths"] >= 200].nlargest(15,"total_auths")

            # Turnaround time
            pa_tat = pa.copy()
            pa_tat["request_date"]  = pd.to_datetime(pa_tat["request_date"],  errors="coerce")
            pa_tat["decision_date"] = pd.to_datetime(pa_tat["decision_date"], errors="coerce")
            pa_tat["tat"] = (pa_tat["decision_date"] - pa_tat["request_date"]).dt.days
            tat_by_payer = pa_tat.groupby("payer_id")["tat"].mean().reset_index()
            tat_by_payer.columns = ["payer_id","avg_tat"]
            pa_perf = pa_perf.merge(tat_by_payer, on="payer_id", how="left")

            c1,c2,c3 = st.columns(3)
            best_payer = pa_perf.loc[pa_perf["approval_rate"].idxmax(), "payer_id"] if not pa_perf.empty else "N/A"
            worst_payer = pa_perf.loc[pa_perf["denial_rate"].idxmax(), "payer_id"] if not pa_perf.empty else "N/A"
            avg_approval = pa_perf["approval_rate"].mean()
            with c1: st.markdown(kpi_card("Avg Approval Rate", f"{avg_approval:.1f}", suffix="%",
                                           color=C["success"] if avg_approval>=80 else C["warning"]), unsafe_allow_html=True)
            with c2: st.markdown(kpi_card("Best Payer",  best_payer[:12],  color=C["success"]), unsafe_allow_html=True)
            with c3: st.markdown(kpi_card("Worst Payer", worst_payer[:12], color=C["danger"]),  unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter: approval rate vs turnaround
            st.markdown("#### Payer Scorecard — Approval Rate vs Turnaround Time")
            fig = px.scatter(pa_perf, x="avg_tat", y="approval_rate",
                             size="total_auths", color="denial_rate",
                             color_continuous_scale="RdYlGn_r",
                             range_color=[0,30],
                             hover_data=["payer_id","total_auths","approved","denied"],
                             text="payer_id",
                             labels={"avg_tat":"Avg Turnaround (days)",
                                     "approval_rate":"Approval Rate %",
                                     "denial_rate":"Denial Rate %",
                                     "total_auths":"Total Auths"})
            fig.add_hline(y=80, line_dash="dash", line_color=C["success"],
                          annotation_text="80% approval target", annotation_position="right")
            fig.add_vline(x=5, line_dash="dash", line_color=C["warning"],
                          annotation_text="5-day target")
            fig.update_traces(textposition="top center", textfont_size=8)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=380,
                              margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
            story("Ideal payers are top-left: high approval rate, fast turnaround. "
                  "Bottom-right payers (slow + low approval) are candidates for contract renegotiation. "
                  "Bubble size = total authorization volume.")

            # Payer comparison bar
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("#### Approval Rate by Payer")
                pa_sorted = pa_perf.sort_values("approval_rate", ascending=True)
                fig = px.bar(pa_sorted, x="approval_rate", y="payer_id", orientation="h",
                             color="approval_rate", color_continuous_scale="RdYlGn",
                             range_color=[50,100],
                             text=pa_sorted["approval_rate"].apply(lambda x: f"{x:.1f}%"),
                             labels={"approval_rate":"Approval Rate %","payer_id":"Payer"})
                fig.add_vline(x=80, line_dash="dash", line_color=C["warning"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", height=380,
                                  margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            with col_p2:
                st.markdown("#### Avg Turnaround Days by Payer")
                tat_sorted = pa_perf.dropna(subset=["avg_tat"]).sort_values("avg_tat", ascending=True)
                fig = px.bar(tat_sorted, x="avg_tat", y="payer_id", orientation="h",
                             color="avg_tat", color_continuous_scale="RdYlGn_r",
                             range_color=[0,15],
                             text=tat_sorted["avg_tat"].apply(lambda x: f"{x:.1f}d"),
                             labels={"avg_tat":"Avg Days","payer_id":"Payer"})
                fig.add_vline(x=5, line_dash="dash", line_color=C["success"],
                              annotation_text="5-day target")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", height=380,
                                  margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            # Payer volume stacked bar
            st.markdown("#### Auth Outcome Distribution by Payer")
            pa_stacked = pa_perf.sort_values("total_auths", ascending=False).head(12)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Approved", x=pa_stacked["payer_id"], y=pa_stacked["approved"],
                                 marker_color=C["success"]))
            fig.add_trace(go.Bar(name="Denied",   x=pa_stacked["payer_id"], y=pa_stacked["denied"],
                                 marker_color=C["danger"]))
            fig.add_trace(go.Bar(name="Pending",  x=pa_stacked["payer_id"], y=pa_stacked["pending"],
                                 marker_color=C["warning"]))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=280,
                              margin=dict(l=0,r=0,t=10,b=0), barmode="stack",
                              legend=dict(orientation="h", y=1.1),
                              xaxis=dict(tickangle=-30))
            st.plotly_chart(fig, use_container_width=True)

        # Procedure revenue by payer (from gold)
        if not proc_rev.empty:
            st.markdown("#### Procedure Reimbursement Rates — Top CPT Codes")
            top_proc = proc_rev.nlargest(15,"utilization_count").copy()
            top_proc["label"] = top_proc["cpt_code"] + " " + top_proc["procedure_description"].str[:20].fillna("")
            top_proc = top_proc.sort_values("avg_reimbursement_rate_pct")
            fig = px.bar(top_proc, x="avg_reimbursement_rate_pct", y="label", orientation="h",
                         color="avg_reimbursement_rate_pct", color_continuous_scale="RdYlGn",
                         range_color=[40,100],
                         text=top_proc["avg_reimbursement_rate_pct"].apply(lambda x: f"{x:.1f}%"),
                         labels={"avg_reimbursement_rate_pct":"Reimb. Rate %","label":"Procedure"})
            fig.add_vline(x=75, line_dash="dash", line_color=C["warning"],
                          annotation_text="75% target")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=380,
                              margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # Final billing insights
        st.markdown("---")
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left">
              <div class="kpi-label">💡 Revenue Opportunity</div>
              <div style="font-size:1.4rem;font-weight:700;color:{C['success']};margin:0.4rem 0">
                {fmt_currency(total_denied_amt * 0.35)}
              </div>
              <div style="font-size:0.8rem;color:#9E9E9E">
                Estimated recoverable from denied claims<br>
                (35% industry average overturn rate)
              </div>
            </div>""", unsafe_allow_html=True)
        with col_i2:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left">
              <div class="kpi-label">⚠️ AR at Risk</div>
              <div style="font-size:1.4rem;font-weight:700;color:{C['danger']};margin:0.4rem 0">
                {fmt_currency(outstanding[outstanding['days_since']>90]['billed_amount'].sum())}
              </div>
              <div style="font-size:0.8rem;color:#9E9E9E">
                Outstanding claims over 90 days<br>
                High write-off risk — immediate action needed
              </div>
            </div>""", unsafe_allow_html=True)
        with col_i3:
            patient_collectible = cl[cl["status_clean"].isin(["Pending","Rejected"])]["patient_responsibility"].sum()
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left">
              <div class="kpi-label">👤 Patient Balance</div>
              <div style="font-size:1.4rem;font-weight:700;color:{C['purple']};margin:0.4rem 0">
                {fmt_currency(patient_collectible)}
              </div>
              <div style="font-size:0.8rem;color:#9E9E9E">
                Collectible patient responsibility<br>
                on pending/rejected claims
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4B5563; font-size:0.78rem; padding:1rem 0;'>
  🏥 <b>Medical Billing Intelligence Platform</b> &nbsp;|&nbsp;
  Built with PySpark · Delta Lake · Apache Airflow · Streamlit · Plotly &nbsp;|&nbsp;
  Data: Medallion Architecture (Bronze → Silver → Gold)
</div>
""", unsafe_allow_html=True)

