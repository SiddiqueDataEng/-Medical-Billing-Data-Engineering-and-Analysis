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


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PATIENT DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Patient Demographics":
    st.markdown("# 👥 Patient Demographics")
    st.markdown("*Understanding who your patients are — age, gender, geography, lifestyle, and risk factors*")
    st.markdown("---")

    pts = load_silver("patients", ["patient_id","age","gender","race","state","zip_code",
                                    "smoking_status","bmi_category","income_bracket",
                                    "marital_status","preferred_language","employment_status"])
    risk = load_gold("kpi_patient_risk")

    if pts.empty:
        st.warning("No patient data available.")
        st.stop()

    total_pts = len(pts)
    avg_age   = pts["age"].dropna().mean()
    pct_female = (pts["gender"].str.upper() == "FEMALE").mean() * 100 if "gender" in pts.columns else 0

    high_risk = 0
    if not risk.empty and "risk_tier" in risk.columns:
        high_risk = (risk["risk_tier"] == "High").sum()

    story("Demographics drive everything in healthcare — from care protocols to payer mix to revenue projections. "
          "Understanding your patient population helps predict demand, allocate resources, and design targeted outreach programs.")

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Total Patients", fmt_num(total_pts), color=C["accent"]), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Average Age", f"{avg_age:.0f}", suffix=" yrs", color=C["teal"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Female Patients", f"{pct_female:.1f}", suffix="%", color=C["purple"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("High-Risk Patients", fmt_num(high_risk), color=C["danger"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        section("🎂 Age Distribution")
        fig = px.histogram(pts.dropna(subset=["age"]), x="age", nbins=25,
                           color_discrete_sequence=[C["accent"]],
                           labels={"age":"Age","count":"Patients"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=260,
                          margin=dict(l=0,r=0,t=10,b=0), bargap=0.05)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚧ Gender Split")
        g = pts["gender"].str.title().value_counts().reset_index()
        g.columns = ["gender","count"]
        fig = px.pie(g, values="count", names="gender", hole=0.55,
                     color_discrete_sequence=[C["accent"], C["purple"], C["teal"]])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=260, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        section("🌍 Race / Ethnicity")
        r = pts["race"].str.title().value_counts().head(8).reset_index()
        r.columns = ["race","count"]
        fig = px.bar(r, x="count", y="race", orientation="h",
                     color="count", color_continuous_scale="Blues",
                     labels={"count":"Patients","race":""})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=260,
                          margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(categoryorder="total ascending"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        section("🚬 Smoking Status")
        sm = pts["smoking_status"].str.title().value_counts().reset_index()
        sm.columns = ["status","count"]
        colors_sm = {"Never":C["success"],"Former":C["warning"],"Current":C["danger"],"Unknown":C["muted"]}
        fig = px.pie(sm, values="count", names="status", hole=0.5,
                     color="status", color_discrete_map=colors_sm)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=240, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        section("⚖️ BMI Category")
        bmi = pts["bmi_category"].str.title().value_counts().reset_index()
        bmi.columns = ["category","count"]
        bmi_colors = {"Normal":C["success"],"Overweight":C["warning"],
                      "Obese":C["danger"],"Morbidly Obese":"#B71C1C","Underweight":C["teal"]}
        fig = px.bar(bmi, x="category", y="count",
                     color="category", color_discrete_map=bmi_colors,
                     labels={"count":"Patients","category":""})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=240,
                          margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        section("💼 Employment Status")
        emp = pts["employment_status"].str.title().value_counts().reset_index()
        emp.columns = ["status","count"]
        fig = px.pie(emp, values="count", names="status", hole=0.5,
                     color_discrete_sequence=QUAL)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          height=240, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Geographic distribution
    section("🗺️ Patient Distribution by State")
    state_counts = pts["state"].value_counts().reset_index()
    state_counts.columns = ["state","count"]
    fig = px.choropleth(state_counts, locations="state", locationmode="USA-states",
                        color="count", scope="usa",
                        color_continuous_scale="Blues",
                        labels={"count":"Patients"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      geo_bgcolor="rgba(0,0,0,0)", height=350,
                      margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Risk tier
    if not risk.empty and "risk_tier" in risk.columns:
        section("🎯 Patient Risk Stratification")
        col_r1, col_r2 = st.columns([1,2])
        with col_r1:
            rt = risk["risk_tier"].value_counts().reset_index()
            rt.columns = ["tier","count"]
            tier_colors = {"High":C["danger"],"Medium":C["warning"],"Low":C["success"]}
            fig = px.pie(rt, values="count", names="tier", hole=0.6,
                         color="tier", color_discrete_map=tier_colors)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col_r2:
            if "risk_score" in risk.columns and "age" in risk.columns:
                sample = risk.sample(min(3000, len(risk)), random_state=42)
                fig = px.scatter(sample, x="age", y="risk_score",
                                 color="risk_tier",
                                 color_discrete_map=tier_colors,
                                 opacity=0.5, size_max=4,
                                 labels={"age":"Age","risk_score":"Risk Score","risk_tier":"Risk Tier"})
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", height=260,
                                  margin=dict(l=0,r=0,t=10,b=0))
                fig.update_traces(marker_size=3)
                st.plotly_chart(fig, use_container_width=True)

        pct_high = high_risk / len(risk) * 100 if len(risk) > 0 else 0
        if pct_high > 20:
            alert(f"{pct_high:.1f}% of patients are High-Risk ({fmt_num(high_risk)} patients). "
                  "Prioritise care management outreach for chronic condition management and preventive interventions.")
        else:
            insight(f"Risk distribution is healthy — only {pct_high:.1f}% High-Risk. "
                    "Continue preventive care programmes to maintain this profile.")

    # Income bracket
    section("💰 Income Bracket Distribution")
    if "income_bracket" in pts.columns:
        inc = pts["income_bracket"].value_counts().reset_index()
        inc.columns = ["bracket","count"]
        fig = px.bar(inc, x="bracket", y="count",
                     color="count", color_continuous_scale="Viridis",
                     labels={"count":"Patients","bracket":"Income Bracket"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=240,
                          margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — CLINICAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
