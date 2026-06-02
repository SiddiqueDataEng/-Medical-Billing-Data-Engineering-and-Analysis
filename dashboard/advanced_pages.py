"""
advanced_pages.py
=================
Three advanced dashboard pages:
  1. Advanced Visuals  — sunburst, sankey, treemap, animated scatter, 3D surface, parallel coords
  2. PDF Report        — dynamic report generation with header/footer, charts embedded
  3. AI Ad-Hoc Query   — NLP -> pandas query -> auto chart selection
"""
import io, os, textwrap, datetime, base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from fpdf import FPDF
import openai

# ── paths (relative to project root) ─────────────────────────
BASE   = Path(__file__).resolve().parent.parent
GOLD   = BASE / "delta" / "gold"
SILVER = BASE / "delta" / "silver"

# ── colour palette (matches app.py) ───────────────────────────
C = {
    "accent":  "#3B82F6", "success": "#22C55E", "warning": "#F59E0B",
    "danger":  "#EF4444", "purple":  "#A855F7", "teal":    "#06B6D4",
    "gold":    "#EAB308", "muted":   "#64748B",
}
QUAL = px.colors.qualitative.Set2

CHART_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0),
    font=dict(family="Inter", color="#E2E8F0"),
)

# ── Data loaders ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _gold(t):
    p = GOLD/t; files = list(p.glob("**/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files]) if files else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def _silver(t, cols=None):
    p = SILVER/t
    if not p.exists(): return pd.DataFrame()
    files = list(p.glob("**/*.parquet"))
    frames = []
    for f in files:
        try: frames.append(pd.read_parquet(f, columns=cols) if cols else pd.read_parquet(f))
        except: pass
    return pd.concat(frames) if frames else pd.DataFrame()

def fmt_c(v):
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def story(t): st.markdown(f'<div class="story-box">💡 {t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box">✅ {t}</div>', unsafe_allow_html=True)
def alert(t): st.markdown(f'<div class="alert-box">⚠️ {t}</div>', unsafe_allow_html=True)
def section(t): st.markdown(f'<div class="section-header">{t}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE: ADVANCED VISUALS
# ═══════════════════════════════════════════════════════════════
def page_advanced_visuals():
    st.markdown("# 🌌 Advanced Analytics")
    st.markdown("*Multi-dimensional and animated visualisations revealing patterns invisible in standard charts*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌞 Sunburst",
        "🔀 Sankey Flow",
        "🗺️ Treemap",
        "🎬 Animated Scatter",
        "📐 Parallel Coords",
        "📊 Revenue 3D Surface",
    ])

    # ── TAB 1: SUNBURST ────────────────────────────────────────
    with tab1:
        section("🌞 Revenue Sunburst — Billing Hierarchy")
        story("A sunburst chart reveals the full hierarchy of your revenue: "
              "from claim status → CPT code → line status → dollars. "
              "Click any segment to drill down.")

        cl = _silver("claim_lines", ["billed_amount","paid_amount","allowed_amount",
                                      "line_status","cpt_code","adjustment_reason_code"])
        if not cl.empty:
            cl["status_clean"] = cl["line_status"].str.strip().str.title()
            cl.loc[cl["status_clean"].str.contains("Paid",na=False),   "status_clean"] = "Paid"
            cl.loc[cl["status_clean"].str.contains("Denied",na=False), "status_clean"] = "Denied"
            cl.loc[cl["status_clean"].str.contains("Pending",na=False),"status_clean"] = "Pending"
            cl.loc[cl["status_clean"].str.contains("Reject",na=False), "status_clean"] = "Rejected"
            cl.loc[cl["status_clean"].str.contains("Appeal",na=False), "status_clean"] = "Appealed"
            cl.loc[cl["status_clean"].str.contains("Void",na=False),   "status_clean"] = "Void"
            cl.loc[cl["status_clean"].str.contains("Adjust",na=False), "status_clean"] = "Adjusted"

            # Top 8 CPT by volume
            top_cpt = cl["cpt_code"].value_counts().head(8).index.tolist()
            cl_top = cl[cl["cpt_code"].isin(top_cpt)].copy()
            cl_top["adj_code"] = cl_top["adjustment_reason_code"].fillna("None").str.strip().str.upper()

            sun_data = cl_top.groupby(["status_clean","cpt_code","adj_code"])["billed_amount"].sum().reset_index()
            sun_data.columns = ["Status","CPT","Adj Code","Billed"]
            sun_data = sun_data[sun_data["Billed"] > 0]

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig = px.sunburst(sun_data,
                    path=["Status","CPT","Adj Code"],
                    values="Billed",
                    color="Status",
                    color_discrete_map={
                        "Paid":C["success"],"Denied":C["danger"],"Pending":C["warning"],
                        "Rejected":"#FF5722","Appealed":C["purple"],"Void":C["muted"],"Adjusted":C["teal"]
                    },
                    title="Claim Status → CPT Code → Adjustment Code",
                    branchvalues="total",
                )
                fig.update_traces(textinfo="label+percent parent", insidetextfont=dict(size=11))
                fig.update_layout(**CHART_LAYOUT, height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col_s2:
                # Patient demographics sunburst
                pts = _silver("patients", ["age","gender","race","smoking_status","bmi_category"])
                if not pts.empty:
                    pts["age_group"] = pd.cut(pts["age"].dropna(),
                        bins=[0,18,35,50,65,100],
                        labels=["0-18","19-35","36-50","51-65","65+"])
                    pts_grp = pts.dropna(subset=["age_group"]).groupby(
                        ["age_group","gender","smoking_status"]).size().reset_index(name="count")
                    fig2 = px.sunburst(pts_grp,
                        path=["age_group","gender","smoking_status"],
                        values="count",
                        color="age_group",
                        color_discrete_sequence=QUAL,
                        title="Patient Population: Age Group → Gender → Smoking Status",
                        branchvalues="total",
                    )
                    fig2.update_traces(textinfo="label+percent parent", insidetextfont=dict(size=11))
                    fig2.update_layout(**CHART_LAYOUT, height=500)
                    st.plotly_chart(fig2, use_container_width=True)

            # Prior auth sunburst
            pa = _silver("prior_authorizations", ["status","service_type","payer_id","urgency"])
            if not pa.empty:
                section("Prior Authorization Hierarchy")
                pa["urgency_clean"] = pa["urgency"].str.strip().str.title()
                pa.loc[pa["urgency_clean"].str.contains("Routine",na=False),"urgency_clean"] = "Routine"
                pa_grp = pa.groupby(["status","service_type","urgency_clean"]).size().reset_index(name="count")
                pa_grp = pa_grp[pa_grp["count"] >= 50]
                fig3 = px.sunburst(pa_grp,
                    path=["status","service_type","urgency_clean"],
                    values="count",
                    color="status",
                    color_discrete_map={
                        "APPROVED":C["success"],"DENIED":C["danger"],
                        "PENDING":C["warning"],"EXPIRED":C["muted"],"APPEALED":C["purple"]
                    },
                    title="Prior Auth: Status → Service Type → Urgency",
                    branchvalues="total",
                )
                fig3.update_traces(textinfo="label+percent parent")
                fig3.update_layout(**CHART_LAYOUT, height=450)
                st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 2: SANKEY ──────────────────────────────────────────
    with tab2:
        section("🔀 Revenue Flow Sankey Diagram")
        story("A Sankey diagram shows exactly how money flows through your revenue cycle — "
              "from gross billed, through contractual adjustments, to final payment status. "
              "Wider bands mean more dollars flowing through that path.")

        cl = _silver("claim_lines", ["billed_amount","paid_amount","allowed_amount",
                                      "patient_responsibility","adjustment_amount","line_status"])
        if not cl.empty:
            cl["status_clean"] = cl["line_status"].str.strip().str.title()
            cl.loc[cl["status_clean"].str.contains("Paid",na=False),   "status_clean"] = "Paid"
            cl.loc[cl["status_clean"].str.contains("Denied",na=False), "status_clean"] = "Denied"
            cl.loc[cl["status_clean"].str.contains("Pending",na=False),"status_clean"] = "Pending"
            cl.loc[cl["status_clean"].str.contains("Reject",na=False), "status_clean"] = "Rejected"
            cl.loc[cl["status_clean"].str.contains("Appeal",na=False), "status_clean"] = "Appealed"

            total_billed   = cl["billed_amount"].sum()
            total_allowed  = cl["allowed_amount"].sum()
            total_paid     = cl["paid_amount"].sum()
            total_pt       = cl["patient_responsibility"].sum()
            contractual    = total_billed - total_allowed
            write_off      = max(0, total_allowed - total_paid - total_pt)

            # Node labels
            labels = [
                "Gross Billed",          # 0
                "Contractual Adj.",       # 1
                "Allowed Amount",         # 2
                "Net Collected",          # 3
                "Patient Responsibility", # 4
                "Write-offs/Denials",     # 5
                "Paid Claims",            # 6
                "Denied/Outstanding",     # 7
            ]
            node_colors = [
                "#3B82F6","#EF4444","#F59E0B","#22C55E",
                "#A855F7","#DC2626","#16A34A","#B45309",
            ]
            source = [0, 0, 2, 2, 2, 3]
            target = [1, 2, 3, 4, 5, 6]
            value  = [
                contractual, total_allowed,
                total_paid, total_pt, write_off,
                cl[cl["status_clean"]=="Paid"]["paid_amount"].sum(),
            ]
            link_colors = [
                "rgba(239,68,68,0.4)", "rgba(245,158,11,0.4)",
                "rgba(34,197,94,0.4)", "rgba(168,85,247,0.4)",
                "rgba(220,38,38,0.4)", "rgba(22,163,74,0.4)",
            ]
            fig = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20, thickness=30,
                    label=labels, color=node_colors,
                    line=dict(color="#1E2740", width=1),
                ),
                link=dict(source=source, target=target, value=value, color=link_colors),
            ))
            fig.update_layout(**CHART_LAYOUT, height=520,
                              title=dict(text="Medical Billing Revenue Flow", font_size=15))
            st.plotly_chart(fig, use_container_width=True)

            insight(f"Of {fmt_c(total_billed)} billed: {fmt_c(total_paid)} collected "
                    f"({total_paid/total_billed*100:.1f}%), {fmt_c(contractual)} contractual adjustment, "
                    f"{fmt_c(write_off)} written off.")

    # ── TAB 3: TREEMAP ─────────────────────────────────────────
    with tab3:
        section("🗺️ Revenue Treemap — Hierarchical Revenue by Category")
        story("Treemaps show relative size at a glance. Larger rectangles = more revenue. "
              "Colour intensity shows reimbursement rate — dark red means under-performing.")

        proc = _gold("kpi_procedure_revenue")
        if not proc.empty:
            proc["category"] = proc["cpt_code"].apply(lambda x: (
                "Evaluation & Management" if str(x).startswith("99") else
                "Radiology/Imaging"       if str(x).startswith("7") else
                "Laboratory"              if str(x).startswith("8") else
                "Surgery"                 if str(x).startswith(("2","3","4","5","6")) else
                "Medicine/Other"
            ))
            proc["label"] = proc["cpt_code"] + "<br>" + proc["procedure_description"].str[:20].fillna("")
            fig = px.treemap(proc,
                path=[px.Constant("All Procedures"),"category","label"],
                values="total_paid",
                color="avg_reimbursement_rate_pct",
                color_continuous_scale="RdYlGn",
                range_color=[30,100],
                hover_data={"total_billed":True,"total_paid":True,"utilization_count":True},
                title="Procedure Revenue Treemap — size=total paid, colour=reimbursement rate",
            )
            fig.update_traces(
                textinfo="label+value",
                textfont_size=12,
                marker_line_width=2,
                marker_line_color="#0A0D14",
            )
            fig.update_layout(**CHART_LAYOUT, height=560,
                              coloraxis_colorbar=dict(title="Reimb. %"))
            st.plotly_chart(fig, use_container_width=True)

        # Patient risk treemap
        risk = _gold("kpi_patient_risk")
        if not risk.empty:
            section("Patient Risk Treemap — Population by BMI & Risk Tier")
            risk_grp = risk.groupby(["risk_tier","bmi_category","gender"]).agg(
                count=("patient_id","count"),
                avg_risk=("risk_score","mean"),
                avg_chronic=("chronic_condition_count","mean"),
            ).reset_index()
            risk_grp = risk_grp[risk_grp["count"] >= 20]
            fig2 = px.treemap(risk_grp,
                path=[px.Constant("All Patients"),"risk_tier","bmi_category","gender"],
                values="count",
                color="avg_risk",
                color_continuous_scale="RdYlGn_r",
                range_color=[0,80],
                title="Patient Population — Risk Tier → BMI Category → Gender",
            )
            fig2.update_traces(textinfo="label+value", textfont_size=11,
                               marker_line_width=1, marker_line_color="#0A0D14")
            fig2.update_layout(**CHART_LAYOUT, height=480,
                               coloraxis_colorbar=dict(title="Avg Risk"))
            st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 4: ANIMATED SCATTER ────────────────────────────────
    with tab4:
        section("🎬 Animated Monthly Revenue Performance")
        story("Watch revenue metrics evolve month by month. "
              "Press Play to see how collection rate, denial rate, and volume change over time. "
              "Each bubble is a CPT code — size = volume, colour = reimbursement rate.")

        cl = _silver("claim_lines", ["billed_amount","paid_amount","allowed_amount","service_date","cpt_code"])
        if not cl.empty:
            cl["service_date"] = pd.to_datetime(cl["service_date"], errors="coerce")
            cl = cl.dropna(subset=["service_date"])
            cl["ym"] = cl["service_date"].dt.to_period("M").astype(str)
            cl["year"] = cl["service_date"].dt.year

            # Top 15 CPT codes overall
            top_cpt = cl.groupby("cpt_code")["billed_amount"].sum().nlargest(15).index.tolist()
            cl_top = cl[cl["cpt_code"].isin(top_cpt)].copy()

            anim = cl_top.groupby(["ym","cpt_code"]).agg(
                billed=("billed_amount","sum"),
                paid=("paid_amount","sum"),
                count=("cpt_code","count"),
            ).reset_index()
            anim["collection_rate"] = anim["paid"] / anim["billed"] * 100
            anim["avg_billed"] = anim["billed"] / anim["count"]
            anim = anim[anim["count"] >= 5]

            fig = px.scatter(anim, x="avg_billed", y="collection_rate",
                animation_frame="ym",
                animation_group="cpt_code",
                size="count", color="cpt_code",
                color_discrete_sequence=px.colors.qualitative.Light24,
                size_max=50,
                range_x=[0, anim["avg_billed"].quantile(0.99)],
                range_y=[0, 110],
                text="cpt_code",
                labels={"avg_billed":"Avg Billed per Claim ($)",
                        "collection_rate":"Collection Rate (%)","cpt_code":"CPT Code"},
                title="Monthly: Avg Billed vs Collection Rate by CPT Code",
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.add_hline(y=75, line_dash="dash", line_color=C["warning"],
                          annotation_text="75% target")
            fig.update_layout(**CHART_LAYOUT, height=540, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Animated appointment ops
            appt = _gold("kpi_appointment_ops")
            if not appt.empty:
                section("Animated Appointment Performance by Type")
                appt["ym"] = appt["year"].astype(str) + "-" + appt["month"].astype(str).str.zfill(2)
                appt["completion_rate"] = appt["completed"] / appt["total_scheduled"] * 100
                appt["noshow_rate"]     = appt["no_show"]   / appt["total_scheduled"] * 100
                appt_f = appt[appt["total_scheduled"] >= 5]
                fig2 = px.scatter(appt_f, x="completion_rate", y="noshow_rate",
                    animation_frame="ym",
                    animation_group="appointment_type",
                    size="total_scheduled", color="appointment_type",
                    color_discrete_sequence=QUAL,
                    size_max=40,
                    range_x=[0,110], range_y=[0,50],
                    text="appointment_type",
                    labels={"completion_rate":"Completion Rate (%)","noshow_rate":"No-Show Rate (%)"},
                    title="Monthly: Completion Rate vs No-Show Rate by Appointment Type",
                )
                fig2.update_traces(textposition="top center", textfont_size=9)
                fig2.add_hline(y=10, line_dash="dash", line_color=C["danger"])
                fig2.update_layout(**CHART_LAYOUT, height=480, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 5: PARALLEL COORDINATES ────────────────────────────
    with tab5:
        section("📐 Parallel Coordinates — Multi-Dimensional Patient Risk")
        story("Parallel coordinates show multiple dimensions simultaneously. "
              "Each line is a patient segment. Brush any axis to filter — "
              "this reveals which combinations of risk factors drive highest scores.")

        risk = _gold("kpi_patient_risk")
        if not risk.empty:
            risk_num = risk.copy()
            risk_num["risk_tier_num"] = risk_num["risk_tier"].map({"Low":0,"Medium":1,"High":2})
            risk_num["gender_num"]    = risk_num["gender"].map({"Male":0,"Female":1,"Non-Binary":2}).fillna(0)
            bmi_map = {"Underweight":0,"Normal":1,"Overweight":2,"Obese":3,"Morbidly Obese":4}
            risk_num["bmi_num"] = risk_num["bmi_category"].map(bmi_map).fillna(0)
            smoke_map = {"Never":0,"Former":1,"Current":2,"Unknown":0}
            risk_num["smoke_num"] = risk_num["smoking_status"].map(smoke_map).fillna(0)

            sample = risk_num.sample(min(5000, len(risk_num)), random_state=42)

            fig = go.Figure(go.Parcoords(
                line=dict(
                    color=sample["risk_score"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    cmin=0, cmax=100,
                    colorbar=dict(title="Risk Score", tickfont=dict(color="#E2E8F0")),
                ),
                dimensions=[
                    dict(label="Age",             values=sample["age"].fillna(0),
                         range=[0,100]),
                    dict(label="Chronic Conds",   values=sample["chronic_condition_count"].fillna(0),
                         range=[0,10]),
                    dict(label="Total Diagnoses", values=sample["total_diagnoses"].fillna(0),
                         range=[0,30]),
                    dict(label="Distinct ICD-10", values=sample["distinct_icd10_codes"].fillna(0),
                         range=[0,20]),
                    dict(label="BMI Category\n(0=Under 4=Morbid)", values=sample["bmi_num"],
                         range=[0,4], tickvals=[0,1,2,3,4],
                         ticktext=["Under","Normal","Over","Obese","Morbid"]),
                    dict(label="Smoking\n(0=Never 2=Current)", values=sample["smoke_num"],
                         range=[0,2], tickvals=[0,1,2],
                         ticktext=["Never","Former","Current"]),
                    dict(label="Risk Score",      values=sample["risk_score"].fillna(0),
                         range=[0,100]),
                    dict(label="Risk Tier\n(0=Low 2=High)", values=sample["risk_tier_num"].fillna(0),
                         range=[0,2], tickvals=[0,1,2],
                         ticktext=["Low","Medium","High"]),
                ],
            ))
            fig.update_layout(**CHART_LAYOUT, height=540,
                              title=dict(text="Patient Risk — Multi-Dimensional Profile (brush to filter)", font_size=14))
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6: 3D SURFACE ──────────────────────────────────────
    with tab6:
        section("📊 3D Revenue Surface — Month × CPT × Collected")
        story("A 3D surface reveals peaks and valleys in revenue across time and procedure type. "
              "Peaks are your revenue drivers. Valleys are opportunities. "
              "Rotate the chart to explore different angles.")

        cl = _silver("claim_lines", ["paid_amount","service_date","cpt_code"])
        if not cl.empty:
            cl["service_date"] = pd.to_datetime(cl["service_date"], errors="coerce")
            cl = cl.dropna(subset=["service_date"])
            cl["month_num"] = cl["service_date"].dt.month

            top_cpt = cl.groupby("cpt_code")["paid_amount"].sum().nlargest(10).index.tolist()
            cl_top = cl[cl["cpt_code"].isin(top_cpt)]

            pivot = cl_top.pivot_table(values="paid_amount", index="cpt_code",
                                        columns="month_num", aggfunc="sum", fill_value=0)
            pivot = pivot.reindex(columns=range(1,13), fill_value=0)

            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            z_data = pivot.values / 1000  # in $K

            fig = go.Figure(go.Surface(
                z=z_data,
                x=list(range(1,13)),
                y=pivot.index.tolist(),
                colorscale="Viridis",
                colorbar=dict(title="Revenue ($K)", tickfont=dict(color="#E2E8F0")),
                contours=dict(
                    x=dict(show=True, color="rgba(255,255,255,0.1)"),
                    y=dict(show=True, color="rgba(255,255,255,0.1)"),
                    z=dict(show=True, color="rgba(255,255,255,0.1)"),
                ),
                lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5),
            ))
            fig.update_layout(
                **CHART_LAYOUT, height=560,
                title=dict(text="3D Revenue Surface — CPT Code × Month × Revenue ($K)", font_size=14),
                scene=dict(
                    xaxis=dict(title="Month", tickvals=list(range(1,13)), ticktext=months,
                               gridcolor="#1E2740", color="#94A3B8"),
                    yaxis=dict(title="CPT Code", gridcolor="#1E2740", color="#94A3B8"),
                    zaxis=dict(title="Revenue ($K)", gridcolor="#1E2740", color="#94A3B8"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                scene_camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            )
            st.plotly_chart(fig, use_container_width=True)
            insight("Peaks on the 3D surface identify your highest-revenue CPT codes in peak months. "
                    "Flat or low areas indicate under-utilised services or seasonal demand drops.")


# ═══════════════════════════════════════════════════════════════
#  PAGE: PDF REPORT
# ═══════════════════════════════════════════════════════════════
def _fig_to_png(fig, w=700, h=400):
    """Convert plotly figure to PNG bytes."""
    try:
        return fig.to_image(format="png", width=w, height=h, scale=1.5)
    except Exception:
        return None

class MedBillingPDF(FPDF):
    """Custom PDF with branded header and footer."""
    def __init__(self, org_name="MedBilling Health System", report_title="Revenue Cycle Report"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.org_name     = org_name
        self.report_title = report_title
        self.generated_at = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(15, 25, 15)

    def header(self):
        # Blue header bar
        self.set_fill_color(30, 64, 175)   # blue-800
        self.rect(0, 0, 210, 16, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.set_xy(5, 4)
        self.cell(130, 8, f"  {self.org_name}", ln=0)
        self.set_font("Helvetica", "", 8)
        self.set_xy(140, 4)
        self.cell(65, 8, f"Generated: {self.generated_at}", ln=0, align="R")
        # Subtitle bar
        self.set_fill_color(15, 23, 42)    # slate-900
        self.rect(0, 16, 210, 8, "F")
        self.set_text_color(148, 163, 184) # slate-400
        self.set_font("Helvetica", "I", 8)
        self.set_xy(5, 17)
        self.cell(200, 6, f"  {self.report_title}  |  Medical Billing Intelligence Platform", ln=1)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(59, 130, 246)  # blue-500
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_font("Helvetica", "", 7)
        self.set_text_color(100, 116, 139)
        self.cell(90, 6, "CONFIDENTIAL — Medical Billing Intelligence Platform", ln=0)
        self.cell(90, 6, f"Page {self.page_no()} / {{nb}}", ln=0, align="R")

    def chapter_title(self, title, color=(59,130,246)):
        self.set_fill_color(*color)
        self.set_text_color(255,255,255)
        self.set_font("Helvetica","B",11)
        self.cell(0, 8, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0,0,0)
        self.ln(2)

    def kpi_row(self, kpis):
        """Render a row of KPI boxes. kpis = list of (label, value, color_hex)."""
        box_w   = (self.w - self.l_margin - self.r_margin) / len(kpis)
        start_x = self.l_margin
        y       = self.get_y()
        for label, val, color in kpis:
            r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            self.set_fill_color(15,23,42)
            self.set_draw_color(r,g,b)
            self.set_line_width(0.6)
            self.rect(start_x, y, box_w-2, 18, "FD")
            # Coloured top strip
            self.set_fill_color(r,g,b)
            self.rect(start_x, y, box_w-2, 2.5, "F")
            # Label
            self.set_xy(start_x+1, y+3)
            self.set_font("Helvetica","",6)
            self.set_text_color(148,163,184)
            self.cell(box_w-4, 4, label.upper(), align="C")
            # Value
            self.set_xy(start_x+1, y+7)
            self.set_font("Helvetica","B",10)
            self.set_text_color(r,g,b)
            self.cell(box_w-4, 8, str(val), align="C")
            start_x += box_w
        self.set_text_color(0,0,0)
        self.set_draw_color(0,0,0)
        self.set_line_width(0.2)
        self.ln(22)

    def add_figure(self, fig, caption="", w=180, h=80):
        """Embed a plotly figure as PNG."""
        img_bytes = _fig_to_png(fig, w=int(w*5), h=int(h*5))
        if img_bytes:
            buf = io.BytesIO(img_bytes)
            x = self.l_margin
            self.image(buf, x=x, y=self.get_y(), w=w, h=h)
            self.ln(h+2)
        if caption:
            self.set_font("Helvetica","I",7)
            self.set_text_color(100,116,139)
            self.cell(0,4,caption,new_x="LMARGIN",new_y="NEXT",align="C")
            self.set_text_color(0,0,0)
            self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        """Render a data table."""
        if col_widths is None:
            total_w = self.w - self.l_margin - self.r_margin
            col_widths = [total_w/len(headers)]*len(headers)
        # Header row
        self.set_fill_color(30,64,175)
        self.set_text_color(255,255,255)
        self.set_font("Helvetica","B",7)
        for h_txt, w in zip(headers, col_widths):
            self.cell(w, 6, str(h_txt), border=0, fill=True, align="C")
        self.ln()
        # Data rows
        self.set_font("Helvetica","",7)
        for i,row in enumerate(rows):
            self.set_fill_color(15,23,42) if i%2==0 else self.set_fill_color(22,32,58)
            self.set_text_color(225,232,240)
            for val, w in zip(row, col_widths):
                self.cell(w, 5, str(val), border=0, fill=True, align="C")
            self.ln()
        self.set_text_color(0,0,0)
        self.ln(3)

    def add_narrative(self, text):
        """Add a narrative paragraph."""
        self.set_font("Helvetica","",8)
        self.set_text_color(71,85,105)
        self.multi_cell(0, 5, text)
        self.set_text_color(0,0,0)
        self.ln(2)


def page_pdf_report():
    st.markdown("# 📄 Dynamic PDF Report Generator")
    st.markdown("*Generate a complete, branded revenue cycle report with charts, KPIs, and narrative — download as PDF*")
    st.markdown("---")

    story("This report engine builds a professional PDF containing all key metrics, charts, and "
          "narrative analysis. Configure the report below and click Generate.")

    # ── Report Configuration ──────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        org_name    = st.text_input("Organisation Name", "MedBilling Health System")
        report_type = st.selectbox("Report Type", [
            "Executive Summary", "Revenue Cycle Deep-Dive",
            "Denial & AR Analysis", "Clinical Operations",
        ])
    with col_cfg2:
        period_start = st.date_input("Period Start", datetime.date(2022,1,1))
        period_end   = st.date_input("Period End",   datetime.date(2025,12,31))
    with col_cfg3:
        include_charts   = st.checkbox("Include Charts",       value=True)
        include_tables   = st.checkbox("Include Data Tables",  value=True)
        include_insights = st.checkbox("Include AI Insights",  value=True)
        prepared_by      = st.text_input("Prepared By", "Data Engineering Team")

    st.markdown("---")
    gen_btn = st.button("🚀 Generate PDF Report", type="primary", use_container_width=True)

    if gen_btn:
        with st.spinner("Building report — this takes ~15 seconds..."):
            try:
                # ── Load data ─────────────────────────────────
                cl = _silver("claim_lines", ["billed_amount","paid_amount","allowed_amount",
                                              "patient_responsibility","adjustment_amount",
                                              "adjustment_reason_code","line_status","service_date","cpt_code"])
                pa = _silver("prior_authorizations", ["status","service_type","payer_id","request_date","denial_reason"])
                appt = _gold("kpi_appointment_ops")
                proc = _gold("kpi_procedure_revenue")
                risk = _gold("kpi_patient_risk")

                cl["service_date"] = pd.to_datetime(cl["service_date"], errors="coerce")
                cl = cl.dropna(subset=["service_date"])
                cl = cl[(cl["service_date"] >= pd.Timestamp(period_start)) &
                        (cl["service_date"] <= pd.Timestamp(period_end))]
                cl["status_clean"] = cl["line_status"].str.strip().str.title()
                cl.loc[cl["status_clean"].str.contains("Paid",na=False),   "status_clean"] = "Paid"
                cl.loc[cl["status_clean"].str.contains("Denied",na=False), "status_clean"] = "Denied"
                cl.loc[cl["status_clean"].str.contains("Pending",na=False),"status_clean"] = "Pending"
                cl["ym"] = cl["service_date"].dt.to_period("M").astype(str)

                total_billed  = cl["billed_amount"].sum()
                total_paid    = cl["paid_amount"].sum()
                total_allowed = cl["allowed_amount"].sum()
                total_ar      = cl[~cl["status_clean"].isin(["Paid","Void","Adjusted"])]["billed_amount"].sum()
                denied_amt    = cl[cl["status_clean"]=="Denied"]["billed_amount"].sum()
                coll_rate     = total_paid/total_billed*100 if total_billed>0 else 0
                denial_rate   = denied_amt/total_billed*100 if total_billed>0 else 0
                auth_appr     = 0
                if not pa.empty:
                    auth_appr = (pa["status"]=="APPROVED").sum()/len(pa)*100
                comp_rate = 0
                if not appt.empty:
                    comp_rate = appt["completed"].sum()/appt["total_scheduled"].sum()*100

                # ── Build charts for PDF ───────────────────────
                # Monthly trend
                monthly = cl.groupby("ym").agg(billed=("billed_amount","sum"),paid=("paid_amount","sum")).reset_index().sort_values("ym")
                monthly["coll_rate"] = monthly["paid"]/monthly["billed"]*100

                fig_trend = make_subplots(specs=[[{"secondary_y":True}]])
                fig_trend.add_trace(go.Bar(x=monthly["ym"],y=monthly["billed"],name="Billed",marker_color="#F59E0B",opacity=0.5),secondary_y=False)
                fig_trend.add_trace(go.Bar(x=monthly["ym"],y=monthly["paid"],name="Paid",marker_color="#22C55E",opacity=0.85),secondary_y=False)
                fig_trend.add_trace(go.Scatter(x=monthly["ym"],y=monthly["coll_rate"],name="Coll.%",line=dict(color="#3B82F6",width=2)),secondary_y=True)
                fig_trend.update_layout(template="plotly_white",height=350,barmode="overlay",
                                        margin=dict(l=40,r=40,t=30,b=60),
                                        legend=dict(orientation="h",y=1.1),
                                        xaxis=dict(tickangle=-45,nticks=12),
                                        title="Monthly Revenue: Billed vs Collected")

                # Status pie
                status_data = cl.groupby("status_clean")["billed_amount"].sum().reset_index()
                sc_map = {"Paid":"#22C55E","Denied":"#EF4444","Pending":"#F59E0B","Rejected":"#FF5722","Appealed":"#A855F7","Void":"#64748B","Adjusted":"#06B6D4"}
                fig_status = go.Figure(go.Pie(
                    labels=status_data["status_clean"],values=status_data["billed_amount"],
                    hole=0.5,marker_colors=[sc_map.get(s,"#64748B") for s in status_data["status_clean"]],
                ))
                fig_status.update_layout(template="plotly_white",height=320,
                                          margin=dict(l=20,r=20,t=30,b=20),
                                          title="Revenue by Claim Status")

                # Top CPT bar
                top_cpt = cl.groupby("cpt_code").agg(billed=("billed_amount","sum"),paid=("paid_amount","sum")).nlargest(10,"paid").reset_index()
                top_cpt["reimb"] = top_cpt["paid"]/top_cpt["billed"]*100
                fig_cpt = px.bar(top_cpt.sort_values("paid"),x="paid",y="cpt_code",orientation="h",
                                  color="reimb",color_continuous_scale="RdYlGn",range_color=[30,100],
                                  title="Top 10 CPT Codes by Revenue")
                fig_cpt.update_layout(template="plotly_white",height=320,
                                       margin=dict(l=10,r=40,t=30,b=20),coloraxis_showscale=True)

                # AR aging
                cl["days_since"] = (pd.Timestamp.now() - cl["service_date"]).dt.days
                outstanding = cl[~cl["status_clean"].isin(["Paid","Void","Adjusted"])].copy()
                outstanding["bucket"] = pd.cut(outstanding["days_since"],
                    bins=[0,30,60,90,120,9999],labels=["0-30","31-60","61-90","91-120","120+"])
                aging = outstanding.groupby("bucket")["billed_amount"].sum().reset_index()
                fig_aging = px.bar(aging,x="bucket",y="billed_amount",
                    color="bucket",color_discrete_sequence=["#22C55E","#06B6D4","#F59E0B","#FF5722","#EF4444"],
                    title="AR Aging Buckets")
                fig_aging.update_layout(template="plotly_white",height=280,
                                         margin=dict(l=10,r=10,t=30,b=20),showlegend=False)

                # ── Build PDF ──────────────────────────────────
                pdf = MedBillingPDF(org_name=org_name, report_title=report_type)
                pdf.alias_nb_pages()
                pdf.add_page()

                # Cover-style intro
                pdf.set_font("Helvetica","B",18)
                pdf.set_text_color(30,64,175)
                pdf.cell(0,10,report_type,new_x="LMARGIN",new_y="NEXT",align="C")
                pdf.set_font("Helvetica","",9)
                pdf.set_text_color(100,116,139)
                pdf.cell(0,6,f"Period: {period_start.strftime('%B %d, %Y')} — {period_end.strftime('%B %d, %Y')}",
                         new_x="LMARGIN",new_y="NEXT",align="C")
                pdf.cell(0,6,f"Prepared by: {prepared_by}  |  Confidential",
                         new_x="LMARGIN",new_y="NEXT",align="C")
                pdf.set_text_color(0,0,0)
                pdf.ln(5)

                # Section 1: Executive KPIs
                pdf.chapter_title("1. Executive KPI Summary", color=(30,64,175))
                pdf.kpi_row([
                    ("Gross Billed",   fmt_c(total_billed),  "#F59E0B"),
                    ("Net Collected",  fmt_c(total_paid),    "#22C55E"),
                    ("Collection Rate",f"{coll_rate:.1f}%",  "#3B82F6"),
                    ("Total AR",       fmt_c(total_ar),      "#EF4444"),
                ])
                pdf.kpi_row([
                    ("Denial Rate",    f"{denial_rate:.1f}%","#EF4444"),
                    ("Auth Approval",  f"{auth_appr:.1f}%",  "#22C55E"),
                    ("Appt Completion",f"{comp_rate:.1f}%",  "#A855F7"),
                    ("Allowed Amount", fmt_c(total_allowed), "#06B6D4"),
                ])

                if include_insights:
                    narrative = (
                        f"During the period {period_start} to {period_end}, the organisation billed "
                        f"{fmt_c(total_billed)} and collected {fmt_c(total_paid)}, achieving a collection rate of "
                        f"{coll_rate:.1f}%. The outstanding AR balance stands at {fmt_c(total_ar)}. "
                        f"The denial rate of {denial_rate:.1f}% "
                        + ("exceeds the 10% benchmark — immediate denial management action is recommended." if denial_rate>10
                           else "is within the 10% industry benchmark.")
                        + f" Prior authorization approval rate is {auth_appr:.1f}%."
                    )
                    pdf.add_narrative(narrative)

                # Section 2: Revenue Trend
                if include_charts:
                    pdf.chapter_title("2. Monthly Revenue Trend", color=(16,185,129))
                    pdf.add_figure(fig_trend,
                                   caption="Monthly billed vs collected amounts with collection rate trend line",
                                   h=75)
                    pdf.chapter_title("3. Claim Status Distribution", color=(245,158,11))
                    col1_x = pdf.l_margin
                    y_before = pdf.get_y()
                    st_img = _fig_to_png(fig_status, w=1500, h=1200)
                    cpt_img = _fig_to_png(fig_cpt, w=1500, h=1200)
                    if st_img and cpt_img:
                        pdf.image(io.BytesIO(st_img), x=col1_x, y=y_before, w=87, h=70)
                        pdf.image(io.BytesIO(cpt_img), x=col1_x+90, y=y_before, w=87, h=70)
                        pdf.ln(73)
                    else:
                        pdf.add_figure(fig_status, caption="Claim status distribution", h=65)

                    pdf.chapter_title("4. Accounts Receivable Aging", color=(239,68,68))
                    pdf.add_figure(fig_aging, caption="Outstanding AR by aging bucket", h=60)

                # Section 5: Top CPT Table
                if include_tables:
                    pdf.add_page()
                    pdf.chapter_title("5. Top 10 CPT Codes — Revenue Detail", color=(30,64,175))
                    headers = ["CPT Code","Billed","Paid","Reimb. Rate","Volume"]
                    rows = []
                    for _, row in top_cpt.iterrows():
                        rows.append([
                            row["cpt_code"],
                            fmt_c(row["billed"]),
                            fmt_c(row["paid"]),
                            f"{row['reimb']:.1f}%",
                            str(int(cl[cl['cpt_code']==row['cpt_code']]['cpt_code'].count())),
                        ])
                    pdf.add_table(headers, rows, col_widths=[35,35,35,35,30])

                    # AR aging table
                    if not aging.empty:
                        pdf.chapter_title("6. AR Aging Summary", color=(239,68,68))
                        ag_headers = ["Aging Bucket","Outstanding Amount","% of Total AR"]
                        ag_rows = []
                        for _, row in aging.iterrows():
                            pct = row["billed_amount"]/total_ar*100 if total_ar>0 else 0
                            ag_rows.append([str(row["bucket"]), fmt_c(row["billed_amount"]), f"{pct:.1f}%"])
                        pdf.add_table(ag_headers, ag_rows, col_widths=[55,65,50])

                # Section 7: Denial reasons from prior auth
                if not pa.empty and include_tables:
                    pdf.chapter_title("7. Prior Authorization Denial Reasons", color=(168,85,247))
                    pa_denied = pa[pa["status"]=="DENIED"]
                    if not pa_denied.empty and "denial_reason" in pa_denied.columns:
                        dr = pa_denied["denial_reason"].value_counts().head(8)
                        dr_headers = ["Denial Reason","Count","% of Denials"]
                        dr_rows = [[str(k), str(v), f"{v/len(pa_denied)*100:.1f}%"] for k,v in dr.items()]
                        pdf.add_table(dr_headers, dr_rows, col_widths=[80,35,55])

                # Signature block
                pdf.ln(10)
                pdf.set_draw_color(59,130,246)
                pdf.set_line_width(0.3)
                pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin+60, pdf.get_y())
                pdf.ln(1)
                pdf.set_font("Helvetica","",7)
                pdf.set_text_color(100,116,139)
                pdf.cell(65,4,f"Authorised: {prepared_by}",ln=0)
                pdf.cell(0,4,f"Report ID: MBI-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",align="R")

                # ── Export ────────────────────────────────────
                pdf_out = io.BytesIO()
                pdf_bytes = pdf.output()
                pdf_out = io.BytesIO(pdf_bytes if isinstance(pdf_bytes, bytes) else bytes(pdf_bytes))

                fname = f"MedBilling_{report_type.replace(' ','_')}_{datetime.date.today()}.pdf"
                st.success(f"✅ Report generated successfully — {pdf.page} pages")
                st.download_button(
                    label=f"📥 Download {fname}",
                    data=pdf_out.getvalue(),
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Report generation failed: {e}")
                import traceback; st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════
#  PAGE: AI AD-HOC QUERY
# ═══════════════════════════════════════════════════════════════

# Data catalogue for the AI to reason about
DATA_CATALOGUE = {
    "claim_lines": {
        "desc": "Individual billing line items with amounts, CPT codes, and status",
        "cols": {
            "billed_amount":"float - gross charged amount",
            "paid_amount":"float - amount actually collected",
            "allowed_amount":"float - payer-allowed amount",
            "patient_responsibility":"float - amount owed by patient",
            "adjustment_amount":"float - contractual adjustment",
            "adjustment_reason_code":"str - CARC code e.g. CO-4, PR-3",
            "line_status":"str - Paid|Denied|Pending|Rejected|Appealed|Void|Adjusted",
            "service_date":"date - date of service",
            "cpt_code":"str - procedure code e.g. 99213",
            "units":"int - units of service",
        }
    },
    "prior_authorizations": {
        "desc": "Prior authorization requests with approval/denial status",
        "cols": {
            "status":"str - APPROVED|DENIED|PENDING|EXPIRED|APPEALED",
            "service_type":"str - type of medical service",
            "payer_id":"str - insurance payer identifier",
            "urgency":"str - Routine|Urgent|Emergent",
            "denial_reason":"str - reason for denial if denied",
            "request_date":"date - when auth was requested",
            "decision_date":"date - when decision was made",
            "approved_units":"int - units approved",
        }
    },
    "patients": {
        "desc": "Patient demographics and clinical characteristics",
        "cols": {
            "age":"int - patient age in years",
            "gender":"str - Male|Female|Non-Binary",
            "race":"str - race/ethnicity",
            "state":"str - US state abbreviation",
            "smoking_status":"str - Never|Former|Current",
            "bmi_category":"str - Normal|Overweight|Obese|Morbidly Obese",
            "income_bracket":"str - income range",
            "employment_status":"str - Employed|Retired|Disabled etc",
        }
    },
    "vitals": {
        "desc": "Clinical vital signs recorded per encounter",
        "cols": {
            "systolic_bp":"float - systolic blood pressure mmHg",
            "diastolic_bp":"float - diastolic blood pressure mmHg",
            "heart_rate":"float - heart rate bpm",
            "bmi":"float - body mass index",
            "temperature_f":"float - temperature Fahrenheit",
            "oxygen_saturation":"float - SpO2 percentage",
        }
    },
    "kpi_procedure_revenue": {
        "desc": "Gold-layer procedure revenue aggregations by CPT code",
        "cols": {
            "cpt_code":"str - CPT procedure code",
            "procedure_description":"str - description of procedure",
            "utilization_count":"int - total procedures performed",
            "total_billed":"float - total billed amount",
            "total_paid":"float - total collected amount",
            "avg_billed":"float - average billed per procedure",
            "avg_paid":"float - average paid per procedure",
            "avg_reimbursement_rate_pct":"float - average reimbursement rate percentage",
        }
    },
    "kpi_appointment_ops": {
        "desc": "Gold-layer appointment operations KPIs by year/month/type",
        "cols": {
            "year":"int - year",
            "month":"int - month number",
            "appointment_type":"str - type of appointment",
            "total_scheduled":"int - total appointments scheduled",
            "completed":"int - appointments completed",
            "cancelled":"int - appointments cancelled",
            "no_show":"int - no-show appointments",
            "completion_rate_pct":"float - completion rate percent",
            "no_show_rate_pct":"float - no-show rate percent",
            "avg_duration_minutes":"float - average appointment duration",
        }
    },
    "kpi_patient_risk": {
        "desc": "Gold-layer patient risk scores and chronic condition counts",
        "cols": {
            "patient_id":"str - unique patient identifier",
            "age":"int - patient age",
            "gender":"str - Male|Female|Non-Binary",
            "bmi_category":"str - BMI category",
            "smoking_status":"str - smoking status",
            "chronic_condition_count":"int - number of chronic conditions",
            "total_diagnoses":"int - total diagnoses count",
            "distinct_icd10_codes":"int - distinct ICD-10 codes",
            "risk_score":"float - calculated risk score 0-100",
            "risk_tier":"str - Low|Medium|High",
        }
    },
}

def _build_system_prompt():
    catalogue_text = ""
    for table, info in DATA_CATALOGUE.items():
        catalogue_text += f"\nTable: {table}\nDescription: {info['desc']}\nColumns:\n"
        for col, desc in info["cols"].items():
            catalogue_text += f"  - {col}: {desc}\n"
    return f"""You are a medical billing data analyst assistant.
You have access to these pandas DataFrames (pre-loaded, referenced by variable name):
{catalogue_text}

Rules:
1. Convert the user's natural language question into valid Python/pandas code.
2. The code must assign the final result to a variable called `result`.
3. Use only the tables and columns listed above.
4. For date filtering use: pd.to_datetime(df['col'], errors='coerce')
5. For claim_lines line_status, normalise with: df['line_status'].str.strip().str.title()
   Then: Paid, Denied, Pending, Rejected, Appealed, Void, Adjusted
6. Return ONLY executable Python code, no explanations, no markdown fences.
7. The result should be a pandas DataFrame or Series.
8. Do not import anything — all libraries are pre-loaded (pd, np).
9. Also suggest a chart_type on the last line as a Python comment:
   # chart_type: bar|line|pie|scatter|table|histogram
"""

def _call_openai(question: str, api_key: str) -> str:
    """Call OpenAI API and return generated code."""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": question},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()

def _extract_chart_type(code: str) -> str:
    """Extract chart_type hint from last comment line."""
    for line in reversed(code.strip().split("\n")):
        if "chart_type:" in line:
            for ct in ["bar","line","pie","scatter","table","histogram"]:
                if ct in line:
                    return ct
    return "table"

def _safe_exec(code: str, frames: dict):
    """Safely execute generated code and return result."""
    safe_globals = {"pd": pd, "np": np, **frames}
    local_ns = {}
    exec(code, safe_globals, local_ns)
    return local_ns.get("result", None)

def _auto_chart(result, chart_type: str, question: str):
    """Render the appropriate chart for the result."""
    if result is None or (hasattr(result, "empty") and result.empty):
        st.warning("Query returned no results.")
        return

    if isinstance(result, (int, float, np.integer, np.floating)):
        st.metric("Result", f"{result:,.2f}")
        return

    if isinstance(result, pd.Series):
        result = result.reset_index()
        result.columns = ["Category", "Value"]

    if not isinstance(result, pd.DataFrame):
        st.write(result)
        return

    # Always show data table
    st.dataframe(result.head(100), use_container_width=True, hide_index=True)

    if len(result) == 0 or len(result.columns) < 2:
        return

    num_cols = result.select_dtypes(include=np.number).columns.tolist()
    cat_cols = result.select_dtypes(exclude=np.number).columns.tolist()
    if not num_cols:
        return

    y_col = num_cols[0]
    x_col = cat_cols[0] if cat_cols else result.columns[0]

    st.markdown("#### 📊 Auto-Generated Chart")
    if chart_type == "pie" and len(result) <= 15:
        fig = px.pie(result, names=x_col, values=y_col, hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=question[:80])
        fig.update_layout(**CHART_LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "line":
        fig = px.line(result, x=x_col, y=num_cols,
                      markers=True, title=question[:80],
                      color_discrete_sequence=[C["accent"],C["success"],C["warning"]])
        fig.update_layout(**CHART_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "scatter" and len(num_cols) >= 2:
        fig = px.scatter(result, x=num_cols[0], y=num_cols[1],
                         color=cat_cols[0] if cat_cols else None,
                         size=num_cols[2] if len(num_cols)>2 else None,
                         title=question[:80],
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "histogram":
        fig = px.histogram(result, x=y_col, nbins=30, title=question[:80],
                           color_discrete_sequence=[C["accent"]])
        fig.update_layout(**CHART_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

    else:  # default bar
        result_plot = result.head(30)
        if len(result_plot) > 15:
            fig = px.bar(result_plot, x=y_col, y=x_col, orientation="h",
                         color=y_col, color_continuous_scale="Blues",
                         title=question[:80])
            fig.update_layout(**CHART_LAYOUT, height=max(320, len(result_plot)*22),
                              yaxis=dict(categoryorder="total ascending"),
                              coloraxis_showscale=False)
        else:
            fig = px.bar(result_plot, x=x_col, y=y_col,
                         color=y_col, color_continuous_scale="Blues",
                         title=question[:80], text_auto=True)
            fig.update_layout(**CHART_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def page_ai_query():
    st.markdown("# 🤖 AI Ad-Hoc Query Engine")
    st.markdown("*Ask any question in plain English — AI converts it to a pandas query and renders the result with the best chart*")
    st.markdown("---")

    story("Type your question in natural language. The AI understands medical billing terminology, "
          "knows all available tables and columns, and automatically picks the best chart type. "
          "No SQL or Python knowledge needed.")

    # ── API Key ───────────────────────────────────────────────
    with st.expander("🔑 OpenAI API Key (required)", expanded=True):
        api_key = st.text_input("Enter your OpenAI API key",
                                 type="password",
                                 placeholder="sk-...",
                                 help="Your key is never stored. Get one at platform.openai.com")
        st.caption("Uses gpt-4o-mini — very fast and cost-effective (~$0.0001 per query)")

    # ── Example queries ───────────────────────────────────────
    st.markdown("#### 💡 Example Questions — click to try")
    examples = [
        "What is the total billed and paid amount by CPT code for the top 10 procedures?",
        "Show me the monthly denial rate trend over time",
        "What percentage of prior authorizations were denied by service type?",
        "Which states have the highest number of high-risk patients?",
        "What is the average systolic blood pressure by gender and BMI category?",
        "Show me the top 5 adjustment reason codes by total adjustment amount",
        "What is the no-show rate by appointment type?",
        "Compare collection rates across different claim line statuses",
        "What is the distribution of patient ages for high-risk patients?",
        "Which payers have the lowest prior auth approval rates?",
    ]
    cols = st.columns(2)
    selected_example = None
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💬 {ex[:60]}...", key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    st.markdown("---")

    # ── Query input ───────────────────────────────────────────
    question = st.text_area(
        "Your Question",
        value=selected_example or "",
        height=80,
        placeholder="e.g. What is the total revenue by claim status for the last 12 months?",
    )

    col_btn1, col_btn2 = st.columns([1,4])
    with col_btn1:
        run_btn = st.button("🚀 Run Query", type="primary", use_container_width=True)
    with col_btn2:
        show_code = st.checkbox("Show generated Python code", value=False)

    if run_btn:
        if not api_key or not api_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key above.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("🤔 AI is generating the query..."):
            try:
                generated_code = _call_openai(question, api_key)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                st.stop()

        if show_code:
            st.markdown("#### 🔍 Generated Python Code")
            st.code(generated_code, language="python")

        chart_type = _extract_chart_type(generated_code)

        with st.spinner("⚡ Running query against data..."):
            try:
                # Pre-load all relevant frames
                col_map = {
                    "claim_lines": ["billed_amount","paid_amount","allowed_amount","patient_responsibility",
                                    "adjustment_amount","adjustment_reason_code","line_status","service_date","cpt_code","units"],
                    "prior_authorizations": ["auth_id","status","service_type","payer_id","urgency","denial_reason",
                                              "request_date","decision_date","approved_units"],
                    "patients": ["patient_id","age","gender","race","state","smoking_status","bmi_category",
                                 "income_bracket","employment_status"],
                    "vitals": ["vital_id","systolic_bp","diastolic_bp","heart_rate","bmi","temperature_f","oxygen_saturation"],
                }
                frames = {}
                for tbl, cols in col_map.items():
                    frames[tbl] = _silver(tbl, cols)
                frames["kpi_procedure_revenue"]  = _gold("kpi_procedure_revenue")
                frames["kpi_appointment_ops"]    = _gold("kpi_appointment_ops")
                frames["kpi_patient_risk"]       = _gold("kpi_patient_risk")

                result = _safe_exec(generated_code, frames)

                st.markdown("#### 📋 Query Results")
                _auto_chart(result, chart_type, question)

            except Exception as e:
                st.error(f"Query execution error: {e}")
                st.markdown("**Generated code:**")
                st.code(generated_code, language="python")
                st.info("Try rephrasing your question or check the example queries above.")

    # ── Schema Explorer ───────────────────────────────────────
    with st.expander("📚 Data Schema Reference"):
        for table, info in DATA_CATALOGUE.items():
            st.markdown(f"**`{table}`** — {info['desc']}")
            cols_df = pd.DataFrame(
                [(c, d) for c, d in info["cols"].items()],
                columns=["Column", "Description"]
            )
            st.dataframe(cols_df, use_container_width=True, hide_index=True, height=min(200, len(cols_df)*36+38))
            st.markdown("")
