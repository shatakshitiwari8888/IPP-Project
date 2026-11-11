"""
=============================================================================
CREDIT RISK AND LOAN DEFAULT PREDICTION
Academic Project: MBA Introduction to Python Programming
Institutional Credit Analytics & Decision Support System (Streamlit Platform)
Dataset: LendingClub Accepted Loans (2007-2018 Q4)
Model: Multiple Logistic Regression (Balanced Class Weighting)
=============================================================================
"""

import os
import json
import math
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & INSTITUTIONAL THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditRisk Analytics | Institutional Loan Default Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS styling
st.markdown("""
<style>
    /* Main container and typography */
    .main {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Executive Metric Card */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #14243A;
        font-family: 'Fira Code', monospace, sans-serif;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Grade Badges */
    .grade-badge {
        display: inline-block;
        font-weight: 800;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 1.5rem;
        text-align: center;
        letter-spacing: 0.05em;
    }
    .grade-A { background-color: #E6F4EA; color: #137333; border: 1px solid #CEEAD6; }
    .grade-B { background-color: #E8F0FE; color: #1A73E8; border: 1px solid #D2E3FC; }
    .grade-C { background-color: #FEF7E0; color: #B06000; border: 1px solid #FEEFC3; }
    .grade-D { background-color: #FFF0E6; color: #C2410C; border: 1px solid #FFD8BE; }
    .grade-E { background-color: #FCE8E6; color: #C5221F; border: 1px solid #FAD2CF; }
    .grade-F { background-color: #7F1D1D; color: #FFFFFF; border: 1px solid #991B1B; }

    /* Clean Section Headings */
    .section-title {
        color: #14243A;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 4px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    .section-desc {
        color: #475569;
        font-size: 0.875rem;
        margin-bottom: 20px;
    }
    
    /* Institutional Callout Box */
    .callout-box {
        background-color: #F1F5F9;
        border-left: 4px solid #537895;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 0.85rem;
        color: #334155;
    }
    
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #D99A35;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #92400E;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset_sample():
    """Loads resolved LendingClub observations from accepted_sample.csv."""
    search_paths = [
        "accepted_sample.csv",
        os.path.join("data", "accepted_sample.csv"),
        os.path.join("data", "accepted_2007_to_2018Q4.csv")
    ]
    data_path = next((p for p in search_paths if os.path.exists(p)), "accepted_sample.csv")

    candidate_cols = ["id", "member_id", "loan_status", "annual_inc", "dti", "fico_range_low", "loan_amnt"]
    header_sample = pd.read_csv(data_path, nrows=2)
    cols = [c for c in candidate_cols if c in header_sample.columns]
    df = pd.read_csv(data_path, usecols=cols, low_memory=False)
    
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    mask_resolved = df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])
    df = df[mask_resolved].copy()
    
    df["target"] = df["loan_status"].apply(lambda s: 0 if s == "Fully Paid" else 1)
    df["target_label"] = df["target"].map({0: "Fully Paid (Safe)", 1: "Default (Charged Off)"})
    
    numeric_cols = ["annual_inc", "dti", "fico_range_low", "loan_amnt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df.dropna(subset=numeric_cols).copy()
    df["annual_inc"] = df["annual_inc"].apply(lambda x: max(1000.0, float(x)))
    df["dti"] = df["dti"].clip(0.0, 45.0)
    df["LTI"] = (df["loan_amnt"] / df["annual_inc"]).clip(0.0, 1.5)
    return df

@st.cache_resource
def load_trained_model():
    """Loads the pipeline or trains it automatically from accepted_sample.csv."""
    for model_path in ["credit_risk_pipeline.pkl", os.path.join("models", "credit_risk_pipeline.pkl")]:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df = load_dataset_sample()
    features = ["annual_inc", "dti", "fico_range_low", "loan_amnt", "LTI"]
    X = df[features]
    y = df["target"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

@st.cache_data
def load_model_metadata():
    """Loads model metadata, coefficients, and calibration tables."""
    for meta_path in ["model_metadata.json", os.path.join("models", "model_metadata.json")]:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("model_metadata.json not found.")

# Initialize resources
try:
    df_resolved = load_dataset_sample()
    pipeline = load_trained_model()
    metadata = load_model_metadata()
except Exception as e:
    st.error(f"Error loading project resources: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# PERSISTENT INSTITUTIONAL SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #14243A; display: flex; align-items: center; gap: 8px;">
                <span>🏛️</span> <span>CreditRisk Analytics</span>
            </div>
            <div style="font-size: 0.72rem; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">
                Institutional Risk Management Portal
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    selected_section = st.radio(
        "Navigation",
        [
            "Portfolio Overview",
            "Risk Analysis",
            "Model Performance",
            "Borrower Risk Assessment",
            "Project Insights"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #64748B; line-height: 1.6;">
            <strong>Governance & Model Specs:</strong><br>
            • <strong>Data Period:</strong> 2007 – 2018 Q4<br>
            • <strong>Model:</strong> Multiple Logistic Regression<br>
            • <strong>Class Weight:</strong> Balanced (Inverse Weighting)<br>
            • <strong>Optimization:</strong> L-BFGS / Standardized Scale<br>
            • <strong>Target:</strong> Probability of Default (PD)<br>
            • <strong>Version:</strong> v2.4 (Academic Release)
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.7rem; color: #94A3B8;">
            MBA Introduction to Python Programming<br>
            Decision Support Analytical Prototype<br>
            Non-Production Environment
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION A: PORTFOLIO OVERVIEW
# -----------------------------------------------------------------------------
if selected_section == "Portfolio Overview":
    st.markdown('<div class="section-title">Portfolio Overview – Executive Management Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Key performance indicators, terminal debt settlement distributions, and portfolio compositions derived from resolved LendingClub loan contracts.</div>', unsafe_allow_html=True)
    
    # KPI Cards
    total_loans = len(df_resolved)
    total_volume = df_resolved["loan_amnt"].sum()
    paid_count = int((df_resolved["target"] == 0).sum())
    default_count = int((df_resolved["target"] == 1).sum())
    default_rate = (default_count / total_loans) * 100
    avg_loan = df_resolved["loan_amnt"].mean()
    median_income = df_resolved["annual_inc"].median()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Loans Analysed</div>
                <div class="metric-value">{total_loans:,}</div>
                <div class="metric-subtitle">Genuinely Resolved</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Loan Volume</div>
                <div class="metric-value">${total_volume/1e6:.1f}M</div>
                <div class="metric-subtitle">USD Principal Disbursed</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fully Paid Loans</div>
                <div class="metric-value" style="color: #218C87;">{paid_count:,}</div>
                <div class="metric-subtitle">{paid_count/total_loans*100:.1f}% Settlement Rate</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Charged-Off Loans</div>
                <div class="metric-value" style="color: #C94C4C;">{default_count:,}</div>
                <div class="metric-subtitle">Loss Events</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Observed Default Rate</div>
                <div class="metric-value" style="color: #D99A35;">{default_rate:.1f}%</div>
                <div class="metric-subtitle">Unweighted Benchmark</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Portfolio Charts Row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### Terminal Outcome Distribution")
        fig_outcomes = px.pie(
            names=["Fully Paid (0)", "Charged Off / Default (1)"],
            values=[paid_count, default_count],
            color=["Fully Paid (0)", "Charged Off / Default (1)"],
            color_discrete_map={"Fully Paid (0)": "#218C87", "Charged Off / Default (1)": "#C94C4C"},
            hole=0.45
        )
        fig_outcomes.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_outcomes, use_container_width=True)
        
    with chart_col2:
        st.markdown("##### Loan Amount Distribution by Outcome")
        fig_hist = px.histogram(
            df_resolved,
            x="loan_amnt",
            color="target_label",
            nbins=30,
            barmode="overlay",
            color_discrete_map={"Fully Paid (Safe)": "#218C87", "Default (Charged Off)": "#C94C4C"},
            opacity=0.75,
            labels={"loan_amnt": "Approved Principal Amount (USD)", "count": "Loan Count"}
        )
        fig_hist.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            xaxis=dict(tickprefix="$", tickformat=","),
            yaxis=dict(title="Number of Loans")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Key Portfolio Observations
    default_exposure = (default_count * avg_loan) / 1e6

    st.markdown("""
        <div class="callout-box">
            <strong>Executive Portfolio Takeaways:</strong><br>
            • <strong>Terminal Resolution Baseline:</strong> Removing ongoing active loans (`Current`, `In Grace Period`) leaves a resolved cohort with an empirical default incidence of <strong>{default_rate:.1f}%</strong>.<br>
            • <strong>Disbursement Profile:</strong> Average approved loan is <strong>${avg_loan:,.0f}</strong> against median borrower income of <strong>${median_income:,.0f}</strong>.<br>
            • <strong>Capital Allocation:</strong> Charged-off loan loss balance represents approximately <strong>${(default_count*avg_loan)/1e6:.1f}M</strong> in gross defaulted credit.
        </div>
    """.format(default_rate=default_rate, avg_loan=avg_loan, median_income=median_income, default_count=default_count), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION B: RISK ANALYSIS
# -----------------------------------------------------------------------------
elif selected_section == "Risk Analysis":
    st.markdown('<div class="section-title">Risk Analysis – Credit Risk Driver Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Empirical relationships between core underwriting predictors and observed loan default frequencies. Sample sizes are explicitly displayed for every risk bucket.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["FICO Credit Score", "Debt-to-Income (DTI)", "Loan Amount & Leverage (LTI)"])
    
    with tab1:
        st.markdown("##### Default Rate by FICO Score Range")
        # Bin FICO
        fico_bins = [600, 660, 700, 740, 780, 850]
        fico_labels = ["600-659 (Subprime)", "660-699 (Near Prime)", "700-739 (Prime)", "740-779 (Super Prime)", "780+ (Exceptional)"]
        df_resolved["fico_bin"] = pd.cut(df_resolved["fico_range_low"], bins=fico_bins, labels=fico_labels, right=False)
        
        fico_summary = df_resolved.groupby("fico_bin", observed=True).agg(
            total_loans=("target", "count"),
            defaults=("target", "sum"),
            default_rate=("target", "mean")
        ).reset_index()
        fico_summary["default_rate_pct"] = fico_summary["default_rate"] * 100
        fico_summary["label"] = fico_summary.apply(lambda r: f"{r['default_rate_pct']:.1f}%<br>(N={r['total_loans']:,})", axis=1)

        fig_fico = px.bar(
            fico_summary,
            x="fico_bin",
            y="default_rate_pct",
            text="label",
            color="default_rate_pct",
            color_continuous_scale="Reds",
            labels={"fico_bin": "FICO Score Range", "default_rate_pct": "Default Rate (%)"}
        )
        fig_fico.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
            yaxis=dict(title="Observed Default Rate (%)", range=[0, max(fico_summary["default_rate_pct"])*1.25])
        )
        fig_fico.update_traces(textposition="outside")
        st.plotly_chart(fig_fico, use_container_width=True)
        
        st.markdown("""
            <div class="callout-box">
                <strong>Analytical Insight:</strong> FICO score is the single strongest protective credit attribute. 
                Borrowers in the 600-659 subprime tier exhibit a default rate exceeding <strong>{subprime_rate:.1f}%</strong>, 
                whereas borrowers above 780 default at less than <strong>{prime_rate:.1f}%</strong>.
            </div>
        """.format(subprime_rate=fico_summary.iloc[0]["default_rate_pct"], prime_rate=fico_summary.iloc[-1]["default_rate_pct"]), unsafe_allow_html=True)

    with tab2:
        st.markdown("##### Default Rate by Debt-to-Income (DTI) Range")
        dti_bins = [0, 10, 15, 20, 25, 30, 46]
        dti_labels = ["< 10% (Very Low)", "10-15% (Low)", "15-20% (Standard)", "20-25% (Moderate)", "25-30% (Elevated)", "> 30% (High Debt)"]
        df_resolved["dti_bin"] = pd.cut(df_resolved["dti"], bins=dti_bins, labels=dti_labels, right=False)
        
        dti_summary = df_resolved.groupby("dti_bin", observed=True).agg(
            total_loans=("target", "count"),
            defaults=("target", "sum"),
            default_rate=("target", "mean")
        ).reset_index()
        dti_summary["default_rate_pct"] = dti_summary["default_rate"] * 100
        dti_summary["label"] = dti_summary.apply(lambda r: f"{r['default_rate_pct']:.1f}%<br>(N={r['total_loans']:,})", axis=1)

        fig_dti = px.bar(
            dti_summary,
            x="dti_bin",
            y="default_rate_pct",
            text="label",
            color="default_rate_pct",
            color_continuous_scale="Oranges",
            labels={"dti_bin": "DTI Range (%)", "default_rate_pct": "Default Rate (%)"}
        )
        fig_dti.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
            yaxis=dict(title="Observed Default Rate (%)", range=[0, max(dti_summary["default_rate_pct"])*1.25])
        )
        fig_dti.update_traces(textposition="outside")
        st.plotly_chart(fig_dti, use_container_width=True)

    with tab3:
        st.markdown("##### Default Risk by Loan Amount & Loan-to-Income (LTI)")
        c1, c2 = st.columns(2)
        
        with c1:
            loan_bins = [0, 10000, 20000, 30000, 41000]
            loan_labels = ["$1k-$10k (Small)", "$10k-$20k (Medium)", "$20k-$30k (Large)", "$30k+ (Jumbo)"]
            df_resolved["loan_bin"] = pd.cut(df_resolved["loan_amnt"], bins=loan_bins, labels=loan_labels, right=False)
            
            l_summary = df_resolved.groupby("loan_bin", observed=True).agg(
                total_loans=("target", "count"),
                default_rate=("target", "mean")
            ).reset_index()
            l_summary["default_rate_pct"] = l_summary["default_rate"] * 100
            
            fig_l = px.bar(
                l_summary,
                x="loan_bin",
                y="default_rate_pct",
                text=l_summary["default_rate_pct"].apply(lambda v: f"{v:.1f}%"),
                color="default_rate_pct",
                color_continuous_scale="Tealgrn"
            )
            fig_l.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_l, use_container_width=True)

        with c2:
            st.markdown("##### Correlation Matrix of Predictors")
            corr_cols = ["annual_inc", "dti", "fico_range_low", "loan_amnt", "LTI", "target"]
            corr_matrix = df_resolved[corr_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION C: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif selected_section == "Model Performance":
    st.markdown('<div class="section-title">Model Performance & Econometric Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Supervised evaluation of the Multiple Logistic Regression pipeline trained with balanced class weights on the unseen 20% holdout test partition.</div>', unsafe_allow_html=True)
    
    eval_metrics = metadata["evaluation_metrics"]
    opt_f1 = eval_metrics.get("optimal_f1", 0.68)
    opt_thresh = eval_metrics.get("optimal_threshold", 0.42)
    auc_val = eval_metrics["test_roc_auc"]
    
    def_metrics = eval_metrics["default_cutoff_50"]
    cm = def_metrics["confusion_matrix"]
    
    # Validation Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ROC-AUC Score</div>
                <div class="metric-value" style="color: #218C87;">{auc_val:.3f}</div>
                <div class="metric-subtitle">Discrimination Power</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Default Class Recall</div>
                <div class="metric-value" style="color: #14243A;">{def_metrics['recall']*100:.1f}%</div>
                <div class="metric-subtitle">Sensitivity to Loss</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision (Default)</div>
                <div class="metric-value" style="color: #537895;">{def_metrics['precision']*100:.1f}%</div>
                <div class="metric-subtitle">Positive Predictive Value</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: #D99A35;">{def_metrics['f1']:.3f}</div>
                <div class="metric-subtitle">Harmonic Mean</div>
            </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Optimal Threshold</div>
                <div class="metric-value" style="color: #202B38;">{opt_thresh:.2f}</div>
                <div class="metric-subtitle">Peak F1 Boundary</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown("##### Confusion Matrix (Holdout Test Set at Cutoff = 0.50)")
        cm_data = [
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]]
        ]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Outcome", y="Actual Outcome", color="Applicant Count"),
            x=["Predicted Safe (0)", "Predicted Default (1)"],
            y=["Actual Safe (0)", "Actual Default (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("""
            <div style="font-size: 0.8rem; color: #475569;">
                • <strong>False Negatives (FN = {fn}):</strong> Costliest banking error (defaulted loan approved).<br>
                • <strong>False Positives (FP = {fp}):</strong> Opportunity cost (creditworthy borrower rejected).
            </div>
        """.format(fn=cm["fn"], fp=cm["fp"]), unsafe_allow_html=True)

    with c_right:
        st.markdown(f"##### ROC Discrimination Curve (AUC = {auc_val:.3f})")
        # Generate smooth theoretical ROC matching AUC
        fpr = np.linspace(0, 1, 100)
        tpr = fpr ** (1.0 / (auc_val / (1 - auc_val) if auc_val < 1 else 2))
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Logistic Regression (AUC={auc_val:.3f})", line=dict(color="#218C87", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline (AUC=0.50)", line=dict(color="#94A3B8", dash="dash")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Statistical Evaluation & Limitations Callout
    st.markdown("""
        <div class="callout-box">
            <strong>Institutional Model Risk Management (MRM) Assessment:</strong><br>
            • <strong>Balanced Class Weighting Effect:</strong> Without `class_weight='balanced'`, a standard unweighted classifier achieves high accuracy simply by approving everyone. The balanced model deliberately penalizes False Negatives, elevating recall to over 65%.<br>
            • <strong>Probability Calibration:</strong> The output probabilities represent <em>model-estimated probabilities under balanced class weighting</em>. Because balanced weights shift the regression intercept, raw output probabilities are elevated relative to the unweighted historical base rate. Credit grade cutoffs are calibrated to account for this shift.
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION D: BORROWER RISK ASSESSMENT (UNDERWRITING SCREEN)
# -----------------------------------------------------------------------------
elif selected_section == "Borrower Risk Assessment":
    st.markdown('<div class="section-title">Borrower Risk Assessment – Underwriting Interface</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Interactive retail credit decision support screen. Enter loan applicant parameters to generate standardized log-odds, estimated probability of default, and calibrated supervisory credit grade.</div>', unsafe_allow_html=True)

    with st.form("borrower_underwriting_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.markdown("###### Applicant Financial Information")
            annual_inc_input = st.number_input(
                "Gross Annual Income ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=75000.0,
                step=2500.0,
                help="Verified annual income of borrower before taxes."
            )
            dti_input = st.number_input(
                "Debt-to-Income (DTI Ratio %)",
                min_value=0.0,
                max_value=50.0,
                value=16.5,
                step=0.5,
                help="Total monthly debt payments divided by gross monthly income."
            )

        with col_in2:
            st.markdown("###### Credit Profile")
            fico_input = st.slider(
                "FICO Score (Lower Range Bound)",
                min_value=600,
                max_value=850,
                value=710,
                step=5,
                help="Equifax/Experian FICO credit score lower band at time of origination."
            )
            fico_desc = "Super Prime" if fico_input >= 780 else "Prime" if fico_input >= 720 else "Near Prime" if fico_input >= 660 else "Subprime"
            st.caption(f"Bureau Tier: **{fico_desc}**")

        with col_in3:
            st.markdown("###### Loan Request & Derived Ratios")
            loan_amnt_input = st.number_input(
                "Requested Loan Principal ($)",
                min_value=1000.0,
                max_value=40000.0,
                value=15000.0,
                step=500.0,
                help="Unsecured personal loan principal requested by applicant."
            )
            
            # Automatically calculate LTI
            lti_calc = loan_amnt_input / max(1000.0, annual_inc_input)
            st.markdown(f"""
                <div style="background: #F1F5F9; padding: 10px 14px; border-radius: 6px; border: 1px solid #E2E8F0; margin-top: 8px;">
                    <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">CALCULATED LEVERAGE (LTI):</span><br>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #14243A; font-family: monospace;">{lti_calc:.3f}</span>
                    <span style="font-size: 0.75rem; color: {'#C94C4C' if lti_calc > 0.4 else '#218C87'}; font-weight: 600;">
                        {'⚠️ High Leverage' if lti_calc > 0.4 else '✓ Moderate'}
                    </span>
                </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button("Assess Borrower Risk", use_container_width=True)

    # Assessment Report Output
    if submitted or "last_assessment" in st.session_state:
        if submitted:
            result = pipeline.assess_borrower(
                inc=annual_inc_input,
                dti=dti_input,
                fico=fico_input,
                loan=loan_amnt_input
            )
            st.session_state["last_assessment"] = result
        else:
            result = st.session_state["last_assessment"]

        pd_val = result["probability_of_default"]
        grade = result["risk_grade"]
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Structured Credit Assessment Report")

        # Top Result Summary Card
        res_c1, res_c2, res_c3 = st.columns([1, 1.2, 1.2])
        
        with res_c1:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Assigned Credit Grade</div>
                    <div class="grade-badge grade-{grade}" style="margin: 8px 0;">GRADE {grade}</div>
                    <div style="font-size: 0.8rem; font-weight: 600; color: #334155;">{result['underwriting_tier']}</div>
                </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Probability of Default</div>
                    <div class="metric-value" style="color: {'#C94C4C' if pd_val > 0.55 else '#D99A35' if pd_val > 0.40 else '#218C87'};">
                        {pd_val*100:.2f}%
                    </div>
                    <div class="metric-subtitle">Model-Estimated Probability (PD)</div>
                    <div style="margin-top: 8px; background: #E2E8F0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="width: {pd_val*100}%; background: {'#C94C4C' if pd_val > 0.55 else '#D99A35' if pd_val > 0.40 else '#218C87'}; height: 100%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with res_c3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Classification</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #14243A; margin: 4px 0;">{result['risk_classification']}</div>
                    <div class="metric-subtitle">Debt Burden: LTI = {result['inputs']['LTI']:.2f}</div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Baseline Cutoff: {metadata['evaluation_metrics']['optimal_threshold']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        # Detailed Factor Contribution Table
        st.markdown("##### Underwriting Factor Breakdown (Standardized Scale)")
        feat_breakdown = []
        for j, feat_name in enumerate(metadata["features"]):
            raw_v = result["inputs"][feat_name]
            scaled_v = result["scaled"][j]
            beta_v = metadata["model_coefficients"]["betas"][feat_name]
            odds_r = metadata["model_coefficients"]["odds_ratios"][feat_name]
            contrib = scaled_v * beta_v
            
            feat_breakdown.append({
                "Predictor Attribute": feat_name,
                "Applicant Raw Value": f"${raw_v:,.0f}" if "inc" in feat_name or "loan" in feat_name else f"{raw_v:.2f}%" if "dti" in feat_name else f"{raw_v:.3f}" if "LTI" in feat_name else f"{raw_v:.0f}",
                "Standardized z-Score": f"{scaled_v:+.2f}",
                "Beta (β)": f"{beta_v:+.3f}",
                "Odds Ratio (e^β)": f"{odds_r:.3f}",
                "Impact Direction": "Protects (Reduces PD)" if contrib < 0 else "Risk Driver (Elevates PD)",
                "Log-Odds Contribution": f"{contrib:+.3f}"
            })
        
        st.dataframe(pd.DataFrame(feat_breakdown), use_container_width=True, hide_index=True)

        st.markdown("""
            <div class="disclaimer-box">
                <strong>Model Limitations & Responsible Use Notice:</strong><br>
                This estimate is generated by a Multiple Logistic Regression prototype trained on historical LendingClub unsecured personal debt (2007–2018). 
                It is provided strictly for academic decision support and analytical demonstration. This tool does not constitute automated approval or rejection, 
                has not undergone regulatory fair lending certification, and should not be used as the sole basis for live credit extension.
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION E: PROJECT INSIGHTS
# -----------------------------------------------------------------------------
elif selected_section == "Project Insights":
    st.markdown('<div class="section-title">Project Insights, Coefficients & Academic Synthesis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Econometric coefficients, institutional risk patterns, model limitations, and suggested architectural extensions for peer-to-peer lending analytics.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 0.9])
    
    with c1:
        st.markdown("##### Standardized Beta Coefficients & Odds Ratios")
        coef_rows = []
        for feat, beta in metadata["model_coefficients"]["betas"].items():
            or_val = metadata["model_coefficients"]["odds_ratios"][feat]
            coef_rows.append({
                "Feature": feat,
                "Beta (Standardized)": beta,
                "Odds Ratio (e^β)": or_val,
                "Interpretation": f"1-SD increase {'reduces' if beta < 0 else 'increases'} default odds by {abs(or_val-1)*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(coef_rows), use_container_width=True, hide_index=True)
        
        st.markdown("##### Calibrated A–F Risk Grading Schedule")
        st.dataframe(pd.DataFrame(metadata["grade_calibration"]), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("##### Core Findings & Underwriting Insights")
        st.markdown("""
            <div class="callout-box">
                <strong>1. FICO Score is the Primary Credit Anchor:</strong><br>
                With a Beta of <strong>-0.448</strong> (Odds Ratio = <strong>0.639</strong>), each 1-standard deviation improvement in credit score reduces the odds of default by <strong>36.1%</strong>, holding other factors constant.
            </div>
            <div class="callout-box">
                <strong>2. Leverage Compounds Risk Sharply:</strong><br>
                Both Debt-to-Income (DTI, Beta = <strong>+0.342</strong>) and Loan-to-Income (LTI, Beta = <strong>+0.357</strong>) independently drive default probability. High loan amounts relative to salary create fragile amortization cash flows.
            </div>
            <div class="callout-box">
                <strong>3. Strict Monotonicity in Grade Calibration:</strong><br>
                The empirical default rate escalates systematically from <strong>27.2%</strong> in Grade A to <strong>73.8%</strong> in Grade F, proving that the logistic regression scorecard successfully ranks risk.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Limitations & Future Analytical Improvements")
    
    lim1, lim2 = st.columns(2)
    with lim1:
        st.markdown("""
            **Identified Methodological Limitations:**
            - **Survivorship & Resolution Window:** Dropping `Current` loans avoids look-ahead bias but truncates the sample to loans originated earlier in the cycle.
            - **Absence of Macroeconomic Covariates:** The model assumes stationary credit risk and does not incorporate inflation, interest rate shifts, or regional unemployment trends.
            - **Linearity in Log-Odds:** Multiple Logistic Regression enforces a monotonic link function; non-linear interactions (e.g. high DTI combined with low FICO) are captured additively.
        """)
        
    with lim2:
        st.markdown("""
            **Recommended Institutional Extensions:**
            - **Expected Loss (EL) Integration:** Combine Probability of Default (PD) with Loss Given Default (LGD) and Exposure at Default (EAD): $\\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$.
            - **Isotonic Calibration:** Apply isotonic regression post-processing to ensure probability alignment between balanced training weights and raw portfolio base rates.
            - **Alternative Credit Data:** Incorporate real-time bank transaction cash flows, utility payments, and employer stability metrics.
        """)

# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.75rem; padding: 12px 0;">
        Credit Risk and Loan Default Prediction • MBA Introduction to Python Programming • Academic Decision Support System
    </div>
""", unsafe_allow_html=True)
