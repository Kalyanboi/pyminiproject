"""
Solar Wind Mission Control — Interactive Dashboard
Python Application (Streamlit + Plotly + Pandas + Scikit-Learn)
"""

import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Wind Mission Control Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM THEME & CSS STYLING (Matching the Mission Control Dark Space Theme)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-main: #0b0e14;
        --card-bg: #141a24;
        --cyan-accent: #35c6d8;
        --violet-accent: #9b7bff;
        --text-color: #e9f2ff;
        --muted-color: #8fa6bf;
        --border-color: rgba(148, 163, 184, 0.18);
    }
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-color);
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(20, 26, 36, 0.95), rgba(14, 19, 28, 0.95));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 11px;
        color: var(--muted-color);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 800;
        color: #f2f4f6;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 11px;
        color: var(--cyan-accent);
    }
    .section-header {
        font-size: 18px;
        font-weight: 750;
        color: var(--cyan-accent);
        margin: 20px 0 10px 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .info-box {
        padding: 12px 16px;
        border-left: 3px solid var(--cyan-accent);
        background: rgba(53, 198, 216, 0.08);
        border-radius: 8px;
        color: #c9d8e7;
        font-size: 13px;
        line-height: 1.5;
        margin: 10px 0 16px 0;
    }
    .warn-box {
        padding: 12px 16px;
        border-left: 3px solid #ff9f43;
        background: rgba(255, 159, 67, 0.08);
        border-radius: 8px;
        color: #fbd6b0;
        font-size: 13px;
        line-height: 1.5;
        margin: 10px 0 16px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_dataset(csv_path: str):
    """Load the Solar Wind dataset from CSV with optimized types and precomputed columns."""
    if not os.path.exists(csv_path):
        return None

    # Load data
    df = pd.read_csv(csv_path)

    # Convert timedelta to numeric elapsed minutes and elapsed days if present
    if "timedelta" in df.columns:
        # Expected formats like "0 days 00:00:00" or pandas timedelta strings
        try:
            td = pd.to_timedelta(df["timedelta"])
            df["elapsed_minutes"] = td.dt.total_seconds() / 60.0
            df["elapsed_days"] = df["elapsed_minutes"] / 1440.0
        except Exception:
            # Fallback regex extraction if timedelta format is non-standard
            def parse_td_str(s):
                if pd.isna(s):
                    return np.nan
                match = re.search(r"(\d+)\s*days?\s*(\d+):(\d+):(\d+)", str(s))
                if match:
                    d, h, m, sec = map(float, match.groups())
                    return d * 1440.0 + h * 60.0 + m + sec / 60.0
                return np.nan
            df["elapsed_minutes"] = df["timedelta"].apply(parse_td_str)
            df["elapsed_days"] = df["elapsed_minutes"] / 1440.0

    # Ensure numeric columns are properly coerced
    numeric_cols = [
        "bx_gse", "by_gse", "bz_gse", "theta_gse", "phi_gse",
        "bx_gsm", "by_gsm", "bz_gsm", "theta_gsm", "phi_gsm",
        "bt", "density", "speed", "temperature", "smoothed_ssn",
        "gse_x_ace", "gse_y_ace", "gse_z_ace", "dst"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Precalculate Derived Physical Signals (as defined in FDS Practical suite)
    if all(k in df.columns for k in ["bx_gsm", "by_gsm", "bz_gsm"]):
        df["gsm_vector"] = np.sqrt(df["bx_gsm"]**2 + df["by_gsm"]**2 + df["bz_gsm"]**2)
        df["clock_angle"] = np.degrees(np.arctan2(np.abs(df["by_gsm"]), df["bz_gsm"]))
        df["southward_bz"] = np.maximum(-df["bz_gsm"], 0.0)

    if all(k in df.columns for k in ["density", "speed"]):
        df["dynamic_proxy"] = df["density"] * (df["speed"] ** 2)

    if all(k in df.columns for k in ["speed", "temperature"]):
        with np.errstate(invalid="ignore"):
            df["speed_temp_index"] = df["speed"] / np.sqrt(np.maximum(df["temperature"], 1e-6))

    return df


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & METRIC CARD BUILDER
# -----------------------------------------------------------------------------
def format_num(val, decimals=2):
    if val is None or pd.isna(val) or not np.isfinite(val):
        return "—"
    if decimals == 0:
        return f"{int(round(val)):,}"
    return f"{val:,.{decimals}f}"


def render_metric_card(col, label, value, hint=""):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-val">{value}</div>
                <div class="metric-sub">{hint}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🌌 Solar Wind Mission")
st.sidebar.caption("Analytical Suite & Interactive Control")

# Dataset Path resolution
default_csv_name = "dataset on solarwind.csv"
current_dir = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(current_dir, default_csv_name)

csv_path = st.sidebar.text_input(
    "Dataset CSV Path",
    value=default_path if os.path.exists(default_path) else default_csv_name,
    help="Enter path to 'dataset on solarwind.csv'",
)

if not os.path.exists(csv_path):
    st.error(f"⚠️ Dataset not found at: `{csv_path}`. Please verify the filename or path in the sidebar.")
    st.stop()

# Load Dataset
with st.spinner("Loading and indexing Solar Wind observations..."):
    df = load_dataset(csv_path)

if df is None or len(df) == 0:
    st.error("Failed to load or parse the dataset. Please ensure it is a valid CSV.")
    st.stop()

# Plot sample size for responsive interactivity
sample_size = st.sidebar.slider(
    "Scatter Plot Display Sample",
    min_value=2000,
    max_value=50000,
    value=15000,
    step=1000,
    help="Sampling large point clouds ensures smooth 60fps charting in the browser, exactly like the original dashboard.",
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("## 🌌 Solar Wind Mission Control Interactive Dashboard")
st.caption(
    "Dataset-first analytical mission control dashboard. "
    "Derived values are labelled; complete observations are preserved."
)

# -----------------------------------------------------------------------------
# NAVIGATION TABS (9 Dashboard Tabs)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🌌 Command Center",
    "📈 Deep Dive",
    "🧲 Relationships",
    "🧪 Preprocessing Lab",
    "🤖 Regression Lab",
    "🧬 Attribute Atlas",
    "🗃️ Data Explorer",
    "🚨 Event Detector",
    "🧠 Anomaly Lab",
])

# =============================================================================
# TAB 1: COMMAND CENTER
# =============================================================================
with tabs[0]:
    st.markdown('<div class="section-header">Operational Overview & Data Quality</div>', unsafe_allow_html=True)

    total_rows = len(df)
    total_cols = len([c for c in df.columns if not c.startswith("elapsed_")])
    time_span = df["elapsed_days"].max() if "elapsed_days" in df.columns else 728.2
    speed_cov = (df["speed"].notna().sum() / total_rows) * 100 if "speed" in df.columns else 0
    temp_cov = (df["temperature"].notna().sum() / total_rows) * 100 if "temperature" in df.columns else 0
    
    # Complete rows across original 22 attributes
    orig_22 = [
        "period", "timedelta", "bx_gse", "by_gse", "bz_gse", "theta_gse", "phi_gse",
        "bx_gsm", "by_gsm", "bz_gsm", "theta_gsm", "phi_gsm", "bt", "density",
        "speed", "temperature", "source", "smoothed_ssn", "gse_x_ace", "gse_y_ace",
        "gse_z_ace", "dst"
    ]
    present_orig = [c for c in orig_22 if c in df.columns]
    complete_rows = df[present_orig].dropna().shape[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    render_metric_card(c1, "Observations", format_num(total_rows, 0), "Total records")
    render_metric_card(c2, "Attributes", str(total_cols), "Features in CSV")
    render_metric_card(c3, "Time Span", f"{format_num(time_span, 1)} days", "Elapsed observation")
    render_metric_card(c4, "Speed Coverage", f"{speed_cov:.1f}%", f"{format_num(df['speed'].count(), 0)} observed")
    render_metric_card(c5, "Temp Coverage", f"{temp_cov:.1f}%", f"{format_num(df['temperature'].count(), 0)} observed")
    render_metric_card(c6, "Complete Rows", format_num(complete_rows, 0), f"{(complete_rows/total_rows*100):.3f}% complete cases")

    # Visualizations: Daily Evolution & Source Distribution
    st.markdown('<div class="section-header">Time Evolution & Observation Sources</div>', unsafe_allow_html=True)
    g1, g2 = st.columns([2, 1])

    with g1:
        if "elapsed_days" in df.columns:
            # Aggregate by elapsed day
            daily_agg = df.groupby(df["elapsed_days"].round().astype(int)).agg(
                mean_speed=("speed", "mean"),
                mean_bt=("bt", "mean"),
                mean_density=("density", "mean")
            ).reset_index()

            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_agg["elapsed_days"], y=daily_agg["mean_speed"],
                mode="lines", name="Speed (km/s)", line=dict(color="#35c6d8", width=1.8)
            ))
            fig_daily.add_trace(go.Scatter(
                x=daily_agg["elapsed_days"], y=daily_agg["mean_bt"],
                mode="lines", name="Bt (nT)", yaxis="y2", line=dict(color="#ff9f43", width=1.5)
            ))
            fig_daily.update_layout(
                title="Daily Evolution • Full Dataset Aggregation",
                xaxis=dict(title="Elapsed Days"),
                yaxis=dict(title="Speed (km/s)", side="left"),
                yaxis2=dict(title="Bt (nT)", overlaying="y", side="right"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_daily, width='stretch')
        else:
            st.info("Timedelta column not available for daily aggregation.")

    with g2:
        if "source" in df.columns:
            src_counts = df["source"].fillna("Missing / Unlabelled").value_counts().reset_index()
            src_counts.columns = ["source", "count"]
            fig_pie = px.pie(
                src_counts, values="count", names="source", hole=0.55,
                title="Data Source Composition",
                color_discrete_sequence=["#35c6d8", "#9b7bff", "#ff9f43", "#67c69b"]
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, width='stretch')

    # Scientific Signal Cards
    st.markdown('<div class="section-header">Derived Scientific Signals</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    render_metric_card(sc1, "Mean Speed", f"{format_num(df['speed'].mean())} km/s", "Dataset field")
    render_metric_card(sc2, "Median Density", f"{format_num(df['density'].median())} cm⁻³", "Dataset field")
    render_metric_card(sc3, "Median Temperature", f"{format_num(df['temperature'].median(), 0)} K", "Dataset field")
    render_metric_card(sc4, "Mean |B| GSM", f"{format_num(df['gsm_vector'].mean())} nT", "Derived √(Bx²+By²+Bz²)")
    render_metric_card(sc5, "Mean Dynamic Proxy", f"{format_num(df['dynamic_proxy'].mean(), 0)}", "Derived density × speed²")
    render_metric_card(sc6, "Mean Clock Angle", f"{format_num(df['clock_angle'].mean())}°", "Derived atan2(|By|, Bz)")

    # Data Quality Missingness Overview
    st.markdown('<div class="section-header">Attribute Missingness & Quality Audit</div>', unsafe_allow_html=True)
    miss_df = pd.DataFrame({
        "Attribute": df.columns,
        "Non-Null Count": df.notna().sum(),
        "Missing Count": df.isna().sum(),
        "Coverage (%)": (df.notna().sum() / total_rows * 100).round(2),
    }).sort_values(by="Coverage (%)")

    fig_missing = px.bar(
        miss_df.head(15), x="Coverage (%)", y="Attribute", orientation="h",
        color="Coverage (%)", color_continuous_scale="Viridis",
        title="Lowest 15 Features by Observation Coverage"
    )
    fig_missing.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_missing, width='stretch')


# =============================================================================
# TAB 2: DEEP DIVE
# =============================================================================
with tabs[1]:
    st.markdown('<div class="section-header">Univariate Distribution & Spread Analysis</div>', unsafe_allow_html=True)
    
    candidate_vars = [c for c in ["speed", "density", "temperature", "bt", "dst", "southward_bz"] if c in df.columns]
    selected_var = st.selectbox("Select Attribute to Inspect", candidate_vars, index=0)

    s_data = df[selected_var].dropna()
    v_count = len(s_data)
    v_mean = s_data.mean()
    v_median = s_data.median()
    v_std = s_data.std()
    v_q1 = s_data.quantile(0.25)
    v_q3 = s_data.quantile(0.75)
    v_iqr = v_q3 - v_q1
    v_min = s_data.min()
    v_max = s_data.max()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    render_metric_card(m1, "Count", format_num(v_count, 0), "Observed rows")
    render_metric_card(m2, "Mean", format_num(v_mean), "Arithmetic mean")
    render_metric_card(m3, "Median", format_num(v_median), "50th percentile")
    render_metric_card(m4, "Std Deviation", format_num(v_std), "Dispersion")
    render_metric_card(m5, "IQR", format_num(v_iqr), "Q3 − Q1")
    render_metric_card(m6, "Range", f"[{format_num(v_min)} → {format_num(v_max)}]", "Min to Max")

    st.markdown(
        f'<div class="info-box"><b>Variable Analysis:</b> {selected_var} contains {v_count:,} valid observations. '
        f'The middle 50% lies between {v_q1:.2f} and {v_q3:.2f}. '
        f'Skewness is noticeable when mean ({v_mean:.2f}) differs substantially from median ({v_median:.2f}).</div>',
        unsafe_allow_html=True
    )

    col_h, col_b = st.columns([1.6, 1])
    with col_h:
        # Sample for fast plotting if dataset is massive
        plot_sample = s_data.sample(min(len(s_data), sample_size), random_state=42)
        fig_hist = px.histogram(
            plot_sample, x=selected_var, nbins=60,
            title=f"Distribution Histogram: {selected_var}",
            color_discrete_sequence=["#35c6d8"]
        )
        fig_hist.add_vline(x=v_mean, line_dash="dash", line_color="#ff9f43", annotation_text=f"Mean: {v_mean:.1f}")
        fig_hist.add_vline(x=v_median, line_dash="dot", line_color="#67c69b", annotation_text=f"Median: {v_median:.1f}")
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            q1=[v_q1], median=[v_median], q3=[v_q3],
            lowerfence=[v_min], upperfence=[v_max],
            name=selected_var, marker_color="#9b7bff"
        ))
        fig_box.update_layout(
            title=f"Five-Number Summary Box Plot: {selected_var}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_box, width='stretch')


# =============================================================================
# TAB 3: RELATIONSHIPS
# =============================================================================
with tabs[2]:
    st.markdown('<div class="section-header">Bivariate Correlation & Association Matrix</div>', unsafe_allow_html=True)
    
    corr_candidates = [c for c in ["bx_gse", "by_gse", "bz_gse", "bt", "density", "speed", "temperature", "dst"] if c in df.columns]
    
    r1, r2 = st.columns([1.2, 1.3])
    with r1:
        # Correlation matrix
        corr_mat = df[corr_candidates].corr(method="pearson").round(3)
        fig_heat = px.imshow(
            corr_mat, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Heatmap"
        )
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, width='stretch')

    with r2:
        cx1, cx2 = st.columns(2)
        with cx1:
            scatter_x = st.selectbox("X-Axis Variable", corr_candidates, index=corr_candidates.index("density") if "density" in corr_candidates else 0)
        with cx2:
            scatter_y = st.selectbox("Y-Axis Variable", corr_candidates, index=corr_candidates.index("speed") if "speed" in corr_candidates else 1)

        sub_df = df[[scatter_x, scatter_y]].dropna()
        if len(sub_df) > sample_size:
            sub_df = sub_df.sample(sample_size, random_state=42)

        fig_scat = px.scatter(
            sub_df, x=scatter_x, y=scatter_y, opacity=0.4,
            title=f"{scatter_x} vs. {scatter_y} (Interactive Sample: {len(sub_df):,} pts)",
            color_discrete_sequence=["#35c6d8"]
        )
        fig_scat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scat, width='stretch')

    # Top correlated pairs
    st.markdown("#### Strongest Linear Relationships (Pearson |r|)")
    pairs = []
    for i in range(len(corr_candidates)):
        for j in range(i + 1, len(corr_candidates)):
            c1, c2 = corr_candidates[i], corr_candidates[j]
            val = corr_mat.loc[c1, c2]
            if pd.notna(val):
                pairs.append({"Variable A": c1, "Variable B": c2, "Pearson r": val, "Absolute r": abs(val)})
    pairs_df = pd.DataFrame(pairs).sort_values(by="Absolute r", ascending=False).drop(columns=["Absolute r"])
    st.dataframe(pairs_df.head(10), width='stretch')


# =============================================================================
# TAB 4: PREPROCESSING LAB
# =============================================================================
with tabs[3]:
    st.markdown('<div class="section-header">Normalization Transformations & Discretization</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box"><b>Preprocessing Rationale:</b> Features with distinct physical units (e.g. Speed in km/s vs Temp in Kelvin) '
        'require scaling so distance-based algorithms (like K-Means) do not get biased by attributes with large absolute scales.</div>',
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### 1. Mathematical Normalization Comparison")
        norm_var = st.selectbox("Select Attribute to Normalize", ["speed", "density", "temperature", "bt"], key="norm_sel")
        series = df[norm_var].dropna()
        
        # Calculate sample transformations
        s_min, s_max = series.min(), series.max()
        s_mean, s_std = series.mean(), series.std()
        j_dec = int(np.ceil(np.log10(max(abs(s_min), abs(s_max), 1))))
        
        sample_rows = series.head(10).reset_index(drop=True)
        norm_table = pd.DataFrame({
            "Original Value": sample_rows,
            "Min-Max [0, 1]": ((sample_rows - s_min) / (s_max - s_min)).round(4),
            "Z-Score (μ=0, σ=1)": ((sample_rows - s_mean) / s_std).round(4),
            "Decimal Scaling (÷10^j)": (sample_rows / (10 ** j_dec)).round(6),
        })
        st.dataframe(norm_table, width='stretch')

    with p2:
        st.markdown("#### 2. K-Means Discretization (K = 3 Clusters)")
        km_var = st.selectbox("Select Attribute for Clustering", ["speed", "temperature", "density", "bt"], key="km_sel")
        km_data = df[[km_var]].dropna()
        
        if len(km_data) > 30000:
            km_sample = km_data.sample(30000, random_state=42)
        else:
            km_sample = km_data.copy()

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(km_sample)
        centers = sorted(kmeans.cluster_centers_.flatten())
        
        # Assign sorted labels Low / Med / High
        center_order = np.argsort(kmeans.cluster_centers_.flatten())
        mapping = {center_order[0]: "Low", center_order[1]: "Moderate", center_order[2]: "High"}
        km_sample["Cluster"] = [mapping[c] for c in kmeans.labels_]

        cluster_counts = km_sample["Cluster"].value_counts().reindex(["Low", "Moderate", "High"]).reset_index()
        cluster_counts.columns = ["Cluster Band", "Count"]

        fig_km = px.bar(
            cluster_counts, x="Cluster Band", y="Count",
            title=f"K-Means Discretization ({km_var}, K=3)",
            color="Cluster Band",
            color_discrete_sequence=["#35c6d8", "#ff9f43", "#9b7bff"]
        )
        fig_km.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_km, width='stretch')
        st.caption(f"Cluster centroids (Low → Med → High): {centers[0]:.2f} • {centers[1]:.2f} • {centers[2]:.2f}")


# =============================================================================
# TAB 5: REGRESSION LAB
# =============================================================================
with tabs[4]:
    st.markdown('<div class="section-header">Linear Regression & Missing Value Imputation</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box"><b>Model Setup:</b> Models are trained on known complete cases using an 80/20 train/test holdout split. '
        'Metrics assess predictive fit on observed data before applying predictions to impute missing temperature values.</div>',
        unsafe_allow_html=True
    )

    reg_type = st.radio("Select Regression Model", ["Simple (Temperature from Speed)", "Multiple (Temperature from Speed + Density + Bt)"], horizontal=True)

    if reg_type.startswith("Simple"):
        feat_cols = ["speed"]
    else:
        feat_cols = ["speed", "density", "bt"]

    reg_df = df[feat_cols + ["temperature"]].dropna()
    X = reg_df[feat_cols]
    y = reg_df["temperature"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    rc1, rc2, rc3, rc4 = st.columns(4)
    render_metric_card(rc1, "Test R²", f"{r2:.4f}", "Variance explained")
    render_metric_card(rc2, "Test MAE", f"{format_num(mae, 1)} K", "Mean Absolute Error")
    render_metric_card(rc3, "Test RMSE", f"{format_num(rmse, 1)} K", "Root Mean Squared Error")
    render_metric_card(rc4, "Train / Test Size", f"{len(X_train):,} / {len(X_test):,}", "Complete rows")

    # Display fitted formula
    if len(feat_cols) == 1:
        eq = f"Temperature = {model.intercept_:.2f} + {model.coef_[0]:.4f} × Speed"
    else:
        eq = f"Temperature = {model.intercept_:.2f} + {model.coef_[0]:.4f} × Speed + {model.coef_[1]:.4f} × Density + {model.coef_[2]:.4f} × Bt"
    st.markdown(f"**Fitted Model Equation:** `{eq}`")

    # Actual vs Predicted plot
    plot_points = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    if len(plot_points) > sample_size:
        plot_points = plot_points.sample(sample_size, random_state=42)

    fig_reg = px.scatter(
        plot_points, x="Actual", y="Predicted", opacity=0.4,
        title="Actual vs. Predicted Temperature (Hold-out Test Set)",
        color_discrete_sequence=["#35c6d8"]
    )
    # Perfect prediction line
    line_min = min(plot_points["Actual"].min(), plot_points["Predicted"].min())
    line_max = max(plot_points["Actual"].max(), plot_points["Predicted"].max())
    fig_reg.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max],
        mode="lines", name="Identity Line (y = x)", line=dict(color="#ff9f43", dash="dash")
    ))
    fig_reg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_reg, width='stretch')

    # Imputation Analysis Section
    st.markdown("#### Regression-Based Imputation Capacity")
    missing_temp_mask = df["temperature"].isna()
    can_impute_mask = missing_temp_mask & df["speed"].notna() & df["density"].notna() & df["bt"].notna()
    
    ic1, ic2, ic3 = st.columns(3)
    render_metric_card(ic1, "Missing Temperature Rows", format_num(missing_temp_mask.sum(), 0), "Unobserved in CSV")
    render_metric_card(ic2, "Imputable Rows", format_num(can_impute_mask.sum(), 0), "Predictors fully present")
    render_metric_card(ic3, "Imputation Viability", f"{(can_impute_mask.sum()/missing_temp_mask.sum()*100):.1f}%", "Recoverable with multiple model")


# =============================================================================
# TAB 6: ATTRIBUTE ATLAS
# =============================================================================
with tabs[5]:
    st.markdown('<div class="section-header">Attribute Atlas & Measurement Scales</div>', unsafe_allow_html=True)
    
    scale_dict = {
        "period": ("Categorical", "Nominal", "Discrete"),
        "timedelta": ("Temporal", "Interval", "Continuous"),
        "bx_gse": ("Numerical", "Ratio", "Continuous"),
        "by_gse": ("Numerical", "Ratio", "Continuous"),
        "bz_gse": ("Numerical", "Ratio", "Continuous"),
        "theta_gse": ("Numerical", "Interval", "Continuous"),
        "phi_gse": ("Numerical", "Interval", "Continuous"),
        "bx_gsm": ("Numerical", "Ratio", "Continuous"),
        "by_gsm": ("Numerical", "Ratio", "Continuous"),
        "bz_gsm": ("Numerical", "Ratio", "Continuous"),
        "theta_gsm": ("Numerical", "Interval", "Continuous"),
        "phi_gsm": ("Numerical", "Interval", "Continuous"),
        "bt": ("Numerical", "Ratio", "Continuous"),
        "density": ("Numerical", "Ratio", "Continuous"),
        "speed": ("Numerical", "Ratio", "Continuous"),
        "temperature": ("Numerical", "Ratio", "Continuous"),
        "source": ("Categorical", "Nominal", "Discrete"),
        "smoothed_ssn": ("Numerical", "Ratio", "Continuous"),
        "gse_x_ace": ("Numerical", "Interval", "Continuous"),
        "gse_y_ace": ("Numerical", "Interval", "Continuous"),
        "gse_z_ace": ("Numerical", "Interval", "Continuous"),
        "dst": ("Numerical", "Interval", "Continuous"),
    }

    atlas_records = []
    for col in df.columns:
        if col.startswith("elapsed_"):
            continue
        scale_info = scale_dict.get(col, ("Derived / Numerical", "Ratio", "Continuous"))
        atlas_records.append({
            "Column Name": col,
            "Pandas Dtype": str(df[col].dtype),
            "Scale Type": scale_info[1],
            "Continuity": scale_info[2],
            "Non-Null Count": df[col].notna().sum(),
            "Missing Count": df[col].isna().sum(),
            "Coverage (%)": f"{(df[col].notna().sum() / len(df) * 100):.2f}%",
            "Unique Values": df[col].nunique(),
        })

    atlas_df = pd.DataFrame(atlas_records)
    st.dataframe(atlas_df, use_container_width=True, height=500)


# =============================================================================
# TAB 7: DATA EXPLORER
# =============================================================================
with tabs[6]:
    st.markdown('<div class="section-header">Live Interactive Data Explorer</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">Filter, search, inspect rows, and export sliced datasets dynamically.</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns([1.5, 1, 1])
    with f1:
        search_query = st.text_input("Text Search Across Records", placeholder="Search period, source, etc...")
    with f2:
        src_filter = st.selectbox("Source Filter", ["All Sources"] + list(df["source"].dropna().unique()))
    with f3:
        page_size = st.selectbox("Rows Per Page", [25, 50, 100, 250], index=1)

    # Range filters
    rf1, rf2, rf3, rf4, rf5 = st.columns(5)
    with rf1:
        sp_min = st.number_input("Speed Min (km/s)", value=float(df["speed"].min() if "speed" in df.columns else 0.0))
        sp_max = st.number_input("Speed Max (km/s)", value=float(df["speed"].max() if "speed" in df.columns else 1000.0))
    with rf2:
        den_min = st.number_input("Density Min", value=float(df["density"].min() if "density" in df.columns else 0.0))
        den_max = st.number_input("Density Max", value=float(df["density"].max() if "density" in df.columns else 150.0))
    with rf3:
        tmp_min = st.number_input("Temp Min (K)", value=float(df["temperature"].min() if "temperature" in df.columns else 0.0))
        tmp_max = st.number_input("Temp Max (K)", value=float(df["temperature"].max() if "temperature" in df.columns else 3500000.0))
    with rf4:
        bt_min = st.number_input("Bt Min (nT)", value=float(df["bt"].min() if "bt" in df.columns else 0.0))
        bt_max = st.number_input("Bt Max (nT)", value=float(df["bt"].max() if "bt" in df.columns else 100.0))
    with rf5:
        dst_min = st.number_input("Dst Min (nT)", value=float(df["dst"].min() if "dst" in df.columns else -400.0))
        dst_max = st.number_input("Dst Max (nT)", value=float(df["dst"].max() if "dst" in df.columns else 100.0))

    # Apply filters on sample
    filtered_df = df.copy()
    if src_filter != "All Sources":
        filtered_df = filtered_df[filtered_df["source"] == src_filter]

    if "speed" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["speed"].isna()) | ((filtered_df["speed"] >= sp_min) & (filtered_df["speed"] <= sp_max))]
    if "density" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["density"].isna()) | ((filtered_df["density"] >= den_min) & (filtered_df["density"] <= den_max))]
    if "temperature" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["temperature"].isna()) | ((filtered_df["temperature"] >= tmp_min) & (filtered_df["temperature"] <= tmp_max))]
    if "bt" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["bt"].isna()) | ((filtered_df["bt"] >= bt_min) & (filtered_df["bt"] <= bt_max))]
    if "dst" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["dst"].isna()) | ((filtered_df["dst"] >= dst_min) & (filtered_df["dst"] <= dst_max))]

    if search_query:
        mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
        filtered_df = filtered_df[mask]

    st.write(f"**Matches Found:** `{len(filtered_df):,}` rows out of `{len(df):,}` records")

    # Column selection
    all_displayable = [c for c in df.columns if not c.startswith("elapsed_")]
    default_selected = [c for c in ["timedelta", "speed", "density", "temperature", "bt", "dst", "source"] if c in all_displayable]
    selected_cols = st.multiselect("Display Columns", all_displayable, default=default_selected)

    # Render table page
    page_df = filtered_df[selected_cols].head(page_size)
    st.dataframe(page_df, width='stretch')

    # CSV Download button
    csv_buffer = io.StringIO()
    page_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇ Download Filtered Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="solarwind_filtered_explorer.csv",
        mime="text/csv",
    )


# =============================================================================
# TAB 8: EVENT DETECTOR
# =============================================================================
with tabs[7]:
    st.markdown('<div class="section-header">Solar Wind Space Weather Event Detector</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box">Events are detected when key physical thresholds indicating geomagnetic storms, '
        'interplanetary shocks, or Coronal Mass Ejections (CMEs) are triggered.</div>',
        unsafe_allow_html=True
    )

    ev_cols = st.columns(6)
    with ev_cols[0]:
        th_speed = st.number_input("High Speed (≥ km/s)", value=600.0, step=10.0)
    with ev_cols[1]:
        th_den = st.number_input("High Density (≥ cm⁻³)", value=10.0, step=1.0)
    with ev_cols[2]:
        th_temp = st.number_input("High Temp (≥ K)", value=200000.0, step=10000.0)
    with ev_cols[3]:
        th_bt = st.number_input("Strong Bt (≥ nT)", value=12.0, step=1.0)
    with ev_cols[4]:
        th_bz = st.number_input("Southward Bz (≥ nT)", value=5.0, step=1.0)
    with ev_cols[5]:
        th_dst = st.number_input("Dst Storm (≤ nT)", value=-50.0, step=5.0)

    # Compute triggers
    hit_speed = df["speed"] >= th_speed
    hit_den = df["density"] >= th_den
    hit_temp = df["temperature"] >= th_temp
    hit_bt = df["bt"] >= th_bt
    hit_bz = df["southward_bz"] >= th_bz if "southward_bz" in df.columns else pd.Series(False, index=df.index)
    hit_dst = df["dst"] <= th_dst

    trigger_matrix = pd.concat([hit_speed, hit_den, hit_temp, hit_bt, hit_bz, hit_dst], axis=1)
    trigger_matrix.columns = ["Speed", "Density", "Temperature", "Bt", "Southward Bz", "Dst"]
    
    trigger_counts = trigger_matrix.sum(axis=1)
    any_trigger = (trigger_counts >= 1).sum()
    multi_trigger = (trigger_counts >= 2).sum()
    severe_trigger = (trigger_counts >= 3).sum()

    ev_c1, ev_c2, ev_c3, ev_c4, ev_c5, ev_c6 = st.columns(6)
    render_metric_card(ev_c1, "Any Trigger", format_num(any_trigger, 0), "≥ 1 rule met")
    render_metric_card(ev_c2, "Multi-Trigger", format_num(multi_trigger, 0), "≥ 2 rules met")
    render_metric_card(ev_c3, "Severe Event", format_num(severe_trigger, 0), "≥ 3 rules met")
    render_metric_card(ev_c4, "Any-Trigger Rate", f"{(any_trigger / len(df) * 100):.2f}%", "Of total dataset")
    render_metric_card(ev_c5, "Multi-Trigger Rate", f"{(multi_trigger / len(df) * 100):.2f}%", "Of total dataset")
    render_metric_card(ev_c6, "Severe Rate", f"{(severe_trigger / len(df) * 100):.2f}%", "Of total dataset")

    # Bar chart of condition trigger counts
    rule_breakdown = pd.DataFrame({
        "Rule": ["High Speed", "High Density", "High Temperature", "Strong Bt", "Southward Bz", "Dst Storm"],
        "Triggered Rows": [hit_speed.sum(), hit_den.sum(), hit_temp.sum(), hit_bt.sum(), hit_bz.sum(), hit_dst.sum()]
    })
    fig_rule = px.bar(
        rule_breakdown, x="Rule", y="Triggered Rows",
        color="Rule", title="Event Trigger Counts Under Configured Thresholds",
        color_discrete_sequence=["#35c6d8", "#ff9f43", "#ff6b81", "#9b7bff", "#67c69b", "#e056fd"]
    )
    fig_rule.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rule, width='stretch')

    # Top severe event observations
    st.markdown("#### Top Extreme Event Observations")
    event_df = df[trigger_counts >= 2].copy()
    if len(event_df) > 0:
        event_df["Active Triggers"] = trigger_counts[trigger_counts >= 2]
        display_event_cols = [c for c in ["timedelta", "speed", "density", "temperature", "bt", "southward_bz", "dst", "Active Triggers"] if c in event_df.columns]
        st.dataframe(event_df[display_event_cols].sort_values(by="Active Triggers", ascending=False).head(25), width='stretch')


# =============================================================================
# TAB 9: ANOMALY LAB
# =============================================================================
with tabs[8]:
    st.markdown('<div class="section-header">Statistical Anomaly Detection (Z-Score & IQR)</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box">Detect outliers using statistical thresholds. Z-score identifies points deviating by '
        'k standard deviations (μ ± kσ); IQR identifies points outside [Q1 - k·IQR, Q3 + k·IQR].</div>',
        unsafe_allow_html=True
    )

    an_c1, an_c2, an_c3 = st.columns(3)
    with an_c1:
        an_method = st.selectbox("Detection Method", ["Z-score", "Interquartile Range (IQR)"])
    with an_c2:
        an_var = st.selectbox("Target Metric", ["All Physical Variables", "speed", "density", "temperature", "bt", "dst"])
    with an_c3:
        an_param = st.number_input("Threshold / Multiplier (k)", value=3.0 if an_method == "Z-score" else 1.5, step=0.1)

    eval_vars = ["speed", "density", "temperature", "bt", "dst"] if an_var == "All Physical Variables" else [an_var]
    
    anomaly_masks = {}
    for col in eval_vars:
        if col in df.columns:
            s = df[col]
            if an_method == "Z-score":
                mean, std = s.mean(), s.std()
                anomaly_masks[col] = (s < mean - an_param * std) | (s > mean + an_param * std)
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                anomaly_masks[col] = (s < q1 - an_param * iqr) | (s > q3 + an_param * iqr)

    an_combined = pd.DataFrame(anomaly_masks).any(axis=1)
    an_total = an_combined.sum()

    ak1, ak2, ak3, ak4 = st.columns(4)
    render_metric_card(ak1, "Anomalous Observations", format_num(an_total, 0), "Met outlier criteria")
    render_metric_card(ak2, "Anomaly Rate", f"{(an_total / len(df) * 100):.2f}%", "Of total records")
    render_metric_card(ak3, "Detection Engine", an_method, f"k = {an_param:.1f}")
    render_metric_card(ak4, "Variables Monitored", str(len(eval_vars)), ", ".join(eval_vars))

    # Anomaly breakdown chart
    an_counts_df = pd.DataFrame({
        "Feature": list(anomaly_masks.keys()),
        "Anomalies": [anomaly_masks[k].sum() for k in anomaly_masks.keys()]
    })
    fig_an = px.bar(
        an_counts_df, x="Feature", y="Anomalies", color="Feature",
        title=f"Anomaly Counts by Feature ({an_method}, k={an_param})",
        color_discrete_sequence=["#35c6d8", "#ff9f43", "#9b7bff", "#ff6b81", "#67c69b"]
    )
    fig_an.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_an, width='stretch')

    # Top anomalies table
    st.markdown("#### Most Extreme Anomalous Records")
    an_rows = df[an_combined].copy()
    if len(an_rows) > 0:
        disp_cols = [c for c in ["timedelta", "speed", "density", "temperature", "bt", "dst"] if c in an_rows.columns]
        st.dataframe(an_rows[disp_cols].head(25), width='stretch')


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8fa6bf; font-size: 12px;'>"
    "Solar Wind Mission Control Analytics • Python Interactive Dashboard • Dataset-First Architecture"
    "</div>",
    unsafe_allow_html=True
)
