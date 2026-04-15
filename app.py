"""
app.py — Streamlit Dashboard
─────────────────────────────
Interactive EDA dashboard for the Global Health & Socioeconomic Indicators dataset.

Sections:
  1. Data Quality Overview   — missing values, raw vs clean comparison
  2. Descriptive Statistics  — summary table with key metrics
  3. Correlation Explorer    — interactive heatmap
  4. Distribution Analysis   — histogram for any indicator
  5. Country Rankings        — top/bottom countries per metric
  6. Global Trends           — year-over-year line charts
  7. Scatter & Regression    — explore bivariate relationships
  8. Income Group Comparison — box plots across economic tiers
  9. Multi-Country Comparison — overlay trends for selected nations

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from src.cleaning import clean_pipeline, RAW_PATH, CLEAN_PATH
from src.analysis import (
    load_clean_data, descriptive_summary, correlation_matrix,
    country_rankings, yearly_trends, add_income_groups,
    group_comparison, top_correlations,
)
from src.visualization import (
    correlation_heatmap, distribution_plot, country_bar_chart,
    trend_line, scatter_with_regression, income_group_boxplot,
    missing_data_chart, multi_country_trend,
)

# ── Page configuration ──
st.set_page_config(
    page_title="Global Health EDA Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ── Sidebar controls ──
st.sidebar.title("🌍 Global Health EDA")
st.sidebar.markdown("Interactive exploration of messy real-world health data.")


@st.cache_data
def get_data():
    """Load raw and cleaned datasets, running the pipeline if needed."""
    if not CLEAN_PATH.exists():
        clean_pipeline()
    raw = pd.read_csv(RAW_PATH)
    clean = load_clean_data()
    clean = add_income_groups(clean)
    return raw, clean


raw_df, df = get_data()

# Available numeric columns for analysis
METRIC_OPTIONS = [
    "life_expectancy", "gdp_per_capita", "infant_mortality",
    "health_exp_pct_gdp", "physicians_per_1000", "co2_per_capita",
    "electricity_access_pct", "literacy_rate", "unemployment_rate",
]

# ────────────────────────────────────────────
#  Section 1: Data Quality Overview
# ────────────────────────────────────────────
st.title("🌍 Global Health & Socioeconomic Indicators — EDA Dashboard")
st.markdown("""
This dashboard demonstrates end-to-end data wrangling and exploratory analysis
on an intentionally messy dataset. The raw data contains **inconsistent country names,
mixed-type columns, impossible values, duplicates, and scattered missing data** —
all of which are cleaned programmatically before analysis.
""")

st.header("📊 Data Quality Overview")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Data")
    st.metric("Rows", raw_df.shape[0])
    st.metric("Columns", raw_df.shape[1])
    st.plotly_chart(missing_data_chart(raw_df, "Missing Data — Raw"), use_container_width=True)
with col2:
    st.subheader("Cleaned Data")
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    st.plotly_chart(missing_data_chart(df, "Missing Data — Cleaned"), use_container_width=True)

st.info(f"🧹 Cleaning removed **{raw_df.shape[0] - df.shape[0]}** rows "
        f"(duplicates + unmapped countries) and fixed column types, "
        f"impossible values, and inconsistent dates.")

# ────────────────────────────────────────────
#  Section 2: Descriptive Statistics
# ────────────────────────────────────────────
st.header("📈 Descriptive Statistics")
desc = descriptive_summary(df)
st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

# ────────────────────────────────────────────
#  Section 3: Correlation Explorer
# ────────────────────────────────────────────
st.header("🔗 Correlation Explorer")

corr = correlation_matrix(df)
st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

st.subheader("Strongest Relationships")
top_corr = top_correlations(df, n=8)
st.dataframe(top_corr, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────
#  Section 4: Distribution Analysis
# ────────────────────────────────────────────
st.header("📐 Distribution Analysis")
dist_metric = st.selectbox("Select indicator:", METRIC_OPTIONS, key="dist")
st.plotly_chart(distribution_plot(df, dist_metric), use_container_width=True)

# ────────────────────────────────────────────
#  Section 5: Country Rankings
# ────────────────────────────────────────────
st.header("🏆 Country Rankings")
rank_col1, rank_col2 = st.columns(2)
with rank_col1:
    rank_metric = st.selectbox("Metric:", METRIC_OPTIONS, key="rank")
with rank_col2:
    rank_year = st.selectbox("Year:", [None] + sorted(df["Year"].unique().tolist()), key="rank_year")

ranked = country_rankings(df, metric=rank_metric, year=rank_year)
st.plotly_chart(country_bar_chart(ranked, rank_metric), use_container_width=True)

# ────────────────────────────────────────────
#  Section 6: Global Trends
# ────────────────────────────────────────────
st.header("📅 Global Trends Over Time")
trend_metric = st.selectbox("Select indicator:", METRIC_OPTIONS, key="trend")
trend = yearly_trends(df, metric=trend_metric)
st.plotly_chart(trend_line(trend, trend_metric), use_container_width=True)

# ────────────────────────────────────────────
#  Section 7: Scatter & Regression
# ────────────────────────────────────────────
st.header("🔬 Scatter Plot & Regression")
scat_col1, scat_col2 = st.columns(2)
with scat_col1:
    x_var = st.selectbox("X-axis:", METRIC_OPTIONS, index=1, key="scat_x")
with scat_col2:
    y_var = st.selectbox("Y-axis:", METRIC_OPTIONS, index=0, key="scat_y")

colour_by = st.checkbox("Colour by income group", value=True)
fig = scatter_with_regression(
    df, x_var, y_var,
    color_col="income_group" if colour_by else None,
)
st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────
#  Section 8: Income Group Comparison
# ────────────────────────────────────────────
st.header("💰 Income Group Comparison")
box_metric = st.selectbox("Select indicator:", METRIC_OPTIONS, key="box")
st.plotly_chart(income_group_boxplot(df, box_metric), use_container_width=True)

# Show group summary table
st.subheader("Group Summary")
st.dataframe(group_comparison(df, metric=box_metric), use_container_width=True)

# ────────────────────────────────────────────
#  Section 9: Multi-Country Comparison
# ────────────────────────────────────────────
st.header("🌐 Multi-Country Comparison")
all_countries = sorted(df["Country"].unique().tolist())
default_countries = ["United States", "China", "India", "Brazil", "Germany"]
defaults = [c for c in default_countries if c in all_countries]

selected = st.multiselect("Select countries:", all_countries, default=defaults)
mc_metric = st.selectbox("Select indicator:", METRIC_OPTIONS, key="mc")

if selected:
    st.plotly_chart(multi_country_trend(df, selected, mc_metric), use_container_width=True)
else:
    st.warning("Select at least one country to compare.")

# ── Footer ──
st.markdown("---")
st.markdown(
    "**Built by Arie DeKraker** · "
    "[GitHub](https://github.com/HaveACigar/eda-visualization-dashboard) · "
    "Data is synthetically generated for demonstration purposes."
)
