"""
visualization.py
────────────────
Visualization module for the Global Health dashboard.

Provides reusable chart-building functions using Plotly for interactivity.
Each function returns a Plotly Figure object that can be displayed in
Streamlit via st.plotly_chart() or rendered standalone.

Chart types included:
  • Correlation heatmap
  • Distribution histograms
  • Country comparison bar charts
  • Time series trend lines
  • Scatter plots with regression lines
  • Income group box plots
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# ── Consistent colour palette for the dashboard ──
PALETTE = px.colors.qualitative.Set2
INCOME_COLORS = {
    "High": "#2196F3",
    "Upper-Middle": "#4CAF50",
    "Lower-Middle": "#FF9800",
    "Low": "#F44336",
    "Unknown": "#9E9E9E",
}


def correlation_heatmap(corr_matrix, title="Indicator Correlation Matrix"):
    """
    Interactive heatmap of pairwise correlations.
    Strong positive = blue, strong negative = red.
    Annotations show the correlation coefficient on each cell.
    """
    labels = corr_matrix.columns.tolist()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        width=700, height=600,
        xaxis_tickangle=-45,
        margin=dict(l=120, b=120),
    )
    return fig


def distribution_plot(df, column, nbins=40, title=None):
    """
    Histogram with KDE curve overlay for any numeric column.
    Shows the distribution shape, central tendency, and spread.
    """
    fig = px.histogram(
        df, x=column, nbins=nbins, marginal="box",
        title=title or f"Distribution of {column}",
        color_discrete_sequence=["#1976D2"],
        opacity=0.75,
    )
    fig.update_layout(
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Count",
        showlegend=False,
    )
    return fig


def country_bar_chart(ranked_df, metric, top_n=15, title=None):
    """
    Horizontal bar chart showing top-N countries ranked by a metric.
    Expects output from analysis.country_rankings().
    """
    top = ranked_df.head(top_n)
    fig = px.bar(
        top, x=metric, y="Country", orientation="h",
        title=title or f"Top {top_n} Countries by {metric.replace('_', ' ').title()}",
        color=metric,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 30),
        coloraxis_showscale=False,
    )
    return fig


def trend_line(trend_df, metric, title=None):
    """
    Line chart showing a global metric trend over time.
    Expects output from analysis.yearly_trends().
    """
    col = f"mean_{metric}"
    fig = px.line(
        trend_df, x="Year", y=col,
        title=title or f"Global {metric.replace('_', ' ').title()} Over Time",
        markers=True,
        color_discrete_sequence=["#1976D2"],
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=metric.replace("_", " ").title(),
    )
    return fig


def scatter_with_regression(df, x_col, y_col, color_col=None, title=None):
    """
    Scatter plot of two indicators with an OLS trendline.
    Optionally colour points by a categorical column (e.g., income_group).

    Great for exploring relationships like GDP vs Life Expectancy.
    """
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        trendline="ols",
        title=title or f"{y_col.replace('_',' ').title()} vs {x_col.replace('_',' ').title()}",
        opacity=0.6,
        color_discrete_map=INCOME_COLORS if color_col == "income_group" else None,
        hover_data=["Country", "Year"],
    )
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
    )
    return fig


def income_group_boxplot(df, metric, title=None):
    """
    Box plot comparing a metric across income groups.
    Shows median, IQR, and outliers for each group.
    """
    # Order groups logically
    order = ["High", "Upper-Middle", "Lower-Middle", "Low", "Unknown"]
    fig = px.box(
        df, x="income_group", y=metric,
        title=title or f"{metric.replace('_',' ').title()} by Income Group",
        color="income_group",
        color_discrete_map=INCOME_COLORS,
        category_orders={"income_group": order},
    )
    fig.update_layout(
        xaxis_title="Income Group",
        yaxis_title=metric.replace("_", " ").title(),
        showlegend=False,
    )
    return fig


def missing_data_chart(df, title="Missing Data by Column"):
    """
    Bar chart showing the percentage of missing values per column.
    Essential for understanding data quality before imputation.
    """
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    fig = px.bar(
        x=missing.values, y=missing.index, orientation="h",
        title=title,
        labels={"x": "Missing %", "y": "Column"},
        color=missing.values,
        color_continuous_scale="OrRd",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    return fig


def multi_country_trend(df, countries, metric, title=None):
    """
    Overlaid line chart comparing a metric over time for selected countries.
    Enables direct comparison of country-level trajectories.
    """
    subset = df[df["Country"].isin(countries)]
    yearly = subset.groupby(["Year", "Country"])[metric].mean().reset_index()
    fig = px.line(
        yearly, x="Year", y=metric, color="Country",
        title=title or f"{metric.replace('_',' ').title()} — Country Comparison",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=metric.replace("_", " ").title(),
    )
    return fig
