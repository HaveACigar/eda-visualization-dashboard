"""
analysis.py
───────────
Statistical analysis module for the cleaned Global Health dataset.

Provides functions for:
  • Descriptive statistics & distribution summaries
  • Correlation analysis across indicators
  • Country-level aggregations and rankings
  • Year-over-year trend detection
  • Group comparisons (e.g., high-income vs low-income nations)

All functions expect the CLEANED DataFrame (output of cleaning.clean_pipeline).
"""

import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_PATH = Path(__file__).parent.parent / "data" / "processed" / "global_health_clean.csv"

# GDP thresholds for income group classification (World Bank-inspired)
INCOME_THRESHOLDS = {
    "Low": (0, 1_045),
    "Lower-Middle": (1_046, 4_095),
    "Upper-Middle": (4_096, 12_695),
    "High": (12_696, float("inf")),
}


def load_clean_data(path=None):
    """Load the cleaned dataset, parsing dates automatically."""
    src = path or CLEAN_PATH
    df = pd.read_csv(src, parse_dates=["last_updated"])
    return df


def descriptive_summary(df):
    """
    Compute a comprehensive descriptive summary for all numeric indicators.

    Returns a DataFrame with count, mean, std, min, max, median, skewness,
    and the percentage of missing values per column.
    """
    numeric = df.select_dtypes(include=[np.number])
    desc = numeric.describe().T
    desc["median"] = numeric.median()
    desc["skew"] = numeric.skew()
    desc["missing_pct"] = (numeric.isnull().sum() / len(df) * 100).round(2)
    return desc


def correlation_matrix(df):
    """
    Compute pairwise Pearson correlations between all numeric indicators.
    Useful for identifying which health/economic factors move together.
    """
    numeric_cols = [
        "life_expectancy", "gdp_per_capita", "infant_mortality",
        "health_exp_pct_gdp", "physicians_per_1000", "co2_per_capita",
        "electricity_access_pct", "literacy_rate", "unemployment_rate",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    return df[cols].corr()


def classify_income_group(gdp):
    """Assign a World Bank-style income group based on GDP per capita."""
    if pd.isna(gdp):
        return "Unknown"
    for group, (lo, hi) in INCOME_THRESHOLDS.items():
        if lo <= gdp <= hi:
            return group
    return "Unknown"


def add_income_groups(df):
    """
    Add an 'income_group' column derived from the country's median GDP
    across all available years. This enables grouped comparisons.
    """
    # Compute each country's median GDP across years
    country_gdp = df.groupby("Country")["gdp_per_capita"].median()
    income_map = country_gdp.apply(classify_income_group)
    df = df.copy()
    df["income_group"] = df["Country"].map(income_map)
    return df


def country_rankings(df, metric="life_expectancy", year=None, ascending=False):
    """
    Rank countries by a given metric. Optionally filter to a specific year.

    Parameters
    ----------
    df : DataFrame
    metric : str — column to rank by
    year : int or None — if provided, filter to that year
    ascending : bool — False = highest first (default)

    Returns a DataFrame: Country, metric value, rank.
    """
    subset = df if year is None else df[df["Year"] == year]
    ranked = (
        subset.groupby("Country")[metric]
        .median()
        .sort_values(ascending=ascending)
        .reset_index()
    )
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def yearly_trends(df, metric="life_expectancy"):
    """
    Compute the global mean of a metric per year to show trends over time.
    Also returns the year-over-year percentage change.
    """
    trend = df.groupby("Year")[metric].mean().reset_index()
    trend.columns = ["Year", f"mean_{metric}"]
    trend["yoy_change_pct"] = trend[f"mean_{metric}"].pct_change() * 100
    return trend


def group_comparison(df, group_col="income_group", metric="life_expectancy"):
    """
    Compare a metric across groups (e.g., income groups).
    Returns mean, median, std, count per group.
    """
    return (
        df.groupby(group_col)[metric]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )


def top_correlations(df, n=10):
    """
    Extract the top-N strongest pairwise correlations (excluding self-correlations).
    Useful for quickly spotting the most meaningful relationships.
    """
    corr = correlation_matrix(df)
    # Unstack and filter out diagonal + duplicates
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["Var1", "Var2", "Correlation"]
    pairs["abs_corr"] = pairs["Correlation"].abs()
    return pairs.nlargest(n, "abs_corr").drop(columns="abs_corr")


if __name__ == "__main__":
    df = load_clean_data()
    print("=== Descriptive Summary ===")
    print(descriptive_summary(df).to_string())
    print("\n=== Top 10 Correlations ===")
    print(top_correlations(df).to_string(index=False))
    print("\n=== Life Expectancy by Income Group ===")
    df = add_income_groups(df)
    print(group_comparison(df).to_string())
