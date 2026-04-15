"""
cleaning.py
───────────
Data wrangling module for the Global Health & Socioeconomic Indicators dataset.

This module tackles the intentionally messy raw data:
  • Column names with leading/trailing whitespace
  • Country names with inconsistent capitalisation, typos, and aliases
  • Mixed types in numeric columns (strings like "$1,234" in gdp_per_capita)
  • Impossible negative values in health metrics
  • Impossible outlier values (e.g., life_expectancy > 120)
  • Duplicate rows
  • Inconsistent date formats in last_updated
  • Scattered NaN values across all indicator columns

Each function is designed to be called in sequence via `clean_pipeline()`,
but can also be used independently for demonstration purposes.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ── Paths ──
RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "global_health_messy.csv"
CLEAN_PATH = Path(__file__).parent.parent / "data" / "processed" / "global_health_clean.csv"

# ── Canonical country name mapping ──
# Handles typos, alternate spellings, casing issues, and aliases
COUNTRY_MAP = {
    "united states": "United States",
    "usa": "United States",
    "canada": "Canada",
    "méxico": "Mexico",
    "mexico": "Mexico",
    "brasil": "Brazil",
    "brazil": "Brazil",
    "argentina": "Argentina",
    "colombia": "Colombia",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "france": "France",
    "germany": "Germany",
    "italy": "Italy",
    "spain": "Spain",
    "nigeria": "Nigeria",
    "south africa": "South Africa",
    "kenya": "Kenya",
    "egypt": "Egypt",
    "china": "China",
    "india": "India",
    "japan": "Japan",
    "australia": "Australia",
    "new zealand": "New Zealand",
    "indonesia": "Indonesia",
    "thailand": "Thailand",
    "russia": "Russia",
    "turkey": "Turkey",
    "saudi arabia": "Saudi Arabia",
    "south korea": "South Korea",
    "sweden": "Sweden",
    "norway": "Norway",
    "netherlands": "Netherlands",
    "poland": "Poland",
    "chile": "Chile",
    "peru": "Peru",
    "philippines": "Philippines",
    "vietnam": "Vietnam",
    "pakistan": "Pakistan",
    "bangladesh": "Bangladesh",
    "ethiopia": "Ethiopia",
    "ghana": "Ghana",
    "morocco": "Morocco",
    "tanzania": "Tanzania",
    "dr congo": "DR Congo",
    "malaysia": "Malaysia",
    "singapore": "Singapore",
    "israel": "Israel",
    "ireland": "Ireland",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
}


def fix_column_names(df):
    """
    Strip leading/trailing whitespace from column names.
    The raw data has columns like ' life_expectancy' and 'unemployment_rate '.
    """
    df.columns = df.columns.str.strip()
    return df


def standardize_countries(df):
    """
    Normalise country names using a canonical mapping.

    Raw issues handled:
      - 'united states', 'USA', ' United States ' → 'United States'
      - 'brasil' → 'Brazil'
      - 'Czechia' → 'Czech Republic'
      - Extra whitespace stripped before lookup
    """
    df["Country"] = (
        df["Country"]
        .str.strip()
        .str.lower()
        .map(COUNTRY_MAP)
    )
    # Flag any unmapped countries so they can be investigated
    unmapped = df["Country"].isna().sum()
    if unmapped > 0:
        print(f"⚠️  {unmapped} rows have unmapped country names — dropping them.")
        df = df.dropna(subset=["Country"])
    return df


def fix_gdp_column(df):
    """
    Convert gdp_per_capita from mixed types to float.

    Raw issues handled:
      - '$1,234' → 1234.0
      - '45,678.90' → 45678.90
      - ' 12345 ' → 12345.0
      - 'N/A' → NaN
    """
    def _parse_gdp(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.upper() in ("N/A", "NA", "NULL", "NONE", ""):
            return np.nan
        # Remove dollar signs and commas
        s = re.sub(r"[$,]", "", s)
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["gdp_per_capita"] = df["gdp_per_capita"].apply(_parse_gdp)
    return df


def remove_duplicates(df):
    """
    Drop exact duplicate rows. The raw data has ~3% duplicated rows injected.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"🗑️  Removed {before - after} duplicate rows ({before} → {after})")
    return df


def fix_impossible_values(df):
    """
    Replace physically impossible values with NaN:
      - Negative infant_mortality (should be ≥ 0)
      - life_expectancy > 120 (injected outliers)
      - Percentages outside 0–100 range
    """
    # Negative infant mortality → NaN
    neg_mask = df["infant_mortality"] < 0
    df.loc[neg_mask, "infant_mortality"] = np.nan
    print(f"🔧 Fixed {neg_mask.sum()} negative infant_mortality values")

    # Impossible life expectancy (> 120 years)
    outlier_mask = df["life_expectancy"] > 120
    df.loc[outlier_mask, "life_expectancy"] = np.nan
    print(f"🔧 Fixed {outlier_mask.sum()} impossible life_expectancy outliers")

    # Percentage columns clamped to [0, 100]
    pct_cols = ["health_exp_pct_gdp", "electricity_access_pct", "literacy_rate", "unemployment_rate"]
    for col in pct_cols:
        if col in df.columns:
            invalid = (df[col] < 0) | (df[col] > 100)
            df.loc[invalid, col] = np.nan

    return df


def parse_dates(df):
    """
    Parse the last_updated column which has mixed date formats:
      - '2023-05-14', '05/14/2023', '14-May-2023', 'May 14, 2023', '2023/05/14'

    pandas.to_datetime with infer_datetime_format handles most of these.
    """
    df["last_updated"] = pd.to_datetime(df["last_updated"], format="mixed", dayfirst=False)
    return df


def clean_pipeline(input_path=None, output_path=None):
    """
    Run the full cleaning pipeline and save the processed dataset.

    Returns the cleaned DataFrame for downstream use.
    """
    src = input_path or RAW_PATH
    dst = output_path or CLEAN_PATH

    print(f"📂 Loading raw data from {src}")
    df = pd.read_csv(src)
    print(f"   Raw shape: {df.shape}")

    # Apply each cleaning step in order
    df = fix_column_names(df)
    df = standardize_countries(df)
    df = fix_gdp_column(df)
    df = remove_duplicates(df)
    df = fix_impossible_values(df)
    df = parse_dates(df)

    # Ensure correct dtypes for numeric columns
    numeric_cols = [
        "life_expectancy", "gdp_per_capita", "population",
        "infant_mortality", "health_exp_pct_gdp", "physicians_per_1000",
        "co2_per_capita", "electricity_access_pct", "literacy_rate",
        "unemployment_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort for consistency
    df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

    # Save
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"\n✅ Cleaned data saved to {dst}")
    print(f"   Clean shape: {df.shape}")
    print(f"   Unique countries: {df['Country'].nunique()}")
    print(f"   Year range: {df['Year'].min()} – {df['Year'].max()}")
    print(f"   Remaining nulls:\n{df.isnull().sum()}")

    return df


if __name__ == "__main__":
    clean_pipeline()
