"""
generate_messy_data.py
──────────────────────
Generates a realistic but intentionally messy dataset simulating global health
indicators across countries and years. The messiness forces real-world data
wrangling: inconsistent formatting, missing values, duplicates, mixed types,
encoding issues, and more.

Dataset theme: Global Health & Socioeconomic Indicators (2000–2023)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Configuration ──
N_COUNTRIES = 60
YEARS = list(range(2000, 2024))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Country names with intentional inconsistencies that require cleaning
# (duplicates with different capitalisation, typos, extra whitespace)
COUNTRIES_MESSY = [
    "United States", "united states", "USA", " United States ",
    "Canada", "canada", "México", "Mexico",
    "Brazil", "brasil", "Argentina", "Colombia",
    "United Kingdom", "UK", "united kingdom",
    "France", "Germany", "germany ", "Italy", "Spain",
    "Nigeria", "South Africa", "south africa", "Kenya", "Egypt",
    "China", "china", "India", "india", "Japan",
    "Australia", "New Zealand", "Indonesia", "Thailand",
    "Russia", "Turkey", "Saudi Arabia", "South Korea",
    "Sweden", "Norway", "Netherlands", "Poland",
    "Chile", "Peru", "Philippines", "Vietnam",
    "Pakistan", "Bangladesh", "Ethiopia", "Ghana",
    "Morocco", "Tanzania", "DR Congo", "DR congo",
    "Malaysia", "Singapore", "Israel", "Ireland",
    "Czech Republic", "Czechia",  # same country, different names
]

# Canonical mapping for reference (NOT used in generation — this is the puzzle)
CANONICAL = {
    "united states": "United States", "usa": "United States",
    " united states ": "United States",
    "canada": "Canada", "méxico": "Mexico", "mexico": "Mexico",
    "brasil": "Brazil", "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "germany ": "Germany", "south africa": "South Africa",
    "china": "China", "india": "India",
    "dr congo": "DR Congo", "czechia": "Czech Republic",
}


def _random_with_holes(n, low, high, null_pct=0.08):
    """Generate float array with realistic missing values."""
    vals = np.random.uniform(low, high, n)
    mask = np.random.random(n) < null_pct
    vals[mask] = np.nan
    return vals


def _inject_string_numbers(series, pct=0.05):
    """
    Replace some numeric values with string representations to simulate
    data entry errors (e.g., '$1,234' instead of 1234).
    """
    s = series.copy()
    idx = s.dropna().sample(frac=pct, random_state=42).index
    for i in idx:
        val = s[i]
        # Randomly pick a bad format
        fmt = np.random.choice(["comma", "dollar", "space", "text"])
        if fmt == "comma":
            s[i] = f"{val:,.2f}"
        elif fmt == "dollar":
            s[i] = f"${val:,.0f}"
        elif fmt == "space":
            s[i] = f" {val} "
        else:
            s[i] = "N/A"
    return s


def generate():
    rows = []
    for country in COUNTRIES_MESSY:
        for year in YEARS:
            # ~15% chance a country-year row is simply missing
            if np.random.random() < 0.15:
                continue
            rows.append({
                "Country": country,
                "Year": year,
                # life_expectancy: realistic range 50-85 years
                "life_expectancy": round(np.random.uniform(50, 85), 1),
                # gdp_per_capita: range $500–$80,000 (will be stringified below)
                "gdp_per_capita": round(np.random.uniform(500, 80000), 2),
                # population: range 500K–1.4B
                "population": int(np.random.uniform(5e5, 1.4e9)),
                # infant_mortality_rate per 1000 births
                "infant_mortality": round(np.random.uniform(2, 80), 1),
                # health_expenditure_pct of GDP
                "health_exp_pct_gdp": round(np.random.uniform(2, 18), 2),
                # physicians_per_1000
                "physicians_per_1000": round(np.random.uniform(0.02, 6.0), 2),
                # co2_emissions_tons_per_capita
                "co2_per_capita": round(np.random.uniform(0.1, 20), 2),
                # access_to_electricity (%)
                "electricity_access_pct": round(np.random.uniform(10, 100), 1),
                # literacy_rate (%)
                "literacy_rate": round(np.random.uniform(30, 100), 1),
                # unemployment_rate (%)
                "unemployment_rate": round(np.random.uniform(1, 30), 1),
            })

    df = pd.DataFrame(rows)

    # ── Inject messiness ──

    # 1. Mixed types in gdp_per_capita — some strings with $ and commas
    df["gdp_per_capita"] = _inject_string_numbers(
        df["gdp_per_capita"].astype(object), pct=0.08
    )

    # 2. Negative values (impossible for these indicators)
    neg_idx = df.sample(frac=0.02, random_state=7).index
    df.loc[neg_idx, "infant_mortality"] = -df.loc[neg_idx, "infant_mortality"]

    # 3. Outlier injection — impossibly high life expectancy
    outlier_idx = df.sample(frac=0.01, random_state=99).index
    df.loc[outlier_idx, "life_expectancy"] = np.random.uniform(120, 200, len(outlier_idx))

    # 4. Duplicate rows (~3%)
    dup_rows = df.sample(frac=0.03, random_state=11)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 5. Sprinkle additional NaN across all numeric columns (~8%)
    for col in ["life_expectancy", "infant_mortality", "health_exp_pct_gdp",
                "physicians_per_1000", "co2_per_capita", "electricity_access_pct",
                "literacy_rate", "unemployment_rate"]:
        mask = np.random.random(len(df)) < 0.08
        df.loc[mask, col] = np.nan

    # 6. Inconsistent date formats in a "last_updated" column
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y", "%Y/%m/%d"]
    dates = pd.date_range("2023-01-01", "2024-06-30", periods=len(df))
    df["last_updated"] = [
        d.strftime(np.random.choice(date_formats)) for d in dates
    ]

    # 7. Random whitespace in column names (a classic headache)
    df.rename(columns={
        "life_expectancy": " life_expectancy",
        "unemployment_rate": "unemployment_rate ",
        "co2_per_capita": " co2_per_capita ",
    }, inplace=True)

    # 8. Shuffle to destroy any ordering
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "global_health_messy.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} rows → {out_path}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    return df


if __name__ == "__main__":
    generate()
