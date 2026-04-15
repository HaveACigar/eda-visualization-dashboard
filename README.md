# Exploratory Data Analysis & Visualization Dashboard

An end-to-end EDA project demonstrating core Data Science skills: data cleaning, statistical analysis, and interactive visualization using a **synthetically generated messy dataset** that mirrors real-world data quality challenges.

## Skills Demonstrated

- **Data Cleaning & Wrangling** — handling 12+ types of data quality issues including mixed types, inconsistent naming, impossible values, duplicates, and scattered nulls
- **Statistical Analysis** — descriptive summaries, correlation analysis, income group classification, country rankings, and year-over-year trend detection
- **Interactive Visualization** — 8 Plotly chart types rendered in a Streamlit dashboard with user-selectable metrics, filters, and country comparisons
- **Modular Python Architecture** — cleanly separated `cleaning.py`, `analysis.py`, and `visualization.py` modules
- **Reproducibility** — virtual environments, pinned requirements, and documented pipeline

## Dataset: Global Health & Socioeconomic Indicators

Synthetically generated dataset covering **46 countries** from **2000–2023** with 11 indicators:

| Indicator | Description |
|-----------|-------------|
| `life_expectancy` | Average life expectancy (years) |
| `gdp_per_capita` | GDP per capita (USD) |
| `population` | Total population |
| `infant_mortality` | Deaths per 1,000 live births |
| `health_exp_pct_gdp` | Health expenditure as % of GDP |
| `physicians_per_1000` | Physicians per 1,000 people |
| `co2_per_capita` | CO₂ emissions (tons per capita) |
| `electricity_access_pct` | Population with electricity access (%) |
| `literacy_rate` | Adult literacy rate (%) |
| `unemployment_rate` | Unemployment rate (%) |

### Data Quality Issues (Intentional)

The raw dataset contains **12+ categories of messiness** to demonstrate wrangling:

1. Inconsistent country names (casing, typos, aliases, whitespace)
2. Mixed types in numeric columns (`$1,234` strings, `N/A`)
3. Negative values in health metrics (impossible)
4. Outlier injection (life expectancy 120–200)
5. ~3% duplicate rows
6. ~8% NaN scattered across all columns
7. 5 different date formats in `last_updated`
8. Leading/trailing whitespace in column names

## Project Structure

```
├── data/
│   ├── raw/global_health_messy.csv        # The intentionally messy dataset
│   └── processed/global_health_clean.csv  # Output of cleaning pipeline
├── src/
│   ├── generate_messy_data.py             # Dataset generator script
│   ├── cleaning.py                        # 7-step data wrangling pipeline
│   ├── analysis.py                        # Statistical analysis functions
│   └── visualization.py                   # Plotly chart builders (8 types)
├── app.py                                 # Streamlit dashboard (9 sections)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate fresh messy data (optional)

```bash
python src/generate_messy_data.py
```

### Run the cleaning pipeline

```bash
python src/cleaning.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

## Deploy to Cloud Run

This repository includes CI/CD for Cloud Run in [`.github/workflows/deploy-cloud-run.yml`](.github/workflows/deploy-cloud-run.yml).

### Prerequisites

- A Google Cloud project with Cloud Run + Cloud Build enabled
- Repository secret in this repo:
	- `GCP_SA_KEY` (service account JSON with Cloud Run deploy permissions)

### Deploy

- Push to `main` (or run workflow manually via `workflow_dispatch`)
- The workflow builds the container and deploys service `eda-visualization-dashboard`
- The workflow logs print the live URL as `Cloud Run URL: https://...`

### Portfolio Embed

Use the deployed Cloud Run URL as the value for:

- `REACT_APP_EDA_DASHBOARD_URL`

in your portfolio website repo secrets, then redeploy the portfolio site.

## Dashboard Sections

1. **Data Quality Overview** — raw vs cleaned comparison with missing-data charts
2. **Descriptive Statistics** — full summary table with skewness & missing %
3. **Correlation Explorer** — interactive heatmap + top relationships
4. **Distribution Analysis** — histogram + box plot for any indicator
5. **Country Rankings** — top-N countries filterable by metric and year
6. **Global Trends** — year-over-year line charts
7. **Scatter & Regression** — bivariate analysis with OLS trendline
8. **Income Group Comparison** — box plots across World Bank-style tiers
9. **Multi-Country Comparison** — overlay selected countries over time

## License

MIT
