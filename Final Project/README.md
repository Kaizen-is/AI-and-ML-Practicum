# Customer Monthly Spend Forecasting

AI/ML Fundamentals Capstone Project — Individual Project Track

## Problem Statement

Online retailers don't know in advance how much revenue to expect from each individual customer next month. This makes it hard for marketing to prioritize retention campaigns and for finance to plan cash flow — the only "forecast" typically available is guesswork or a flat average across all customers, with nothing personalized.

This project forecasts each customer's total transaction spend for the **upcoming calendar month**, based on their prior purchase history, so a retailer's marketing/finance team can move from guessing to data-driven retention and revenue planning.

## Project Track

Individual Project Track (original project idea, not a predefined field-based scenario).

## Dataset Source

- **Dataset:** [Online Retail II (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/502/online+retail+ii), mirrored on Kaggle as [`mashlyn/online-retail-ii-uci`](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- **Content:** ~1,067,000 invoice-line transactions from a UK-based online retailer, December 2009 – December 2011
- **Fields used:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- Downloaded programmatically in the notebook via `kagglehub.dataset_download(...)` — no manual download step required.

## ML Task Type

**Supervised regression (time-aware).**

- **Input (X):** a customer's aggregated purchase behaviour up to and including month *t* — recent spend, order frequency, recency, rolling trend, tenure, country, seasonality.
- **Output (y):** predicted total spend (£) for that customer in month *t+1*.
- **Target column:** `next_month_spend`.

## Project Pipeline / Architecture

```
Raw transactions (Online Retail II)
        │
        ▼
   EDA & Data Cleaning
   (drop missing CustomerID, cancellations, non-positive qty/price, cap outliers)
        │
        ▼
   Feature Engineering
   (aggregate to customer-month grid, RFM features, rolling stats, recency,
    tenure, seasonality → target = next month's spend, shifted per customer)
        │
        ▼
   Time-based Train / Validation / Test Split
   (strict chronological split — no shuffling, no leakage)
        │
        ▼
   Preprocessing (ColumnTransformer: StandardScaler + OneHotEncoder)
        │
        ▼
   Baselines → 5 Base Learners → Hyperparameter Tuning → Stacking Ensemble
   (vs. a manual inverse-RMSE weighted blend, as a comparison point)
        │
        ▼
   Final Model Selection (on validation) → One-time Test Evaluation
        │
        ▼
   Error Analysis + Saved Model Artifacts (joblib)
        │
        ▼
   Demonstration (reloads saved artifacts, runs example predictions)
```

The entire pipeline — from raw data to a working demo — runs end-to-end in a single Colab notebook (T4 runtime), `project_notebook.ipynb`.

## Models / Approaches Tested

| Model | Role |
|---|---|
| Naive (repeat last month's spend) | Baseline #1 |
| Ridge Regression | Baseline #2 (linear) |
| Random Forest Regressor | Base learner (tuned via `RandomizedSearchCV`) |
| Hist Gradient Boosting Regressor | Base learner (tuned via `RandomizedSearchCV`) |
| K-Nearest Neighbors Regressor | Base learner |
| Support Vector Regressor (SVR) | Base learner |
| **StackingRegressor** (Ridge meta-learner over the 5 base learners) | Main model |
| Manual inverse-RMSE weighted blend of the same 5 base learners | Comparison model |

All hyperparameter tuning used `TimeSeriesSplit` cross-validation (never a random split), to stay consistent with the project's no-leakage rule.

## Final Model and Justification

**Final model:** `<FILL IN — the model printed by "Selected final model (lowest validation RMSE): ..." in Section 14>` (chosen strictly by lowest **validation**-set RMSE, before the test set was ever touched).

**Test-set results:**

| Metric | Naive Baseline | Final Model |
|---|---|---|
| RMSE (£) | 1408.43 | 1052.52 |
| Improvement over baseline | — | **25.3%** |

This exceeds the project's success criterion (≥15% RMSE improvement over the naive baseline, as defined in the approved Project Brief).

## Evaluation Metrics and Results

- **RMSE** (primary metric — same units as spend, penalizes large misses)
- **MAE** (average absolute error, easier to interpret)
- **MAPE** (scale-independent; computed only on customers with non-zero actual spend, due to the heavy right-skew of customer spending seen in EDA)

Full per-model comparison table (baselines, all 5 base learners, stacking ensemble, manual blend) is generated in Section 14 of the notebook (`results_log` → `pd.DataFrame`). Error analysis (Section 15) shows the model is accurate for typical customers, with the largest absolute errors concentrated among a small number of high-spend/wholesale-style customers.

## Installation Instructions

The notebook is designed to run in **Google Colab** with a T4 GPU runtime (GPU isn't strictly required for these models, but matches the approved project setup).

1. Open `project_notebook.ipynb` in Google Colab.
2. Run the setup cell (Section 0) — installs/imports all required packages. If running locally instead of Colab:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib kagglehub openpyxl
```

3. The dataset is downloaded automatically via `kagglehub` (Section 2) — you need a Kaggle account and API credentials configured (either `kagglehub.login()` in-notebook, or `KAGGLE_USERNAME`/`KAGGLE_KEY` set as Colab secrets).

## Training / Fine-Tuning Instructions

Run the notebook **top to bottom** (`Runtime → Run all` in Colab):

- Sections 0–6: setup, EDA, cleaning, feature engineering, train/val/test split.
- Sections 7–11: preprocessing, baselines, 5 base learners, hyperparameter tuning.
- Sections 12–14: stacking ensemble, manual blend, final model selection and test evaluation.
- Sections 15–17: error analysis and saving model artifacts to `artifacts/`.

No manual steps are required between cells; each cell only depends on variables defined earlier in the same top-to-bottom run.

## Demo / Inference Instructions

The notebook has a clearly separated **Demonstration** section at the end (after Section 19), marked with a visible divider. It is self-contained:

1. It does **not** reuse any variable from the training sections above it — it reloads the saved model (`artifacts/final_spend_forecast_model.joblib`) and metadata (`artifacts/model_metadata.joblib`) from disk.
2. To verify this yourself: `Runtime → Restart runtime`, then run only the cells from the Demonstration section onward (the `artifacts/` folder already exists from your one full run).
3. It exposes a single reusable function, `predict_next_month_spend(customer_features: dict) -> float`, with input validation (raises a clear `ValueError` on missing fields) and safe output handling (predictions are clipped to be non-negative).

## Example Input and Output

```python
predict_next_month_spend({
    "monetary_this_month": 320.0, "orders_this_month": 2, "distinct_products": 8,
    "rolling_mean_3m": 300.0, "rolling_std_3m": 45.0, "frequency_to_date": 14,
    "tenure_months": 9, "recency_months": 0, "month_of_year": 5, "Country": "United Kingdom",
})
# Output: predicted next-month spend ≈ £<value printed by the notebook>
```

Three worked examples (a typical customer, a high-value frequent customer, and a recently-quiet customer) are run in the Demonstration section, alongside an invalid-input example that is rejected with a clear error message instead of crashing.

## Known Limitations

- Cold-start customers (fewer than `MIN_TENURE_MONTHS` months of purchase history) are explicitly out of scope and are not forecast by this model.
- The dataset is heavily UK-dominated; forecasts for customers in smaller international markets rest on far less data and are less reliable.
- The model reflects one company's 2009–2011 UK/EU customer base and would need retraining for a different retailer, region, or time period.
- Evaluated on 3 held-out months only; a longer test window would give a more robust estimate of real-world performance.
- The largest absolute errors occur for high-spend/wholesale-style customers — forecasts for a company's largest accounts should be sanity-checked by a human, not used unmonitored.

## Responsible AI Considerations

- **Bias / representativeness:** see UK-dominance limitation above — model confidence is not uniform across countries.
- **Fairness / appropriate use:** intended to *support planning* (retention campaigns, revenue estimates), not to *automatically* deprioritize or exclude customers with a lower predicted spend — a low forecast can simply reflect a normal, less-frequent purchase cadence rather than churn risk.
- **Privacy:** `CustomerID` and transaction history are potentially sensitive commercial data. This project uses a public, anonymized academic release for learning purposes only; a real deployment would need to run inside the company's own access-controlled data environment.

## Repository Structure

```
.
├── README.md
├── project_notebook.ipynb   # full pipeline: EDA → training → evaluation → demo
├── artifacts/                # generated by running the notebook (model + metadata)
└── requirements.txt
```

## Author

`Islom Amanullayev`
