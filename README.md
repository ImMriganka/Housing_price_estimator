# 🏙️ Housing Price Estimator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![CatBoost](https://img.shields.io/badge/CatBoost-Best_Model-yellow?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A full-stack machine learning project that predicts residential property prices in Gurgaon, India — complete with an interactive web application featuring price prediction, market analytics, property recommendations, and statistical insights.**

### 🔴 Live Demo → [nestimate.streamlit.app on Hugging Face](https://huggingface.co/spaces/IMriganka/nestimate)

[Features](#-features) • [App Modules](#-app-modules) • [ML Pipeline](#-ml-pipeline) • [Project Structure](#-project-structure) • [Getting Started](#-getting-started) • [Results](#-results)

</div>

---

## 🌟 Features

| Module | Description |
|--------|-------------|
| 🏠 **Price Predictor** | Estimate property price instantly using 15+ inputs |
| 📊 **Analytics** | Explore market trends across sectors, BHK types, and furnishing levels |
| 🔍 **Recommender** | Find similar properties or search by landmark proximity |
| 💡 **Insights** | Statistical analysis of what actually drives property prices |

---

## 🖥️ App Modules

### 🏠 Price Predictor
Enter property details and get an instant market price estimate powered by a CatBoost model tuned with Optuna Bayesian optimization.

**Inputs:** Property type · Sector (107 options) · Bedrooms · Bathrooms · Balconies · Built-up area · Floor number · Age/Possession · Furnishing level · Luxury score · Servant room · Store room

**Output:** Estimated price in Crores with a ±10% confidence band

---

### 📊 Analytics
8 interactive Plotly charts exploring the Gurgaon property market across 3,554 listings:

- 📍 **Sector-wise Price per Sqft** — top N sectors ranked (adjustable slider)
- ☁️ **Amenities Word Cloud** — most common facilities across 247 premium projects
- 📈 **Area vs Price Scatter** — with BHK coloring, filter by flat/house
- 🥧 **BHK Distribution Pie** — overall or filtered by sector
- 📦 **Price Range by BHK** — box plot showing spread per bedroom count
- 🏘️ **Flat vs House Distribution** — overlaid KDE histograms
- 🪑 **Furnishing vs Price** — avg price by furnishing level
- ✨ **Luxury Score vs Price** — scatter with OLS trendline

---

### 🔍 Recommender
Content-based recommendation system built on 247 curated Gurgaon apartment projects.

**Tab 1 — Similar Properties**
- Similarity = **60% amenity match** (MultiLabelBinarizer → cosine similarity) + **40% attribute match** (price range + BHK)
- Filters: max budget, BHK type, number of results
- Each card shows: similarity score, shared amenities, exclusive amenities, nearby landmarks

**Tab 2 — Landmark Radius Search**
- Search across 990+ real landmarks (airports, malls, hospitals, metro stations)
- Adjust radius (0.5–50 km), sort by distance or price
- Results shown as a sortable table + bar chart

---

### 💡 Insights
Statistical deep-dive using Ridge Regression + OLS (statsmodels) on log-price:

- **Market KPIs** — total listings, avg/median price, avg price per sqft, model R²
- **Feature Importance** — Ridge coefficients showing which attributes push prices up or down
- **Sector Analysis** — top 10 premium vs. top 10 affordable sectors side-by-side
- **Price Breakdown** — 5 tabs: BHK · Property Type · Furnishing · Age · Luxury Category
- **Area vs Price Regression** — scatter with trendline + Pearson correlation
- **Statistical Significance** — OLS p-values with p=0.05 threshold, plain-English summary

---

## 🤖 ML Pipeline

### Models Evaluated

| Model | MAE (Cr) | R² | Notes |
|-------|----------|----|-------|
| **CatBoost (Optuna)** | **0.457** | **0.867** | ✅ Best model |
| Extra Trees (baseline) | 0.463 | 0.860 | — |
| XGBoost (default) | 0.468 | 0.858 | — |
| Stacking (XGB+LGB+CB+ET+RF) | 0.471 | 0.845 | — |
| ExtraTrees (Optuna) | 0.474 | 0.856 | — |
| LightGBM (Optuna) | 0.487 | 0.826 | — |
| XGBoost (Optuna) | 0.499 | 0.832 | Optuna over-tuned |

> All models trained with 5-fold CV. Target variable: `log1p(price)` → back-transformed with `expm1`.

---

### Preprocessing Stack

```
Raw Data (3,554 rows · 18 features)
        │
        ▼
  ColumnTransformer
  ├── StandardScaler        → numeric features
  ├── OrdinalEncoder        → property_type, balcony, floor_category, luxury_category
  ├── OneHotEncoder         → agePossession, furnishing_type
  └── TargetEncoder         → sector (smoothing=100 to handle rare sectors)
        │
        ▼
  CatBoostRegressor (Optuna: 60 trials · depth=7 · lr=0.046)
```

---

### Feature Engineering

| Feature | Formula | Why |
|---------|---------|-----|
| `log_area` | `log1p(built_up_area)` | Compress right tail |
| `sqrt_area` | `√built_up_area` | Intermediate scale |
| `log_area_sq` | `log_area²` | Non-linear area signal without outlier amplification |
| `bath_bed_ratio` | `bathroom / bedRoom` | Luxury layout signal |
| `area_per_room` | `area / total_rooms` | Spaciousness proxy |
| `amenity_score` | `has_servant + has_store` | Premium amenity indicator |

---

### Key Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Optuna CV data leakage | TargetEncoder fitted on full train set before CV splits → inflated R², worse test MAE | Wrapped in `Pipeline` so encoder refits per fold |
| `passthrough=True` in stacking | Ridge meta-learner overfit 35+ raw features instead of blending predictions | Set `passthrough=False` |
| `area_sq = built_up_area²` | Quadratically amplified errors for luxury properties | Replaced with `log_area²` |
| `TargetEncoder smoothing=10` | Rare sectors (<5 listings) wildly overfit | Increased to `smoothing=100` |

---

## 📁 Project Structure

```
Housing_price_estimator/
│
├── 1_data_preprocessing/          # Raw data ingestion & cleaning
│   ├── data_preprocessing_flats*.ipynb
│   ├── data_preprocessing_houses*.ipynb
│   ├── missing_value_imputation*.ipynb
│   ├── outlier_management*.ipynb
│   ├── flats.csv / houses.csv / gurgaon_properties.csv   ← raw data
│   └── gurgaon_properties_cleaned_v*.csv                 ← cleaned outputs
│
├── 2_eda/                         # Exploratory Data Analysis
│   ├── eda_univariate_analysis.ipynb
│   ├── eda_multivariate_analysis*.ipynb
│   └── eda_pandas_profiling.ipynb
│
├── 3_feature_engineering/         # Feature creation & selection
│   ├── feature_engineering.ipynb
│   ├── feature_selection*.ipynb
│   ├── appartments.csv                                    ← 247 curated projects
│   └── gurgaon_properties_*.csv                          ← engineered datasets
│
├── 4_modelling/                   # Model training & evaluation
│   ├── baseline_model*.ipynb
│   ├── model_selection*.ipynb
│   ├── model_improvement_advanced.ipynb
│   ├── model_final_mae025.ipynb                          ← final model notebook
│   ├── pipeline_final.pkl                                ← trained pipeline
│   └── df_final.pkl                                      ← training features
│
└── real_estate_app/               # Streamlit web application
    ├── app.py                                            ← entry point
    ├── pages/
    │   ├── 1_Price_Predictor.py
    │   ├── 2_Analytics.py
    │   ├── 3_Recommender.py
    │   └── 4_Insights.py
    ├── Dockerfile
    ├── requirements.txt
    └── *.csv / *.pkl                                     ← app data files
```

---

## 🚀 Getting Started

### 🔴 Try the Live App

**[https://huggingface.co/spaces/IMriganka/nestimate](https://huggingface.co/spaces/IMriganka/nestimate)**

No setup needed — runs directly in your browser.

---

### Run Locally

```bash
# Clone the repo
git clone https://github.com/ImMriganka/Housing_price_estimator.git
cd Housing_price_estimator/real_estate_app

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

### Run with Docker

```bash
cd real_estate_app

# Build
docker build -t gurgaon-property-app .

# Run
docker run -p 8501:8501 gurgaon-property-app
```

---

### Deploy on Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) — choose **Streamlit** SDK
2. Clone the Space repo and copy `real_estate_app/` files into it
3. Push — Hugging Face installs `requirements.txt` and launches automatically ✅

---

## 📊 Results

```
Dataset       : 3,554 Gurgaon residential properties
Price range   : ₹0.07 Cr — ₹31.50 Cr  (mean ₹2.44 Cr)
Sectors       : 107 unique sectors

Best Model    : CatBoost (Optuna, 60 trials)
MAE           : 0.457 Crores  (~±₹45.7 Lakhs)
RMSE          : 1.011 Crores
R²  (test)    : 0.867
R²  (5-fold)  : 0.896
MAPE          : 18.4%
```

> **Note on MAE floor:** The practical lower bound with this dataset is ~0.40 Cr. ~60% of remaining error is irreducible variance — market timing, exact plot location, and property condition are not captured in the available features. Adding geocoordinates or recent transaction comparables would push MAE lower.

---

## 🛠️ Tech Stack

<div align="center">

| Category | Libraries |
|----------|-----------|
| **Web App** | Streamlit |
| **ML Models** | CatBoost · XGBoost · LightGBM · scikit-learn |
| **Hyperparameter Tuning** | Optuna |
| **Data** | Pandas · NumPy |
| **Visualization** | Plotly · Matplotlib · Seaborn · WordCloud |
| **Statistics** | Statsmodels |
| **Encoding** | category_encoders (TargetEncoder) |
| **Containerization** | Docker |

</div>

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Made with ❤️ by [ImMriganka](https://github.com/ImMriganka)

⭐ Star this repo if you found it useful!

</div>
