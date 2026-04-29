import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ── Load model & training data ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("pipeline_final.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("df_final.pkl", "rb") as f:
        X_train = pickle.load(f)
    return pipeline, X_train

pipeline, X_train = load_artifacts()

# ── Feature engineering (must match notebook exactly) ─────────────────────────
def engineer_features(df):
    df = df.copy()
    df["log_area"]  = np.log1p(df["built_up_area"])
    df["sqrt_area"] = np.sqrt(df["built_up_area"])
    df["area_sq"]   = df["built_up_area"] ** 2

    df["total_rooms"]    = df["bedRoom"] + df["bathroom"]
    df["bath_bed_ratio"] = df["bathroom"] / (df["bedRoom"] + 1e-6)
    df["area_per_room"]  = df["built_up_area"] / (df["total_rooms"] + 1)

    df["has_servant"]   = (df["servant room"] > 0).astype(int)
    df["has_store"]     = (df["store room"]   > 0).astype(int)
    df["amenity_score"] = df["has_servant"] + df["has_store"]

    df["area_bin"] = pd.cut(
        df["built_up_area"],
        bins=[0, 500, 800, 1200, 2000, np.inf],
        labels=["tiny", "small", "mid", "large", "luxury"],
    ).astype(str)

    return df


def predict_price(inputs: dict) -> float:
    row = pd.DataFrame([inputs])
    for col in X_train.columns:
        if col not in row.columns:
            if X_train[col].dtype in [np.float64, np.int64, float, int]:
                row[col] = X_train[col].mean()
            else:
                row[col] = X_train[col].mode()[0]
    row = row[X_train.columns]
    row = engineer_features(row)
    return float(np.expm1(pipeline.predict(row)[0]))


# ── Sector list ────────────────────────────────────────────────────────────────
SECTORS = sorted([
    "dwarka expressway", "gwal pahari", "manesar", "sector 1", "sector 102",
    "sector 103", "sector 104", "sector 105", "sector 106", "sector 107",
    "sector 108", "sector 109", "sector 10a", "sector 11", "sector 110",
    "sector 111", "sector 112", "sector 113", "sector 12", "sector 13",
    "sector 14", "sector 15", "sector 17", "sector 17a", "sector 17b",
    "sector 2", "sector 21", "sector 22", "sector 23", "sector 24",
    "sector 25", "sector 26", "sector 27", "sector 28", "sector 3",
    "sector 3 phase 2", "sector 3 phase 3 extension", "sector 30",
    "sector 31", "sector 33", "sector 36", "sector 36a", "sector 37",
    "sector 37c", "sector 37d", "sector 38", "sector 39", "sector 4",
    "sector 40", "sector 41", "sector 43", "sector 45", "sector 46",
    "sector 47", "sector 48", "sector 49", "sector 5", "sector 50",
    "sector 51", "sector 52", "sector 53", "sector 54", "sector 55",
    "sector 56", "sector 57", "sector 58", "sector 59", "sector 6",
    "sector 60", "sector 61", "sector 62", "sector 63", "sector 63a",
    "sector 65", "sector 66", "sector 67", "sector 67a", "sector 68",
    "sector 69", "sector 7", "sector 70", "sector 70a", "sector 71",
    "sector 72", "sector 73", "sector 74", "sector 76", "sector 77",
    "sector 78", "sector 79", "sector 8", "sector 80", "sector 81",
    "sector 82", "sector 82a", "sector 83", "sector 84", "sector 85",
    "sector 86", "sector 88a", "sector 88b", "sector 89", "sector 9",
    "sector 90", "sector 91", "sector 92", "sector 93", "sector 95",
    "sector 99", "sector 99a", "sector 9a", "sohna road", "sohna road road",
])

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏠 Price Predictor")
st.markdown("Estimate the market price of a residential property in Gurgaon.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", ["flat", "house"])
    sector        = st.selectbox("Sector", SECTORS, index=SECTORS.index("sector 47"))
    bedrooms      = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms     = st.number_input("Bathrooms", min_value=1, max_value=12, value=2)
    balcony       = st.selectbox("Balconies", ["0", "1", "2", "3", "3+"])
    built_up_area = st.number_input("Built-up Area (sq ft)", min_value=100, max_value=15000, value=1200)

with col2:
    floor_num    = st.number_input("Floor Number", min_value=0, max_value=51, value=5)
    age          = st.selectbox("Age / Possession", [
        "New Property", "Relatively New", "Moderately Old", "Old Property", "Under Construction"
    ])
    furnishing   = st.selectbox("Furnishing", ["unfurnished", "semifurnished", "furnished"])
    luxury_score = st.slider("Luxury Score", min_value=0, max_value=174, value=50,
                             help="0 = basic amenities, 174 = ultra-luxury")
    servant_room = st.checkbox("Servant Room")
    store_room   = st.checkbox("Store Room")

st.divider()

def luxury_category(score):
    if score < 50:    return "Low"
    elif score < 150: return "Medium"
    return "High"

def floor_category(floor):
    if floor <= 2:    return "Low Floor"
    elif floor <= 10: return "Mid Floor"
    return "High Floor"

if st.button("Predict Price", type="primary", use_container_width=True):
    inputs = {
        "property_type":   property_type,
        "sector":          sector,
        "bedRoom":         int(bedrooms),
        "bathroom":        int(bathrooms),
        "balcony":         str(balcony),
        "floorNum":        float(floor_num),
        "agePossession":   age,
        "built_up_area":   float(built_up_area),
        "servant room":    int(servant_room),
        "store room":      int(store_room),
        "furnishing_type": furnishing,
        "luxury_score":    int(luxury_score),
        "luxury_category": luxury_category(luxury_score),
        "floor_category":  floor_category(floor_num),
        "study room": 0,
        "pooja room": 0,
        "others":     0,
    }

    with st.spinner("Estimating..."):
        price = predict_price(inputs)

    low  = round(price * 0.90, 2)
    high = round(price * 1.10, 2)

    st.success(f"### Estimated Price: ₹ {price:.2f} Crores")
    st.caption(f"Typical market range: ₹ {low} Cr — ₹ {high} Cr  *(± 10% confidence band)*")

    st.divider()
    st.markdown("**Input Summary**")
    summary = {
        "Property Type": property_type.title(),
        "Sector":        sector.title(),
        "Bedrooms":      bedrooms,
        "Bathrooms":     bathrooms,
        "Built-up Area": f"{built_up_area} sq ft",
        "Floor":         floor_num,
        "Age":           age,
        "Furnishing":    furnishing.title(),
        "Luxury Score":  luxury_score,
    }
    st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

st.divider()
st.caption("Model: CatBoost (Optuna-tuned) · Dataset: 3,554 Gurgaon properties · MAE ≈ 0.46 Cr")
