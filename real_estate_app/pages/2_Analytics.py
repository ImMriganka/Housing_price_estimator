import ast

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("gurgaon_properties_missing_value_imputation.csv")
    # map furnishing int codes to labels (stored as int in this CSV)
    df["furnishing_type"] = df["furnishing_type"].replace(
        {0: "Unfurnished", 1: "Semifurnished", 2: "Furnished",
         0.0: "Unfurnished", 1.0: "Semifurnished", 2.0: "Furnished"}
    )
    return df

@st.cache_data
def load_facilities():
    apt = pd.read_csv("appartments.csv")
    items = []
    for val in apt["TopFacilities"].dropna():
        try:
            items.extend(ast.literal_eval(val))
        except Exception:
            items.append(str(val))
    return " ".join(items)

df = load_data()
facility_text = load_facilities()

st.title("📊 Analytics")
st.markdown("Explore Gurgaon property market trends across sectors, property types, and configurations.")
st.divider()

# ── 1. Sector-wise Avg Price per Sqft ─────────────────────────────────────────
st.header("Sector-wise Avg Price per Sqft")

sector_df = (
    df.groupby("sector")[["price", "price_per_sqft", "built_up_area"]]
    .mean()
    .round(2)
    .reset_index()
    .sort_values("price_per_sqft", ascending=False)
)

top_n = st.slider("Show top N sectors", min_value=10, max_value=len(sector_df), value=30, step=5)
fig1 = px.bar(
    sector_df.head(top_n),
    x="sector", y="price_per_sqft",
    color="price_per_sqft",
    color_continuous_scale="RdYlGn_r",
    labels={"price_per_sqft": "Avg Price/Sqft (₹)", "sector": "Sector"},
    title=f"Top {top_n} Sectors by Avg Price per Sqft",
    height=450,
)
fig1.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── 2. Facilities Word Cloud ───────────────────────────────────────────────────
st.header("Top Property Amenities")

wc = WordCloud(
    width=1200, height=400,
    background_color="white",
    colormap="Blues",
    stopwords={"s", "and"},
    min_font_size=12,
    collocations=False,
).generate(facility_text)

fig_wc, ax_wc = plt.subplots(figsize=(14, 4))
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
plt.tight_layout(pad=0)
st.pyplot(fig_wc)

st.divider()

# ── 3. Area vs Price scatter ───────────────────────────────────────────────────
st.header("Area vs Price")

prop_type = st.selectbox("Property Type", ["flat", "house"], key="scatter_type")
fig2 = px.scatter(
    df[df["property_type"] == prop_type],
    x="built_up_area", y="price",
    color="bedRoom",
    color_continuous_scale="Viridis",
    labels={"built_up_area": "Built-up Area (sq ft)", "price": "Price (Cr)", "bedRoom": "BHK"},
    title=f"Area vs Price — {prop_type.title()}s",
    opacity=0.7,
    height=480,
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── 4. BHK distribution pie ────────────────────────────────────────────────────
st.header("BHK Distribution")

sector_options = ["Overall"] + sorted(df["sector"].unique().tolist())
selected_sector = st.selectbox("Filter by Sector", sector_options, key="bhk_sector")

pie_df = df if selected_sector == "Overall" else df[df["sector"] == selected_sector]
fig3 = px.pie(
    pie_df,
    names="bedRoom",
    title=f"BHK Split — {selected_sector}",
    color_discrete_sequence=px.colors.sequential.RdBu,
    hole=0.35,
)
fig3.update_traces(textinfo="percent+label")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── 5. BHK price box plot ──────────────────────────────────────────────────────
st.header("Price Range by BHK")

fig4 = px.box(
    df[df["bedRoom"] <= 6],
    x="bedRoom", y="price",
    color="bedRoom",
    color_discrete_sequence=px.colors.qualitative.Safe,
    labels={"bedRoom": "BHK", "price": "Price (Cr)"},
    title="Price Distribution by Number of Bedrooms",
    height=450,
)
fig4.update_layout(showlegend=False)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── 6. Flat vs House price distribution ───────────────────────────────────────
st.header("Price Distribution: Flat vs House")

fig5, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df[df["property_type"] == "house"]["price"], label="House",
             kde=True, color="#e07b54", bins=40, ax=ax)
sns.histplot(df[df["property_type"] == "flat"]["price"],  label="Flat",
             kde=True, color="#5b8db8", bins=40, ax=ax)
ax.set_xlabel("Price (Crores)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
st.pyplot(fig5)

st.divider()

# ── 7. Furnishing vs Avg Price ─────────────────────────────────────────────────
st.header("Avg Price by Furnishing Type")

furn_df = df.groupby("furnishing_type")["price"].mean().round(2).reset_index()
furn_df.columns = ["Furnishing", "Avg Price (Cr)"]
fig6 = px.bar(
    furn_df.sort_values("Avg Price (Cr)", ascending=False),
    x="Furnishing", y="Avg Price (Cr)",
    color="Avg Price (Cr)",
    color_continuous_scale="Blues",
    title="Average Price by Furnishing Type",
    height=380,
)
fig6.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── 8. Luxury Score vs Price ───────────────────────────────────────────────────
st.header("Luxury Score vs Price")

fig7 = px.scatter(
    df, x="luxury_score", y="price",
    color="property_type",
    color_discrete_map={"flat": "#5b8db8", "house": "#e07b54"},
    labels={"luxury_score": "Luxury Score", "price": "Price (Cr)", "property_type": "Type"},
    title="Does Luxury Score Drive Price?",
    opacity=0.6,
    trendline="ols",
    height=450,
)
st.plotly_chart(fig7, use_container_width=True)

st.divider()
st.caption("Data: 3,554 Gurgaon residential properties · gurgaon_properties_missing_value_imputation.csv")
