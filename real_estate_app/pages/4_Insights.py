import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Market Insights",
    page_icon="💡",
    layout="wide",
)

# ── Data & model ───────────────────────────────────────────────────────────────
@st.cache_data
def load_and_model():
    df_raw = pd.read_csv("gurgaon_properties_post_feature_selection_v2.csv")

    # ── Preprocessing (mirrors sample code) ───────────────────────────────────
    df = df_raw.drop(columns=["store room", "floor_category", "balcony"])
    df["agePossession"] = df["agePossession"].replace({
        "Relatively New":     "New",
        "New Property":       "New",
        "Moderately Old":     "Old",
        "Old Property":       "Old",
        "Under Construction": "Under Construction",
    })
    df["luxury_num"] = df["luxury_category"].replace({"Low": 0, "Medium": 1, "High": 2})
    df["luxury_num"] = df["luxury_num"].infer_objects(copy=False).astype(int)

    encoded = pd.get_dummies(
        df, columns=["sector", "agePossession", "property_type"], drop_first=True
    )
    encoded = encoded.apply(lambda c: c.astype(int) if c.dtype == bool else c)

    X = encoded.drop(columns=["price", "luxury_category"])
    y_log = np.log1p(encoded["price"])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Ridge coefficients
    ridge = Ridge(alpha=0.0001)
    ridge.fit(X_scaled, y_log)
    coef_df = (
        pd.DataFrame({"feature": X.columns, "coef": ridge.coef_})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
    )

    # OLS p-values
    X_const = sm.add_constant(X_scaled)
    ols = sm.OLS(y_log, X_const).fit()
    pval_df = (
        pd.DataFrame({"feature": ols.pvalues.index, "pvalue": ols.pvalues.values})
        .query("feature != 'const'")
        .sort_values("pvalue")
        .reset_index(drop=True)
    )
    pval_df["significant"] = pval_df["pvalue"] < 0.05

    return df_raw, coef_df, pval_df, ols


df, coef_df, pval_df, ols_model = load_and_model()

# ── Page ───────────────────────────────────────────────────────────────────────
st.title("💡 Market Insights")
st.markdown("Statistical analysis of what drives property prices in Gurgaon.")
st.divider()

# ── 1. Key Metrics ─────────────────────────────────────────────────────────────
st.subheader("Market Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Listings",    f"{len(df):,}")
m2.metric("Avg Price",         f"₹ {df['price'].mean():.2f} Cr")
m3.metric("Median Price",      f"₹ {df['price'].median():.2f} Cr")
price_per_sqft = (df['price'] * 1e7) / df['built_up_area']
m4.metric("Avg Price/Sqft",    f"₹ {price_per_sqft.mean():,.0f}")
m5.metric("Model R²",          f"{ols_model.rsquared:.3f}")

st.divider()

# ── 2. Feature Importance (Ridge coefs) ───────────────────────────────────────
st.subheader("What Drives Property Prices?")
st.caption(
    "Ridge regression coefficients on log-price — larger bar = stronger impact. "
    "Sector dummies excluded for clarity."
)

# split into non-sector and sector features
non_sector = coef_df[~coef_df["feature"].str.startswith("sector_")].head(20).copy()
non_sector["direction"] = non_sector["coef"].apply(lambda x: "Increases price" if x > 0 else "Decreases price")

# clean up feature names for display
label_map = {
    "built_up_area":         "Built-up Area",
    "property_type_house":   "Property Type: House",
    "bathroom":              "No. of Bathrooms",
    "bedRoom":               "No. of Bedrooms",
    "servant room":          "Servant Room",
    "luxury_num":            "Luxury Score",
    "agePossession_New":     "Age: New",
    "agePossession_Under Construction": "Under Construction",
    "furnishing_type":       "Furnishing Level",
    "floorNum":              "Floor Number",
}
non_sector["label"] = non_sector["feature"].apply(lambda x: label_map.get(x, x))

fig_coef = px.bar(
    non_sector.sort_values("coef"),
    x="coef", y="label",
    orientation="h",
    color="direction",
    color_discrete_map={"Increases price": "#2ecc71", "Decreases price": "#e74c3c"},
    labels={"coef": "Coefficient (log-price scale)", "label": ""},
    title="Top Property Attributes by Price Impact",
    height=420,
)
fig_coef.update_layout(showlegend=True, legend_title_text="")
st.plotly_chart(fig_coef, use_container_width=True)

st.divider()

# ── 3. Sector Price Analysis ───────────────────────────────────────────────────
st.subheader("Sector-Level Price Analysis")

sector_stats = (
    df.groupby("sector")["price"]
    .agg(["mean", "median", "count"])
    .rename(columns={"mean": "Avg Price (Cr)", "median": "Median Price (Cr)", "count": "Listings"})
    .reset_index()
    .sort_values("Avg Price (Cr)", ascending=False)
)

tab_sec1, tab_sec2 = st.tabs(["Premium vs Affordable", "Full Sector Table"])

with tab_sec1:
    col_top, col_bot = st.columns(2)

    with col_top:
        fig_top = px.bar(
            sector_stats.head(10),
            x="Avg Price (Cr)", y="sector",
            orientation="h",
            color="Avg Price (Cr)",
            color_continuous_scale="Greens",
            title="Top 10 Premium Sectors",
            height=370,
            text="Avg Price (Cr)",
        )
        fig_top.update_traces(texttemplate="₹%{text:.1f}Cr", textposition="outside")
        fig_top.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

    with col_bot:
        fig_bot = px.bar(
            sector_stats.tail(10).sort_values("Avg Price (Cr)"),
            x="Avg Price (Cr)", y="sector",
            orientation="h",
            color="Avg Price (Cr)",
            color_continuous_scale="Reds_r",
            title="Top 10 Affordable Sectors",
            height=370,
            text="Avg Price (Cr)",
        )
        fig_bot.update_traces(texttemplate="₹%{text:.2f}Cr", textposition="outside")
        fig_bot.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bot, use_container_width=True)

with tab_sec2:
    st.dataframe(
        sector_stats.reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Avg Price (Cr)":    st.column_config.NumberColumn(format="₹ %.2f Cr"),
            "Median Price (Cr)": st.column_config.NumberColumn(format="₹ %.2f Cr"),
            "Listings":          st.column_config.NumberColumn(format="%d"),
        },
        height=450,
    )

st.divider()

# ── 4. Price Drivers Breakdown ─────────────────────────────────────────────────
st.subheader("Price Breakdown by Key Attributes")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "BHK", "Property Type", "Furnishing", "Age", "Luxury Category"
])

with tab1:
    bhk_df = (
        df.groupby("bedRoom")["price"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"bedRoom": "BHK", "mean": "Avg", "median": "Median"})
    )
    fig_bhk = go.Figure()
    fig_bhk.add_trace(go.Bar(x=bhk_df["BHK"].astype(str), y=bhk_df["Avg"],
                             name="Avg Price", marker_color="#3498db"))
    fig_bhk.add_trace(go.Bar(x=bhk_df["BHK"].astype(str), y=bhk_df["Median"],
                             name="Median Price", marker_color="#85c1e9"))
    fig_bhk.update_layout(barmode="group", xaxis_title="BHK",
                           yaxis_title="Price (Cr)", height=380,
                           title="Avg vs Median Price by BHK")
    st.plotly_chart(fig_bhk, use_container_width=True)

with tab2:
    pt_df = (
        df.groupby("property_type")["price"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"property_type": "Type", "mean": "Avg", "median": "Median", "count": "Count"})
    )
    fig_pt = px.bar(
        pt_df, x="Type", y=["Avg", "Median"],
        barmode="group",
        color_discrete_sequence=["#2ecc71", "#27ae60"],
        labels={"value": "Price (Cr)", "variable": ""},
        title="Flat vs House Price Comparison",
        height=380,
        text_auto=".2f",
    )
    st.plotly_chart(fig_pt, use_container_width=True)
    st.caption(f"Houses: {pt_df[pt_df['Type']=='house']['Count'].values[0]:,} listings | "
               f"Flats: {pt_df[pt_df['Type']=='flat']['Count'].values[0]:,} listings")

with tab3:
    furn_map = {0.0: "Unfurnished", 1.0: "Semifurnished", 2.0: "Furnished"}
    furn_df = df.copy()
    furn_df["Furnishing"] = furn_df["furnishing_type"].map(furn_map)
    furn_agg = (
        furn_df.groupby("Furnishing")["price"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "Avg", "median": "Median"})
    )
    fig_furn = px.bar(
        furn_agg, x="Furnishing", y=["Avg", "Median"],
        barmode="group",
        color_discrete_sequence=["#e67e22", "#f0b27a"],
        labels={"value": "Price (Cr)", "variable": ""},
        title="Price by Furnishing Type",
        height=380,
        text_auto=".2f",
    )
    st.plotly_chart(fig_furn, use_container_width=True)

with tab4:
    age_df = (
        df.groupby("agePossession")["price"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "Avg", "median": "Median"})
        .sort_values("Avg", ascending=False)
    )
    fig_age = px.bar(
        age_df, x="agePossession", y=["Avg", "Median"],
        barmode="group",
        color_discrete_sequence=["#9b59b6", "#d7bde2"],
        labels={"agePossession": "Age / Possession", "value": "Price (Cr)", "variable": ""},
        title="Price by Property Age",
        height=380,
        text_auto=".2f",
    )
    st.plotly_chart(fig_age, use_container_width=True)

with tab5:
    lux_order = ["Low", "Medium", "High"]
    lux_df = (
        df.groupby("luxury_category")["price"]
        .agg(["mean", "median"])
        .reindex(lux_order)
        .reset_index()
        .rename(columns={"mean": "Avg", "median": "Median"})
    )
    fig_lux = px.bar(
        lux_df, x="luxury_category", y=["Avg", "Median"],
        barmode="group",
        color_discrete_sequence=["#f39c12", "#fad7a0"],
        labels={"luxury_category": "Luxury Category", "value": "Price (Cr)", "variable": ""},
        title="Price by Luxury Category",
        height=380,
        text_auto=".2f",
    )
    st.plotly_chart(fig_lux, use_container_width=True)

st.divider()

# ── 5. Area vs Price Regression ────────────────────────────────────────────────
st.subheader("Built-up Area vs Price")

prop_filter = st.radio("Filter by", ["All", "Flat only", "House only"], horizontal=True)
plot_df = df.copy()
if prop_filter == "Flat only":
    plot_df = df[df["property_type"] == "flat"]
elif prop_filter == "House only":
    plot_df = df[df["property_type"] == "house"]

fig_area = px.scatter(
    plot_df,
    x="built_up_area", y="price",
    color="bedRoom",
    color_continuous_scale="Viridis",
    trendline="ols",
    trendline_scope="overall",
    trendline_color_override="red",
    opacity=0.5,
    labels={"built_up_area": "Built-up Area (sq ft)", "price": "Price (Cr)", "bedRoom": "BHK"},
    title="Area vs Price with Linear Trendline",
    height=480,
)
st.plotly_chart(fig_area, use_container_width=True)

# show correlation
corr = plot_df[["built_up_area", "price"]].corr().iloc[0, 1]
st.caption(f"Pearson correlation between area and price: **{corr:.3f}**")

st.divider()

# ── 6. Statistical Significance ───────────────────────────────────────────────
st.subheader("Statistically Significant Price Drivers")
st.caption(
    "OLS regression on log-price. Features with p-value < 0.05 are statistically significant. "
    f"Model R² = **{ols_model.rsquared:.3f}**, Adj R² = **{ols_model.rsquared_adj:.3f}**"
)

# show only non-sector features for readability, top 15 by significance
display_pval = (
    pval_df[~pval_df["feature"].str.startswith("sector_")]
    .head(15)
    .copy()
)
display_pval["label"] = display_pval["feature"].apply(lambda x: label_map.get(x, x))
display_pval["p-value"] = display_pval["pvalue"].apply(lambda p: f"{p:.2e}")
display_pval["Significant?"] = display_pval["significant"].apply(lambda x: "Yes" if x else "No")

fig_pval = px.bar(
    display_pval,
    x="pvalue", y="label",
    orientation="h",
    color="Significant?",
    color_discrete_map={"Yes": "#2ecc71", "No": "#e74c3c"},
    log_x=True,
    title="Feature Significance (lower p-value = stronger statistical evidence)",
    labels={"pvalue": "p-value (log scale)", "label": ""},
    height=420,
)
fig_pval.add_vline(x=0.05, line_dash="dash", line_color="gray",
                   annotation_text="p = 0.05 threshold")
fig_pval.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_pval, use_container_width=True)

# Plain-English summary
st.markdown("**Key Takeaways from Statistical Analysis**")
top_sig = display_pval[display_pval["Significant?"] == "Yes"]["label"].tolist()
if top_sig:
    st.success(
        f"The most statistically significant price drivers are: **{', '.join(top_sig[:5])}**. "
        "These factors have the strongest evidence of a real relationship with property price."
    )

st.divider()
st.caption("Data: 3,554 Gurgaon properties · OLS + Ridge regression on log1p(price)")
