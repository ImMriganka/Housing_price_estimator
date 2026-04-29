import ast
import json
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(
    page_title="Property Recommender",
    page_icon="🔍",
    layout="wide",
)

# ── Data loading & preprocessing ───────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    apt = pd.read_csv("appartments.csv")

    def safe_parse(raw):
        if not isinstance(raw, str) or raw.strip() in ("", "nan"):
            return None
        try:
            return ast.literal_eval(raw)
        except Exception:
            try:
                return json.loads(raw.replace("'", '"'))
            except Exception:
                return None

    def parse_distance_m(s):
        s = str(s).lower()
        m = re.search(r"([\d.]+)\s*(km|meter|m\b)", s)
        if not m:
            return None
        val = float(m.group(1))
        return val * 1000 if "km" in m.group(2) else val

    def extract_min_price(raw):
        d = safe_parse(raw)
        if not isinstance(d, dict):
            return None
        prices = []
        for info in d.values():
            if not isinstance(info, dict):
                continue
            pr = info.get("price-range", "")
            nums = re.findall(r"[\d.]+", pr.replace(",", ""))
            if nums:
                val = float(nums[0])
                if "L" in pr:
                    val /= 100
                prices.append(val)
        return min(prices) if prices else None

    def extract_max_price(raw):
        d = safe_parse(raw)
        if not isinstance(d, dict):
            return None
        prices = []
        for info in d.values():
            if not isinstance(info, dict):
                continue
            pr = info.get("price-range", "")
            nums = re.findall(r"[\d.]+", pr.replace(",", ""))
            if len(nums) >= 2:
                val = float(nums[-1])
                if "Cr" not in pr and "L" in pr:
                    val /= 100
                prices.append(val)
        return max(prices) if prices else None

    def extract_bhk(raw):
        d = safe_parse(raw)
        if not isinstance(d, dict):
            return []
        bhks = []
        for k in d.keys():
            nums = re.findall(r"\d+", k)
            if nums:
                bhks.append(int(nums[0]))
        return sorted(set(bhks))

    def extract_facilities(raw):
        r = safe_parse(raw)
        return r if isinstance(r, list) else []

    def extract_loc_advantages(raw):
        d = safe_parse(raw)
        if not isinstance(d, dict):
            return {}
        return {k: parse_distance_m(v) for k, v in d.items()
                if parse_distance_m(v) is not None}

    apt["min_price"]     = apt["PriceDetails"].apply(extract_min_price)
    apt["max_price"]     = apt["PriceDetails"].apply(extract_max_price)
    apt["bhk_list"]      = apt["PriceDetails"].apply(extract_bhk)
    apt["facilities"]    = apt["TopFacilities"].apply(extract_facilities)
    apt["loc_dict"]      = apt["LocationAdvantages"].apply(extract_loc_advantages)
    apt["nearby"]        = apt["NearbyLocations"].apply(
        lambda x: safe_parse(x) if isinstance(x, str) else []
    )
    apt = apt.dropna(subset=["PropertyName"]).reset_index(drop=True)

    # ── Build cosine similarity on amenities ───────────────────────────────────
    mlb = MultiLabelBinarizer()
    fac_matrix = mlb.fit_transform(apt["facilities"])
    cosine_fac = cosine_similarity(fac_matrix)

    # ── Build cosine similarity on property attributes ─────────────────────────
    # numeric: min_price, max_price, min_bhk, max_bhk
    def safe_num(series, fill):
        return series.fillna(fill).values.reshape(-1, 1)

    min_bhk = apt["bhk_list"].apply(lambda x: min(x) if x else 0)
    max_bhk = apt["bhk_list"].apply(lambda x: max(x) if x else 0)

    attr = np.hstack([
        (apt["min_price"].fillna(apt["min_price"].median()).values.reshape(-1, 1)
         / apt["min_price"].max()),
        (apt["max_price"].fillna(apt["max_price"].median()).values.reshape(-1, 1)
         / apt["max_price"].max()),
        (min_bhk.values.reshape(-1, 1) / max_bhk.max()),
        (max_bhk.values.reshape(-1, 1) / max_bhk.max()),
    ])
    cosine_attr = cosine_similarity(attr)

    # ── Weighted blend (amenities 60% + attributes 40%) ───────────────────────
    cosine_combined = 0.6 * cosine_fac + 0.4 * cosine_attr

    return apt, mlb, cosine_combined


apt, mlb, cosine_combined = load_and_prepare()

ALL_PROPERTIES   = sorted(apt["PropertyName"].tolist())
ALL_LOCATIONS    = sorted({
    loc
    for loc_dict in apt["loc_dict"]
    for loc in loc_dict.keys()
})


# ── Helper: recommend ──────────────────────────────────────────────────────────
def recommend(property_name, top_n=5, budget_max=None, bhk_filter=None):
    idx = apt[apt["PropertyName"] == property_name].index[0]
    sim_scores = list(enumerate(cosine_combined[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in sim_scores[1:]:
        row = apt.iloc[i]

        if budget_max is not None:
            if pd.notna(row["min_price"]) and row["min_price"] > budget_max:
                continue

        if bhk_filter is not None:
            if not any(b in row["bhk_list"] for b in bhk_filter):
                continue

        # common amenities with selected property
        sel_fac   = set(apt.iloc[idx]["facilities"])
        row_fac   = set(row["facilities"])
        common    = sel_fac & row_fac
        exclusive = row_fac - sel_fac

        results.append({
            "name":        row["PropertyName"],
            "score":       round(score * 100, 1),
            "min_price":   row["min_price"],
            "max_price":   row["max_price"],
            "bhk":         row["bhk_list"],
            "facilities":  row["facilities"],
            "common_fac":  sorted(common),
            "new_fac":     sorted(exclusive),
            "nearby":      row["nearby"] if isinstance(row["nearby"], list) else [],
            "sub_name":    row["PropertySubName"],
        })
        if len(results) == top_n:
            break

    return results


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔍 Property Recommender")
st.markdown("Find similar properties or search by landmark proximity.")
st.divider()

tab1, tab2 = st.tabs(["Similar Properties", "Nearby Landmark Search"])

# ── TAB 1: Property Recommender ────────────────────────────────────────────────
with tab1:
    st.subheader("Find Properties Similar to One You Like")

    c1, c2 = st.columns([2, 1])
    with c1:
        selected = st.selectbox("Select a property", ALL_PROPERTIES, key="rec_prop")
    with c2:
        top_n = st.slider("Number of recommendations", 3, 10, 5, key="rec_n")

    st.markdown("**Optional filters**")
    f1, f2, f3 = st.columns(3)
    with f1:
        budget = st.number_input(
            "Max budget (Crores)", min_value=0.0, max_value=50.0,
            value=0.0, step=0.5,
            help="0 = no budget filter",
        )
    with f2:
        bhk_options = [1, 2, 3, 4, 5, 6]
        bhk_sel = st.multiselect("BHK required", bhk_options,
                                  help="Leave empty for no filter")
    with f3:
        st.write("")  # spacer

    if st.button("Recommend", type="primary", use_container_width=True, key="btn_rec"):
        budget_max  = budget if budget > 0 else None
        bhk_filter  = bhk_sel if bhk_sel else None

        with st.spinner("Finding similar properties..."):
            recs = recommend(selected, top_n, budget_max, bhk_filter)

        if not recs:
            st.warning("No properties match the current filters. Try relaxing budget or BHK constraints.")
        else:
            # Show selected property details
            sel_row = apt[apt["PropertyName"] == selected].iloc[0]
            with st.expander(f"Selected: {selected}", expanded=False):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Min Price", f"₹ {sel_row['min_price']:.2f} Cr"
                           if pd.notna(sel_row["min_price"]) else "N/A")
                sc2.metric("BHK Types", ", ".join(str(b) for b in sel_row["bhk_list"])
                           if sel_row["bhk_list"] else "N/A")
                sc3.metric("Amenities", len(sel_row["facilities"]))
                st.markdown("**Top Amenities:** " +
                            ", ".join(sel_row["facilities"][:6]))

            st.markdown(f"### Top {len(recs)} Recommendations")
            st.caption("Similarity = 60% amenity match + 40% price & BHK match")

            for rank, rec in enumerate(recs, 1):
                with st.container(border=True):
                    h1, h2, h3, h4 = st.columns([3, 1.2, 1.2, 1])
                    h1.markdown(f"**{rank}. {rec['name']}**")
                    h2.metric("Similarity", f"{rec['score']}%")
                    price_str = (
                        f"₹ {rec['min_price']:.2f} Cr"
                        if pd.notna(rec["min_price"]) else "N/A"
                    )
                    h3.metric("From", price_str)
                    bhk_str = (", ".join(str(b) for b in rec["bhk"])
                               if rec["bhk"] else "N/A")
                    h4.metric("BHK", bhk_str)

                    if rec["sub_name"] and isinstance(rec["sub_name"], str):
                        st.caption(rec["sub_name"])

                    ac1, ac2 = st.columns(2)
                    with ac1:
                        if rec["common_fac"]:
                            st.markdown(
                                "**Shared amenities:** "
                                + ", ".join(f"`{f}`" for f in rec["common_fac"][:6])
                                + ("..." if len(rec["common_fac"]) > 6 else "")
                            )
                    with ac2:
                        if rec["new_fac"]:
                            st.markdown(
                                "**Additional amenities:** "
                                + ", ".join(f"`{f}`" for f in rec["new_fac"][:5])
                                + ("..." if len(rec["new_fac"]) > 5 else "")
                            )

                    if rec["nearby"]:
                        st.caption("Nearby: " + " · ".join(rec["nearby"][:4]))

# ── TAB 2: Landmark Radius Search ─────────────────────────────────────────────
with tab2:
    st.subheader("Find Properties Near a Landmark")

    lc1, lc2 = st.columns([3, 1])
    with lc1:
        sel_loc = st.selectbox("Select Landmark", ALL_LOCATIONS, key="loc_sel")
    with lc2:
        radius_km = st.number_input("Radius (km)", min_value=0.5,
                                     max_value=50.0, value=5.0, step=0.5)

    sort_by = st.radio(
        "Sort results by",
        ["Distance", "Price (low to high)", "Price (high to low)"],
        horizontal=True,
    )

    if st.button("Search", type="primary", use_container_width=True, key="btn_loc"):
        radius_m = radius_km * 1000
        matches = []
        for _, row in apt.iterrows():
            dist = row["loc_dict"].get(sel_loc)
            if dist is not None and dist <= radius_m:
                matches.append({
                    "Property":    row["PropertyName"],
                    "Distance (km)": round(dist / 1000, 1),
                    "Min Price (Cr)": row["min_price"],
                    "BHK Types":   ", ".join(str(b) for b in row["bhk_list"])
                                   if row["bhk_list"] else "N/A",
                    "Amenities":   len(row["facilities"]),
                })

        if not matches:
            st.warning(f"No properties found within {radius_km} km of **{sel_loc}**. Try a larger radius.")
        else:
            df_res = pd.DataFrame(matches)
            if sort_by == "Distance":
                df_res = df_res.sort_values("Distance (km)")
            elif sort_by == "Price (low to high)":
                df_res = df_res.sort_values("Min Price (Cr)")
            else:
                df_res = df_res.sort_values("Min Price (Cr)", ascending=False)

            st.success(f"Found **{len(df_res)}** properties within {radius_km} km of **{sel_loc}**")
            st.dataframe(
                df_res.reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "Distance (km)": st.column_config.NumberColumn(format="%.1f km"),
                    "Min Price (Cr)": st.column_config.NumberColumn(
                        format="₹ %.2f Cr", help="Starting price"
                    ),
                    "Amenities": st.column_config.NumberColumn(help="Number of listed amenities"),
                },
            )

            # Quick summary chart
            if len(df_res) >= 3:
                import plotly.express as px
                fig = px.bar(
                    df_res.sort_values("Distance (km)").head(15),
                    x="Property", y="Distance (km)",
                    color="Min Price (Cr)",
                    color_continuous_scale="Blues",
                    title=f"Properties within {radius_km} km of {sel_loc}",
                    height=380,
                )
                fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=True)
                st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data: 247 curated Gurgaon projects · Similarity = amenity TF-IDF + price/BHK attributes")
