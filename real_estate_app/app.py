import streamlit as st

st.set_page_config(
    page_title="Gurgaon Property Hub",
    page_icon="🏙️",
    layout="centered",
)

st.title("🏙️ Gurgaon Property Hub")
st.markdown(
    """
    Welcome! Use the sidebar to navigate between modules.

    | Page | Description |
    |---|---|
    | **Price Predictor** | Estimate the market price of any residential property |
    | **Analytics** | Explore market trends, sector comparisons, and property insights |
    """
)
