# Works on both arm64 (Apple Silicon) and amd64 (Linux/cloud)
FROM python:3.11-slim

# System deps needed by lightgbm, catboost, matplotlib, wordcloud
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files the app actually needs at runtime
COPY app.py .
COPY pages/ pages/
COPY appartments.csv .
COPY gurgaon_properties_missing_value_imputation.csv .
COPY gurgaon_properties_post_feature_selection_v2.csv .
COPY pipeline_final.pkl .
COPY df_final.pkl .

EXPOSE 8501

# Disable Streamlit telemetry and browser auto-open
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
