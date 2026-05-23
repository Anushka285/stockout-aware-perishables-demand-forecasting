import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from datetime import datetime
from pathlib import Path


# ---------- Paths and core objects ----------

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../Capstone_PIM
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "Data" / "Processed"

config_path = MODEL_DIR / "model_config.joblib"
config = joblib.load(config_path)

# Normalize feature names for backward compatibility
final_features = list(config["final_features"])

# Old names in models vs new names in data
name_map = {
    "iscensoredday": "is_censored_day",
    "isnotstocked": "is_not_stocked",
}
# Keep original names in final_features (they must match what models expect)
# but when building rows, we will create both versions.

target_obs = config.get("target_obs", "sale_amount")
target_latent = config.get("target_latent", "latent_demand_recovered")

segments = ["ShortLife", "MediumLife", "LongLife"]

# Load models
obs_models = {
    seg: lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_obs_{seg}.txt"))
    for seg in segments
}
latent_models = {
    seg: lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_latent_{seg}.txt"))
    for seg in segments
}

# Load eval data (for charts)
eval_path = DATA_DIR / "eval_holdout.parquet"
eval_df = pd.read_parquet(eval_path) if eval_path.exists() else None

# Your date column in eval_df
DATE_COL = "dt"


def assign_shelf_life(third_cat_id: int) -> str:
    if third_cat_id <= 80:
        return "ShortLife"
    elif third_cat_id <= 160:
        return "MediumLife"
    else:
        return "LongLife"


# Final segment metrics (replace with your real numbers if different)
segment_metrics = pd.DataFrame(
    [
        {"Segment": "ShortLife", "Target": "Observed", "WAPE": 36.18, "R2": 0.7074},
        {"Segment": "MediumLife", "Target": "Observed", "WAPE": 37.80, "R2": 0.5984},
        {"Segment": "LongLife", "Target": "Observed", "WAPE": 32.04, "R2": 0.9228},
    ]
)


# ---------- Page config ----------

st.set_page_config(
    page_title="Perishable Inventory Forecasting",
    page_icon="📦",
    layout="wide",
)

st.title("Perishable Inventory Forecasting – Perishables Demand & Stockouts")

st.markdown(
    """
This interactive dashboard uses **shelf‑life segmented LightGBM models** on FreshRetailNet‑50K to forecast:

- **Observed sales** (censored by stockouts).
- **Recovered demand** (approximate true demand under full in‑stock conditions).
"""
)


# ---------- Sidebar: inputs ----------

st.sidebar.header("Product & Location")

c1, c2, c3 = st.sidebar.columns(3)
with c1:
    city_id = st.number_input("city_id", min_value=0, value=0, step=1)
with c2:
    store_id = st.number_input("store_id", min_value=0, value=0, step=1)
with c3:
    management_group_id = st.number_input("mgmt_group_id", min_value=0, value=0, step=1)

c4, c5, c6, c7 = st.sidebar.columns(4)
with c4:
    first_category_id = st.number_input("first_category_id", min_value=0, value=5, step=1)
with c5:
    second_category_id = st.number_input("second_category_id", min_value=0, value=6, step=1)
with c6:
    third_category_id = st.number_input("third_category_id", min_value=0, value=65, step=1)
with c7:
    product_id = st.number_input("product_id", min_value=0, value=38, step=1)

st.sidebar.header("Context & Covariates")

date_input = st.sidebar.date_input("Date", value=datetime(2024, 3, 28))

discount = st.sidebar.number_input("discount", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
holiday_flag = st.sidebar.checkbox("Holiday?", value=False)
activity_flag = st.sidebar.checkbox("Promotion/Activity?", value=False)

precpt = st.sidebar.number_input("precpt", min_value=0.0, value=1.7, step=0.1)
avg_temperature = st.sidebar.number_input("avg_temperature", min_value=-10.0, value=15.5, step=0.5)
avg_humidity = st.sidebar.number_input("avg_humidity", min_value=0.0, max_value=100.0, value=73.5, step=1.0)
avg_wind_level = st.sidebar.number_input("avg_wind_level", min_value=0.0, value=2.0, step=0.1)

stockout_hours = st.sidebar.number_input(
    "stockout_hours (06–22)",
    min_value=0,
    max_value=17,
    value=0,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Recovered demand estimates demand as if the product had been fully in stock "
    "during the 06–22 selling window."
)


# ---------- Tabs ----------

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Single forecast", "📊 Segment performance", "📈 Predicted vs observed", "📁 Upload new data"]
)


# ---------- Tab 1: single forecast ----------

with tab1:
    st.subheader("Single‑day forecast for a product & store")

    if st.button("Predict", key="predict_single"):
        dt = pd.to_datetime(date_input)
        dayofweek = dt.dayofweek
        isweekend = int(dayofweek in [5, 6])
        month = dt.month

        shelflifebucket = assign_shelf_life(int(third_category_id))

        # Base row using new names
        row = {
            "city_id": city_id,
            "store_id": store_id,
            "management_group_id": management_group_id,
            "first_category_id": first_category_id,
            "second_category_id": second_category_id,
            "third_category_id": third_category_id,
            "product_id": product_id,
            "dt": dt,
            "sale_amount": 0.0,  # placeholder
            "discount": discount,
            "holiday_flag": int(holiday_flag),
            "activity_flag": int(activity_flag),
            "precpt": precpt,
            "avg_temperature": avg_temperature,
            "avg_humidity": avg_humidity,
            "avg_wind_level": avg_wind_level,
            "stockout_hours": stockout_hours,
            "stock_hour6_22_cnt": stockout_hours,
            "is_not_stocked": int(stockout_hours == 0),
            "is_censored_day": int(stockout_hours >= 3),
            "is_partial_stockout": int(0 < stockout_hours < 17),
            "shelf_life_bucket": shelflifebucket,
            "dayofweek": dayofweek,
            "isweekend": isweekend,
            "month": month,
        }

        # Also create the old names expected by models, mapping from the new names
        row["iscensoredday"] = row["is_censored_day"]
        row["isnotstocked"] = row["is_not_stocked"]

        X = pd.DataFrame([row])[final_features].fillna(0)

        obs_model = obs_models[shelflifebucket]
        latent_model = latent_models[shelflifebucket]

        obs_pred = float(obs_model.predict(X)[0])
        latent_pred = float(latent_model.predict(X)[0])

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Observed sales forecast (kg/day)", f"{obs_pred:.2f}")
        with k2:
            st.metric("Recovered demand forecast (kg/day)", f"{latent_pred:.2f}")
        with k3:
            st.metric("Shelf‑life segment", shelflifebucket)

        st.markdown("### Input summary")
        st.dataframe(pd.DataFrame([row]))
    else:
        st.info("Set sidebar inputs and click **Predict** to see forecasts.")


# ---------- Tab 2: segment‑level performance ----------

with tab2:
    st.subheader("Model accuracy by shelf‑life segment")

    st.markdown(
        """
Each row summarizes the **LightGBM model trained for that shelf‑life bucket** on the hold‑out set:

- **WAPE** (Weighted Absolute Percentage Error) – business‑friendly error measure.
- **R²** – variance explained.
"""
    )

    st.dataframe(segment_metrics.style.format({"WAPE": "{:.2f}%", "R2": "{:.4f}"}))

    col_wape, col_r2 = st.columns(2)

    with col_wape:
        st.markdown("#### WAPE by segment")
        st.bar_chart(
            segment_metrics.set_index("Segment")["WAPE"],
            width="stretch",
        )

    with col_r2:
        st.markdown("#### R² by segment")
        st.bar_chart(
            segment_metrics.set_index("Segment")["R2"],
            width="stretch",
        )

    st.markdown(
        """
**Interpretation**:

- **ShortLife** items (highly perishable) are hardest to predict.
- **MediumLife** items sit in the middle.
- **LongLife** items are most stable with lowest WAPE and highest R².

This justifies training **separate models per shelf‑life bucket** instead of one global model.
"""
    )


# ---------- Tab 3: predicted vs observed over time ----------

with tab3:
    st.subheader("Predicted vs observed sales over time (eval set)")

    if eval_df is None:
        st.warning("Eval holdout parquet not found – time‑series plots are disabled.")
    elif DATE_COL not in eval_df.columns:
        st.error(f"Date column '{DATE_COL}' not found in eval_holdout.parquet.")
    else:
        eval_df[DATE_COL] = pd.to_datetime(eval_df[DATE_COL])

        date_range = st.date_input(
            "Date range (eval set)",
            value=(eval_df[DATE_COL].min(), eval_df[DATE_COL].max()),
        )

        if not isinstance(date_range, (list, tuple)):
            start_date = end_date = date_range
        else:
            start_date, end_date = date_range

        if st.button("Show time series", key="show_ts"):
            mask = (
                (eval_df["city_id"] == city_id)
                & (eval_df["store_id"] == store_id)
                & (eval_df["product_id"] == product_id)
                & (eval_df[DATE_COL] >= pd.to_datetime(start_date))
                & (eval_df[DATE_COL] <= pd.to_datetime(end_date))
            )
            ts = eval_df.loc[mask].copy().sort_values(DATE_COL)

            if ts.empty:
                st.warning("No eval data for this product/store in the selected range.")
            else:
                if "shelf_life_bucket" in ts.columns:
                    ts["shelflifebucket"] = ts["shelf_life_bucket"]
                else:
                    ts["shelflifebucket"] = ts["third_category_id"].apply(assign_shelf_life)

                ts["obs_pred"] = np.nan
                for seg in segments:
                    rows = ts["shelflifebucket"] == seg
                    if rows.any():
                        X_seg = ts.loc[rows, final_features].fillna(0)
                        ts.loc[rows, "obs_pred"] = obs_models[seg].predict(X_seg)

                chart_df = ts[[DATE_COL, "sale_amount", "obs_pred"]].set_index(DATE_COL)
                chart_df = chart_df.rename(
                    columns={"sale_amount": "Observed sales", "obs_pred": "Predicted sales"}
                )

                st.line_chart(chart_df, width="stretch")

                st.markdown("#### Stockout days for this product")
                if "stockout_hours" in ts.columns:
                    stockout_days = ts[ts["stockout_hours"] > 0][[DATE_COL, "stockout_hours"]]
                    if stockout_days.empty:
                        st.write("No stockouts in this period.")
                    else:
                        st.dataframe(stockout_days)
                else:
                    st.info("Column 'stockout_hours' not present in eval_holdout.parquet.")


# ---------- Tab 4: upload new data for batch scoring ----------

with tab4:
    st.subheader("Upload new daily data for batch prediction")

    st.markdown(
        """
Upload a CSV or Parquet file with the same columns used in modelling
(city/store/product, covariates, censoring features).  
The app will compute **observed sales** and **recovered demand** forecasts
for each row.
"""
    )

    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded)
        else:
            new_df = pd.read_parquet(uploaded)

        st.write("Preview of uploaded data:")
        st.dataframe(new_df.head())

        if st.button("Run batch predictions", key="run_batch"):
            df = new_df.copy()

            if "shelf_life_bucket" in df.columns:
                df["shelflifebucket"] = df["shelf_life_bucket"]
            elif "third_category_id" in df.columns:
                df["shelflifebucket"] = df["third_category_id"].apply(assign_shelf_life)
            else:
                st.error("Need 'shelf_life_bucket' or 'third_category_id' to assign segment.")
                st.stop()

            # create old flag names if needed
            if "is_censored_day" in df.columns:
                df["iscensoredday"] = df["is_censored_day"]
            if "is_not_stocked" in df.columns:
                df["isnotstocked"] = df["is_not_stocked"]

            df["obs_pred"] = np.nan
            df["latent_pred"] = np.nan

            for seg in segments:
                rows = df["shelflifebucket"] == seg
                if rows.any():
                    X_seg = df.loc[rows, final_features].fillna(0)
                    df.loc[rows, "obs_pred"] = obs_models[seg].predict(X_seg)
                    df.loc[rows, "latent_pred"] = latent_models[seg].predict(X_seg)

            st.success("Batch predictions computed.")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )