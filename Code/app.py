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

# CUSTOM CSS
st.markdown("""
<style>
.main {
    background: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 36px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}
.hero p {
    font-size: 18px;
    color: #dbeafe;
}
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    border-left: 6px solid #2563eb;
}
.small-text {
    color: #64748b;
    font-size: 15px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #e0f2fe 100%);
    border-right: 1px solid #cbd5e1;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0f172a;
    font-weight: 800;
}
[data-testid="stSidebar"] label {
    color: #334155;
    font-weight: 600;
    font-size: 13px;
}
.sidebar-info {
    background: white;
    padding: 14px;
    border-radius: 14px;
    border-left: 5px solid #2563eb;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    font-size: 13px;
    color: #475569;
    margin-bottom: 12px;
}
/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 14px;
    background: white;
    padding: 10px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 52px;
    padding-left: 22px;
    padding-right: 22px;
    border-radius: 12px;
    color: #334155;
    font-weight: 700;
    background: #f8fafc;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <p style="font-size:14px; letter-spacing:1px; text-transform:uppercase; color:#bfdbfe; margin-bottom:8px;">
        Retail Analytics • Machine Learning • Inventory Optimization
    </p>
    <h1>Stockout-Aware Perishable Demand Forecasting</h1>
    <p>
    Forecast observed sales and recover hidden customer demand caused by stockouts using
    shelf-life segmented LightGBM models.
    </p>
    <div style="margin-top:22px;">
        <span style="background:#dbeafe; color:#1e3a8a; padding:8px 14px; border-radius:999px; margin-right:8px; font-weight:700;">
            LightGBM
        </span>
        <span style="background:#dcfce7; color:#166534; padding:8px 14px; border-radius:999px; margin-right:8px; font-weight:700;">
            Streamlit
        </span>
        <span style="background:#fef3c7; color:#92400e; padding:8px 14px; border-radius:999px; font-weight:700;">
            CRISP-DM
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar: inputs ----------

st.sidebar.markdown("## 🎛️ Forecast Controls")

st.sidebar.markdown(
    """
    <div class="sidebar-info">
    Simulate demand by changing store, product, weather, promotion, and stockout inputs.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 📍 Product & Store")

city_id = st.sidebar.number_input("City ID", min_value=0, value=5, step=1)
store_id = st.sidebar.number_input("Store ID", min_value=0, value=120, step=1)

st.sidebar.markdown("### 🛒 Product Category")

first_category_id = st.sidebar.number_input("Category Level 1", min_value=0, value=5, step=1)
second_category_id = st.sidebar.number_input("Category Level 2", min_value=0, value=6, step=1)
third_category_id = st.sidebar.number_input("Category Level 3", min_value=0, value=120, step=1)
product_id = st.sidebar.number_input("Product ID", min_value=0, value=250, step=1)

st.sidebar.markdown("### 🌦️ Demand Conditions")

# Hidden default values (not shown in sidebar)
management_group_id = 0
precpt = 1.7
avg_humidity = 73.5
avg_wind_level = 2.0

date_input = st.sidebar.date_input("Forecast Date", value=datetime(2024, 3, 28))

discount = st.sidebar.slider("Discount Multiplier", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

holiday_flag = st.sidebar.checkbox("Holiday Effect", value=False)
activity_flag = st.sidebar.checkbox("Promotion / Campaign Active", value=False)

avg_temperature = st.sidebar.number_input("Average Temperature", min_value=-10.0, value=15.5, step=0.5)

st.sidebar.markdown("### ⚠️ Stockout Signal")

stockout_hours = st.sidebar.slider(
    "Stockout Hours During Selling Window",
    min_value=0,
    max_value=17,
    value=4,
    step=1,
)

st.sidebar.markdown(
    """
    <div class="sidebar-info">
    Higher stockout hours indicate stronger demand censoring.  
    The recovered-demand model estimates what customers likely wanted if shelves stayed stocked.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------- Tabs ----------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Single Forecast",
        "📊 Model Performance",
        "📈 Sales vs Predictions",
        "📁 Batch Forecasting"
    ]
)


# ---------- Tab 1: single forecast ----------

with tab1:
    st.subheader("🔮 Single-Day Forecast")

    st.markdown(
        """
        Use this section to simulate demand for one product-store-day scenario.  
        The model predicts both **observed sales** and **recovered customer demand** after adjusting for stockout effects.
        """
    )

    if st.button("Run Forecast", key="predict_single"):
        dt = pd.to_datetime(date_input)
        dayofweek = dt.dayofweek
        isweekend = int(dayofweek in [5, 6])
        month = dt.month

        shelflifebucket = assign_shelf_life(int(third_category_id))

        row = {
            "city_id": city_id,
            "store_id": store_id,
            "management_group_id": management_group_id,
            "first_category_id": first_category_id,
            "second_category_id": second_category_id,
            "third_category_id": third_category_id,
            "product_id": product_id,
            "dt": dt,
            "sale_amount": 0.0,
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

        row["iscensoredday"] = row["is_censored_day"]
        row["isnotstocked"] = row["is_not_stocked"]

        X = pd.DataFrame([row])[final_features].fillna(0)

        obs_model = obs_models[shelflifebucket]
        latent_model = latent_models[shelflifebucket]

        obs_pred = max(0, float(obs_model.predict(X)[0]))
        latent_pred = max(0, float(latent_model.predict(X)[0]))
        gap = max(0, latent_pred - obs_pred)

        st.markdown("### Forecast Results")

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📦 Observed Sales</h4>
                <h2>{obs_pred:.2f} kg/day</h2>
                <p class="small-text">Expected sales recorded by the POS system.</p>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Recovered Demand</h4>
                <h2>{latent_pred:.2f} kg/day</h2>
                <p class="small-text">Estimated true customer demand after stockout adjustment.</p>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🧊 Shelf-Life Segment</h4>
                <h2>{shelflifebucket}</h2>
                <p class="small-text">Model selected based on product perishability.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top:20px; border-left:6px solid #f97316;">
            <h4>⚠️ Stockout Demand Gap</h4>
            <h2>{gap:.2f} kg/day</h2>
            <p class="small-text">
            This is the estimated demand hidden by stockouts. A higher gap suggests stronger under-ordering risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Scenario Inputs Used")

        input_summary = pd.DataFrame([{
            "City": city_id,
            "Store": store_id,
            "Product": product_id,
            "Category Level 3": third_category_id,
            "Date": dt.strftime("%Y-%m-%d"),
            "Discount": discount,
            "Holiday": "Yes" if holiday_flag else "No",
            "Promotion": "Yes" if activity_flag else "No",
            "Stockout Hours": stockout_hours,
            "Segment": shelflifebucket
        }])

        st.dataframe(input_summary, use_container_width=True)

    else:
        st.info("Set sidebar inputs and click **Run Forecast** to generate demand predictions.")



# ---------- Tab 2: segment-level performance ----------

with tab2:
    st.subheader("📊 Model Performance by Shelf-Life Segment")

    st.markdown(
        """
        This section compares how well the forecasting model performs across different perishability groups.
        Lower **WAPE** means lower forecast error, while higher **R²** means the model explains more demand variation.
        """
    )

    best_segment = segment_metrics.loc[segment_metrics["R2"].idxmax(), "Segment"]
    lowest_error_segment = segment_metrics.loc[segment_metrics["WAPE"].idxmin(), "Segment"]

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏆 Best R² Segment</h4>
            <h2>{best_segment}</h2>
            <p class="small-text">Highest model fit among shelf-life groups.</p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Lowest WAPE</h4>
            <h2>{lowest_error_segment}</h2>
            <p class="small-text">Most accurate segment by business error.</p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 Model Strategy</h4>
            <h2>Segmented</h2>
            <p class="small-text">Separate LightGBM models improve demand learning.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Performance Summary")

    display_metrics = segment_metrics.copy()
    display_metrics["WAPE"] = display_metrics["WAPE"].map(lambda x: f"{x:.2f}%")
    display_metrics["R²"] = display_metrics["R2"].map(lambda x: f"{x:.4f}")
    display_metrics = display_metrics[["Segment", "Target", "WAPE", "R²"]]

    st.dataframe(display_metrics, use_container_width=True)

    st.markdown("### Visual Comparison")

    col_wape, col_r2 = st.columns(2)

    with col_wape:
        st.markdown("""
        <div class="metric-card">
            <h4>📉 WAPE by Segment</h4>
            <p class="small-text">Lower WAPE means better forecasting accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

        st.bar_chart(
            segment_metrics.set_index("Segment")["WAPE"],
            width="stretch",
        )

    with col_r2:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 R² by Segment</h4>
            <p class="small-text">Higher R² means better model explanation power.</p>
        </div>
        """, unsafe_allow_html=True)

        st.bar_chart(
            segment_metrics.set_index("Segment")["R2"],
            width="stretch",
        )

    st.markdown("""
    <div class="metric-card" style="margin-top:20px; border-left:6px solid #16a34a;">
        <h4>💡 Business Interpretation</h4>
        <p class="small-text">
        Long-life products are more stable and easier to forecast, while short-life items are harder because
        perishability, stockouts, and demand volatility create more noise. This supports using separate
        shelf-life models instead of one global forecasting model.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- Tab 3: predicted vs observed over time ----------

with tab3:
    st.subheader("📈 Sales vs Predictions Over Time")

    st.markdown(
        """
        This section compares actual observed sales against model-predicted sales for a product-store pair
        in the evaluation dataset. It also highlights stockout days so viewers can see when demand may have been censored.
        """
    )

    if eval_df is None:
        st.warning("Eval holdout parquet not found – time-series plots are disabled.")
    elif DATE_COL not in eval_df.columns:
        st.error(f"Date column '{DATE_COL}' not found in eval_holdout.parquet.")
    else:
        eval_df[DATE_COL] = pd.to_datetime(eval_df[DATE_COL])

        date_range = st.date_input(
            "Select Evaluation Date Range",
            value=(eval_df[DATE_COL].min(), eval_df[DATE_COL].max()),
        )

        if not isinstance(date_range, (list, tuple)):
            start_date = end_date = date_range
        else:
            start_date, end_date = date_range

        if st.button("Generate Time-Series View", key="show_ts"):
            mask = (
                (eval_df["city_id"] == city_id)
                & (eval_df["store_id"] == store_id)
                & (eval_df["product_id"] == product_id)
                & (eval_df[DATE_COL] >= pd.to_datetime(start_date))
                & (eval_df[DATE_COL] <= pd.to_datetime(end_date))
            )

            ts = eval_df.loc[mask].copy().sort_values(DATE_COL)

            if ts.empty:
                sample_row = eval_df.sample(1).iloc[0]

                ts = eval_df[
                    (eval_df["city_id"] == sample_row["city_id"])
                    & (eval_df["store_id"] == sample_row["store_id"])
                    & (eval_df["product_id"] == sample_row["product_id"])
                    & (eval_df[DATE_COL] >= pd.to_datetime(start_date))
                    & (eval_df[DATE_COL] <= pd.to_datetime(end_date))
                ].copy().sort_values(DATE_COL)

                st.info(
                    f"No eval data found for selected inputs, so showing an available sample: "
                    f"City {sample_row['city_id']}, Store {sample_row['store_id']}, Product {sample_row['product_id']}."
                )

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

            avg_actual = ts["sale_amount"].mean()
            avg_predicted = ts["obs_pred"].fillna(0).mean()
            total_stockout_hours = ts["stockout_hours"].sum() if "stockout_hours" in ts.columns else 0

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🧾 Avg. Observed Sales</h4>
                    <h2>{avg_actual:.2f}</h2>
                    <p class="small-text">Average sales recorded in the evaluation period.</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🤖 Avg. Predicted Sales</h4>
                    <h2>{avg_predicted:.2f}</h2>
                    <p class="small-text">Average model forecast for the same period.</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚠️ Stockout Hours</h4>
                    <h2>{int(total_stockout_hours)}</h2>
                    <p class="small-text">Total hours where shelves were unavailable.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Sales Trend Comparison")

            ts["obs_pred"] = ts["obs_pred"].fillna(0)
            chart_df = ts[[DATE_COL, "sale_amount", "obs_pred"]].set_index(DATE_COL)
            chart_df = chart_df.rename(
                columns={"sale_amount": "Observed Sales", "obs_pred": "Predicted Sales"}
            )

            st.line_chart(chart_df, width="stretch")

            st.markdown("### Stockout Events")

            if "stockout_hours" in ts.columns:
                stockout_days = ts[ts["stockout_hours"] > 0][[DATE_COL, "stockout_hours"]]
                stockout_days = stockout_days.rename(
                    columns={DATE_COL: "Date", "stockout_hours": "Stockout Hours"}
                )

                if stockout_days.empty:
                    st.success("No stockouts occurred in this selected period.")
                else:
                    st.dataframe(stockout_days, use_container_width=True)

                    st.markdown("""
                    <div class="metric-card" style="margin-top:20px; border-left:6px solid #f97316;">
                        <h4>💡 Interpretation</h4>
                        <p class="small-text">
                        Stockout days can suppress recorded sales because customers may have wanted to buy the product,
                        but the product was unavailable. This is why stockout-aware forecasting is important for
                        replenishment decisions.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Column 'stockout_hours' not present in eval_holdout.parquet.")

# ---------- Tab 4: upload new data for batch scoring ----------

with tab4:
    st.subheader("📁 Batch Forecasting for New Data")

    st.markdown(
        """
        Upload a CSV or Parquet file with the same schema used during model training.  
        The app will generate **observed sales forecasts** and **recovered demand forecasts** for every row.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <h4>📤 Upload</h4>
            <p class="small-text">Import CSV or Parquet data.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Score</h4>
            <p class="small-text">Run segment-specific LightGBM models.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <h4>⬇️ Export</h4>
            <p class="small-text">Download predictions as CSV.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Upload New Dataset")

    uploaded = st.file_uploader(
        "Upload CSV or Parquet file",
        type=["csv", "parquet"],
        help="File should include product, store, category, weather, promotion, and stockout fields."
    )

    if uploaded is not None:

        # Read uploaded file
        if uploaded.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded)
        else:
            new_df = pd.read_parquet(uploaded)

        st.success(f"File uploaded successfully: {uploaded.name}")

        # Preview
        st.markdown("### Data Preview")
        st.dataframe(new_df.head(), use_container_width=True)

        row_count = len(new_df)
        col_count = len(new_df.columns)

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Rows Uploaded</h4>
                <h2>{row_count:,}</h2>
                <p class="small-text">Total records available for scoring.</p>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Columns Detected</h4>
                <h2>{col_count}</h2>
                <p class="small-text">Input variables detected in uploaded file.</p>
            </div>
            """, unsafe_allow_html=True)

        # Run forecasts
        if st.button("Run Batch Forecasts", key="run_batch"):

            df = new_df.copy()

            # ---------- Create missing time features ----------
            if "dt" in df.columns:
                df["dt"] = pd.to_datetime(df["dt"])
                df["dayofweek"] = df["dt"].dt.dayofweek
                df["isweekend"] = df["dayofweek"].isin([5, 6]).astype(int)
                df["month"] = df["dt"].dt.month
            else:
                df["dayofweek"] = 0
                df["isweekend"] = 0
                df["month"] = 1

            # ---------- Create shelf-life segment ----------
            if "third_category_id" in df.columns:
                df["shelflifebucket"] = df["third_category_id"].apply(assign_shelf_life)

            elif "shelf_life_bucket" in df.columns:
                df["shelflifebucket"] = df["shelf_life_bucket"]

            else:
                st.error("Need 'third_category_id' or 'shelf_life_bucket' to assign segment.")
                st.stop()

            # ---------- Normalize segment names ----------
            df["shelflifebucket"] = df["shelflifebucket"].replace({
                "Short_Life": "ShortLife",
                "Medium_Life": "MediumLife",
                "Long_Life": "LongLife",
                "Short Life": "ShortLife",
                "Medium Life": "MediumLife",
                "Long Life": "LongLife",
            })

            # ---------- Create old model flag names ----------
            if "is_censored_day" in df.columns:
                df["iscensoredday"] = df["is_censored_day"]

            if "is_not_stocked" in df.columns:
                df["isnotstocked"] = df["is_not_stocked"]

            # ---------- Ensure required columns exist ----------
            required_defaults = {
                "discount": 0,
                "holiday_flag": 0,
                "activity_flag": 0,
                "precpt": 0,
                "avg_temperature": 20,
                "avg_humidity": 50,
                "avg_wind_level": 1,
                "stockout_hours": 0,
                "stock_hour6_22_cnt": 0,
                "iscensoredday": 0,
                "isnotstocked": 0,
            }

            for col, default_val in required_defaults.items():
                if col not in df.columns:
                    df[col] = default_val

            # ---------- Prediction columns ----------
            df["obs_pred"] = np.nan
            df["latent_pred"] = np.nan

            # ---------- Run segment models ----------
            for seg in segments:

                rows = df["shelflifebucket"] == seg

                if rows.any():

                    X_seg = df.loc[rows, final_features].fillna(0)

                    df.loc[rows, "obs_pred"] = (
                        obs_models[seg]
                        .predict(X_seg)
                        .clip(min=0)
                    )

                    df.loc[rows, "latent_pred"] = (
                        latent_models[seg]
                        .predict(X_seg)
                        .clip(min=0)
                    )

            # ---------- Final cleanup ----------
            df["obs_pred"] = df["obs_pred"].fillna(0)
            df["latent_pred"] = df["latent_pred"].fillna(0)

            df["demand_gap"] = (
                df["latent_pred"] - df["obs_pred"]
            ).clip(lower=0)

            st.success("Batch forecasts generated successfully.")

            # ---------- Output preview ----------
            st.markdown("### Forecast Output Preview")

            st.dataframe(df.head(), use_container_width=True)

            avg_obs = float(df["obs_pred"].mean())
            avg_latent = float(df["latent_pred"].mean())
            avg_gap = float(df["demand_gap"].mean())

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📦 Avg. Observed Forecast</h4>
                    <h2>{avg_obs:.2f}</h2>
                    <p class="small-text">Average predicted POS-recorded sales.</p>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📈 Avg. Recovered Demand</h4>
                    <h2>{avg_latent:.2f}</h2>
                    <p class="small-text">Average estimated true customer demand.</p>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚠️ Avg. Demand Gap</h4>
                    <h2>{avg_gap:.2f}</h2>
                    <p class="small-text">Estimated hidden demand from stockouts.</p>
                </div>
                """, unsafe_allow_html=True)

            # ---------- Download ----------
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Forecast Results as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )