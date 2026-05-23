# Stockout-Aware Perishables Inventory Forecasting

**App link =** https://stockout-aware-perishables-inv-forecasting.streamlit.app/ 

*A segmented machine-learning approach to the censored-demand newsvendor problem*

This project builds a **stockout-aware demand forecasting system** for perishable retail inventory using the public **FreshRetailNet‑50K** dataset. The system explicitly models **censored demand** caused by stockouts and produces two parallel forecasts:

- **Observed sales** – what the POS system will record (censored when shelves empty).
- **Recovered latent demand** – an estimate of true customer demand under full in-stock availability.

The final solution uses **shelf-life segmented LightGBM models** (Short, Medium, Long life products) and is exposed through an interactive **Streamlit dashboard**.

---

## 1. Business problem

Perishable categories (fresh produce, dairy, meat, bakery) contribute a large share of supermarket revenue but are extremely sensitive to stocking decisions:

- Under‑ordering → **empty shelves**, lost sales, and lower customer trust.
- Over‑ordering → **spoilage and write-offs**.

Standard forecasting pipelines train on **raw historical sales**. When a product sells out mid-day, the recorded sales are lower than actual demand, so the model learns to **repeat stockouts**. This project tackles that structural issue by:

- **Flagging and quantifying stockouts** at the hourly level.
- **Recovering latent demand** for stockout days.
- **Segmenting by shelf life** and training separate models for Short, Medium, and Long life items.

---

## 2. Dataset

- **Source**: [FreshRetailNet‑50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (4.5M daily SKU-store records).
- **Granularity**: one row per **store–product–day**.
- **Key features** (after preprocessing):
  - Hierarchy: `city_id`, `store_id`, `management_group_id`, `first_category_id`, `second_category_id`, `third_category_id`, `product_id`
  - Time: `dt`, `dayofweek`, `isweekend`, `month`
  - Target: `sale_amount` (daily units sold)
  - Censoring & stockouts:
    - `stockout_hours` (business hours 06–22 with no stock)
    - `is_not_stocked`, `is_censored_day`, `is_partial_stockout`
    - `hours_sale`, `hours_stock_status` (24‑dim arrays)
  - Drivers: `discount`, `holiday_flag`, `activity_flag`
  - Weather: `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`
  - Segment: `shelf_life_bucket` (ShortLife, MediumLife, LongLife)

Preprocessed splits are stored under:

- `Data/Processed/train_split.parquet`
- `Data/Processed/val_split.parquet`
- `Data/Processed/eval_holdout.parquet`

---

## 3. Modelling approach

### 3.1 Segmentation

Products are segmented by **shelf‑life bucket**:

- `ShortLife` – highly perishable SKUs (e.g., leafy greens).
- `MediumLife` – moderate shelf life.
- `LongLife` – stable items with low spoilage (e.g., root vegetables).

Each segment receives its own model to respect different demand patterns and censoring behavior.

### 3.2 Demand recovery

To mimic the “latent demand” focus of the FreshRetailNet‑50K paper, the pipeline builds a simple but explicit **recovered demand** target:

- If a product is in stock for only a fraction of business hours, daily sales are scaled up based on hours in stock (with caps to avoid extreme inflation).
- This yields `latent_demand_recovered`, used alongside the original `sale_amount`.

### 3.3 Models

For each shelf‑life segment, the project trains and compares:

- **LightGBM** (final production model)
- Random Forest
- Ridge linear regression (DLinear‑style baseline)
- Tabular MLP (deep baseline)

The final system selects **LightGBM per segment** as the primary model family and trains:

- One model per segment on **observed sales** (`lgbm_obs_*.txt`)
- One model per segment on **recovered demand** (`lgbm_latent_*.txt`)

Model files are stored in `models/`:

- `lgbm_obs_ShortLife.txt`, `lgbm_obs_MediumLife.txt`, `lgbm_obs_LongLife.txt`
- `lgbm_latent_ShortLife.txt`, `lgbm_latent_MediumLife.txt`, `lgbm_latent_LongLife.txt`
- `model_config.joblib` (feature list & config)

---

## 4. Repository structure

```text
Stockout-Aware-Perishables-Forecasting/
├─ Code/
│  ├─ 01_EDA.ipynb              # Data understanding & exploratory analysis
│  ├─ 02_Modelling.ipynb        # Feature engineering & model training
│  └─ app.py                    # Streamlit dashboard
├─ Data/
│  └─ Processed/
│     ├─ train_split.parquet
│     ├─ val_split.parquet
│     └─ eval_holdout.parquet
├─ models/
│  ├─ lgbm_obs_ShortLife.txt
│  ├─ lgbm_obs_MediumLife.txt
│  ├─ lgbm_obs_LongLife.txt
│  ├─ lgbm_latent_ShortLife.txt
│  ├─ lgbm_latent_MediumLife.txt
│  ├─ lgbm_latent_LongLife.txt
│  └─ model_config.joblib
├─ requirements.txt
└─ README.md
```

> Raw FreshRetailNet‑50K data is not shipped in this repo due to size; see the dataset link above and the notebooks for how to recreate the processed splits.

---

## 5. How to run the Streamlit app locally

### 5.1 Clone the repository

```bash
git clone https://github.com/dipanshuparashar902/Stockout-Aware-Perishables-Forecasting.git
cd Stockout-Aware-Perishables-Forecasting
```

### 5.2 Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.4 Run the app

```bash
cd Code
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 6. Dashboard overview

The Streamlit app provides four main views:

1. **🔮 Single forecast**  
   - Choose city, store, product, date, and covariates in the sidebar.  
   - See:
     - Observed sales forecast (kg/day)
     - Recovered demand forecast (kg/day)
     - Shelf‑life segment
   - Input summary table for reproducibility.

2. **📊 Segment performance**  
   - Table and bar charts for WAPE and R² by shelf‑life segment (Short, Medium, Long).  
   - Highlights that:
     - Short‑life items are hardest to forecast.
     - Long‑life items achieve the best accuracy (lowest WAPE, highest R²).

3. **📈 Predicted vs observed**  
   - Select a product, store, and date range from the eval set.  
   - Plot observed vs predicted daily sales over time.  
   - Show days with stockouts (“empty shelf” periods) and their stockout hours.

4. **📁 Upload new data**  
   - Upload a CSV or Parquet file with the same schema as the processed data.  
   - The app returns:
     - Observed sales forecasts
     - Recovered demand forecasts  
   - Download predictions as CSV for integration into planning tools.




<img width="1874" height="914" alt="c1" src="https://github.com/user-attachments/assets/cc91ff3e-3302-4a85-a841-08891bce9c7c" />
<img width="1792" height="893" alt="c2" src="https://github.com/user-attachments/assets/fba4ed1d-33e7-4f1f-882c-86ff5a576d5a" />
<img width="1594" height="892" alt="c3" src="https://github.com/user-attachments/assets/803a5775-29c6-40e4-bc54-3797f3ec5e9d" />
<img width="1551" height="728" alt="c4" src="https://github.com/user-attachments/assets/8416f7fb-075d-4365-b216-fed4da309abb" />
<img width="1535" height="718" alt="c5" src="https://github.com/user-attachments/assets/eb9274d9-755b-465e-9322-aecb9aeceacd" />
<img width="1513" height="883" alt="c6" src="https://github.com/user-attachments/assets/584930cd-48c4-4d7b-97ed-5954ab17000b" />
<img width="1730" height="986" alt="c7" src="https://github.com/user-attachments/assets/08850f0d-0295-4cd3-9e41-2bb819784ed6" />
<img width="1516" height="792" alt="c8" src="https://github.com/user-attachments/assets/9871ac18-5fe2-4a1f-b44b-f17f43f0ae76" />
<img width="1481" height="832" alt="c9" src="https://github.com/user-attachments/assets/5d1a9333-174d-411c-8ba9-5828b6580693" />
<img width="1892" height="781" alt="c10" src="https://github.com/user-attachments/assets/9fff8426-c545-4edc-9137-77f7170e8e60" />
<img width="1481" height="832" alt="c" src="https://github.com/user-attachments/assets/3d7300d3-78c0-486f-a24a-6f53e77adc15" />

---

## 7. Results (high level)

On the validation and holdout sets, the final shelf‑life segmented LightGBM models achieve:

- **ShortLife**: WAPE ≈ 36%, R² ≈ 0.71  
- **MediumLife**: WAPE ≈ 38%, R² ≈ 0.60  
- **LongLife**: WAPE ≈ 32%, R² ≈ 0.92  

These models significantly outperform linear baselines and simple tabular MLPs for this censored, stockout‑heavy perishable retail dataset.

---

## 8. Future work

- Integrate more advanced latent‑demand recovery models (e.g., SSA, TFT, ImputeFormer) from the FreshRetailNet‑50K paper.  
- Extend segmentation to include product category and store format.  
- Deploy as a containerized service (e.g., FastAPI + Streamlit) for integration into real replenishment systems.

---

## 9. Acknowledgements

- **Dataset**: FreshRetailNet‑50K by Dingdong Inc.  
- **Inspiration**: *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.*  
- **Capstone**: Business Analytics Capstone, University of North Texas.
