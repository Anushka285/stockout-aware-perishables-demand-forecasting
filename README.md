# 📦 Stockout-Aware Perishable Demand Forecasting

### Machine Learning System for Perishable Retail Inventory Forecasting & Demand Recovery

An interactive machine learning dashboard that forecasts:

- 📈 **Observed product sales**
- 🧠 **Recovered hidden customer demand**
- ⚠️ **Stockout-driven demand gaps**

using **shelf-life segmented LightGBM models** trained on the **FreshRetailNet-50K** retail dataset.

---

# 🚀 Live Demo

### 🌐 Streamlit Application

https://stockout-aware-perishables-demand-forecasting.streamlit.app/

---

# 📸 Application Preview

## 🔮 Forecast Dashboard
<img width="1126" height="913" alt="image" src="https://github.com/user-attachments/assets/e6bacbb4-643a-43bf-8aee-ccecf603a620" />

## 📊 Model Performance Analytics
<img width="1124" height="909" alt="image" src="https://github.com/user-attachments/assets/d487ecfe-216e-4913-8177-dc5f703fc642" />

## 📈 Sales vs Predictions
<img width="1126" height="905" alt="image" src="https://github.com/user-attachments/assets/5ee9d6d0-003d-484b-a3ab-d12e84284589" />

## 📁 Batch Forecasting
<img width="1915" height="1019" alt="image" src="https://github.com/user-attachments/assets/158ca238-d9c7-465c-8430-6cdf672ac265" />
<img width="1902" height="908" alt="image" src="https://github.com/user-attachments/assets/8e54bd6b-baf6-40d2-b042-8393a64ffb34" />
<img width="1900" height="903" alt="image" src="https://github.com/user-attachments/assets/b2b54210-4ab9-4f2b-9817-77e38ed68f6d" />

---

# 📌 Project Overview

Perishable retail categories such as:

- fresh produce
- dairy
- bakery
- meat

are highly sensitive to inventory decisions.

Traditional forecasting systems rely only on recorded sales data. However, when products go out of stock, customers cannot purchase them, causing sales data to underestimate true demand.

This project builds a **stockout-aware demand forecasting system** that estimates both:

- what was actually sold
- what customers likely wanted to buy if products had remained fully stocked

The final system combines:

- shelf-life segmentation
- stockout-aware feature engineering
- machine learning forecasting
- demand recovery estimation
- interactive business dashboards

---

# 🎯 Business Problem

Retailers face two major inventory risks:

### Under-Ordering
- Empty shelves
- Lost sales
- Lower customer satisfaction

### Over-Ordering
- Product spoilage
- Waste
- Increased inventory costs

The challenge becomes even harder for highly perishable products because stockouts censor true customer demand.

This project addresses that challenge by building forecasting models that explicitly account for stockout behavior.

---

# 🧠 Solution Approach

The project follows the **CRISP-DM framework**:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

# ⚙️ Core Forecasting Strategy

## Shelf-Life Segmentation

Instead of training one global forecasting model, products are grouped into:

- 🥬 ShortLife
- 🥛 MediumLife
- 🥔 LongLife

Each segment receives its own machine learning model to better capture perishability patterns and demand behavior.

---

## Demand Recovery

The system estimates hidden customer demand by identifying stockout periods and recovering demand that may have been lost when products were unavailable.

Two forecasts are generated:

### 📦 Observed Sales
What the POS system records.

### 🧠 Recovered Demand
Estimated true customer demand under full-stock conditions.

---

# 🤖 Machine Learning Models

The final production system uses:

- LightGBM (primary model)
- Random Forest
- Ridge Regression baseline
- Tabular MLP baseline

The deployed dashboard uses **segment-specific LightGBM models** for forecasting.

---

# 📊 Model Performance

| Segment | WAPE | R² |
|---|---|---|
| ShortLife | ~36% | ~0.71 |
| MediumLife | ~38% | ~0.60 |
| LongLife | ~32% | ~0.92 |

---

## Key Insights

- Long-life products are easier to forecast because demand is more stable.
- Highly perishable products show greater volatility and stockout sensitivity.
- Segment-specific modeling improves performance compared to one global forecasting model.

---

# 🗂️ Dataset

## FreshRetailNet-50K

The dataset contains:

- 4.5M+ retail records
- store-product-day observations
- promotions & discounts
- weather variables
- stockout indicators
- temporal features
- shelf-life characteristics

### Key Features Used

| Category | Features |
|---|---|
| Product Hierarchy | city_id, store_id, category IDs, product_id |
| Temporal | dayofweek, month, weekend |
| Promotions | discount, activity_flag |
| Weather | temperature, humidity, precipitation |
| Stockouts | stockout_hours, censored-day flags |
| Target | sale_amount |

---

# 🖥️ Interactive Dashboard Features

## 🔮 Single Forecast Simulation

Generate forecasts for:

- observed sales
- recovered demand
- stockout demand gap
- shelf-life segment

using interactive business inputs.

---

## 📊 Segment Performance Analytics

Compare forecasting quality across shelf-life groups using:

- WAPE
- R²
- business KPI summaries

---

## 📈 Sales vs Predictions Analysis

Visualize:

- observed vs predicted sales
- stockout periods
- demand behavior over time

---

## 📁 Batch Forecasting

Upload CSV or Parquet datasets to:

- generate bulk forecasts
- estimate recovered demand
- export prediction results instantly

---

# 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | LightGBM, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |
| Model Serialization | Joblib |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

# 📂 Repository Structure

```text
stockout-aware-perishable-demand-forecasting/
│
├── Code/
│   ├── app.py
│   ├── 01_EDA.ipynb
│   └── 02_Modelling.ipynb
│
├── Data/
│
├── models/
│
├── README.md
├── requirements.txt
└── Capstone_Final_Report.docx
```

---

# 🚀 Run Locally

## Clone Repository

```bash
git clone https://github.com/Anushka285/stockout-aware-perishable-demand-forecasting.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run Code/app.py
```

---

# 💡 Business Impact

This project demonstrates how machine learning can support:

- retail inventory optimization
- demand recovery under stockouts
- perishable supply chain planning
- replenishment decision-making
- forecasting analytics for retail operations

---

# 🧠 Skills Demonstrated

- Machine Learning
- Demand Forecasting
- Retail Analytics
- Inventory Optimization
- Streamlit Dashboard Development
- Feature Engineering
- Time-Series Analysis
- Business Intelligence
- Interactive ML Deployment

---

# 🙏 Acknowledgements

- Dataset: FreshRetailNet-50K by Dingdong Inc.
- Inspired by the FreshRetailNet-50K latent demand recovery research paper.
- Developed as part of a Business Analytics Capstone project at the University of North Texas.
