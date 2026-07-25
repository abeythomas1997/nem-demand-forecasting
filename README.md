# ⚡ Australian NEM Operational Demand Forecasting using Machine Learning

## Overview

This project develops an end-to-end machine learning pipeline to forecast 30-minute ahead operational electricity demand for the Victorian region (VIC1) of the Australian National Electricity Market (NEM).

The objective is not only to build an accurate forecasting model, but also to demonstrate how production-quality forecasting systems are designed, monitored, and maintained throughout the complete machine learning lifecycle.

The project includes data engineering, feature engineering, model selection, hyperparameter tuning, inference, monitoring, and automated retraining decisions.

---

# Business Problem

Accurate short-term electricity demand forecasting is essential for maintaining power system reliability and supporting efficient market operation.

Electricity retailers, network operators and market operators rely on demand forecasts to:

- Schedule generation resources
- Maintain grid stability
- Balance electricity supply and demand
- Reduce operating costs
- Support renewable energy integration
- Improve operational decision making


---

# Dataset

### Operational Demand

- Source: Australian Energy Market Operator (AEMO)
- Region: VIC1
- Resolution: 30-minute intervals

### Weather Data

Weather variables used include:

- Temperature
- Apparent temperature
- Relative humidity
- Wind speed
- Cloud cover
- Solar radiation
- Precipitation
- Rain indicator

---

# Project Architecture

```
                AEMO Demand Data
                        │
                        ▼
                Weather Data
                        │
                        ▼
             Data Validation & Cleaning
                        │
                        ▼
             Feature Engineering
                        │
                        ▼
          Train / Validation / Test Split
                        │
                        ▼
              Multiple ML Models
                        │
                        ▼
          Hyperparameter Optimisation
                        │
                        ▼
             Best XGBoost Model
                        │
                        ▼
                 Model Persistence
                        │
                        ▼
                 Inference Pipeline
                        │
                        ▼
              Production Monitoring
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Data Quality                 Feature Drift
         ▼                             ▼
              Prediction Performance
                        │
                        ▼
              Retraining Decision
```

---

# Machine Learning Pipeline

The project follows a complete production-style ML workflow.

## 1. Data Preparation

- Data validation
- Missing value handling
- Duplicate removal
- Timestamp processing
- Weather integration

---

## 2. Feature Engineering

Calendar Features

- Year
- Month
- Day
- Hour
- Minute
- Day of week
- Weekend indicator

Lag Features

- Lag 1
- Lag 2
- Lag 48
- Lag 336

Rolling Statistics

- Rolling mean
- Rolling minimum
- Rolling maximum
- Rolling standard deviation

Cyclical Features

- Time sine/cosine
- Day sine/cosine
- Month sine/cosine

Weather Features

- Temperature
- Apparent temperature
- Humidity
- Wind speed
- Solar radiation
- Cloud cover
- Rainfall

---

# Models Evaluated

The following models were compared:

- Baseline Persistence Model
- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost

Hyperparameter tuning was performed to identify the best-performing model.

---

# Final Model

**Model**

XGBoost Regressor

### Test Performance

| Metric | Value |
|---------|-------|
| MAE | **61.04 MW** |
| RMSE | **86.85 MW** |
| MAPE | **1.17%** |
| R² | **0.9897** |

---

# Model Monitoring

The project includes a lightweight monitoring framework.

### Data Quality Monitoring

- Missing value detection

### Feature Drift Monitoring

Population Stability Index (PSI)

Status levels:

- PASS
- WARNING
- FAIL

### Prediction Monitoring

- MAE
- RMSE
- MAPE

---

# Automated Retraining Decision

The system evaluates whether retraining is required based on:

- Model performance
- Feature drift
- Data quality

Possible outcomes:

- KEEP_CURRENT_MODEL
- RETRAIN_MODEL

Example:

```
Monitoring Status

WARNING

Performance

PASS

Retraining Decision

KEEP_CURRENT_MODEL
```

---

# Project Structure

```
Energy-Vic/

│
├── data/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── data
│   ├── features
│   ├── models
│   ├── evaluation
│   ├── inference
│   ├── monitoring
│   ├── tracking
│   └── utils
│
├── tests/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run the complete pipeline

```bash
python main.py
```

The pipeline automatically performs:

- Data preparation
- Feature engineering
- Model training
- Hyperparameter tuning
- Final evaluation
- Inference
- Monitoring
- Retraining decision
- Output generation

---

# Outputs

The pipeline generates:

- Model comparison
- Test metrics
- Predictions
- Actual vs Predicted plots
- Residual analysis
- Error analysis by hour/day/month
- Monitoring reports
- Model manifest
- Experiment history

---

# Technologies Used

Programming

- Python

Machine Learning

- Scikit-learn
- XGBoost

Data Processing

- Pandas
- NumPy

Visualisation

- Matplotlib

Software Engineering

- Git
- GitHub
- Pytest
- GitHub Actions

---

# Future Improvements

Potential production enhancements include:

- Probabilistic forecasting
- Multi-step forecasting
- Deep learning models (LSTM / Temporal Fusion Transformer)
- MLflow model registry
- Docker deployment
- FastAPI prediction service
- Cloud deployment
- Automated scheduled retraining
- Live weather API integration

---

# Author

**Abey Thomas**

Master of Analytics

Background in Electrical & Electronics Engineering with professional experience in the energy sector.

Interested in:

- Energy Analytics
- Machine Learning
- Forecasting
- MLOps
- Power Systems
- AI for Sustainable Energy

---

## License

This project is provided for educational and portfolio purposes.