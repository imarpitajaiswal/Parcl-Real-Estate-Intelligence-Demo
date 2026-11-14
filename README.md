# Parcl | Real Estate Buyer Intelligence

### Machine Learning-Based Buyer Segmentation & Investment Profiling

**Python | Pandas | NumPy | Scikit-learn | Plotly | Streamlit | Business Intelligence**

[Live Dashboard](https://parcl-real-estate-intelligence.streamlit.app) |
[GitHub Repository](https://github.com/imarpitajaiswal/Parcl-Real-Estate-Intelligence-Demo)

> **Public Demo Disclosure:** This repository uses independently
> generated synthetic data. Buyer segment labels are simulated,
> not predictions from the original machine learning model.

---

## 1. Executive Summary

Parcl Real Estate Buyer Intelligence is an end-to-end data analytics
and machine learning project developed as part of the Unified Mentor
internship.

The project explores how property transaction records can be
transformed into customer-level behavioral features, interpretable
buyer segments, and interactive business intelligence.

The original internship research investigated unsupervised machine
learning. This public Streamlit application demonstrates the
analytical interface using fictional transaction records.

## 2. Business Problem

Real estate businesses need to understand purchasing behavior,
transaction frequency, property preferences, spending patterns,
and geographic activity.

Without structured analytics, identifying meaningful customer
groups and communicating insights can be challenging.

This project explores how data-driven segmentation can support
better customer understanding and business reporting.

## 3. Project Objectives

1. Validate buyer and property datasets.
2. Clean and prepare transaction records.
3. Engineer buyer-level behavioral features.
4. Explore unsupervised machine learning.
5. Evaluate clustering performance.
6. Interpret descriptive buyer profiles.
7. Develop an interactive business intelligence dashboard.
8. Provide a privacy-conscious synthetic public demonstration.

## 4. Original Research Workflow

The internship research followed these stages:

1. Data validation and cleaning
2. Buyer-level feature engineering
3. Feature preprocessing and scaling
4. K-Means clustering
5. Hierarchical clustering
6. Model optimization and evaluation
7. Buyer segment profiling
8. Dashboard development and quality assurance

The original model evaluation artifacts are not included
in this public repository.

## 5. Buyer Segmentation Framework

The research explored descriptive profiles including:

| Segment | Analytical Focus |
|---|---|
| High-Volume Property Buyers | Purchasing frequency and transaction activity |
| Mixed-Portfolio Buyers | Residential and commercial property participation |
| Residential-Focused Buyers | Residential property purchasing patterns |

These categories do not establish customer income, wealth,
investment suitability, or financial risk.

## 6. Interactive Dashboard

The public dashboard contains five modules.

### Executive Overview

Summary metrics, simulated transaction spending,
buyer segment distribution, and aggregated exports.

### Buyer Segmentation

Interactive customer profiles, segment comparisons,
and purchasing behavior analysis.

### Investment Intelligence

Property spending distributions, transaction values,
and purchase-pattern visualizations.

### Geographic Analytics

Geographic buyer representation and regional comparisons.

### ML Model Insights

Research methodology, simulated segment distribution,
and interpretation limitations.

## 7. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| ML Research | Scikit-learn |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

Scikit-learn was used in the original research workflow.
The public dashboard does not train or serve a clustering model.

## 8. Repository Structure

```text
Parcl-Real-Estate-Intelligence-Demo/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── demo_data/
    ├── buyer_segments.csv
    ├── properties_cleaned.csv
    └── demo_metadata.json
```

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/imarpitajaiswal/Parcl-Real-Estate-Intelligence-Demo.git
cd Parcl-Real-Estate-Intelligence-Demo
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```

## 10. Analytical Applications

The demonstration illustrates how interactive analytics can help
explore questions such as:

- How does purchasing activity vary across buyer groups?
- Which simulated segments account for higher spending?
- How do property preferences differ between groups?
- How does geographic representation vary?
- How can transaction data support customer profiling?

All public dashboard metrics are derived from fictional records.

## 11. Limitations

- Public data is synthetic, not observed market data.
- Public segment labels are simulated rather than trained predictions.
- Financial metrics represent fictional transactions.
- Original ML evaluation results are not reproduced in this demo.
- Descriptive segments do not establish causality.
- Segment membership does not verify investment suitability.
- Results must not be used for real investment decisions.

## 12. Future Enhancements

Potential extensions include:

- Deployment of trained models on authorized datasets
- Expanded clustering stability evaluation
- Temporal purchasing analysis
- Geographic mapping enhancements
- Automated dashboard testing
- Additional data governance documentation

## 13. Author

**Arpita Jaiswal**

MCA — Artificial Intelligence & Machine Learning

**GitHub:** [imarpitajaiswal](https://github.com/imarpitajaiswal)

**Live Application:** [Parcl Real Estate Intelligence](https://parcl-real-estate-intelligence.streamlit.app)

---

**Data Ethics:** This public repository contains fictional demonstration
data. Original internship buyer and property records are not distributed.
