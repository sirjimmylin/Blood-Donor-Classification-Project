# GEMINI.md - Project Context

This project is a supervised machine learning application for classifying blood donors and identifying different stages of Hepatitis based on clinical blood laboratory measurements.

## Project Overview
- **Goal:** Classify patients into five categories: `Blood Donor`, `Hepatitis`, `Fibrosis`, `Cirrhosis`, and `Suspect Blood Donor` (though mainly focused on donor vs. hepatitis stages).
- **Dataset:** UCI Machine Learning Repository "HCV data".
- **Models:** Logistic Regression, SVC, Random Forest, and XGBoost.
- **Evaluation Metric:** The project uses F2-score (implied by file names like `model_f2_scores.pkl` and `f2weightedbaseline.png`) to prioritize recall over precision, which is critical in medical diagnosis.

## Technical Stack
- **Language:** Python 3.12.7
- **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`.
- **Visualization:** `matplotlib`, `plotly`.
- **Environment Management:** Conda (`project.yml`).

## Directory Structure
- `data/`: Contains raw CSV data (`blooddonor.csv`) and preprocessed data (`preprocessed_data.pkl`).
- `src/`: 
  - `project.ipynb`: Main notebook for data analysis, preprocessing, and model training.
  - `projectbaseline.ipynb`: Baseline calculations and correlation analysis.
  - `oldproject.ipynb`: Defunct/archive notebook.
- `results/`: Pickled model outputs and performance comparisons (LR, SVC, RF, XGB).
- `figures/`: Visualizations including feature distributions, confusion matrices, and feature importance (Permutation Importance, XGBoost Weight/Gain).
- `report/`: Final project documentation in Markdown and PDF.

## Workflow & Development
- **Environment Setup:** 
  ```bash
  conda env create -f project.yml
  conda activate data1030
  ```
- **Analysis:** Most development happens within Jupyter Notebooks in the `src/` directory. 
- **Preprocessing:** Data cleaning and transformations are stored in `data/preprocessed_data.pkl`.
- **Hyperparameter Tuning:** Models are tuned before saving final results to the `results/` directory.

## Key Files
- `src/project.ipynb`: The primary entry point for current research and implementation.
- `project.yml`: Definition for the reproducible Conda environment.
- `README.md`: High-level project overview and data dictionary.
