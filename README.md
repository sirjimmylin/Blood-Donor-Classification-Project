# 🩸 Hepatitis C Classification & Blood Donor Screening
### A Supervised Machine Learning Approach to Early Detection of Liver Disease

[![Python 3.12](https://img.shields.io/badge/Python-3.12.7-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-red.svg)](https://xgboost.readthedocs.io/)
[![Brown University](https://img.shields.io/badge/Institution-Brown%20University%20DSI-brown.svg)](https://dsi.brown.edu/)
[![Report](https://img.shields.io/badge/Report-PDF-red.svg)](report/finalreport.pdf)

> **Author:** Jimmy Lin  
> **Institution:** Data Science Institute, Brown University  
> **Repository:** [sirjimmylin/Blood-Donor-Classification-Project](https://github.com/sirjimmylin/Blood-Donor-Classification-Project.git)  
> **Interactive Project Showcase:** [Live Web Portfolio Page](https://sirjimmylin.github.io/Blood-Donor-Classification-Project/) | [`docs/index.html`](docs/index.html)

---

## 📌 Project Overview

Blood donor screening is a cornerstone of transfusion safety and preventative healthcare. Hepatitis C virus (HCV) infections frequently progress silently through hepatic fibrosis to irreversible cirrhosis without overt early symptoms. 

This project develops an end-to-end supervised machine learning pipeline to classify individuals across the complete **hepatitis disease spectrum** using routine, minimally invasive clinical blood laboratory tests. 

### Key Clinical & Technical Results
* **Dataset:** 615 patient instances with 12 clinical attributes from the [UCI Machine Learning Repository HCV Dataset](https://doi.org/10.24432/C5D612).
* **Metric Alignment:** Optimized for a **weighted F2F_2F2 score** to penalize life-threatening False Negatives (missing a hepatitis infection) twice as heavily as False Positives.
* **Top Performance:** **XGBoost Classifier** achieved an F2F_2F2 score of **0.9453 ± 0.0209**, outperforming baseline (0.8447) by **+4.76 standard deviations**.
* **Biomarker Interpretability:** Permutation importance and XGBoost gain metrics identified **AST (Aspartate Aminotransferase)** and **CHE (Cholinesterase)** as primary drivers of model decisions, corroborating established clinical consensus on hepatocellular necrosis and impaired liver synthesis.

---

## 📊 Model Comparison & Benchmarks

All models were evaluated using stratified 5-fold cross-validation with grid search hyperparameter optimization:

| Model Architecture | Feature Strategy | Weighted F2F_2F2 Score | Std Dev | Improvement vs. Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (Majority Class)** | Predicts "Blood Donor" | 0.8447 | 0.0000 | Reference Baseline |
| **Logistic Regression** | Reduced Features (Subset 1) | 0.9235 | 0.0261 | +3.02 F2\sigmaF2 |
| **Support Vector Classifier (Linear)** | Reduced Features (Subset 1) | 0.9313 | 0.0261 | +3.31 F2\sigmaF2 |
| **Random Forest Classifier** | Full Features, Ensemble | 0.9387 | 0.0200 | +3.60 F2\sigmaF2 |
| **XGBoost Classifier** 🏆 | Full Features, Gradient Boosted | **0.9453** | **0.0209** | **+4.76 F2\sigmaF2 (Winner)** |

---

## 🔬 Dataset & Clinical Attributes

The dataset comprises **615 patient records** classified into five disease progression categories:

| Target Category | Clinical Description | Patient Count | Percentage |
| :--- | :--- | :---: | :---: |
| `0=Blood Donor` | Healthy volunteer blood donors | 533 | 86.7% |
| `0s=suspect Blood Donor` | Donors with borderline or unconfirmed lab markers | 7 | 1.1% |
| `1=Hepatitis` | Active hepatitis infection | 24 | 3.9% |
| `2=Fibrosis` | Intermediate hepatic scarring | 21 | 3.4% |
| `3=Cirrhosis` | Advanced, irreversible end-stage liver damage | 30 | 4.9% |

### Laboratory Biomarkers
* **Demographics:** Age (years), Sex (`m`/`f`)
* **Liver Enzymes:**
  * **AST (Aspartate Aminotransferase, U/L):** Released into bloodstream upon liver tissue damage; key indicator of hepatocellular necrosis.
  * **ALT (Alanine Aminotransferase, U/L):** Liver-specific enzyme indicating acute liver inflammation.
  * **ALP (Alkaline Phosphatase, U/L):** Enzyme associated with bile ducts and biliary clearance.
  * **GGT (Gamma-Glutamyl Transferase, U/L):** Sensitive biomarker for biliary obstruction and alcohol/drug-induced liver stress.
* **Proteins & Metabolites:**
  * **ALB (Albumin, g/L):** Primary protein synthesized by the liver; binds bilirubin.
  * **CHE (Cholinesterase, U/L):** Serum enzyme reflecting liver metabolic and synthetic capacity.
  * **BIL (Bilirubin, μmol/L):** Hemoglobin breakdown byproduct; elevated in cirrhosis and jaundice.
  * **CHOL (Cholesterol, mmol/L):** Serum lipid synthesized and processed by hepatocytes.
  * **CREA (Creatinine, μmol/L):** Waste product monitoring renal function and hepatorenal syndrome.
  * **PROT (Total Protein, g/L):** Aggregate measure of serum globulins and albumin.

---

## 🛠️ Technical Challenges & Methodology

### 1. Severe Class Imbalance (86.7% Majority Class)
* **Challenge:** 533 healthy donors vs. only 7 suspect cases and 21–30 cases per disease stage. A naive majority-class classifier achieves ~87% raw accuracy.
* **Solution:** 
  * Implemented a custom **Stratified 60/20/20 train/validation/test split** ensuring all minority classes are represented across folds.
  * Incorporated balanced class-weight penalties (`class_weight="balanced"`) in linear and margin models.
  * Evaluated models strictly via **F2F_2F2-weighted score** with `zero_division=0` to ensure unpredicted minority classes in cross-validation folds do not produce metric bias.

### 2. Missing Laboratory Data
* **Challenge:** 5 continuous laboratory features contained missing measurements (26 rows total, with ALP missing ~2.9% and CHOL ~1.6%).
* **Solution:**
  * For linear/SVM models, trained **Reduced-Feature Subset Models** on clean feature combinations without missing values.
  * For tree-based models, leveraged **XGBoost's native default-direction routing**, learning optimal branch assignments for missing values during split finding.

### 3. Pathophysiological Model Interpretability
* Evaluated global feature importances using **Permutation Importance** (model-agnostic) alongside **XGBoost Gain & Weight metrics**.
* Confirmed that **AST** and **CHE** dominated decision splits, ensuring the model relies on true biological indicators of disease rather than demographic artifacts.

---

## 📁 Repository Structure

```
Blood-Donor-Classification-Project/
├── README.md                      # Project documentation & summary
├── project.yml                    # Conda reproducible environment specification
├── data/
│   ├── blooddonor.csv             # Raw UCI HCV dataset
│   └── preprocessed_data.pkl      # Preprocessed stratified split tensors
├── src/
│   ├── project.ipynb              # Primary analysis, modeling & interpretation notebook
│   ├── project_backup.ipynb       # Original pre-cleanup notebook backup
│   └── projectbaseline.ipynb      # Initial baseline exploratory calculations
├── figures/                       # High-resolution (300 DPI) publication visualizations
│   ├── category.png               # Target class distribution
│   ├── sexproportion.png          # Hepatitis prevalence by biological sex
│   ├── albumin.png                # Albumin vs. total protein relationship
│   ├── bilirubin.png              # Bilirubin elevation across cirrhosis
│   ├── f2weightedbaseline.png     # Benchmark model comparison chart
│   ├── permutationimportance.png  # Permutation feature importance rankings
│   ├── xgboost_gain_importance.png# XGBoost Gain feature rankings
│   └── xgboost_weight_importance.png# XGBoost Weight feature rankings
├── results/                       # Serialized model scores and GridSearchCV results (.pkl)
├── report/
│   ├── finalreport.md             # Complete academic report in Markdown
│   └── finalreport.pdf            # Compiled PDF publication report
└── docs/
    └── index.html                 # Interactive portfolio showcase webpage
```

---

## 🚀 Environment Setup & Reproducibility

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sirjimmylin/Blood-Donor-Classification-Project.git
   cd Blood-Donor-Classification-Project
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f project.yml
   conda activate data1030
   ```

3. **Launch the Jupyter notebook:**
   ```bash
   jupyter lab src/project.ipynb
   ```

---

## 📚 References & Citation

1. **Dataset Citation:**  
   Lichtinghagen, R., Klawonn, F., & Hoffmann, G. (2020). *HCV data*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5D612](https://doi.org/10.24432/C5D612).
2. **Clinical Research:**  
   Hoffmann, G. F. et al. (2018). *Using machine learning techniques to generate laboratory diagnostic pathways—a case study.* Journal of Laboratory and Precision Medicine.
3. **Biomarker Correlation:**  
   Zhang, H. et al. (2023). *Correlation Between Total Bilirubin, Total Bilirubin/Albumin Ratio with Disease Activity.* International Journal of General Medicine, 16, 273–280.
