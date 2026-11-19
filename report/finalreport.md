# Hepatitis C Progression & Blood Donor Classification
## A Supervised Machine Learning Approach to Early Detection of Liver Disease

**Jimmy Lin**  
*Data Science Institute, Brown University*  
**Repository:** [sirjimmylin/Blood-Donor-Classification-Project](https://github.com/sirjimmylin/Blood-Donor-Classification-Project.git)  
**Interactive Showcase:** [Project Webpage](https://sirjimmylin.github.io/Blood-Donor-Classification-Project/)

---

## 1. Introduction 

### Motivation
Blood donor screening is a cornerstone of transfusion safety and preventative healthcare. Hepatitis C virus (HCV) infections frequently progress silently through hepatic fibrosis to irreversible cirrhosis without overt symptoms. Accurate diagnostic models offer the potential to screen asymptomatic blood donors rapidly and cost-effectively using routine laboratory blood tests, simultaneously protecting the blood supply and expediting clinical intervention before irreversible organ damage occurs.

### Dataset Description
The dataset comprises 615 patient records with 12 clinical attributes from the UCI Machine Learning Repository HCV Dataset [1]. The target variable is `Category`, classifying individuals across the hepatitis progression spectrum: healthy blood donors (`0=Blood Donor`), borderline donor profiles (`0s=suspect Blood Donor`), active hepatitis (`1=Hepatitis`), fibrosis (`2=Fibrosis`), and end-stage cirrhosis (`3=Cirrhosis`). Continuous laboratory features include routine liver enzymes and metabolic biomarkers: Albumin (ALB), Alkaline Phosphatase (ALP), Alanine Aminotransferase (ALT), Aspartate Aminotransferase (AST), Bilirubin (BIL), Cholinesterase (CHE), Cholesterol (CHOL), Creatinine (CREA), Gamma-Glutamyl Transferase (GGT), and Total Protein (PROT), alongside patient Age and biological Sex.

### Previous Work
This dataset originated from research by Lichtinghagen, Klawonn, and Hoffmann (2020), who demonstrated that automated diagnostic pathways from laboratory measurements could detect liver fibrosis and cirrhosis [2]. Their findings emphasized that while machine learning algorithms can detect intricate nonlinear biomarker patterns, models must be clinically interpretable and aligned with pathophysiological domain knowledge to assist medical practitioners effectively.

---

## 2. Exploratory Data Analysis (EDA)

The cohort contains 615 patient instances across 12 predictor features and 1 target attribute.

### Target Variable Distribution
![Category](../figures/category.png)
*Figure 1: Target variable distribution across the 5 clinical categories, exhibiting severe class imbalance.*

As shown in Figure 1 and Table 1, the dataset exhibits severe class imbalance: 86.7% of instances belong to healthy blood donors, while the minority disease classes range from only 7 to 30 instances.

| Target Category | Clinical Description | Patient Count | Percentage |
| :--- | :--- | :---: | :---: |
| `0=Blood Donor` | Healthy volunteer blood donors | 533 | 86.7% |
| `0s=suspect Blood Donor` | Donors with borderline laboratory markers | 7 | 1.1% |
| `1=Hepatitis` | Active hepatitis infection | 24 | 3.9% |
| `2=Fibrosis` | Intermediate hepatic scarring | 21 | 3.4% |
| `3=Cirrhosis` | Advanced, irreversible liver damage | 30 | 4.9% |

*Table 1: Cohort distribution by clinical diagnosis.*

### Feature Distributions & Clinical Relationships
Exploratory pairwise analyses highlighted critical clinical relationships between laboratory markers and disease progression:

![Sex Proportion](../figures/sexproportion.png)
*Figure 2: Proportion of hepatitis diagnoses stratified by biological sex.*

Biological sex represents the single categorical predictor. As illustrated in Figure 2, male donors exhibited a noticeably higher incidence of hepatitis and fibrosis compared to female donors, making biological sex an essential demographic stratifier.

![Albumin](../figures/albumin.png)
*Figure 3: Strong linear relationship between Albumin (ALB) and Total Protein (PROT).*

Between numerical markers, Albumin and Total Protein demonstrated a strong positive linear correlation (Figure 3). Albumin is synthesized exclusively by hepatocytes and represents the largest component of total serum protein, validating expected baseline physiological behavior [3].

![Bilirubin](../figures/bilirubin.png)
*Figure 4: Distribution of serum Bilirubin across clinical stages.*

Bilirubin is a metabolic breakdown product of hemoglobin cleared by the liver. Elevated serum bilirubin reflects biliary impairment and hepatic decompensation. As shown in Figure 4, bilirubin levels spiked selectively in patients with stage 3 Cirrhosis, aligning with established diagnostic criteria for end-stage liver disease [3].

### Missing Laboratory Data
Missing values were confined to 5 continuous laboratory features, affecting 26 of the 615 rows (Table 2).

| Feature | Description | Fraction Missing |
| :--- | :--- | :---: |
| ALB | Albumin | 0.16% |
| ALP | Alkaline Phosphatase | 2.93% |
| ALT | Alanine Aminotransferase | 0.16% |
| CHOL | Cholesterol | 1.63% |
| PROT | Total Protein | 0.16% |

*Table 2: Proportion of missing values per laboratory biomarker.*

---

## 3. Methodology & Modeling Pipeline

### Splitting Strategy
Given the extreme class imbalance, random partitioning would risk omitting minority classes (e.g., `0s=suspect Blood Donor` with only 7 cases) from training or validation folds. A stratified partitioning strategy was developed:
1. An initial stratified split isolated a 20% holdout test set ($N=123$) while preserving exact category proportions.
2. The remaining 80% ($N=492$) was partitioned using 5-fold Stratified K-Fold cross-validation for hyperparameter tuning and model comparison.
3. The resulting partition mirrors a 60% train, 20% validation, and 20% independent test split.

### Preprocessing Pipeline
Continuous features were standardized using `StandardScaler` to zero mean and unit variance, while `Age` was normalized using `MinMaxScaler`. Biological sex was encoded via `OneHotEncoder`. Missing values were addressed through dual strategies:
* For linear and support vector models, **Reduced-Feature Subset Models** were trained on feature combinations containing complete observations.
* For gradient boosted trees, **XGBoost's native default-direction routing** handled missing continuous inputs dynamically during split evaluation.

### Evaluation Metric: Weighted $F_2$ Score
In clinical diagnostic screening, **False Negatives carry catastrophic consequences** (an infected donor's blood entering the supply or a patient missing early therapeutic intervention), whereas False Positives incur comparatively minor follow-up testing costs. 

Standard accuracy is invalid due to the 86.7% majority baseline. The **$F_2$ score** was chosen because it places twice as much weight on recall as precision:
$$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

To account for multi-class proportions without skewing rare classes, a support-weighted average with `zero_division=0` was implemented across all cross-validation evaluations.

### Hyperparameter Tuning
Grid search cross-validation was conducted across four candidate architectures:

| Model Architecture | Hyperparameter Grid | Optimal Setting |
| :--- | :--- | :--- |
| **SVC (Linear)** | $C \in \{0.1, 1, 10\}$, kernel = linear, class_weight = balanced | $C=1$, kernel = linear |
| **Logistic Regression** | $C \in \{0.01, 1, 100\}$, penalty $\in \{L_1, L_2\}$, solver $\in \{\text{liblinear, saga}\}$ | $C=1$, penalty = $L_1$, solver = liblinear |
| **Random Forest** | $n\_estimators \in \{100, 200\}$, max_depth $\in \{\text{None}, 10\}$, max_features $\in \{\text{sqrt}, \text{log2}\}$ | $n\_estimators=100$, max_depth=None, max_features=sqrt |
| **XGBoost** | max_depth $\in \{3, 5, 7\}$, learning_rate $\in \{0.01, 0.1\}$, subsample $\in \{0.8, 1.0\}$ | max_depth=3, learning_rate=0.1, subsample=0.8 |

*Table 3: Hyperparameter optimization search spaces and selected configurations.*

---

## 4. Results & Benchmark Evaluation

### Model Comparison
The majority-class baseline yields an $F_2$-weighted score of 0.8447. Table 4 benchmarks all trained architectures against the baseline:

| Model Architecture | Feature Strategy | Weighted $F_2$ Score | Std Dev | Improvement vs. Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (Majority Class)** | Predicts "Blood Donor" | 0.8447 | 0.0000 | Reference Baseline |
| **Logistic Regression** | Reduced Features (Subset 1) | 0.9235 | 0.0261 | +3.02 $\sigma$ |
| **Support Vector Classifier (Linear)** | Reduced Features (Subset 1) | 0.9313 | 0.0261 | +3.31 $\sigma$ |
| **Random Forest Classifier** | Full Features, Ensemble | 0.9387 | 0.0200 | +3.60 $\sigma$ |
| **XGBoost Classifier** | Full Features, Gradient Boosted | **0.9453** | **0.0209** | **+4.76 $\sigma$ (Best Model)** |

*Table 4: Cross-validated performance benchmarks across model architectures.*

![F2weightedscore](../figures/f2weightedbaseline.png)
*Figure 5: Cross-validated $F_2$-weighted performance comparison across all evaluated models.*

The **XGBoost Classifier achieved the highest score of 0.9453 ± 0.0209**, outperforming the baseline by **4.76 standard deviations**. The tree ensemble architectures systematically outperformed linear models due to their ability to capture nonlinear biomarker interactions without requiring manual interaction engineering.

---

## 5. Model Interpretation & Biomarker Analysis

### Global Feature Importance
To ensure clinical validity, feature importances were evaluated using both model-agnostic **Permutation Importance** and tree-intrinsic **Gain and Weight** metrics:

![Permutation Importance](../figures/permutationimportance.png)
*Figure 6: Top 10 features ranked by Permutation Feature Importance.*

![XGBoost Weight](../figures/xgboost_weight_importance.png)
*Figure 7: Top 10 features ranked by XGBoost split frequency (Weight).*

![XGBoost Gain](../figures/xgboost_gain_importance.png)
*Figure 8: Top 10 features ranked by XGBoost relative information Gain.*

### Clinical Takeaways
1. **AST (Aspartate Aminotransferase):** Ranked #1 by XGBoost Weight and consistently near the top in Gain. AST is an intracellular enzyme released during hepatocellular injury; elevated AST is the hallmark clinical indicator of acute and chronic liver necrosis.
2. **CHE (Cholinesterase):** Emerged as the #1 predictive biomarker by XGBoost Gain. Cholinesterase measures liver protein synthesis and metabolic activity; depressed CHE levels serve as a strong discriminator for advanced fibrosis and cirrhosis.
3. **GGT (Gamma-Glutamyl Transferase):** Ranked highly across all importance metrics, validating its sensitivity to biliary tract inflammation and chronic hepatic injury.
4. **Demographics (Sex and Age):** Biological sex contributed meaningfully to decision boundaries in permutation tests, reflecting the observed male disease prevalence, whereas Age had minimal predictive power once liver enzymes were accounted for.

---

## 6. Discussion & Future Outlook

### Clinical Feasibility
The high diagnostic recall ($F_2 = 0.945$) achieved by XGBoost on standard blood laboratory features demonstrates that machine learning can provide an effective, automated preliminary screening tool for blood donation centers and primary care clinics. Flagging at-risk donors for confirmatory nucleic acid testing (NAT) can reduce diagnostic turnaround latency and prevent inadvertent transfusion transmission.

### Local Interpretability with TreeSHAP
While global feature rankings confirmed domain validity, translating models into clinical practice requires **patient-specific explanations**. For multi-class architectures, computing 3-dimensional SHAP tensors ($\text{samples} \times \text{features} \times \text{classes}$) allows generation of individualized patient force plots. For instance, clinicians can observe whether an individual's elevated risk score was driven primarily by an acute AST elevation (suggesting active hepatitis) versus severely depleted CHE and elevated Bilirubin (indicating advanced cirrhosis).

### Limitations
A key constraint remains the small absolute sample size of minority categories (e.g., 7 suspect donors, 21 fibrosis patients). Future investigations should validate this pipeline across multi-center clinical cohorts with independent prospective testing to evaluate model generalizability across diverse demographic distributions.

---

## 7. References

1. Lichtinghagen, R., Klawonn, F., & Hoffmann, G. (2020). *HCV data* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5D612](https://doi.org/10.24432/C5D612).
2. Hoffmann, G. F., Bietenbeck, A., Lichtinghagen, R., & Klawonn, F. (2018). *Using machine learning techniques to generate laboratory diagnostic pathways—a case study.* Journal of Laboratory and Precision Medicine, 3, 56.
3. Zhang, H., et al. (2023). *Correlation Between Total Bilirubin, Total Bilirubin/Albumin Ratio with Disease Activity.* International Journal of General Medicine, 16, 273–280.
