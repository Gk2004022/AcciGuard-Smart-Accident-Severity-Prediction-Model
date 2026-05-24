# AcciGuard: Smart Vehicle Accident Severity Prediction using Machine Learning

**Author:** Antigravity (Google DeepMind)  
**Project Phase:** Production-Ready Delivery  
**Target Variable:** `accident_severity` (Class 1: Fatal, Class 2: Serious, Class 3: Slight)  
**Dataset Scale:** 1,804,624 Records (2008–2020)  

---

## 1. Executive Summary

This report delivers a production-ready, mathematically rigorous exploratory and predictive framework for road accident severity in the United Kingdom. Analyzing over **1.8 million historical records**, this project bridges the gap between descriptive statistics and actionable machine learning.

Through comprehensive preprocessing, advanced inferential hypothesis testing (**Chi-Square Test of Independence with Cramer's V** and **Kruskal-Wallis H-Test**), and multi-model evaluation, we engineered an optimal predictive pipeline. Our core findings establish that **LightGBM with cost-sensitive Class Weighting** represents the industry-optimal model, yielding **49.78% Recall for Fatal accidents** and **41.19% Recall for Serious accidents**—representing a **17x and 114x performance multiplier**, respectively, over traditional SMOTE oversampling models.

---

## 2. Preprocessing & Feature Engineering Audit

To ensure high data quality and eliminate leakage, a rigorous preprocessing pipeline was audited and executed (derived from `1-preprocessing.ipynb`):
1. **Irrelevant Feature Elimination:** Dropped high-cardinality metadata (`accident_index`, `accident_reference`, `local_authority_highway`, `lsoa_of_accident_location`, `local_authority_ons_district`) that introduce noise and lead to model overfitting.
2. **Temporal Parsing & Engineering:** Unified `date` and `time` into a pandas `datetime64` object. Engineered high-value features:
   * `accident_month` (1–12) and `accident_hour` (0–23).
   * `is_weekend` (Binary indicator derived from day of week).
   * `time_of_day` (Binned into: `Morning` [5-12], `Afternoon` [12-17], `Evening` [17-21], `Night` [21-5]).
   * `season` (Binned into: `Winter`, `Rainy`, `Summer`, `Autumn`).
3. **Imputation of Missing Coordinates:** Latitude and longitude null values were imputed using the group-wise mean of their corresponding `police_force` area, preserving spatial clustering.
4. **Target Alignment:** Mapped standard target variable `accident_severity` (originally 1: Fatal, 2: Serious, 3: Slight) into a 0-indexed format (`0: Fatal`, `1: Serious`, `2: Slight`) to comply with multi-class gradient boosting API constraints (e.g., XGBoost, LightGBM).

---

## 3. Exploratory Data Analysis & Statistical Discoveries

### 3.1 Univariate Analysis
* **Target Class Imbalance:** The target variable is severely skewed, representing a major architectural constraint:
  * **Slight (Class 3 / Indexed 2):** 1,506,722 rows (**83.50%**)
  * **Serious (Class 2 / Indexed 1):** 275,390 rows (**15.26%**)
  * **Fatal (Class 1 / Indexed 0):** 22,512 rows (**1.25%**)
  * *Implication:* Standard accuracy is a useless metric. A dummy classifier predicting "Slight" for every accident achieves 83.5% accuracy but fails completely to predict fatal or serious accidents.
* **Speed Limits:** Speed limits show a highly discrete distribution. **30 mph** is the most common road limit (representing urban residential streets), followed by **60 mph** (rural single-carriageway roads) and **70 mph** (motorways).
* **Hourly Trend:** Accidents peak sharply during morning rush hours (**08:00**) and afternoon rush hours (**15:00 to 17:00**), corresponding to daily school and work commutes.

### 3.2 Bivariate Insights
Standardizing bivariate analyses into **100% Stacked Proportions** reveals crucial safety associations:
* **Urban vs. Rural:** While urban areas experience a higher absolute count of accidents due to density, **rural area accidents have double the proportion of Fatal and Serious outcomes** compared to urban areas. This is driven by higher speeds and delayed emergency response times on rural roads.
* **Speed Limit Relationship:** There is a direct, positive correlation between speed limit and accident seriousness. Roads with **60 mph and 70 mph limits show a significantly larger Fatal percentage** compared to 20 mph and 30 mph zones.
* **Light Conditions:** Accidents occurring in **Darkness (No Lights)** or **Darkness (Lights Unlit)** exhibit a substantially higher Fatal and Serious percentage compared to daylight conditions, highlighting visibility as a primary hazard multiplier.

---

## 4. Advanced Inferential Hypothesis Testing

To mathematically justify feature selection, we executed rigorous inferential statistical tests.

### 4.1 Chi-Square Test of Independence (Categorical Features)
We test the null hypothesis ($H_0$) that each categorical feature is independent of accident severity. Because large sample sizes ($N = 1.8M$) make p-values hyper-sensitive, we calculate **Cramer's V** to measure the true, sample-size-independent effect size (strength of association).

$$\text{Cramer's } V = \sqrt{\frac{\chi^2}{n \times \min(k - 1, r - 1)}}$$

Where $n = 1,804,624$, $k = 3$ (target classes), and $r$ is the unique categories of the predictor.

| Predictive Rank | Categorical Feature | Chi-Square Stat | Degrees of Freedom | p-value | Cramer's V | Association Strength |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `urban_or_rural_area` | 9132.880 | 2 | $0.000\text{e}+00$ | **0.0711** | Weak (Highly Significant) |
| **2** | `speed_limit` | 8954.512 | 12 | $0.000\text{e}+00$ | **0.0498** | Weak (Highly Significant) |
| **3** | `light_conditions` | 5112.441 | 8 | $0.000\text{e}+00$ | **0.0376** | Weak (Highly Significant) |
| **4** | `time_of_day` | 3822.451 | 6 | $0.000\text{e}+00$ | **0.0325** | Weak (Highly Significant) |
| **5** | `junction_detail` | 4124.811 | 18 | $0.000\text{e}+00$ | **0.0338** | Weak (Highly Significant) |
| **6** | `weather_conditions` | 1342.331 | 16 | $1.240\text{e}-271$ | **0.0193** | Negligible (Significant) |
| **7** | `is_weekend` | 644.212 | 2 | $1.340\text{e}-140$ | **0.0189** | Negligible (Significant) |
| **8** | `season` | 240.119 | 6 | $2.110\text{e}-48$ | **0.0082** | Negligible (Significant) |

*Analysis:* All categorical features successfully reject $H_0$ ($p < 0.05$). **`urban_or_rural_area`**, **`speed_limit`**, and **`light_conditions`** represent the most statistically significant drivers of accident severity and are prioritized as core ML features.

### 4.2 Kruskal-Wallis H-Test (Continuous Features)
Continuous variables (`speed_limit`, `accident_hour`) violate the normality and variance homogeneity assumptions of standard ANOVA. We therefore implement the non-parametric **Kruskal-Wallis H-Test** based on ranks.

* **Speed Limit:** $H$-Statistic = **8842.12**, $p$-value = **$0.000\text{e}+00$** (Statistically Significant)
* **Accident Hour:** $H$-Statistic = **412.33**, $p$-value = **$2.440\text{e}-90$** (Statistically Significant)

*Conclusion:* The median speeds and times of accidents vary significantly across Fatal, Serious, and Slight accident groups, scientifically confirming their inclusion in the predictive pipeline.

---

## 5. Machine Learning Pipeline & Imbalance Strategy

We construct a robust machine learning pipeline in `src/model_pipeline.py` using:
1. **One-Hot Encoding:** Low-cardinality categories.
2. **Numeric Scaling:** Standardizing spatial coordinates, hour, and speed limits.
3. **Stratified Split:** Creating an 80/20 train-test split that maintains class ratios.

### 5.1 The Imbalance Showdown: Cost-Sensitive Class Weighting vs. SMOTE
To handle the extreme minority class skew (1.2% Fatal), we evaluate and contrast two methods:
* **Cost-Sensitive Class Weighting:** We penalize minority class errors in the loss function during training, scaling the penalty inversely to the class frequency.
  
$$W_c = \frac{N}{C \times N_c}$$

  * *Advantage:* Computationally instantaneous. Requires no extra memory. Highly robust.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically generates new minority class instances using $k$-nearest neighbors in the feature space until all classes are perfectly balanced in the training set.
  * *Disadvantage:* Explodes memory usage. Increases training time from 7 seconds to 22 seconds, and on the full dataset can trigger Out-of-Memory (OOM) kernel crashes.

---

## 6. Empirical Model Comparison & Leaderboard

To ensure objective evaluation, all models were trained and tested on the exact same stratified splits (using a stable 10% representative subset, representing **180,000+ total rows**).

| Rank | Classifier Model | Accuracy | Macro F1-Score | Macro Recall | Fatal F1 | Serious F1 | Slight F1 | Fatal Recall | Serious Recall | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **LightGBM (Class Weighted)** | **57.01%** | **35.98%** | **50.33%** | **7.48%** | **28.52%** | **71.94%** | **49.78%** | **41.19%** | **7.53s** |
| **2** | **Random Forest (Class Weighted)** | 59.67% | 35.97% | 48.34% | 6.91% | 26.24% | 74.76% | 47.33% | 32.95% | 17.10s |
| **3** | **XGBoost (Class Weighted)** | 55.66% | 35.19% | 50.06% | 7.01% | 27.59% | 70.98% | 52.44% | 38.96% | 22.33s |
| **4** | **Logistic Regression (Weighted)** | 51.40% | 32.84% | 49.64% | 6.05% | 24.55% | 67.93% | **62.22%** | 31.88% | 81.30s |
| **5** | **LightGBM (SMOTE Balanced)** | **83.30%** | **32.02%** | **34.30%** | **4.41%** | **0.72%** | **90.94%** | **2.89%** | **0.36%** | **21.77s** |

---

## 7. Deep Insights & Model Reasoning

### 7.1 Why LightGBM (Class Weighted) is the Winner
1. **The SMOTE Failure Mode:** The LightGBM model trained on SMOTE data achieves a deceptively high **Accuracy of 83.30%**, but a poor **Macro F1-Score of 32.02%**. Looking closely, its **Fatal Recall is a near-zero 2.89%** and its **Serious Recall is a near-zero 0.36%**. 
   * *Reasoning:* SMOTE overfits the synthetic feature space regions of the minority classes. In high-dimensional spaces with overlapping distributions (which is typical for road accident conditions), SMOTE creates noisy synthetic samples that do not generalize well, causing the model to default to predicting "Slight" for almost all real-world test cases.
2. **The Class Weighting Triumph:** LightGBM with cost-sensitive class weights yields a **Fatal Recall of 49.78%** (correctly identifying half of all fatal accidents) and a **Serious Recall of 41.19%**, while maintaining a balanced **Macro F1-Score of 35.98%**.
   * *Reasoning:* Rather than modifying the data distribution (which introduces noise), Class Weighting alters the optimization landscape, forcing tree nodes to split on features that maximize the separation of the minority classes, leading to robust generalization.
3. **Execution Efficiency:** LightGBM trains in **7.53 seconds**—more than **2x faster than Random Forest (17.10s)**, **3x faster than XGBoost (22.33s)**, and **10x faster than Logistic Regression (81.30s)**, making it highly scalable for production retraining pipelines.

---

## 8. Feature Importance: What Drives Severity?

Based on information gain from our top-performing LightGBM classifier:
1. **Latitude & Longitude (Spatial Coordinates):** Ranks as the #1 predictive feature, indicating that the geographical location (e.g. sharp bends, high-risk intersections, poor local road design) is the strongest predictor of accident severity.
2. **Speed Limit:** Roads with higher speed limits heavily correlate with higher information gain, reflecting that kinetic energy dissipation is a direct mathematical driver of fatal impacts.
3. **Accident Hour & Month:** Highlights temporal and seasonal risk factors (e.g., winter night hours represent peak hazard times).
4. **Number of Casualties & Vehicles involved:** Ranks highly, as multi-vehicle crashes significantly increase the likelihood of severe outcomes.

---

## 9. Actionable Policy & Production Deployment Recommendations

### 9.1 Infrastructure & Policy Interventions
* **Rural Speed Management:** Since Rural accidents are twice as likely to result in Serious or Fatal outcomes, transportation planners should install automated speed enforcement cameras on high-speed rural single-carriageway corridors.
* **Smart Street Lighting:** Implement high-intensity smart LED streetlights along high-risk spatial coordinates that exhibit low-light hazards.
* **Targeted Rush-Hour Patrols:** Deploy police force monitoring during peak school/commute hours (**08:00** and **17:00**) at the high-risk geographic coordinates identified by the spatial feature ranking.

### 9.2 Technical Production Architecture
To deploy this road accident prediction model in a live environment:
1. **FastAPI Web Service:** Package the modular `src/model_pipeline.py` preprocessing and LightGBM model into an active FastAPI service. This exposes a `/predict` REST API endpoint that accepts raw accident characteristics and returns real-time severity probabilities.
2. **Automated Retraining Pipeline:** Implement an orchestration pipeline (e.g. Apache Airflow) that retrieves new annual accident data, runs the modular data preprocessing and class-weighted LightGBM script, validates model recall, and updates the active model registry.
3. **Geospatial Dashboard:** Deploy a Streamlit or Dash interactive web application. This imports spatial coordinate predictions, highlighting active high-risk accident "hotspots" on an interactive map for civil engineers and city planners.
