# 🫀 Heart Disease Classification – ML Assignment 2

**BITS Pilani | Work Integrated Learning Programmes Division**  
**M.Tech (AIML/DSE) | Machine Learning | Assignment 2**

---

## 🔗 Submission Links

| Item | Link |
|------|------|
| 🐙 GitHub Repository | https://github.com/ramc96/ml-assignment-2.git |
| 🚀 Live Streamlit App | https://ml-assignment-2git-bsurgcceyfw4sabpu46ma5.streamlit.app/ |

---

## a. Problem Statement

Heart disease is one of the leading causes of death globally. Early and accurate detection can save lives. This project builds and compares multiple machine learning classification models to predict whether a patient is likely to have heart disease, based on clinical and demographic features.

The goal is to evaluate and compare the performance of six different classification algorithms on the same dataset using standard evaluation metrics, and deploy the best-performing pipeline as an interactive web application.

---

## b. Dataset Description

**Dataset:** Heart Failure Prediction Dataset  
**Source:** [Kaggle – Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**License:** Open Database License (ODbL)

| Property | Value |
|----------|-------|
| Total Instances | 918 |
| Total Features | 11 input + 1 target |
| Target Variable | `HeartDisease` (0 = No Disease, 1 = Disease) |
| Task Type | Binary Classification |
| Missing Values | None |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Age of patient in years |
| Sex | Categorical | M = Male, F = Female |
| ChestPainType | Categorical | TA, ATA, NAP, ASY |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mm/dl) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dl (1 = True) |
| RestingECG | Categorical | ECG results (Normal, ST, LVH) |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Slope of peak exercise ST segment |
| HeartDisease | Binary | **Target** – 0 = No Disease, 1 = Disease |

**Class Distribution:**
- No Heart Disease (0): 410 instances (44.7%)
- Heart Disease (1): 508 instances (55.3%)

---

## c. Models Used

### Preprocessing Pipeline
1. Categorical encoding using `LabelEncoder`
2. 80/20 Train-Test split with stratification
3. Feature scaling using `StandardScaler`
4. Random state: 42 for reproducibility

---

### Comparison Table – Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8533 | 0.9201 | 0.8661 | 0.8701 | 0.8681 | 0.7028 |
| Decision Tree | 0.8152 | 0.8129 | 0.8274 | 0.8268 | 0.8271 | 0.6271 |
| K-Nearest Neighbor | 0.8696 | 0.9277 | 0.8786 | 0.8819 | 0.8802 | 0.7358 |
| Naive Bayes | 0.8478 | 0.9157 | 0.8601 | 0.8661 | 0.8631 | 0.6920 |
| Random Forest (Ensemble) | 0.8859 | 0.9498 | 0.8961 | 0.8937 | 0.8949 | 0.7695 |
| XGBoost (Ensemble) | 0.9022 | 0.9587 | 0.9103 | 0.9094 | 0.9098 | 0.8027 |

> ✅ **Best overall model: XGBoost** — highest across all metrics.  
> 🟢 Values highlighted represent best scores in each column.

---

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Achieved a solid baseline accuracy of 85.3% with an impressive AUC of 0.92, indicating good class separability. The model benefits from the near-linear relationships between features like Age, MaxHR, and Oldpeak with the target. It is fast, interpretable, and a reliable baseline. Slightly limited by its linearity assumption. |
| Decision Tree | Performed the weakest among all models with 81.5% accuracy and AUC of 0.81, showing signs of overfitting. Decision trees tend to memorize training patterns and do not generalize as well to test data. Lacks the regularization needed for this dataset without pruning or depth constraints. |
| K-Nearest Neighbor | Performed well with 86.9% accuracy and 0.93 AUC, benefiting from the scaled features. KNN is a non-parametric model that works well when class boundaries are irregular. However, it may be sensitive to noisy features and is computationally expensive at inference time. |
| Naive Bayes | Achieved 84.8% accuracy with an AUC of 0.92. While it assumes feature independence (violated in medical data), the Gaussian NB still performed competitively. It's extremely fast and works surprisingly well as a probabilistic baseline despite the independence assumption being unrealistic. |
| Random Forest (Ensemble) | One of the top performers with 88.6% accuracy and 0.95 AUC. As a bagging ensemble, it reduces variance by averaging across many decision trees, resulting in robust and generalizable predictions. Feature importance analysis shows MaxHR, Oldpeak, and ST_Slope as the most predictive features. |
| XGBoost (Ensemble) | Delivered the best overall performance — 90.2% accuracy, 0.96 AUC, and 0.80 MCC. As a boosting ensemble, it iteratively corrects errors from previous trees. Its gradient-boosted approach handles non-linear relationships and feature interactions very effectively, making it ideal for tabular clinical data like this dataset. |

---

## 🚀 Streamlit App Features

The deployed Streamlit app includes:

- 📁 **CSV Upload** – Upload test data directly in the browser
- 🔽 **Model Dropdown** – Choose one model or compare all 6 simultaneously  
- 📊 **Evaluation Metrics Display** – Accuracy, AUC, Precision, Recall, F1, MCC  
- 🔲 **Confusion Matrix** – Heatmap for each model  
- 📋 **Classification Report** – Detailed per-class metrics  
- 🌟 **Feature Importance** – Available for tree-based models (RF, DT, XGBoost)

---




*Submitted for BITS Pilani M.Tech (AIML/DSE) | Machine Learning | Assignment 2 | Feb 2026*
