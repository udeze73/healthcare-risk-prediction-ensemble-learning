<img width="3780" height="1446" alt="08_github_banner" src="https://github.com/user-attachments/assets/6ff28b56-b3b4-4cc0-a141-301aebea151a" />

# ❤️ Heart Attack Risk Prediction Using Stacked Ensemble Machine Learning

## Project Overview

Heart disease remains one of the leading causes of mortality worldwide. Early identification of high-risk individuals can significantly improve preventive interventions and clinical outcomes.

This project develops an end-to-end machine learning framework for predicting heart attack risk using healthcare survey data. The solution goes beyond traditional classification by implementing a stacked ensemble architecture that combines multiple machine learning algorithms to improve predictive performance, robustness, and generalization.

The project addresses several real-world machine learning challenges including:

- Severe class imbalance
- Feature engineering
- Skewed data distributions
- Multicollinearity
- Ensemble learning
- Model stacking
- Threshold optimization
- Computational efficiency

---

## Problem Statement

The target variable (`HadHeartAttack`) is highly imbalanced:

| Class | Distribution |
|---------|---------|
| No Heart Attack | ~94.5% |
| Heart Attack | ~5.5% |

In datasets like this, standard machine learning models often achieve high accuracy while performing poorly on the minority class.

The primary objective was therefore not to maximize accuracy, but to improve detection of heart attack cases while maintaining a practical balance between precision and recall.

---

## Dataset

### Source

Behavioral Risk Factor Surveillance System (BRFSS) 2022 Heart Disease Dataset

### Target Variable

`HadHeartAttack`

Binary Classification:

- 0 → No Heart Attack
- 1 → Heart Attack

### Dataset Characteristics

- 246,013 observations
- 17:1 class imbalance ratio
- Healthcare and lifestyle-related predictors
- Mixed numerical and categorical features

---

## Exploratory Data Analysis (EDA)

A comprehensive exploratory analysis was conducted to understand:

- Feature distributions
- Missing values
- Class imbalance
- Outliers
- Correlation structure
- Predictor relationships

### Key Findings

- Severe class imbalance (~17:1)
- Multiple highly skewed healthcare variables
- Strong multicollinearity between Weight and BMI
- Several ordinal variables requiring domain-driven encoding

---

## Data Cleaning & Preprocessing

### Outlier Treatment

Outliers were treated using the Interquartile Range (IQR) capping method.

Benefits:

- Reduces influence of extreme observations
- Preserves dataset size
- Improves model stability

---

### Skewness Treatment

Log transformations were applied to highly skewed variables.

Examples:

- PhysicalHealthDays
- MentalHealthDays
- RemovedTeeth

Benefits:

- Produces more normalized distributions
- Improves linear model learning
- Reduces influence of extreme values

---

## Feature Engineering

A major focus of this project was transforming raw healthcare attributes into meaningful numerical representations.

### AgeCategory Mapping

Age ranges were converted into representative numerical ages.

| Original Category | Encoded Value |
|------------------|--------------|
| Age 25–29 | 27 |
| Age 30–34 | 32 |
| Age 50–54 | 52 |
| Age 80+ | 85 |

This preserved the ordinal relationship while allowing models to learn age-related risk patterns.

---

### GeneralHealth Encoding

Natural ordering was preserved through ordinal encoding:

```text
Poor < Fair < Good < Very Good < Excellent
```

---

### RemovedTeeth Mapping

Dental health categories were converted into approximate tooth counts.

Examples:

```text
None Removed → 0
1–5 Removed → 3
6+ Removed → 8
All Removed → 32
```

This transformation captures severity more effectively than standard categorical encoding.

---

### Weight Feature Engineering

Weight values were grouped into clinically meaningful categories:

- Underweight
- Normal Weight
- Overweight
- Obese

This improved interpretability and reduced noise.

---

### SleepHours Standardization

Extreme sleep-hour values were capped to realistic ranges to improve consistency and model reliability.

---

## Feature Selection

Correlation analysis identified strong multicollinearity among:

- WeightInKilograms
- BMI
- WeightCategory

These relationships were carefully evaluated during experimentation to reduce redundancy while preserving predictive power.

---

## Handling Class Imbalance

Class imbalance was one of the most significant challenges in this project.

Several techniques were investigated:

### SMOTE Oversampling

SMOTE produced limited improvement while significantly increasing runtime hence other techniques were explored.

### Class Weight Balancing

Models were trained using:

```python
class_weight='balanced'
```

This forces the algorithm to pay greater attention to minority-class observations.

### XGBoost Class Weighting

For XGBoost:

```python
scale_pos_weight
```

was used to compensate for class imbalance.

### Threshold Optimization

Rather than relying on the default probability threshold of 0.50, threshold tuning was performed to identify decision boundaries that improved minority-class performance.

### Key Finding

Class weighting and threshold optimization produced more practical gains than aggressive oversampling while significantly reducing computational cost.

---

## Modeling Strategy

Five diverse machine learning models were trained as base learners.

### Logistic Regression

**Strengths**

- Strong baseline model
- Interpretable predictions
- Excellent probability calibration

---

### K-Nearest Neighbors (KNN)

**Configuration**

```python
KNeighborsClassifier(
    n_neighbors=15,
    weights='distance'
)
```

**Strengths**

- Non-parametric learning
- Captures local neighborhood patterns
- Introduces diversity into the ensemble

---

### Extra Trees

**Strengths**

- Handles nonlinear relationships
- Robust to noise
- Strong ensemble baseline

---

### XGBoost

**Strengths**

- Gradient boosting framework
- Excellent performance on structured data
- Captures complex feature interactions

---

### LightGBM

**Strengths**

- Fast training
- Efficient boosting algorithm
- Scales well to large datasets

---

## Cross Validation

Stratified K-Fold Cross Validation was used throughout model development.

Benefits:

- Preserves class distribution across folds
- Produces reliable performance estimates
- Improves generalization assessment

---

# Stacked Ensemble Architecture

The core contribution of this project is the implementation of a stacked ensemble learning framework.

## Level 1 — Base Models

The following models generated out-of-fold probability predictions:

- Logistic Regression
- KNN
- Extra Trees
- XGBoost
- LightGBM

Example generated meta-features:

```text
logistic_prob
knn_prob
etratrees_prob
xgboost_prob
```

---

## Level 2 — Meta Model

An Extra Trees Classifier was trained using the probability outputs from the base learners.

Benefits:

- Learns strengths of individual models
- Reduces weaknesses of individual learners
- Improves robustness and generalization

---

## Stacking Workflow

```text
Raw Data
    │
    ▼
Feature Engineering
    │
    ▼
Base Models
(Logistic, KNN, RF, XGBoost, LightGBM)
    │
    ▼
Out-of-Fold Probabilities
    │
    ▼
Meta Features
    │
    ▼
Logistic Regression Model
    │
    ▼
Final Prediction
```

---

## Threshold Optimization

Most classification projects use a default threshold of:

```python
0.50
```

This project instead searched multiple thresholds to maximize F1 Score.

Benefits:

- Better minority-class detection
- Improved precision-recall tradeoff
- More practical decision boundaries

---

## Evaluation Metrics

Because of severe class imbalance, accuracy was not used as the primary evaluation metric.

Primary metrics:

- F1 Score
- Precision
- Recall
- ROC-AUC
- Balanced Accuracy

These metrics provide a more realistic assessment of model effectiveness.

---

## Final Results

### Stacked Ensemble Performance

| Metric | Score |
|----------|---------|
| ROC-AUC | 0.89 |
| Balanced Accuracy | 0.73 |
| Precision | 0.45 |
| Recall | 0.51 |
| F1 Score | 0.48 |

---

## Performance Interpretation

Despite a severe imbalance ratio of approximately 17:1, the model successfully learned meaningful patterns associated with heart attack risk.

### Highlights

✅ Strong discriminatory power (ROC-AUC = 0.89)

✅ Effective minority-class detection

✅ Improved precision-recall balance through threshold optimization

✅ Robust performance across multiple validation folds

✅ Ensemble architecture outperformed individual models

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- Matplotlib
- Seaborn

---

## Repository Structure

```text
heart-attack-risk-prediction-stacked-ensemble/
│
├── data/
├── notebooks/
│   └── heart_attack_prediction.ipynb
│
├── images/
│   ├── project_workflow.png
│   └── stacking_architecture.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│   ├── precision_curve.png
│   └── model_comparison.png
│   ├── feature_importance.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Key Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Handling Imbalanced Data
- Ensemble Learning
- Model Stacking
- Threshold Optimization
- Cross Validation
- Healthcare Analytics
- Machine Learning Pipeline Design
- Predictive Modeling

---

## Future Improvements

Potential enhancements include:

- Hyperparameter optimization using Optuna
- Calibrated probability estimation
- Deployment using FastAPI or Streamlit

---

## Author

### Joshua Udeze

**MBA (Finance & Investment) | Data Scientist | Machine Learning Practitioner**

Passionate about applying machine learning, predictive analytics, and business intelligence to solve real-world problems across healthcare, finance, risk management, and business operations.

### Connect With Me

- LinkedIn: www.linkedin.com/in/joshua-udeze-394531154
- UDEZEJOSHUA@GMAIL.COM

---
⭐ If you found this project useful, consider starring the repository.
