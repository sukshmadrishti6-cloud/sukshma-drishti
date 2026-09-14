# Data Directory — SIH26139

This directory manages biomedical dataset artifacts for the **SIH26139 Hybrid QML Platform**.

---

## 1. Directory Structure

```text
data/
├── raw/         # Immutable raw dataset files (CSV, Parquet, JSON)
├── processed/   # Cleaned, imputed, scaled, and feature-engineered datasets
└── README.md
```

---

## 2. Dataset Specification: Pima Indians Diabetes Database

* **Dataset:** Pima Indians Diabetes Database
* **Rows:** 768
* **Predictors:** 8
* **Target:** Outcome
* **Target values:** 0, 1
* **Expected distribution:** 500 zeros / 268 ones
* **Raw filename:** pima_diabetes.csv
* **Raw path:** data/raw/pima_diabetes.csv
* **SHA-256:** b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac
* **Source:** https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv (Verified UCI Kaggle distribution copy)
* **Original provenance:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), as documented for the Pima Indians Diabetes Database.
* **Acquisition date:** 2026-09-02
* **Current Repository Status:** File is present in `data/raw/pima_diabetes.csv`.
* **Usage & Licensing**: Public research / prototype dataset. For scientific research demonstration only—does not claim direct clinical diagnostic authority.

### Column Schema

| Column Name | Type | Description | Valid Range | Invalid-Zero Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `Pregnancies` | `int` | Number of times pregnant | $\ge 0$ | Legitimate value |
| `Glucose` | `float` | Plasma glucose concentration (2h oral glucose tolerance test) | $> 0$ | `0` = Missing Data |
| `BloodPressure` | `float` | Diastolic blood pressure (mm Hg) | $> 0$ | `0` = Missing Data |
| `SkinThickness` | `float` | Triceps skin fold thickness (mm) | $> 0$ | `0` = Missing Data |
| `Insulin` | `float` | 2-Hour serum insulin ($\mu\text{U/ml}$) | $> 0$ | `0` = Missing Data |
| `BMI` | `float` | Body Mass Index ($\text{weight in kg}/(\text{height in m})^2$) | $> 0$ | `0` = Missing Data |
| `DiabetesPedigreeFunction` | `float` | Diabetes pedigree function (genetic risk score) | $> 0$ | Legitimate value |
| `Age` | `int` | Age in years | $\ge 21$ | Legitimate value |
| **`Outcome`** | `int` | **Target Label (0: No Diabetes, 1: Diabetes)** | `{0, 1}` | **Target Column** |

---

## 3. Data Pipeline & Leakage Prevention Rules

1. **Immutability**: Raw files in `data/raw/` must remain untouched.
2. **Strict Train/Test Split First**: Data is split into 80% train and 20% test folds (`random_seed=42`, stratified on `Outcome`) **BEFORE** fitting any preprocessing transformations.
3. **Leakage-Safe Imputation & Scaling**:
   - Invalid zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are converted to `NaN`.
   - Median imputation values and `StandardScaler` mean/std parameters are computed **STRICTLY on `X_train`**.
   - `X_test` is transformed using `X_train` parameters without viewing test fold statistics.
4. **Target Exclusion**: `Outcome` is strictly isolated as `y` and is never included in feature matrix `X`.
