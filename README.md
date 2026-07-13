# Smoker Status Prediction from Health Insurance Records

A comprehensive Machine Learning study developed as the final project for the **CCDS 322 - Applied Machine Learning** course at the **University of Jeddah**.

This repository implements an end-to-end Machine Learning pipeline to predict an individual's smoking status ("Yes" or "No") based on demographic, lifestyle, and health insurance billing data. The project evaluates and compares the performance, scalability, and interpretability of three distinct algorithm families: **Logistic Regression**, **Support Vector Machines (SVM)**, and **Random Forest**.

---

## 👥 Team Members
* **Rimas Almuntashiri** 
* **Ghadof Aljabri**
* **Lama Alqarashi**
* **Mayar Mousa**
* **Taraf Abdullah**
 
**Date:** May 2025

---

## 📌 Project Overview & Objectives
Smoking status is one of the most critical risk indicators in healthcare analytics, directly correlating with long-term health anomalies and highly inflated insurance premiums. When self-reported smoking data is missing or unreliable, machine learning can recover this status using secondary features already gathered by insurers.

### Core Objectives:
1. **Binary Classification:** Build a robust classifier to identify smokers vs. non-smokers.
2. **Pipeline Architecture:** Implement a clean data preparation workflow to handle missing data, encoding, and feature scaling without data leakage.
3. **Algorithm Comparison:** Benchmark traditional linear baselines against non-linear margin-based models and ensemble methods.
4. **Feature Importance:** Isolate the driving health indicators that determine smoking status.

---

## 🛠️ Technical Pipeline & Methodology

The project follows a 5-step machine learning workflow:

```text
[Raw CSV] ➔ [1. Data Cleaning] ➔ [2. Categorical Encoding] ➔ [3. Stratified Splitting] ➔ [4. Feature Scaling] ➔ [5. Model Training & Evaluation]
```

---

### 1. Data Cleaning & Imputation
* **Missing Data:** Handled **501,166 missing values** (~25% missing data in `medical_history` and `family_medical_history` columns).  
* **Imputation Strategy:** Numerical missing columns were imputed using the **Median**, while categorical columns were filled using the **Mode**.  
* **Duplicates:** Duplicate rows were checked and verified as zero.  

### 2. Categorical Encoding
* **Target Variable:** The target variable (`smoker`) was binary encoded (`yes` = 1, `no` = 0).  
* **Features Encoding:** The remaining categorical features (`gender`, `region`, `medical_history`, `family_medical_history`, `exercise_frequency`, `occupation`, and `coverage_level`) were transformed via **Label Encoding** to make them fully compatible with the machine learning algorithms.  

### 3. Data Splitting
* **Strategy:** Partitioned the dataset into train and test sets using a **stratified split (80% training / 20% testing)** to preserve the perfect 50/50 balance of the target variable across partitions.  

### 4. Feature Scaling
* **Strategy:** Applied `StandardScaler` to normalize numerical features, fitting solely on the training partition and transforming the test partition to strictly avoid any data leakage.  

---

## 📊 Model Performance & Comparative Summary
We trained and cross-evaluated 5 variants across 3 algorithm families:  

### 1. Logistic Regression (Linear Baseline)
* **Setup:** Trained as an interpretable linear baseline using the top two correlated features (`bmi` and `charges`).  
* **Performance:** **74.48% Accuracy** | **74.75% Precision** | **73.97% Recall**.  
* **Verdict:** Provides a highly interpretative boundary but struggles to capture complex, non-linear dependencies between health spending and smoking habits.  

### 2. Support Vector Machine (SVM) — 3 Kernels Evaluated
*Due to the $O(n^2)$ memory scaling complexity of kernel SVMs, training was performed on a stratified 15,000-row sample to keep execution tractable.*  
* **Polynomial Kernel ($C=5, \text{degree}=3$):** **81.17% Accuracy** ➔ *Best Raw Accuracy Model*.  
* **Linear Kernel ($C=5$):** **80.77% Accuracy**.  
* **RBF Kernel ($C=10$):** **80.24% Accuracy**.  

### 3. Random Forest (Ensemble Learning)
* **Setup:** Ensemble of decision trees trained on the full 1M-row dataset, `criterion='entropy'`, `max_depth=20`.  
* **Performance:** **80.79% Accuracy** | **80.51% Precision** | **81.26% Recall**.  
* **Verdict:** Named the **Best Overall Practical Model**. It matches the best SVM accuracy while being infinitely more scalable and computationally efficient on the large 1,000,000-row dataset.

---

## 🔍 Key Findings & Feature Importances

### 1. Driving Predictors:
The Gini/Entropy feature importance from the Random Forest model clearly isolates **Insurance Charges**, **BMI**, and **Age** as the dominant drivers determining an individual's smoking classification.

### 2. The Accuracy-Scale Tradeoff:
While the Polynomial SVM yielded a marginally higher raw accuracy (81.17%), training an SVM on 1,000,000 rows is incredibly resource-intensive. **Random Forest wins overall** because it scales excellently, natively handles tabular categorical data, and remains robust against outliers without strict feature scaling dependencies.

---

## 🖥️ Project Repository Files
* `ML3_Final_Project.ipynb`: The complete executable Jupyter Notebook containing data exploration, visualization plots, and training codes.  
* `Final Report_CCDS223.pdf`: The official detailed academic report.  
