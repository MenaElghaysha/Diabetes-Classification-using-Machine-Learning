# 🩺 Diabetes Classification using Machine Learning

This project focuses on building a **binary classification system** to predict whether a patient has diabetes or not using **machine learning models**, while properly handling **imbalanced data** and deploying the final model using **Streamlit**.

---

## 📌 Project Overview

- **Problem Type:** Binary Classification  
- **Target Variable:** Diabetes (0 = No, 1 = Yes)  
- **Main Challenges:**
  - Imbalanced dataset
  - Model selection & comparison
  - Proper preprocessing
  - Deployment

---

## 🧠 Machine Learning Pipeline

The project follows a clean and modular ML pipeline:

### 1️⃣ Data Preprocessing
- Data cleaning using `FunctionTransformer`
- Handling missing values using `SimpleImputer`
- Feature scaling with `RobustScaler`
- Feature transformation using `PowerTransformer`

### 2️⃣ Handling Imbalanced Data
To solve class imbalance, **SMOTE-based techniques** were applied:
- `SMOTE`
- `SMOTETomek`

These techniques were applied **only on the training data** to avoid data leakage.

---

## 🤖 Models Used

The following models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes (Gaussian)
- XGBoost

Each model was evaluated using multiple metrics.

---

## 📊 Evaluation Metrics

Because the dataset is imbalanced, accuracy alone is not enough.  
The following metrics were used:

- Accuracy (Train & Test)
- Precision
- Recall
- F1-score
- ROC-AUC

The **F1-score** was the primary metric for model selection.

---

## 🔍 Hyperparameter Tuning

- **Technique:** GridSearchCV  
- **Cross Validation:** 5-Fold  
- **Scoring Metric:** F1-score  

Random Forest hyperparameters tuned include:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

---

## 🏆 Best Model

- **Model:** Random Forest Classifier  
- **Why?**
  - Best balance between precision & recall
  - High F1-score
  - Stable performance after SMOTE

---

## 💾 Model Persistence

The trained artifacts were saved using `pickle`:

- `rf_model.pkl` → trained model
- `preprocessor.pkl` → preprocessing pipeline

This allows:
- Reusing the model without retraining
- Easy deployment

---

## 🌐 Streamlit Deployment

A simple and interactive **Streamlit web app** was created where users can:
- Input patient medical data
- Get real-time diabetes prediction
- View prediction confidence

### ▶️ Run the App Locally

```bash
streamlit run app.py
