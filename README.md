# 🏥 Health Care Premium Prediction

A machine learning web application that predicts health insurance premiums based on personal, lifestyle, and medical factors. Built with **Streamlit** and powered by regression models trained on segmented customer data.

---

## 📌 Overview

Health insurance companies use complex risk models to estimate premium costs. This project replicates that process using real-world-style features — segmenting users by age group and applying tailored ML models to predict annual insurance costs accurately.

---

## 🚀 Features

- Predicts health insurance cost based on 12 user inputs
- Age-based model segmentation (Young vs. Rest) for improved accuracy
- Interactive web UI built with Streamlit
- Supports multiple medical history, lifestyle, and demographic inputs
- Trained with XGBoost and scikit-learn pipelines

---
---

## 🧠 Input Features

| Feature              | Type        | Options / Range                                                                 |
|----------------------|-------------|---------------------------------------------------------------------------------|
| Age                  | Numerical   | 18 – 100                                                                        |
| Number of Dependants | Numerical   | 0 – 20                                                                          |
| Income (in Lakhs)    | Numerical   | 0 – 200                                                                         |
| Genetical Risk       | Numerical   | 0 – 5                                                                           |
| Gender               | Categorical | Male, Female                                                                    |
| Marital Status       | Categorical | Married, Unmarried                                                              |
| BMI Category         | Categorical | Normal, Obesity, Overweight, Underweight                                        |
| Smoking Status       | Categorical | No Smoking, Occasional, Regular                                                 |
| Employment Status    | Categorical | Salaried, Self-Employed, Freelancer                                             |
| Region               | Categorical | Northwest, Southeast, Northeast, Southwest                                      |
| Medical History      | Categorical | No Disease, Diabetes, High blood pressure, Heart disease, Thyroid, combinations |
| Insurance Plan       | Categorical | Bronze, Silver, Gold                                                            |

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit
- **ML Models:** XGBoost, scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Language:** Python 3.x

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/RawatXd/Health-Care-Premium-prediction.git
cd Health-Care-Premium-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Model Approach

The prediction pipeline uses **age-based segmentation** to improve model performance:

- **Young model** — Applied to users in a younger age bracket
- **Rest model** — Applied to older users

Each segment has a dedicated trained model (with and without genetic risk factor), stored as serialized artifacts in the `artifacts/` folder.

---

## 📁 Dependencies
joblib==1.3.2
pandas==2.0.2
streamlit==1.22.0
numpy==1.25.0
scikit-learn==1.3.0
xgboost==2.0.3
