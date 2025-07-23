# Employee_salary_prediction
A machine learning project to predict employee salaries and deploy it via Streamlit

This project predicts employee salaries using machine learning based on key features like **Age**, **Gender**, **Education Level**, **Job Title**, and **Years of Experience**. It includes a full ML pipeline — from data preprocessing and model training to deployment via **Streamlit**.

---

## 🚀 Features

- Cleans and preprocesses employee salary data
- Encodes categorical columns using `LabelEncoder`
- Removes salary outliers using the IQR method
- Trains 3 regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Selects and saves the best model (based on R² Score)
- Deploys a Streamlit web app for real-time prediction

---

## ⚙️ Requirements

You need **Python 3.7+** and the following libraries:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib streamlit
```

How to Run the Project
Step 1:

```bash
python Salary.py
```

This will:

Preprocess the data
Train and compare models
Save the best model as best_modelofReg.joblib
Save LabelEncoders and feature column order
Make sure Salary_Data.csv is in the same folder as Salary.py.

Step 2: Launch the Streamlit Web App
```bash
streamlit run app.py
```
Then browser opens & From there, you can input employee details and see predicted salary in real-time.

Notes:
Ensure all .joblib files are present before running app.py
This app can also be deployed online via Streamlit Cloud
