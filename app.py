import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Load data and model
st.set_page_config(page_title="Mohs Hardness Prediction", layout="centered")

df = pd.read_csv("train.csv")
model = joblib.load("best_regression_model.pkl")

# Feature list
FEATURES = ['allelectrons_Total', 'density_Total', 'allelectrons_Average',
            'val_e_Average', 'atomicweight_Average', 'ionenergy_Average',
            'el_neg_chi_Average', 'R_vdw_element_Average', 'R_cov_element_Average',
            'zaratio_Average', 'density_Average']

# Create pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES)
])
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
pipeline.fit(df[FEATURES], df["Hardness"])

def hardness_prediction(input_data):
    prediction = pipeline.predict(pd.DataFrame([input_data]))[0]
    return float(prediction)

# Application title
st.title("🔍 Mohs Hardness Prediction")
st.markdown("Predict the Mohs hardness of a material based on its properties.")

# User input form
with st.form("prediction_form"):
    st.subheader("📊 Model Inputs")
    col1, col2 = st.columns(2)
    
    inputs = {}
    input_params = [
        ("allelectrons_Total", 0, 20000, 100),
        ("density_Total", 0, 10000, 50),
        ("allelectrons_Average", 0, 100, 1),
        ("val_e_Average", 0.0, 10.0, 0.1),
        ("atomicweight_Average", 0, 200, 1),
        ("ionenergy_Average", 0, 100, 1),
        ("el_neg_chi_Average", 0.0, 10.0, 0.1),
        ("R_vdw_element_Average", 0.0, 5.0, 0.01),
        ("R_cov_element_Average", 0.0, 5.0, 0.01),
        ("zaratio_Average", 0.0, 1.0, 0.01),
        ("density_Average", 0, 10, 1)
    ]
    
    for i, (feature, min_v, max_v, step_v) in enumerate(input_params):
        col = col1 if i % 2 == 0 else col2  # Arrange inputs in two columns
        inputs[feature] = col.number_input(feature, min_value=min_v, max_value=max_v, step=step_v)
    
    submitted = st.form_submit_button("🚀 Predict")
    
# Show prediction result
if submitted:
    prediction = hardness_prediction(inputs)
    st.success(f"**Predicted Mohs Hardness: {prediction:.2f}**")