import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("lifespan_prediction_model.pkl")

# Streamlit app layout
def main():
    st.title("Lifespan Prediction App")
    st.write("Enter the details below to predict lifespan.")

    # User inputs
    age = st.number_input("Age", min_value=20, max_value=100, value=30)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    smoking = st.selectbox("Smoking", options=["No", "Yes"])
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

    # Encode categorical inputs
    sex_encoded = 1 if sex == "Male" else 0
    smoking_encoded = 1 if smoking == "Yes" else 0

    # Prediction button
    if st.button("Predict Lifespan"):
        features = np.array([[age, sex_encoded, smoking_encoded, height, weight]])
        prediction = model.predict(features)
        st.write(f"### Predicted Lifespan: {prediction[0]:.2f} years")

# Run the app
if __name__ == "__main__":
    main()