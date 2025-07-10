import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load the recruitment dataset
df = st.file_uploader("Upload file CSV", type="csv")
X = df.drop('HiringDecision', axis=1)
y = df['HiringDecision']

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the SVM model
svm = SVC()
svm.fit(X_train, y_train)

# Step 4: Streamlit UI
st.title("Hiring Decision Prediction App")
st.write("Masukkan data kandidat untuk memprediksi apakah akan diterima atau tidak.")

# Input fields
age = st.number_input("Age", min_value=20.0, max_value=50.0, value=25.0)
gender = st.selectbox("Gender", [0, 1], index=1)
education_level = st.selectbox("Education Level", [1, 2, 3, 4], index=2)
years_experience = st.number_input("Years of Experience", min_value=0.0, value=2.0)
previous_companies = st.number_input("Previous Companies", min_value=0.0, value=3.0)
distance_from_company = st.number_input("Distance From Company", min_value=0.0, max_value=50.0, value=20.0)
interview_score = st.number_input("Interview Score", min_value=0.0, max_value=100.0, value=75.0)
skill_score = st.number_input("Skill Score", min_value=0.0, max_value=100.0, value=75.0)
personality_score = st.number_input("Personality Score", min_value=0.0, max_value=100.0, value=75.0)
recruitment_strategy = st.selectbox("Recruitment Strategy", [1, 2, 3], index=2)

# Step 5: Predict
if st.button('Predict Hiring Decision'):
    input_data = np.array([
        age, gender, education_level, years_experience, previous_companies,
        distance_from_company, interview_score, skill_score,
        personality_score, recruitment_strategy
    ]).reshape(1, -1)

    # Predict
    prediction = svm.predict(input_data)[0]

    # Display result
    result = "Diterima ✅" if prediction == 1 else "Tidak Diterima ❌"
    st.success(f"Hasil Prediksi: Kandidat {result}")

    # Display accuracy
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi Model: {accuracy * 100:.2f}%")
    
    # DISINI PREDIKSINYA BENER AKURASI 72.33%
