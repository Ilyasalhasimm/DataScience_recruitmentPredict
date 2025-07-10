import streamlit as st
import pandas as pd
import numpy as np
import joblib  # untuk load model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Load model yang sudah dilatih
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl") 

# Load data hanya untuk uji akurasi (jangan di-train ulang)
# df = pd.read_csv(r'dataset\recruitment_data.csv')
# X = df.drop('HiringDecision', axis=1)
# y = df['HiringDecision']

# Step 2: Split the data untuk evaluasi model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Streamlit UI
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

# Step 4: Predict
if st.button('Predict Hiring Decision'):
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'EducationLevel': education_level,
        'ExperienceYears': years_experience,
        'PreviousCompanies': previous_companies,
        'DistanceFromCompany': distance_from_company,
        'InterviewScore': interview_score,
        'SkillScore': skill_score,
        'PersonalityScore': personality_score,
        'RecruitmentStrategy': recruitment_strategy
    }])
    
    numerical_cols = ['Age', 'ExperienceYears', 'DistanceFromCompany',
                    'InterviewScore', 'SkillScore', 'PersonalityScore']
    
    
    input_scaled = scaler.transform(input_df[numerical_cols])
    
    other_features = input_df.drop(columns=numerical_cols).values
    final_input = np.hstack([input_scaled, other_features])
    
    prediction = model.predict(final_input)[0]
    result = "Diterima ✅" if prediction == 1 else "Tidak Diterima ❌"
    st.success(f"Hasil Prediksi: Kandidat {result}")
    
    
    # X_test_scaled = scaler.transform(X_test[numerical_cols])
    # other_test_features = X_test.drop(columns=numerical_cols).values
    # X_test_final = np.hstack([X_test_scaled, other_test_features])

    # y_pred = model.predict(X_test_final)
    # # Tampilkan akurasi model
    # accuracy = accuracy_score(y_test, y_pred)
    # st.write(f"Akurasi Model: {accuracy * 100:.2f}%")
    
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
    
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
    # DISINI SEMUA KANDIDAT HASIL PREDIKSI NYA DI TERIMA ?????????
