import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.title("Hiring Decision Prediction App")
st.write("Upload dataset dan masukkan data kandidat untuk memprediksi apakah akan diterima atau tidak.")
# Tombol unduh dataset
dataset_url = "https://drive.google.com/file/d/10gBXi7-waVW-TDMt2GOVZNp6bWkCVFnH/view?usp=drive_link"  # Ganti link ini sesuai kebutuhan

st.markdown(
    f"""
    <a href="{dataset_url}" target="_blank">
        <button style="background-color: #4CAF50; color: white; padding: 8px 16px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">
            📥 Unduh Dataset di Sini
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

# Step 1: Upload & Load the recruitment dataset
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data yang diupload:")
    st.dataframe(df)

    try:
        # Step 2: Split features and label
        X = df.drop('HiringDecision', axis=1)
        y = df['HiringDecision']

        # Step 3: Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Train the SVM model
        svm = SVC()
        svm.fit(X_train, y_train)

        # Input fields
        st.subheader("Masukkan Data Kandidat:")
        age = st.number_input("Age", min_value=20.0, max_value=50.0, value=25.0)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        education_level = st.selectbox("Education Level", [1, 2, 3, 4])
        years_experience = st.number_input("Years of Experience", min_value=0.0, value=2.0)
        previous_companies = st.number_input("Previous Companies", min_value=0.0, value=3.0)
        distance_from_company = st.number_input("Distance From Company", min_value=0.0, max_value=50.0, value=20.0)
        interview_score = st.number_input("Interview Score", min_value=0.0, max_value=100.0, value=75.0)
        skill_score = st.number_input("Skill Score", min_value=0.0, max_value=100.0, value=75.0)
        personality_score = st.number_input("Personality Score", min_value=0.0, max_value=100.0, value=75.0)
        recruitment_strategy = st.selectbox("Recruitment Strategy", [1, 2, 3])

        # Step 5: Predict
        if st.button('Predict Hiring Decision'):
            input_data = np.array([[
                age, gender, education_level, years_experience, previous_companies,
                distance_from_company, interview_score, skill_score,
                personality_score, recruitment_strategy
            ]])

            prediction = svm.predict(input_data)[0]
            result = "Diterima ✅" if prediction == 1 else "Tidak Diterima ❌"
            st.success(f"Hasil Prediksi: Kandidat {result}")

            # Display accuracy
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.info(f"Akurasi Model: {accuracy * 100:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses dataset: {e}")
else:
    st.warning("Silakan upload file CSV terlebih dahulu.")
