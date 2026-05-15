import streamlit as st

from src.pipeline.training_pipeline import CustomData, PredictPipeline


st.title("Student Performance Prediction")


Student_Age = st.number_input("Student Age")
Sex = st.selectbox("Sex", ["Male", "Female"])
Graduated_High_School_Type = st.selectbox("High School Type", ["Public", "Private"])
Scholarship_Type = st.selectbox("Scholarship", ["Yes", "No"])
Additional_Work = st.selectbox("Additional Work", ["Yes", "No"])
Sports_Activity = st.selectbox("Sports Activity", ["Yes", "No"])
Transportation = st.selectbox("Transportation", ["Bus", "Car", "Walk"])
Weekly_Study_Hours = st.number_input("Weekly Study Hours")
Attendance = st.number_input("Attendance")
Reading = st.number_input("Reading Score")
Notes = st.number_input("Notes Score")
Listening_in_Class = st.number_input("Listening Score")
Project_Work = st.selectbox("Project Work", ["Yes", "No"])
if st.button("Predict"):

    data = CustomData(
        Student_Age,
        Sex,
        Graduated_High_School_Type,
        Scholarship_Type,
        Additional_Work,
        Sports_Activity,
        Transportation,
        Weekly_Study_Hours,
        Attendance,
        Reading,
        Notes,
        Listening_in_Class,
        Project_Work
    )

    pred_df = data.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()

    result = predict_pipeline.predict(pred_df)

    st.write("Predicted Grade:", result[0])