import streamlit as st
import pandas as pd

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from api_key import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = ""

from src.pipeline.prediction_pipeline import PredictPipeline


def numeric_score_to_grade(score):
    if score >= 4.75:
        return "AA"
    if score >= 4.25:
        return "AB"
    if score >= 3.75:
        return "BB"
    if score >= 3.25:
        return "BC"
    if score >= 2.75:
        return "CC"
    if score >= 2.25:
        return "CD"
    if score >= 1.75:
        return "DD"
    if score >= 1.25:
        return "DF"
    return "FF"


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)


# =====================================
# CUSTOM CSS
# =====================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }.stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1E1E1E;
        margin-top: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# =====================================
# TITLE
# =====================================
st.title("🎓 AI Student Performance Prediction System")

st.write(
    "Predict student academic performance using Machine Learning + AI Assistant"
)


# =====================================
# SIDEBAR CHATBOT
# =====================================
st.sidebar.title("🤖 AI Study Assistant")
api_key = GROQ_API_KEY

if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

if not api_key:
    api_key = st.sidebar.text_input("Groq API key", type="password")

user_question = st.sidebar.text_area(
    "Ask AI anything about studies, grades, learning, AI, or education"
)


if st.sidebar.button("Ask AI"):
    if Groq is None:
        st.sidebar.error("Groq package is not installed. Run: pip install groq")
    elif not api_key:
        st.sidebar.warning("Please enter your Groq API key.")
    elif not user_question.strip():
        st.sidebar.warning("Please enter a question.")
    else:

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful educational AI assistant."
                },
                {
                    "role": "user",
                    "content": user_question
                }
            ]
        )

        answer = response.choices[0].message.content

        st.sidebar.success(answer)
# =====================================
# INPUT FORM
# =====================================
col1, col2 = st.columns(2)

with col1:

    Student_Age = st.selectbox(
        "Student Age",
        [
            "18-21",
            "22-25",
            "above 26"
        ]
    )

    Sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    High_School_Type = st.selectbox(
        "High School Type",
        [
            "Private",
            "Public",
            "Other"
        ]
    )

    Scholarship = st.selectbox(
        "Scholarship",
        [
            "Yes",
            "No"
        ]
    )

    Additional_Work = st.selectbox(
        "Additional Work",
        ["Yes", "No"]
    )

    Sports_activity = st.selectbox(
        "Sports Activity",
        ["Yes", "No"]
    )


with col2:

    Transportation = st.selectbox(
        "Transportation",
        [
            "Bus",
            "Private", "Walk"
        ]
    )

    Weekly_Study_Hours = st.slider(
        "Weekly Study Hours",
        1,
        40,
        10
    )

    Attendance = st.selectbox(
        "Attendance",
        [
            "Always",
            "Sometimes",
            "Never"
        ]
    )

    Reading = st.selectbox(
        "Reading Habit",
        [
            "Always",
            "Sometimes",
            "Never"
        ]
    )
    Notes = st.selectbox(
        "Taking Notes",
        [
            "Yes",
            "No"
        ]
    )

    Listening_in_Class = st.selectbox(
        "Listening in Class",
        [
            "Yes",
            "No"
        ]
    )

    Project_work = st.selectbox(
        "Project Work",
        [
            "Yes",
            "No"
        ]
    )
# =====================================
# PREDICTION BUTTON
# =====================================
if st.button("Predict Student Performance"):

    try:

        data = pd.DataFrame({
            "Student_Age": [Student_Age],
            "Sex": [Sex],
            "High_School_Type": [High_School_Type],
            "Scholarship": [Scholarship],
            "Additional_Work": [Additional_Work],
            "Sports_activity": [Sports_activity],
            "Transportation": [Transportation],
            "Weekly_Study_Hours": [Weekly_Study_Hours],
            "Attendance": [Attendance],
            "Reading": [Reading],
            "Notes": [Notes],
            "Listening_in_Class": [Listening_in_Class],
            "Project_work": [Project_work],
        })

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(data)

        numeric_score = float(result.ravel()[0])
        predicted_grade = numeric_score_to_grade(numeric_score)

        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>📊 Prediction Result</h2>
                <h1>Grade: {predicted_grade}</h1>
                <h3>Numeric Score: {numeric_score:.2f}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")
