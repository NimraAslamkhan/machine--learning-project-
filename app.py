import streamlit as st
import pandas as pd
import groq

from api_key import GROQ_API_KEY
from src.pipeline.prediction_pipeline import PredictPipeline


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI SaaS Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ===============================
# UI STYLE
# ===============================
st.markdown("""
<style>
/* IMPORT A PLAYFUL YET CLEAN FONT */
@import url('https://googleapis.com');

/* MAIN BACKGROUND (BRIGHT & AIRY) */
.block-container {
    padding: 2.5rem;
    background-color: #FFFFFF;
    color: #1E293B;
}

/* TEXT - ROUNDED & FRIENDLY */
html, body, [class*="css"]  {
    color: #334155 !important;
    font-family: 'Quicksand', sans-serif;
}

/* HEADINGS (PLAYFUL BLUE) */
h1, h2, h3 {
    color: #4F46E5 !important; /* Vibrant Indigo */
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* SIDEBAR (SOFT PASTEL BLUE) */
[data-testid="stSidebar"] {
    background-color: #F0F9FF; 
    border-right: 2px solid #E0F2FE;
}

/* INPUT BOXES (EXTRA ROUNDED) */
input, textarea, .stSelectbox, .stSlider {
    border-radius: 12px !important;
    border: 2px solid #E2E8F0 !important;
}

/* CARD STYLE (SOFT FLOATING LOOK) */
.dashboard-card {
    background: #FFFFFF;
    border: none;
    padding: 25px;
    border-radius: 24px; /* Super rounded corners */
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.1); /* Subtle blue glow */
}

/* METRICS (BRIGHT & COLORFUL) */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
    border: none;
    padding: 20px;
    border-radius: 20px;
    transition: transform 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-5px); /* Bouncy effect */
}

/* METRIC LABEL */
div[data-testid="metric-container"] label {
    color: #6366F1 !important;
    font-weight: 600;
}

/* BUTTON (FUN GRADIENT & BOUNCY) */
.stButton>button {
    width: 100%;
    border-radius: 15px;
    height: 3.5em;
    background: linear-gradient(135deg, #6366F1, #A855F7); /* Indigo to Purple */
    color: white;
    font-size: 18px;
    font-weight: 700;
    border: none;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #4F46E5, #9333EA);
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
}

/* SUCCESS BOX (BUBBLY GREEN) */
.stSuccess {
    background-color: #DCFCE7 !important;
    color: #15803D !important;
    border-radius: 15px;
    border: 1px solid #BBF7D0;
}

</style>
""", unsafe_allow_html=True)
# ===============================
# HEADER
# ===============================
st.title("🎓 AI SaaS Student Performance Dashboard")
st.write("Predict + Analyze + Chat with AI Assistant")

# ===============================
# SIDEBAR CHATBOT
# ===============================
st.sidebar.title("🤖 AI Chat Assistant")

api_key = GROQ_API_KEY or st.secrets.get("GROQ_API_KEY", "")
if not api_key:
   api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
user_question = st.sidebar.text_area("Ask anything about studies or AI")

if st.sidebar.button("Ask AI"):

    if not api_key:
        st.sidebar.warning("Please enter API key")
    elif not user_question.strip():
        st.sidebar.warning("Please enter a question")

    else:
        try:
            client = groq.Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an AI education assistant."},
                    {"role": "user", "content": user_question}
                ]
            )

            st.sidebar.success(response.choices[0].message.content)

        except Exception as e:
            st.sidebar.error(str(e))

# ===============================
# INPUT FORM
# ===============================
st.markdown("## 📊 Student Input Panel")

col1, col2 = st.columns(2)

with col1:
    Student_Age = st.selectbox("Age", ["18-21", "22-25", "26+"])
    Sex = st.selectbox("Gender", ["Male", "Female"])
    High_School_Type = st.selectbox("School Type", ["Private", "Public", "Other"])
    Scholarship = st.selectbox("Scholarship", ["Yes", "No"])
    Weekly_Study_Hours = st.slider("Study Hours", 1, 50, 10)

with col2:
    Attendance = st.selectbox("Attendance", ["Always", "Sometimes", "Never"])
    Reading = st.selectbox("Reading", ["Always", "Sometimes", "Never"])
    Notes = st.selectbox("Notes", ["Yes", "No"])
    Listening_in_Class = st.selectbox("Listening", ["Yes", "No"])
    Project_work = st.selectbox("Project Work", ["Yes", "No"])

# ===============================
# GRADE FUNCTION
# ===============================
def numeric_to_grade(score):
    if score >= 4.75: return "AA"
    if score >= 4.25: return "AB"
    if score >= 3.75: return "BB"
    if score >= 3.25: return "BC"
    if score >= 2.75: return "CC"
    if score >= 2.25: return "CD"
    if score >= 1.75: return "DD"
    if score >= 1.25: return "DF"
    return "FF"

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 Predict Performance"):

    data = pd.DataFrame({
        "Student_Age": [Student_Age],
        "Sex": [Sex],
        "High_School_Type": [High_School_Type],
        "Scholarship": [Scholarship],
        "Additional_Work": ["No"],
        "Sports_activity": ["No"],
        "Transportation": ["Bus"],
        "Weekly_Study_Hours": [Weekly_Study_Hours],
        "Attendance": [Attendance],
        "Reading": [Reading],
        "Notes": [Notes],
        "Listening_in_Class": [Listening_in_Class],
        "Project_work": [Project_work],
    })

    pipeline = PredictPipeline()
    result = pipeline.predict(data)

    score = float(result[0])
    grade = numeric_to_grade(score)

    # ===============================
    # DASHBOARD OUTPUT (SAAS STYLE)
    # ===============================
    st.markdown("## 📈 Dashboard Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("🎓 Grade", grade)
    col2.metric("📊 Score", f"{score:.2f}")
    col3.metric(
        "Status",
        "Excellent" if score > 4 else "Good" if score > 3 else "Needs Improvement"
    )

    st.markdown("---")

    st.success("Prediction completed successfully 🚀")