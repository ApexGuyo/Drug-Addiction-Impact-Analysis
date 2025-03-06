import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

# Load trained models
with open("xgb_severity.pkl", "rb") as file:
    severity_model = pickle.load(file)

with open("xgb_relapse.pkl", "rb") as file:
    relapse_model = pickle.load(file)

# Load encoders
with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

with open("severity_encoder.pkl", "rb") as file:
    severity_encoder = pickle.load(file)

with open("relapse_encoder.pkl", "rb") as file:
    relapse_encoder = pickle.load(file)

# Feature list (must match model training)
feature_columns = ["Age", "Gender", "Education Level", "Employment Status", "Location", "Drug Type", "Frequency",
                   "Duration (Years)", "Mental Health", "Physical Health", "Monthly Income (Ksh)",
                   "Family Background", "Peer Influence", "Rehab Attendance", "Counseling Sessions"]

categorical_columns = ["Gender", "Education Level", "Employment Status", "Location", "Drug Type",
                       "Frequency", "Mental Health", "Physical Health", "Family Background",
                       "Peer Influence", "Rehab Attendance"]

# Streamlit UI
st.set_page_config(page_title="Addiction Prediction App", layout="wide")
st.title("🧠 Addiction Severity & Relapse Prediction")
st.markdown(
    "## Predict addiction severity and relapse risk based on various factors.")

# Sidebar for user input
st.sidebar.header("User Input Features")
st.sidebar.markdown("Enter patient details below:")

user_input = {
    "Age": st.sidebar.number_input("Age", min_value=10, max_value=100, step=1),
    "Gender": st.sidebar.selectbox("Gender", label_encoders["Gender"].classes_),
    "Education Level": st.sidebar.selectbox("Education Level", label_encoders["Education Level"].classes_),
    "Employment Status": st.sidebar.selectbox("Employment Status", label_encoders["Employment Status"].classes_),
    "Location": st.sidebar.selectbox("Location", label_encoders["Location"].classes_),
    "Drug Type": st.sidebar.selectbox("Drug Type", label_encoders["Drug Type"].classes_),
    "Frequency": st.sidebar.selectbox("Usage Frequency", label_encoders["Frequency"].classes_),
    "Duration (Years)": st.sidebar.number_input("Duration (Years)", min_value=0, max_value=50, step=1),
    "Mental Health": st.sidebar.selectbox("Mental Health Condition", label_encoders["Mental Health"].classes_),
    "Physical Health": st.sidebar.selectbox("Physical Health Condition", label_encoders["Physical Health"].classes_),
    "Monthly Income (Ksh)": st.sidebar.number_input("Monthly Income (Ksh)", min_value=0, step=500),
    "Family Background": st.sidebar.selectbox("Family Background", label_encoders["Family Background"].classes_),
    "Peer Influence": st.sidebar.selectbox("Peer Influence", label_encoders["Peer Influence"].classes_),
    "Rehab Attendance": st.sidebar.selectbox("Rehab Attendance", label_encoders["Rehab Attendance"].classes_),
    "Counseling Sessions": st.sidebar.number_input("Counseling Sessions Attended", min_value=0, max_value=100, step=1)
}

user_input_df = pd.DataFrame([user_input])

# Encode categorical input
for col in categorical_columns:
    user_input_df[col] = label_encoders[col].transform(user_input_df[col])

user_input_df = user_input_df.apply(pd.to_numeric)

# Prediction button
if st.sidebar.button("Predict"):
    severity_pred = severity_model.predict(user_input_df)
    relapse_pred = relapse_model.predict(user_input_df)

    severity_label = severity_encoder.inverse_transform([severity_pred[0]])[0]
    relapse_label = "Yes" if relapse_pred[0] == 1 else "No"

    st.success("### 🔍 Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric(label="**Addiction Severity Level**", value=severity_label)
    col2.metric(label="**Relapse Probability**", value=relapse_label)
    style_metric_cards()

# Sample Data Visualization
st.markdown("---")
st.subheader("📊 Data Insights and Trends")

try:
    df = pd.read_csv("data.csv")  # Load sample dataset for visualization
    fig1 = px.histogram(df, x="Age", nbins=20,
                        title="Age Distribution of Patients", color="Severity")
    fig2 = px.bar(df, x="Drug Type", y="Frequency",
                  title="Drug Type vs Frequency of Use", color="Severity")
    fig3 = px.box(df, x="Employment Status", y="Monthly Income (Ksh)",
                  title="Income Distribution by Employment Status")
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
except FileNotFoundError:
    st.warning(
        "⚠️ Data file not found. Please ensure 'data.csv' is in the project directory.")

st.markdown("---")
st.markdown("💡 *This application helps NGOs and rehabilitation centers analyze patient trends and make data-driven decisions.*")
