import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write("An AIML mini project that estimates a student's expected final marks from study habits and previous performance.")
st.info("Educational demonstration only. Predictions are estimates, not guarantees.")

@st.cache_resource
def train_model():
    data = pd.read_csv(Path(__file__).with_name("student_data.csv"))
    X = data[["Study_Hours", "Attendance", "Previous_Marks"]]
    y = data["Final_Marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, mean_absolute_error(y_test, predictions), r2_score(y_test, predictions), data

model, mae, r2, data = train_model()
st.subheader("Enter student details")
study = st.slider("Average study hours per day", 0.0, 12.0, 4.0, 0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous = st.slider("Previous exam marks", 0, 100, 60)

if st.button("Predict performance", type="primary"):
    sample = pd.DataFrame([[study, attendance, previous]],
        columns=["Study_Hours", "Attendance", "Previous_Marks"])
    prediction = float(model.predict(sample)[0])
    prediction = max(0, min(100, prediction))
    st.metric("Estimated final marks", f"{prediction:.1f} / 100")
    if prediction >= 75:
        st.success("Estimated performance band: High")
    elif prediction >= 50:
        st.warning("Estimated performance band: Moderate")
    else:
        st.error("Estimated performance band: Needs support")

st.divider()
st.subheader("Model evaluation")
c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error", f"{mae:.2f}")
c2.metric("R² score", f"{r2:.2f}")
st.caption("Evaluation uses a held-out test split from the included illustrative dataset.")
st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)
