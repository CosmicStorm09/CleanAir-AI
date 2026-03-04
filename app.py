import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("CleanAir AI")
st.subheader("Simple Air Quality Prediction & Health Guide")

st.write(
"""
CleanAir AI predicts short-term air quality using historical AQI data
and provides simple health recommendations.
"""
)

# Sample AQI data
data = {
    "day": [1,2,3,4,5,6,7],
    "AQI": [180,200,220,210,230,250,240]
}

df = pd.DataFrame(data)

# Train simple model
X = df[["day"]]
y = df["AQI"]

model = LinearRegression()
model.fit(X,y)

next_day = st.slider("Select future day for prediction",8,14)

prediction = model.predict([[next_day]])[0]

st.write(f"### Predicted AQI: {int(prediction)}")

# Simple advisory
if prediction < 100:
    advice = "Air quality is good. Outdoor activities are safe."
elif prediction < 200:
    advice = "Moderate pollution. Sensitive groups should take care."
else:
    advice = "High pollution expected. Avoid heavy outdoor activity."

st.write("### Health Advice")
st.write(advice)

st.caption("Prototype version of CleanAir AI")
