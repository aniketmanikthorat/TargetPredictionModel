# app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("RandomForest_model.pkl")
df = pd.read_csv("final_balanced_dataset.csv")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📄 Raw Data", "📊 Graphs and Charts", "🎯 Prediction"])

# Page 1: Raw Data
if page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(df)

# Page 2: Graphs and Charts
elif page == "📊 Graphs and Charts":
    st.title("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("🔹 Hit vs Miss Count")
        st.bar_chart(df['target_hit'].value_counts())

    with col2:
        st.write("🔹 Experience Level Distribution")
        st.bar_chart(df['shooter_experience'].value_counts())

    st.write("🔹 Pairplot (Sample - 100 Random Records)")
    fig = sns.pairplot(df.sample(100), hue="target_hit", diag_kind="kde")
    st.pyplot(fig)

# Page 3: Prediction
elif page == "🎯 Prediction":
    st.title("🎯 Target Hit Prediction App")

    # User input form
    shooter_experience = st.selectbox("Shooter Experience", ["New", "Intermediate", "Expert"])
    distance_to_target = st.slider("Distance to Target (meters)", 100, 1000, 500)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 40.0, 10.0)
    wind_direction = st.selectbox("Wind Direction", ["Headwind", "Tailwind", "Crosswind"])
    is_target_moving = st.checkbox("Is the Target Moving?")
    target_speed = st.slider("Target Speed (if moving)", 0.0, 30.0, 10.0) if is_target_moving else 0.0
    time_of_day = st.selectbox("Time of Day", ["Morning", "Noon", "Evening"])
    weapon_type = st.selectbox("Weapon Type", ["Sniper", "Assault Rifle"])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "shooter_experience": shooter_experience,
            "distance_to_target": distance_to_target,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "is_target_moving": is_target_moving,
            "target_speed": target_speed,
            "time_of_day": time_of_day,
            "weapon_type": weapon_type
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Prediction: {'Hit ✅' if prediction else 'Miss ❌'}")
