import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Highway Safety Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, 'data')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_PATH, 'cleaned_accident_data.csv'))

# Load model
@st.cache_resource
def load_model():
    with open(os.path.join(DATA_PATH, 'best_model.pkl'), 'rb') as f:
        return pickle.load(f)

# Load encoders
@st.cache_resource
def load_encoders():
    with open(os.path.join(DATA_PATH, 'encoders.pkl'), 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()
encoders = load_encoders()

# Sidebar
st.sidebar.title("🚗 Highway Safety")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 State Analysis", "🤖 Predict Severity"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

all_states = ['All'] + sorted(df['state'].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", all_states)

all_severity = ['All'] + sorted(df['severity'].unique().tolist())
selected_severity = st.sidebar.selectbox("Select Severity", all_severity)

filtered_df = df.copy()
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['state'] == selected_state]
if selected_severity != 'All':
    filtered_df = filtered_df[filtered_df['severity'] == selected_severity]

# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🚗 National Highway Incident & Safety Analytics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Accidents", len(filtered_df))
    with col2:
        st.metric("States Covered", filtered_df['state'].nunique())
    with col3:
        st.metric("Most Dangerous Hour", f"{filtered_df['hour'].value_counts().idxmax()}:00")
    with col4:
        st.metric("Most Common Severity", filtered_df['severity'].value_counts().idxmax())

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accidents by Severity")
        severity_counts = filtered_df['severity'].value_counts()
        fig = px.pie(values=severity_counts.values,
                     names=severity_counts.index,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Accidents by Weather")
        weather_counts = filtered_df['weather'].value_counts()
        fig = px.bar(x=weather_counts.index,
                     y=weather_counts.values,
                     color=weather_counts.values,
                     color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accidents by Hour of Day")
        hour_counts = filtered_df['hour'].value_counts().sort_index()
        fig = px.bar(x=hour_counts.index,
                     y=hour_counts.values,
                     color=hour_counts.values,
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Accidents by Month")
        month_counts = filtered_df.groupby(['month', 'month_name']).size().reset_index(name='count')
        month_counts = month_counts.sort_values('month')
        fig = px.line(month_counts, x='month_name', y='count',
                      markers=True, color_discrete_sequence=['coral'])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 - STATE ANALYSIS
# ============================================================
elif page == "📊 State Analysis":
    st.title("📊 State Wise Accident Analysis")
    st.markdown("---")

    st.subheader("Total Accidents by State")
    state_counts = filtered_df['state'].value_counts().reset_index()
    state_counts.columns = ['state', 'count']
    fig = px.bar(state_counts, x='state', y='count',
                 color='count', color_continuous_scale='YlOrRd')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Severity by State")
        severity_state = pd.crosstab(filtered_df['state'], filtered_df['severity'])
        fig = px.bar(severity_state, barmode='group',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Weather by State")
        weather_state = pd.crosstab(filtered_df['state'], filtered_df['weather'])
        fig = px.bar(weather_state, barmode='stack',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(filtered_df, use_container_width=True)

# ============================================================
# PAGE 3 - PREDICTION
# ============================================================
elif page == "🤖 Predict Severity":
    st.title("🤖 Accident Severity Predictor")
    st.markdown("---")
    st.markdown("Fill in the details below to predict accident severity")

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox("State", sorted(df['state'].unique()))
        weather = st.selectbox("Weather", sorted(df['weather'].unique()))
        vehicle_type = st.selectbox("Vehicle Type", sorted(df['vehicle_type'].unique()))
        light_condition = st.selectbox("Light Condition", sorted(df['light_condition'].unique()))

    with col2:
        driver_sex = st.selectbox("Driver Gender", sorted(df['driver_sex'].unique()))
        driver_age = st.slider("Driver Age", 18, 80, 30)
        car_age = st.slider("Car Age (years)", 0, 30, 5)
        engine_size = st.slider("Engine Size (cc)", 500, 5000, 1500)

    with col3:
        hour = st.slider("Hour of Accident", 0, 23, 12)
        week_day = st.selectbox("Day of Week", sorted(df['week_day'].unique()))
        casualty_type = st.selectbox("Casualty Type", sorted(df['casualty_type'].unique()))

    st.markdown("---")

    if st.button("🔮 Predict Severity", use_container_width=True):
        try:
            input_data = {
                'state': encoders['state'].transform([state])[0],
                'weather': encoders['weather'].transform([weather])[0],
                'vehicle_type': encoders['vehicle_type'].transform([vehicle_type])[0],
                'driver_sex': encoders['driver_sex'].transform([driver_sex])[0],
                'driver_age': driver_age,
                'car_age': car_age,
                'hour': hour,
                'week_day': encoders['week_day'].transform([week_day])[0],
                'light_condition': encoders['light_condition'].transform([light_condition])[0],
                'casualty_type': encoders['casualty_type'].transform([casualty_type])[0],
                'engine_size': engine_size
            }

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

            severity_map = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
            predicted_severity = severity_map.get(prediction, str(prediction))

            if predicted_severity == 'Minor':
                st.success(f"✅ Predicted Severity: {predicted_severity}")
            elif predicted_severity == 'Moderate':
                st.warning(f"⚠️ Predicted Severity: {predicted_severity}")
            else:
                st.error(f"🚨 Predicted Severity: {predicted_severity}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
