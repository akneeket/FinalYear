import streamlit as st
import numpy as np
import tensorflow as tf
import google.generativeai as genai

# Configure API key for Gemini
genai.configure(api_key='***************************************')

# Load your pre-trained LSTM model (replace with your actual model path)
model = tf.keras.models.load_model('lstm.h5')

# Function to predict speed (in m/s, converted to km/h)
def predict_speed(attributes):
    attributes = np.array(attributes)
    attributes = attributes.reshape(1, 1, 9)  # Reshaping for LSTM model
    predicted_speed_m_s = model.predict(attributes)[0][0]  # Predict speed in m/s
    predicted_speed_kmh = predicted_speed_m_s  # Convert m/s to km/h
    return predicted_speed_kmh

# Function to interact with the LLM
def analyze_session(data):
    prompt = (
        f"Details: {data['details']}\n"
        f"Attributes: Sleep: {data['Sleep']}, Training Hours: {data['Training Hours']}, "
        f"Rest Days: {data['Rest Days']}, Hydration: {data['Hydration']}, "
        f"BMI: {data['BMI']}, Age: {data['Age']}, Recovery Hours: {data['Recovery Hours']}\n"
        f"Predicted Speed: {data['predicted_speed']} km/h, Actual Speed: {data['actual_speed']} km/h\n"
        f"Please provide insights and suggestions for improvement."
    )
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
    return response.text

# Streamlit UI Setup
st.set_page_config(page_title="Athlete Performance Prediction", layout="centered", initial_sidebar_state="auto")

# Custom CSS to enhance appearance
st.markdown("""
    <style>
        body {
            background-color: #f4f7fa;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 18px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div, .stSlider, .stForm textarea {
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            text-align: center;
            font-family: 'Arial', sans-serif;
            color: #4CAF50;
        }
        .stTextInput input {
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
        }
        .stSlider {
            font-size: 16px;
        }
        .stForm textarea {
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
        }
        .stMarkdown {
            font-size: 18px;
            line-height: 1.6;
        }
        .stInfo {
            font-size: 18px;
            background-color: #e7f9e7;
        }
    </style>
    """, unsafe_allow_html=True)

# Step 1: Attribute input
st.title('🏃 Athlete Performance Prediction 🏃‍♂️')
st.write(
    "Welcome to the Athlete Speed Prediction app. Please provide the necessary attributes below to predict your performance and gain insights for improvement.")

st.header('Step 1: Provide Your Attributes')

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        sleep = st.slider('Sleep (hours)', 0, 12, 7)
        rest_days = st.slider('Rest Days', 0, 7, 2)
        age = st.slider('Age', 16, 50, 25)
    with col2:
        hydration = st.slider('Hydration (liters)', 0, 5, 2)
    with col3:
        training_hours = st.slider('Training Hours', 0, 10, 3)
        bmi = st.slider('BMI', 15, 40, 22)
        recovery_hours = st.slider('Recovery Hours', 0, 12, 6)
        actual_speed_kmh = st.number_input('Enter actual current speed (km/h):', min_value=0.0, step=0.1)

    submit = st.form_submit_button("Predict Speed")

# Store predicted speed in session state
if 'predicted_speed_kmh' not in st.session_state:
    st.session_state.predicted_speed_kmh = None

if submit:
    # Predict speed
    attributes = [
        sleep, training_hours, rest_days, hydration, bmi, age, recovery_hours,
        0, 0  # Placeholder values for missing features
    ]
    predicted_speed_kmh = predict_speed(attributes)
    st.session_state.predicted_speed_kmh = predicted_speed_kmh

    st.success(f"🏃 Predicted Speed: {predicted_speed_kmh:.2f} km/h")

    if actual_speed_kmh > 0:
        improvement = ((actual_speed_kmh - predicted_speed_kmh) / predicted_speed_kmh) * 100
        st.info(f"📈 Improvement: {improvement:.2f}%")

# Step 2: Session feedback and LLM analysis
st.header('Step 2: Provide Session Feedback')

st.write(
    "Please share details about your training session and any difficulties faced to receive personalized suggestions.")

with st.form("feedback_form"):
    session_details = st.text_area("Describe the session and difficulties faced:", height=150)
    send_data = st.form_submit_button("Send Data for Analysis")

if send_data:
    # Use predicted speed from session state
    predicted_speed_kmh = st.session_state.predicted_speed_kmh

    if actual_speed_kmh > 0 and predicted_speed_kmh is not None:
        data = {
            'Sleep': sleep,
            'Training Hours': training_hours,
            'Rest Days': rest_days,
            'Hydration': hydration,
            'BMI': bmi,
            'Age': age,
            'Recovery Hours': recovery_hours,
            'predicted_speed': predicted_speed_kmh,
            'actual_speed': actual_speed_kmh,
            'details': session_details
        }
        # Get LLM response
        analysis = analyze_session(data)

        # Store analysis in session state
        st.session_state.analysis = analysis
        st.session_state.session_details = session_details

        # Create button to display analysis
        st.button('View Analysis', key='show_analysis', on_click=lambda: st.session_state.update({'show_analysis': True}))

# Step 3: Display Analysis Section
if 'show_analysis' in st.session_state and st.session_state.show_analysis:
    st.markdown("### 💡 Your Personalized Recommendations")
    st.markdown(f"**Analysis:** {st.session_state.analysis}")
    st.markdown(f"#### 📋 Session Details\n{st.session_state.session_details}")
