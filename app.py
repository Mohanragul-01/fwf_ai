import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
import cv2

# Set the page title
st.set_page_config(page_title="User Mood Feedback System", page_icon="😊")

# Define class names
class_names = ['Contempt', 'Happy', 'Anger', 'Fear', 'Sadness', 'Surprise', 'Disgust']

# Load the three models
@st.cache_resource
def load_models():
    model1 = load_model("user_interaction_model_1.h5")
    model2 = load_model("system_metrics_model_2.h5")
    model3 = load_model("emotion_recognition_model_3.h5")
    
    # Extract intermediate models for confidence levels
    intermediate_model_1 = Model(inputs=model1.input, outputs=model1.get_layer('intermediate_output').output)
    intermediate_model_2 = Model(inputs=model2.input, outputs=model2.get_layer('intermediate_output').output)
    
    return intermediate_model_1, intermediate_model_2, model3

intermediate_model_1, intermediate_model_2, model3 = load_models()

# Streamlit UI
st.title("User Mood Feedback System")
st.write("Predict the user's mood based on interaction, system metrics, and facial emotions.")

# Input fields for Model 1: User Interaction
st.header("User Interaction Features")
scroll_speed = st.number_input("Scroll Speed", min_value=0.0, value=50.0, step=0.1)
tab_switched = st.number_input("Tabs Switched", min_value=0, value=5)
session_duration = st.number_input("Session Duration", min_value=0.0, value=1000.0, step=0.1)
click_events = st.number_input("Click Events", min_value=0, value=20)
idle_time = st.number_input("Idle Time", min_value=0.0, value=100.0, step=0.1)
interactive_time = st.number_input("Interactive Time", value=500.0, step=0.1)

# Input fields for Model 2: System Metrics
st.header("System Metrics Features")
cpu_utilization = st.number_input("CPU Utilization", min_value=0.0, value=100.0, step=0.1)
disk_io = st.number_input("Disk IO", min_value=0.0, value=50.0, step=0.1)
memory_usage = st.number_input("Memory Usage", min_value=0.0, value=1500.0, step=0.1)
network_throughput = st.number_input("Network Throughput", min_value=0.0, value=20.0, step=0.1)

# Input field for Model 3: Facial Emotion Image
st.header("Upload a Facial Emotion Image")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# Predict Button
if st.button("Predict Mood"):
    # Prepare input for Model 1
    user_interaction_data = pd.DataFrame({
        "scroll_speed": [scroll_speed],
        "tab_switched": [tab_switched],
        "session_duration": [session_duration],
        "click_events": [click_events],
        "idle_time": [idle_time],
        "interactive_time": [interactive_time],
    })
    
    # Prepare input for Model 2
    system_metrics_data = pd.DataFrame({
        "CPU_Utilization": [cpu_utilization],
        "Disk_IO": [disk_io],
        "Memory_Usage": [memory_usage],
        "Network_Throughput": [network_throughput],
    })
    
    # Predict confidence scores from Model 1 and Model 2
    confidence_scores_1 = intermediate_model_1.predict(user_interaction_data)
    confidence_scores_2 = intermediate_model_2.predict(system_metrics_data)
    
    # Initialize total confidence scores
    total_confidence_scores = np.sum(confidence_scores_1, axis=0) + np.sum(confidence_scores_2, axis=0)

    # Process image for Model 3
    if uploaded_file is not None:
        # Read and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)   # Add batch dimension

        # Predict confidence scores from Model 3
        confidence_scores_3 = model3.predict(img)
        total_confidence_scores += np.sum(confidence_scores_3[:, :7], axis=0)
    else:
        st.warning("Please upload a facial emotion image for Model 3.")
    
    # Determine final mood
    predicted_class_idx = np.argmax(total_confidence_scores)
    predicted_mood = class_names[predicted_class_idx]

    # Display results
    st.subheader("Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {total_confidence_scores[i]:.4f}")
    
    st.subheader(f"Predicted Mood: {predicted_mood}")

    # Display the rating bar
    mood_mapping = {
        'Contempt': 5,
        'Happy': 7,
        'Anger': 2,
        'Fear': 3,
        'Sadness': 1,
        'Surprise': 6,
        'Disgust': 4
    }

    rating = mood_mapping.get(predicted_mood, 3)  # Default to neutral if mood is not mapped

    # Display the emoji and rating bar
    if predicted_mood == 'Contempt':
        emoji = '😡'
        color = 'red'
    elif predicted_mood == 'Happy':
        emoji = '😊'
        color = 'green'
    elif predicted_mood == 'Anger':
        emoji = '😠'
        color = 'orange'
    elif predicted_mood == 'Fear':
        emoji = '😨'
        color = 'orange'
    elif predicted_mood == 'Sadness':
        emoji = '😢'
        color = 'red'
    elif predicted_mood == 'Surprise':
        emoji = '😲'
        color = 'yellow'
    elif predicted_mood == 'Disgust':
        emoji = '🤢'
        color = 'red'
    else:
        emoji = '😐'
        color = 'yellow'

    # Rating bar with color indicator
    st.markdown(f"### Mood Rating: {emoji}")
    st.slider("User Experience", min_value=1, max_value=7, value=rating, step=1, key="mood_rating")

    # Display the color-coded bar with the corresponding mood scale
    st.markdown("""
        <style>
        .stSlider>div {
            background-color: """ + color + """;
            border-radius: 25px;
        }
        .stSlider>div>div {
            background-color: """ + color + """;
            padding: .5rem;
            border-radius: 25px;
        }
        .stSlider>div>div>div.st-ba {
            background-color: #FFF;
            padding: .25rem;
            margin: 1rem;
            margin-bottom: 0;
        }
        .stSlider>div>div.st-emotion-cache-7ti8k2 {
            margin-left: 1rem;
            margin-right: 1rem;
            border-radius: 25px;
        }
        </style>
    """, unsafe_allow_html=True)
