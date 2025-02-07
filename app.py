import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from groq import Groq  # Using Groq LLM for chatbot

# Load environment variables
load_dotenv()

# Configure Groq AI
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the trained ML model
MODEL_PATH = "random_forest.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
else:
    st.error("Trained model not found! Make sure to train and save it first.")
    st.stop()

# Initialize Streamlit app
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧘", layout="centered")
st.title("🧘 Mental Health Support Chatbot")

st.markdown("""
    Welcome to the Mental Health Support Chatbot.
    This tool can help assess your mental health and provide guidance for better well-being.
    All responses are kept anonymous.
    """)

# Questionnaire for prediction
st.subheader("📝 Mental Health Assessment Questionnaire")

# Collect user input through questions
age = st.slider("What is your age?", 18, 99, 25)
gender = st.selectbox("Select your gender:", ["male", "female", "other"])
anonymity = st.selectbox("Do you prefer anonymity when seeking help?", ["Yes", "No"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Are mental health care options available?", ["Yes", "No", "Not sure"])
family_history = st.selectbox("Do you have a family history of mental health conditions?", ["Yes", "No"])
mental_health_consequence = st.selectbox("Do you think discussing mental health might have negative consequences at work?", ["Yes", "No"])
obs_consequence = st.selectbox("Do you think there are negative consequences for discussing mental health openly?", ["Yes", "No"])
seek_help = st.selectbox("Have you ever sought mental health treatment?", ["Yes", "No"])
supervisor = st.selectbox("Do you feel comfortable talking to your supervisor about mental health?", ["Yes", "No"])
work_interfere = st.selectbox("How often does your mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"])

# Encode categorical inputs
encoding_map = {
    "Yes": 1, "No": 0, "Don't know": -1, "Not sure": -1,
    "male": 0, "female": 1, "other": 2,
    "Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3
}

# Collect responses and build features for prediction
user_features = {
    "Age": age,
    "Gender": encoding_map[gender],
    "anonymity": encoding_map[anonymity],
    "benefits": encoding_map.get(benefits, -1),
    "care_options": encoding_map.get(care_options, -1),
    "family_history": encoding_map[family_history],
    "mental_health_consequence": encoding_map[mental_health_consequence],
    "obs_consequence": encoding_map[obs_consequence],
    "seek_help": encoding_map[seek_help],
    "supervisor": encoding_map[supervisor],
    "work_interfere": encoding_map[work_interfere]
}

# Prediction function
def predict_mental_health(features):
    input_df = pd.DataFrame([features])

    # Ensure feature alignment
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)
    return "Need Treatment" if prediction[0] == 1 else "Don't Need Treatment"

if st.button("🧘 Submit Questionnaire"):
    prediction = predict_mental_health(user_features)

    st.subheader("🧾 Prediction Result")
    st.write(f"The assessment indicates: **{prediction}**")

# Chatbot Section
st.subheader("💬 Mental Health Chatbot")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to get Groq response
def get_groq_response(question):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Input field
user_input = st.text_input("Ask for advice or share your thoughts:", key="input")
submit_chat = st.button("🗨️ Ask the Bot")

if submit_chat and user_input:
    response = get_groq_response(user_input)

    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", user_input))
    st.session_state['chat_history'].append(("Bot", response))

    # Display response
    st.subheader("🤖 Chatbot Response")
    st.write(response)

# Display chat history
st.subheader("📜 Chat History")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
