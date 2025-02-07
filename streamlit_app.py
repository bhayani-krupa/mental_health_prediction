# import streamlit as st
# import pandas as pd
# import joblib  # To load the trained model
# import numpy as np

# # Load the trained model
# model = joblib.load("random_forest.pkl")  # Ensure you have saved this model

# # Define the input fields
# st.title("Mental Health Condition Predictor")

# st.write("Enter the details below to predict if you might need mental health treatment.")

# # User inputs
# age = st.slider("Age", 18, 100, 25)  # Slider for age input
# gender = st.selectbox("Gender", ["Male", "Female", "Other"])
# self_employed = st.selectbox("Self Employed", ["Yes", "No"])
# family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
# work_interfere = st.selectbox("Work Interference", ["Often", "Rarely", "Never", "Sometimes"])
# no_employees = st.selectbox("Number of Employees", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
# remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
# tech_company = st.selectbox("Work in a Tech Company?", ["Yes", "No"])
# benefits = st.selectbox("Does your company provide mental health benefits?", ["Yes", "No", "Don't know"])
# care_options = st.selectbox("Care Options Available?", ["Yes", "No", "Not sure"])
# wellness_program = st.selectbox("Wellness Program at Work?", ["Yes", "No", "Don't know"])
# seek_help = st.selectbox("Does your company encourage seeking help?", ["Yes", "No", "Don't know"])
# anonymity = st.selectbox("Is seeking help anonymous?", ["Yes", "No", "Don't know"])
# leave = st.selectbox("How easy is it to take leave for mental health?", ["Very Easy", "Somewhat Easy", "Don't Know", "Somewhat Difficult", "Very Difficult"])
# mental_health_consequence = st.selectbox("Negative consequences for mental health disclosure?", ["Yes", "No", "Maybe"])
# phys_health_consequence = st.selectbox("Negative consequences for physical health disclosure?", ["Yes", "No", "Maybe"])
# coworkers = st.selectbox("Would you discuss mental health with coworkers?", ["Yes", "No", "Some of them"])
# supervisor = st.selectbox("Would you discuss mental health with your supervisor?", ["Yes", "No", "Some of them"])
# mental_health_interview = st.selectbox("Would you bring up mental health in an interview?", ["Yes", "No", "Maybe"])
# phys_health_interview = st.selectbox("Would you bring up physical health in an interview?", ["Yes", "No", "Maybe"])
# mental_vs_physical = st.selectbox("Do you think mental health is treated the same as physical health?", ["Yes", "No", "Don't know"])
# obs_consequence = st.selectbox("Observed negative consequences of mental health issues at work?", ["Yes", "No"])

# # Convert categorical inputs into numerical values (this must match how your model was trained)
# gender_map = {"Male": 0, "Female": 1, "Other": 2}
# binary_map = {"Yes": 1, "No": 0}
# leave_map = {"Very Easy": 0, "Somewhat Easy": 1, "Don't Know": 2, "Somewhat Difficult": 3, "Very Difficult": 4}
# freq_map = {"Often": 0, "Rarely": 1, "Never": 2, "Sometimes": 3}
# emp_map = {"1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3, "500-1000": 4, "More than 1000": 5}
# coworker_map = {"Yes": 0, "No": 1, "Some of them": 2}

# # Convert user inputs
# input_data = np.array([
#     age,
#     gender_map[gender],
#     binary_map[self_employed],
#     binary_map[family_history],
#     freq_map[work_interfere],
#     emp_map[no_employees],
#     binary_map[remote_work],
#     binary_map[tech_company],
#     binary_map[benefits],
#     binary_map[care_options],
#     binary_map[wellness_program],
#     binary_map[seek_help],
#     binary_map[anonymity],
#     leave_map[leave],
#     binary_map[mental_health_consequence],
#     binary_map[phys_health_consequence],
#     coworker_map[coworkers],
#     coworker_map[supervisor],
#     binary_map[mental_health_interview],
#     binary_map[phys_health_interview],
#     binary_map[mental_vs_physical],
#     binary_map[obs_consequence]
# ]).reshape(1, -1)  # Reshape for prediction

# # Predict button
# if st.button("Predict Mental Health Condition"):
#     prediction = model.predict(input_data)
    
#     # Assuming 1 = Needs Treatment, 0 = No Treatment Required
#     if prediction[0] == 1:
#         st.error("The model predicts that you **might need mental health treatment.** Please consider consulting a professional.")
#     else:
#         st.success("The model predicts that you **might not need mental health treatment.** However, always prioritize your well-being and seek help if needed.")

import streamlit as st
import joblib
import numpy as np
import re

# Load trained model
model = joblib.load("random_forest.pkl")

# Chatbot intro
st.title("🧠 Mental Health Chatbot")
st.write("Hello! I'm here to help you understand your mental health condition. Describe your symptoms, and I'll try to assist you.")

# Function to extract features from user input
def extract_features(user_input):
    """
    Extracts relevant features from user input and converts them into numerical format.
    """
    # Example keyword matching (can be improved with NLP)
    age_match = re.search(r"(\d+)\s*years?\s*old", user_input)
    age = int(age_match.group(1)) if age_match else 25  # Default age

    gender = 0 if "male" in user_input.lower() else 1 if "female" in user_input.lower() else 2
    self_employed = 1 if "self-employed" in user_input.lower() else 0
    family_history = 1 if "family history" in user_input.lower() else 0
    work_interfere = 0 if "often" in user_input.lower() else 3 if "sometimes" in user_input.lower() else 1
    remote_work = 1 if "remote" in user_input.lower() else 0
    tech_company = 1 if "tech" in user_input.lower() else 0
    benefits = 1 if "benefits" in user_input.lower() else 0
    care_options = 1 if "care options" in user_input.lower() else 0
    wellness_program = 1 if "wellness" in user_input.lower() else 0
    seek_help = 1 if "seek help" in user_input.lower() else 0
    anonymity = 1 if "anonymous" in user_input.lower() else 0
    leave = 0 if "easy leave" in user_input.lower() else 4 if "difficult" in user_input.lower() else 2
    mental_health_consequence = 1 if "fear consequences" in user_input.lower() else 0
    phys_health_consequence = 1 if "physical health issues" in user_input.lower() else 0
    coworkers = 0 if "talk to coworkers" in user_input.lower() else 1
    supervisor = 0 if "talk to supervisor" in user_input.lower() else 1
    mental_health_interview = 1 if "interview mental health" in user_input.lower() else 0
    phys_health_interview = 1 if "interview physical health" in user_input.lower() else 0
    mental_vs_physical = 1 if "mental and physical same" in user_input.lower() else 0
    obs_consequence = 1 if "observed consequences" in user_input.lower() else 0
    no_employees = 0  # Default value

    if "1-5 employees" in user_input.lower():
        no_employees = 1
    elif "6-25 employees" in user_input.lower():
        no_employees = 2
    elif "26-100 employees" in user_input.lower():
        no_employees = 3
    elif "100-500 employees" in user_input.lower():
        no_employees = 4
    elif "500+ employees" in user_input.lower():
        no_employees = 5


    return np.array([
        age, gender, self_employed, family_history, work_interfere,
        remote_work, tech_company, benefits, care_options, wellness_program,
        seek_help, anonymity, leave, mental_health_consequence, phys_health_consequence,
        coworkers, supervisor, mental_health_interview, phys_health_interview,
        mental_vs_physical, obs_consequence, no_employees
    ]).reshape(1, -1)

# Chatbot input and response handling
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Describe your mental health concerns...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process input and make prediction
    features = extract_features(user_input)
    prediction = model.predict(features)[0]

    # Generate chatbot response
    response = (
        "Based on your input, **you might need mental health treatment.** Please consider consulting a professional."
        if prediction == 1
        else "Based on your input, **you might not need mental health treatment.** However, always prioritize your well-being."
    )

    # Display chatbot message
    with st.chat_message("assistant"):
        st.markdown(response)

    # Store chatbot response
    st.session_state.messages.append({"role": "assistant", "content": response})
