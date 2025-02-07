
Prerequisites:
1. Install necessary packages by running:
2. Pip install pandas numpy scikit-learn matplotlib streamlit shap
3. Ensure the following files are available: 
      1. Random_forest.pkl: Pre-trained Random Forest Model
      2. predict_model_health.py/.ipynb: Inference Script

Running the Script:
1. Open a terminal or command prompt.
2. Execute the script using
3. The system will prompt users for mental health-related questionnaire responses and 
provide a prediction.

 UI/CLI Usage Instructions
Using the Streamlit UI:
1. Run the chatbot and questionnaire app with: 
2. streamlit run app.py
3. Mental Health Questionnaire:
o Answer questions such as age, gender, family history, and work interference.
o Click the "Submit Questionnaire" button to receive a prediction.
4. Chatbot Interaction:
o Ask mental health-related questions or seek advice.
o The chatbot, powered by Groq LLM, provides empathetic and context-aware 
responses.
