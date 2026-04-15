import streamlit as st
import joblib
import os
from src.train import preprocess_text

# Set page configuration
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# Load the saved model and vectorizer
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(models_dir, 'naive_bayes_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        return None, None

def main():
    st.title("📧 Spam Email/SMS Detection System")
    st.markdown("""
        Welcome to the Spam Detection app!
        Enter your message below to check if it's **Spam** or **Not Spam (Ham)**.
    """)
    
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("⚠️ Models not found! Please run `python src/train.py` first to generate the models.")
        return
    
    # Text input area for the user
    user_input = st.text_area("Enter your message here:", height=150)
    
    # Prediction button
    if st.button("Predict 🚀", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a valid message to proceed.")
        else:
            with st.spinner("Analyzing message..."):
                # 1. Preprocess the input text
                clean_text = preprocess_text(user_input)
                
                # 2. Transform the text into numerical features
                text_features = vectorizer.transform([clean_text])
                
                # 3. Predict using the compiled model
                prediction = model.predict(text_features)[0]
                prediction_prob = model.predict_proba(text_features)[0]
                
                # 4. Display results
                st.markdown("---")
                st.subheader("Result:")
                
                if prediction == 1:
                    st.error("🚨 **This message is classified as: SPAM**")
                else:
                    st.success("✅ **This message is classified as: NOT SPAM (HAM)**")
                
                # Show confidence scores horizontally
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Probability of Not Spam", value=f"{prediction_prob[0]*100:.2f}%")
                with col2:
                    st.metric(label="Probability of Spam", value=f"{prediction_prob[1]*100:.2f}%")

if __name__ == "__main__":
    main()
