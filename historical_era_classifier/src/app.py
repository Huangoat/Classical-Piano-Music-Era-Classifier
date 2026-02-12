import streamlit as st
import joblib
import pandas as pd
import tempfile
import os
from pathlib import Path

from realtime_feature_extracter import extract_features_from_midi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = (PROJECT_ROOT / "models" / "model.pkl")

# Page config
st.set_page_config(
    page_title="Classical Piano Music Era Classifier",
    page_icon="🎼",
    layout="centered"
)

st.title("🎼 Classical Piano Music Era Classifier")
@st.cache_resource
def load_classifier_model():
    return joblib.load(MODEL_DIR)

classifier_model = load_classifier_model()

#Upload file
uploaded_file = st.file_uploader(
    "Upload a MIDI file",
    type=["mid", "midi"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        #Feature Extraction
        features = extract_features_from_midi(tmp_path)

        if features is None or features.empty:
            st.error("Could not extract features from this MIDI file.")
        else:
            # Using model to predict
            prediction = classifier_model.predict(features)[0]
        
            st.subheader("🎶 Predicted Era")
            st.success(f"**{prediction}**")

            #Probability breakdown
            if hasattr(classifier_model, "predict_proba"):
                probs = classifier_model.predict_proba(features)[0]
                classes = classifier_model.classes_
                
                confidence = probs.max()
                st.write(f"**Confidence:** {confidence:.2%}")

                prob_df = pd.DataFrame({
                    "Era": classes,
                    "Probability": probs
                }).sort_values(by="Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Era"))

    except Exception as e:
        st.error(f"Error during prediction: {e}")

    finally:
        os.remove(tmp_path)

else:
    st.info("Upload a MIDI file to get started.")
