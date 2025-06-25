import gradio as gr
import joblib
import numpy as np

from extracted_features import extract_all_features

# Load the trained model
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("parkinsons_scaler.pkl")

# Prediction function
def predict_parkinsons(audio_file):
    if audio_file is None:
        return "Please provide an audio file."

    features = extract_all_features(audio_file)
    if features is None:
        return "Failed to extract features. Please try a different audio sample."

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]


    if prediction == 1:
        result = "🧠 Parkinson's Detected"
    else:
        result = "✅ Normal Speech"
        
    if proba is not None:
        result += f"\nConfidence: {proba[prediction]*100:.2f}%"

    return result

# Gradio UI
iface = gr.Interface(
    fn=predict_parkinsons,
    inputs = gr.Audio(type="filepath", label="Upload or Record Voice (.wav preferred)"),
    outputs="text",
    title="Parkinson's Voice Predictor",
    description="Upload or record a short voice clip to check for signs of Parkinson's disease using voice-based ML analysis.",
    live=False
)

if __name__ == "__main__":
    iface.launch()
