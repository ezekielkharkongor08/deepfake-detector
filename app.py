import streamlit as st
import torch
import tempfile
from utils import preprocess_audio, load_model

st.set_page_config(page_title="Deepfake Detector")

st.title("🎧 Audio Deepfake Detection")

model = load_model("model.pth")

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "flac"]
)

if uploaded_file:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        x = preprocess_audio(path)

        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs).item()

    if pred == 0:
        st.success("Real Audio")
    else:
        st.error("Deepfake Audio")

    st.write("Confidence:", float(probs[0][pred]))
