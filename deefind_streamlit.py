import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️")
st.title("🕵️ Deepfake Detection Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# from PIL import Image
# image = Image.open(uploaded_file)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width= True)

    if st.button("Detect Deepfake"):
        with st.spinner("Analyzing..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            try:
                response = requests.post(API_URL, files=files)
            except Exception as e:
                st.error(f"Could not connect to API: {e}")
            else:
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction received")
                    st.json(result)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
