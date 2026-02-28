import streamlit as st
import requests

st.title("PPE Verification System For Workers")

badge_id = st.text_input("Badge ID")

first = st.text_input("First Name")
last = st.text_input("Last Name")
image = st.file_uploader("Upload Badge Image", type=["jpg", "png"])

if st.button("LOGIN") and image:
    files = {"file": image.getvalue()}
    data = {"first_name": first, "last_name": last}

    response = requests.post(
        "http://localhost:8000/analyze-image",
        files={"file": image},
        data=data
    )

    result = response.json()
    # streamlit run app.py --server.port 8501
    
    # st.write("Extracted Name:", result["extracted_name"])
    # st.write("Match Score:", result["name_match_score"])
    # st.write("Face Detected:", result["face_detected"])
    # st.write("Anomaly:", result["anomaly"])