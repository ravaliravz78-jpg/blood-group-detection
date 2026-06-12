import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import gdown

def set_background(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}

        h1, h2, h3, h4, h5, h6, p, div, label, span {{
            color: black !important;
        }}

        div[data-testid="metric-container"] {{
            background-color: white;
            border: 2px solid #e0e0e0;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }}

        .main-title {{
            text-align: center;
            color: #B71C1C !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -----------------------------
# MODEL DOWNLOAD
# -----------------------------
MODEL_PATH = "blood_group_model.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1WK3-NsMi4ye9gWqSpLnc8ztcYMxSBmBT"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model(MODEL_PATH)

classes = ['A', 'AB', 'B', 'O']

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🩸 Navigation")

st.sidebar.info("""
Fingerprint Based Blood Group Detection

Mini Project

Using CNN Deep Learning
""")

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "home":

    set_background("#FFFFFF")

    st.markdown("""
    # 🩸 Fingerprint Blood Group Detection System

    ### Deep Learning Based Healthcare Application

    Predict Blood Group using Fingerprint Images and CNN
    """)

    st.image(
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef",
        use_container_width=True
    )

    st.success("Welcome to the Fingerprint Blood Group Detection System")

    st.write("### Team Members")

    st.write("• Ravali")
    st.write("• Anusha")
    st.write("• Yeshashwi")

    if st.button("🚀 Get Started"):
        st.session_state.page = "about"
        st.rerun()
# -----------------------------
# ABOUT PAGE
# -----------------------------
elif st.session_state.page == "about":

    set_background("#E8F5E9")

    st.title("📖 About Project")

    st.info("CNN-based healthcare application for blood group prediction.")

    st.write("""
        This project predicts blood groups using
        fingerprint images and a Convolutional
        Neural Network (CNN).

        Technologies Used:

        • Python
        • TensorFlow
        • Keras
        • OpenCV
        • Streamlit

        Dataset:
        Fingerprint Blood Group Dataset
        """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        if st.button("➡ Continue"):
            st.session_state.page = "upload"
            st.rerun()
# -----------------------------
# UPLOAD PAGE
# -----------------------------
elif st.session_state.page == "upload":

    set_background("#E3F2FD")

    st.title("🔍 Fingerprint Analysis")

    st.success("Upload a fingerprint image to begin prediction.")

    uploaded_file = st.file_uploader(
        "Upload Fingerprint Image",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    # Navigation Buttons
    nav1, nav2 = st.columns(2)

    with nav1:
        if st.button("⬅ Back to About"):
            st.session_state.page = "about"
            st.rerun()

    with nav2:
        if st.button("👨‍💻 Team Details"):
            st.session_state.page = "team"
            st.rerun()

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_GRAYSCALE
        )

        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = img.reshape(1, 128, 128, 1)

        with st.spinner("Analyzing Fingerprint..."):
            prediction = model.predict(img)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.image(
            uploaded_file,
            caption="Uploaded Fingerprint"
        )

        # Metric Cards
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Blood Group",
                classes[predicted_class]
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        # Report Download
        report = f"""
Fingerprint Blood Group Detection Report

Predicted Blood Group: {classes[predicted_class]}

Confidence Score: {confidence:.2f}%

Generated By:
Fingerprint Blood Group Detection System
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="blood_group_report.txt",
            mime="text/plain"
        )

        blood_info = {
            "A": "Group A blood contains A antigens.",
            "AB": "Group AB is Universal Recipient.",
            "B": "Group B blood contains B antigens.",
            "O": "Group O is Universal Donor."
        }

        st.subheader("Blood Group Information")
        st.write(
            blood_info[classes[predicted_class]]
        )

        st.subheader("Prediction Summary")

        st.table({
            "Parameter": [
                "Predicted Group",
                "Confidence"
            ],
            "Value": [
                classes[predicted_class],
                f"{confidence:.2f}%"
            ]
        })

        st.subheader("Prediction Probability Analysis")

        prob_df = pd.DataFrame({
            "Blood Group": classes,
            "Probability (%)": prediction[0] * 100
        })

        st.bar_chart(
            prob_df.set_index("Blood Group")
        )

        st.write("---")

        st.subheader("Dataset Statistics")

        st.write("""
Total Images Used: 6000

A Group : 1500
AB Group : 1500
B Group : 1500
O Group : 1500
""")

        st.write("---")

        st.subheader("Model Information")

        st.write("""
Model Type: Convolutional Neural Network (CNN)

Input Image Size: 128 × 128 pixels

Training Images: 6000

Training Epochs: 5

Output Classes: A, AB, B, O

Framework: TensorFlow & Keras
""")

        st.subheader("Project Information")

        st.write("""
Dataset Used: Fingerprint Blood Group Dataset

Algorithm: Convolutional Neural Network (CNN)

Classes:
• A
• AB
• B
• O
""")

# -----------------------------
# TEAM PAGE
# -----------------------------
elif st.session_state.page == "team":

    set_background("#FFF3E0")

    st.title("👨‍💻 Team Information")

    st.balloons()

    st.write("""
        Project Title:
        Fingerprint Blood Group Detection System

        Team Members:

        • Ravali
        • Anusha
        • Yeshashwi

        Department:
        CSE (AI & ML)

        College:
        Methodist College of Engineering and Technology
        """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "upload"
            st.rerun()

    with col2:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()
# -----------------------------
# FOOTER
# -----------------------------
st.write("---")

st.caption(
    "Developed by Ravali, Anusha, Yeshashwi | Fingerprint Blood Group Detection System"
)
