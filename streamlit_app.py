import streamlit as st
import os
from werkzeug.utils import secure_filename
from prediction import predict_flower

# Streamlit app title
st.title("Flower Identification App")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure the upload folder exists
    UPLOAD_FOLDER = 'static/uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Save the uploaded file
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Predict the flower type
    predicted_label, confidence = predict_flower(file_path)
    if predicted_label:
        st.success(f"Predicted Flower: {predicted_label} with {confidence:.2f}% confidence.")
    else:
        st.warning("The flower cannot be confidently recognized. Please try another image.")

else:
    st.text("Upload an image to get started.")