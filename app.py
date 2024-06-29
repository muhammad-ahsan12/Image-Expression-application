import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import os

# Set up the Google API key environment variable
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho'

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

# Streamlit app
st.sidebar.title("🖼️ Upload Your Image Here")
uploaded_file = st.sidebar.file_uploader("📂 Select an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open("temp_image.jpg")
    st.sidebar.image(image, caption='🖼️ Uploaded Image.', width=300)  # Display the uploaded image in the sidebar
    st.sidebar.success("✅ Image Uploaded Successfully")
# Main section
st.title("💬🖼️ Image Expression Analysis App")

if uploaded_file is not None:
    text = st.text_input("❓ Ask a Question About the Image")
    if st.button("🔮 Predict"):
        # Prepare the message for the language model
        message = HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": "temp_image.jpg"}
            ]
        )

        # Get prediction
        with st.spinner('🔄 Analyzing the image...'):
            response = llm.invoke([message])
        
        # Display the prediction result
        st.subheader("🧠 Model Prediction")
        necessary_text = response.content
        st.write(necessary_text)

        # Optionally, delete the temporary file after use
        os.remove("temp_image.jpg")
else:
    st.write("📤 Upload an image in the sidebar to start analysis.")
