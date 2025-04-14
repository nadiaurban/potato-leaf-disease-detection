import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

# Define class labels
CLASS_NAMES = ["Fungi", "Pest", "Healthy"]  # Matches your model's label encoding

# Load the model using TFSMLayer
model = keras.layers.TFSMLayer(
    "model.savedmodel",
    call_endpoint="serving_default"
)

def resize_image(image_path, width):
    img = Image.open(image_path)
    img.thumbnail((width, width))  # Maintain aspect ratio
    return img


# Sidebar for model information
with st.sidebar:
    st.title("ℹ️ About the Model")
    st.write(
        """
        **Potato Leaf Disease Classification App** 🌿  
        This app helps classify potato leaves into three categories:
        - 🦠 **Fungi**: Indicates fungal infection on the leaf.
        - 🐛 **Pest**: Signs of pest infestation.
        - ✅ **Healthy**: No disease detected.

        **🧠 Model Details:**
        - Model trained using **Teachable Machine**.
        - Dataset: *Potato Leaf Disease Dataset in Uncontrolled Environments from Kaggle.com*.
        - Model Type: **CNN (Convolutional Neural Network)**.
        - Input Size: **150x150 pixels**.
        - Output: **3-class classification**.
        """
    )
    # Example Images (Replace with actual image paths)
    st.write("### 🦠 Fungi")
    st.image(resize_image("images/fungi_example.jpg", 300), caption="Fungal Infection")

    st.write("### 🐛 Pest")
    st.image(resize_image("images/pest_example.jpg", 300), caption="Pest Infestation")

    st.write("### ✅ Healthy")
    st.image(resize_image("images/healthy_example.jpg", 300), caption="Healthy Leaf")


   
    st.caption("📝 Use the file uploader to analyze a potato leaf image.")

# Custom CSS to style the sidebar background color
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #c6e6a8; /* Green background */
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: black !important; /* Ensures text is readable */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)  # Ensure correct data type

# Main app UI
# Custom CSS to style the header
st.markdown(
    """
    <style>
        /* Center-align and style the title */
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #2E5A4D;
        }

        /* Style for main text */
        .main-text {
            font-size: 18px;
            text-align: center;
            color: #3A5A40;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header section
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🍃 Potato Leaf Disease Detection</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-text">Upload an image of a potato leaf to check if it is affected by <b>Fungi</b>, <b>Pests</b>, or is <b>Healthy</b>.</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader("📤 Upload a potato leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Resize the image for display
    max_width = 300  # Set desired max width (adjust as needed)
    image.thumbnail((max_width, max_width))  # Maintain aspect ratio

    st.image(image, caption="📸 Uploaded Image", use_container_width=False)  # Disable full width

    # Preprocess image
    processed_image = preprocess_image(image)

    # Get predictions from the model
    prediction = model(processed_image)

    # Extract tensor from dictionary output
    if isinstance(prediction, dict):
        prediction = prediction[list(prediction.keys())[0]]  # Automatically extract first key

    predicted_class = np.argmax(prediction)  # Get index of highest confidence
    confidence = np.max(prediction)  # Confidence score

    # Display prediction
    st.write("### 🏆 Prediction:")
    st.success(f"**Class: {CLASS_NAMES[predicted_class]}** (Confidence: {confidence:.2%})")

# Add empty space before footer
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)




# Footer Text
st.markdown(
    """
    <div style='text-align: left;'>
        <p>©Created by Nadia Urban for Shanghai Thomas School.<br>CNN model trained with Teachable Machine</p>
    </div>
    """,
    unsafe_allow_html=True
)

# School Logo
st.image("school_logo.png", width=150)
