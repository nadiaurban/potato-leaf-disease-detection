# Potato Leaf Disease Detection App

This project is an example app that demonstrates how to deploy an image recognition model for detecting potato leaf diseases.  
It was built with middle school students as part of a hands-on activity, where they trained their models using [Teachable Machine](https://teachablemachine.withgoogle.com/).

## Overview

The app detects three types of potato leaf conditions:
- **Fungi:** Indicates a fungal infection.
- **Pest:** Shows signs of pest infestation.
- **Healthy:** Indicates that the leaf is healthy.

### How It Works

- **Model Loading:**  
  The app uses a pre-trained CNN model (saved with TensorFlow/Keras) to make predictions.
  
- **Image Input:**  
  Users can upload an image of a potato leaf using the file uploader.

- **Prediction:**  
  The app processes the image, feeds it to the model, and displays the predicted class along with the confidence score.

## Files Included

- `app.py`: The main Streamlit application code.
- `requirements.txt`: The list of required Python packages.
- `README.md`: This file.
- `.streamlit/config.toml`: Configuration file (if you need to force the light theme).
- Various image files for examples (located in the `images/` folder) and the saved model (in `model.savedmodel/`).

## How to Run the App Locally

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your-username>/student-app-template.git
   cd student-app-template
