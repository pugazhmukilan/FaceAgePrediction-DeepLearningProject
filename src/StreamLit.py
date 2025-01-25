import streamlit as st
import tensorflow as tf
from PIL import Image
import io

# Function to load the model and make predictions
def load_and_predict(image_path, model_path="Models/pretrained_final_model.h5"):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    
    # Make the prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = tf.argmax(predictions[0]).numpy()
    
    # Generate class names for age groups from 0-4 to 100 (in steps of 5 years)
    classname = [f"{i}-{i+4}" for i in range(0, 101, 5)]
    
    return classname[predicted_class]

# Streamlit UI
st.title("Age Prediction from Image")
st.write("Upload an image of a face, and the model will predict the age group.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    
    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Call the prediction function
    st.write("Processing...")
    
    # Use the uploaded image file to make the prediction
    predicted_age = load_and_predict(uploaded_file)

    # Display the result
    st.write(f"The predicted age group for the uploaded image is: {predicted_age}")


