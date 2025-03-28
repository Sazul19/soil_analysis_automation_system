import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from utils.config import MUNSELL_CSV_PATH, TFLITE_MODELS_DIR, SOIL_LABELED_DIR

# Activate automatic conversion between pandas and R dataframes
pandas2ri.activate()

# Import the munsell package
munsell = importr('munsell')

# Load Munsell color data from CSV
munsell_df = pd.read_csv(MUNSELL_CSV_PATH)

# Define the R function to match Munsell colors
match_munsell_color_r = robjects.r("""
function(lab_color) {
    library(munsell)
    lab_color <- matrix(lab_color, nrow = 1, ncol = 3)
    colnames(lab_color) <- c('L', 'A', 'B')
    munsell_chip <- closestMunsell(lab_color)
    return(as.character(munsell_chip))
}
""")

def preprocess_soil_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    lab = tf.image.rgb_to_lab(image / 255.0)
    
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = lab[..., 0]
    l_channel_clahe = clahe.apply((l_channel * 255).astype(np.uint8)).astype(np.float32) / 255.0
    
    # Normalize channels
    l_norm = l_channel_clahe / 100.0
    a_norm = (lab[..., 1] + 128) / 255.0
    b_norm = (lab[..., 2] + 128) / 255.0
    
    processed_image = np.stack([l_norm, a_norm, b_norm], axis=-1)
    
    return processed_image

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_soil_class(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], np.array([image], dtype=np.float32))
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    return predicted_class

def match_munsell_color(lab_color):
    # Convert LAB color to Munsell color using munsell package
    lab_color_r = robjects.FloatVector(lab_color)
    munsell_chip = match_munsell_color_r(lab_color_r)
    return str(munsell_chip[0])

def validate_location(latitude, longitude):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    if location:
        return location.address
    else:
        return "Location not found"

def predict_and_validate_soil_type(model_path, image_path, location_data):
    # Load the model
    interpreter = load_tflite_model(model_path)
    
    # Preprocess the image
    image = preprocess_soil_image(image_path)
    
    # Predict soil type
    predicted_class = predict_soil_class(interpreter, image)
    
    # Get predicted LAB color
    lab_color = np.array([
        image[..., 0].mean(),
        image[..., 1].mean(),
        image[..., 2].mean()
    ])
    
    # Match with Munsell color
    munsell_color = match_munsell_color(lab_color)
    
    # Validate with location data
    location_address = validate_location(location_data['latitude'], location_data['longitude'])
    
    return predicted_class, munsell_color, location_address

if __name__ == "__main__":
    model_path = f'{TFLITE_MODELS_DIR}/soil_classifier.tflite'
    image_path = f'{SOIL_LABELED_DIR}/class1/sample.jpg'
    location_data = {'latitude': 37.7749, 'longitude': -122.4194}  # Example location data
    
    predicted_class, munsell_color, location_address = predict_and_validate_soil_type(model_path, image_path, location_data)
    print(f'Predicted Class: {predicted_class}')
    print(f'Matched Munsell Color: {munsell_color}')
    print(f'Location Address: {location_address}')