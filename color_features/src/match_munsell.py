import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from .data_preprocessing import LABPreprocessing
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from utils.config import MUNSELL_CSV_PATH, FINAL_MODEL_PATH, TFLITE_MODELS_DIR

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

def match_munsell_color(lab_color):
    # Convert LAB color to Munsell color using munsell package
    lab_color_r = robjects.FloatVector(lab_color)
    munsell_chip = match_munsell_color_r(lab_color_r)
    return str(munsell_chip[0])

def predict_and_match_soil_type(model_path, image_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
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
    
    # Predict soil type
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = np.argmax(prediction)
    
    # Get predicted LAB color
    predicted_lab = np.array([l_norm.mean(), a_norm.mean(), b_norm.mean()])
    
    # Match with Munsell color
    munsell_color = match_munsell_color(predicted_lab)
    
    return predicted_class, munsell_color

if __name__ == "__main__":
    model_path = f'{TFLITE_MODELS_DIR}/soil_classifier.tflite'
    image_path = f'{SOIL_LABELED_DIR}/class1/sample.jpg'
    
    predicted_class, munsell_color = predict_and_match_soil_type(model_path, image_path)
    print(f'Predicted Class: {predicted_class}')
    print(f'Matched Munsell Color: {munsell_color}')