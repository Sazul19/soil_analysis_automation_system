import tensorflow as tf
from tensorflow.keras.models import load_model
from .data_preprocessing import LABPreprocessing
from .data_augmentation import build_data_pipeline
from utils.config import SOIL_TEST_DIR, FINAL_MODEL_PATH

def evaluate_model(model_path, data_dir):
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',
        seed=42
    ).map(build_data_pipeline(), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

if __name__ == "__main__":
    evaluate_model(FINAL_MODEL_PATH, SOIL_TEST_DIR)