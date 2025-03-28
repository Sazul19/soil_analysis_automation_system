import tensorflow as tf

# ----------------------------
# 2. Data Augmentation
# ----------------------------
def lab_augment(image):
    # Random lighting variations in L channel
    image[..., 0] = tf.image.random_brightness(image[..., 0], max_delta=25 / 100.0)
    # Color consistency in AB channels
    image[..., 1:] = tf.clip_by_value(
        image[..., 1:] + tf.random.normal([2], mean=0, stddev=10 / 255.0),
        0, 1
    )
    return image

def build_data_pipeline():
    # Data augmentation for LAB space
    lab_augment = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ])
    
    def _preprocess(image):
        # Convert to LAB and apply augmentations
        lab = LABPreprocessing()(image)
        return lab_augment(lab)
    
    return _preprocess