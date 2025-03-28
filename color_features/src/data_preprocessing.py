import tensorflow as tf
import cv2
import numpy as np

def clahe_normalization(image):
    """Adaptive histogram equalization for lighting invariance"""
    # Convert to LAB color space
    lab = tf.image.rgb_to_xyz(image)
    lab = tf.clip_by_value(lab, 1e-8, 1.0)
    
    # CLAHE approximation using TensorFlow ops
    l_channel = lab[..., 0]
    l_normalized = tf.image.per_image_standardization(l_channel)
    l_normalized = tf.clip_by_value(l_normalized, -2.0, 2.0)
    l_normalized = (l_normalized + 2.0) / 4.0  # Scale to 0-1 range
    
    # Merge channels
    return tf.concat([l_normalized, lab[..., 1:3]], axis=-1)

def color_augmentations(image):
    """Apply random color transformations"""
    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Add Gaussian noise
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.1)
    image = image + noise
    return tf.clip_by_value(image, 0.0, 1.0)

def geometric_augmentations(image):
    """Apply random geometric transformations"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_rotation(image, 0.2)
    image = tf.image.random_crop(image, size=(224, 224, 3))
    return image

def preprocess_image(image_path, training=True):
    """Full preprocessing pipeline for raw image files"""
    # Load and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize with aspect ratio preservation
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)
    
    if training:
        # Apply geometric augmentations
        image = geometric_augmentations(image)
        
        # Apply color augmentations
        image = color_augmentations(image)
    else:
        # Center crop for validation/inference
        image = tf.image.central_crop(image, central_fraction=0.8)
        image = tf.image.resize(image, (224, 224))
    
    # Lighting normalization
    image = clahe_normalization(image)
    
    # Convert to LAB-like features
    image = tf.image.rgb_to_xyz(image)
    return image

def create_dataset_pipeline(image_paths, batch_size=32, training=True):
    """Create optimized tf.data pipeline with preprocessing"""
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    
    # Shuffle only for training
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Parallel processing
    dataset = dataset.map(
        lambda x: preprocess_image(x, training=training), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batching and prefetching
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Example usage --------------------------------------------------------------
if __name__ == "__main__":
    # Sample image paths (replace with your actual paths)
    train_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
    val_images = ["path/to/val_image1.jpg"]

    # Create datasets
    train_ds = create_dataset_pipeline(train_images, training=True)
    val_ds = create_dataset_pipeline(val_images, training=False)

    # Visualize preprocessed images
    import matplotlib.pyplot as plt

    def denormalize(image):
        """Convert normalized image back to displayable format"""
        return (image * 255).numpy().astype(np.uint8)

    for batch in train_ds.take(1):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i in range(4):
            img = denormalize(batch[i])
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].axis('off')
        plt.show()