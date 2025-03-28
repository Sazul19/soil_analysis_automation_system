
# Now use absolute imports
from color_features.model import mobile_soil_classifier, optimize_for_mobile
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from color_features.model import mobile_soil_classifier, optimize_for_mobile
from color_features.data_augmentation import build_data_pipeline
from color_features.utils.config import GENERAL_IMAGES_DIR, CHECKPOINTS_DIR, FINAL_MODEL_PATH

if __name__ == "__main__":
    # Initialize model
    model = mobile_soil_classifier(num_classes=100)  # Adjust num_classes based on general dataset
    model = optimize_for_mobile(model)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        FINAL_MODEL_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train (example)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        GENERAL_IMAGES_DIR,
        image_size=(224, 224),
        batch_size=32,
        label_mode=None,  # No labels for general images
        shuffle=True,
        seed=42
    ).map(build_data_pipeline(), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Since there are no labels, we will use a dummy label for compatibility
    train_ds = train_ds.map(lambda x: (x, tf.zeros(tf.shape(x)[0], dtype=tf.int32)))
    
    # Dummy validation dataset (since we don't have labels)
    val_ds = train_ds.take(10)  # Take a small subset for validation
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save(FINAL_MODEL_PATH)