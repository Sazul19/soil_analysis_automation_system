import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .model import mobile_soil_classifier, optimize_for_mobile
from .data_augmentation import build_data_pipeline
from utils.config import SOIL_LABELED_DIR, CHECKPOINTS_DIR, FINAL_MODEL_PATH, TFLITE_MODELS_DIR

if __name__ == "__main__":
    # Load pre-trained general model
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    
    # Unfreeze some layers for fine-tuning
    for layer in model.layers[:-10]:  # Adjust based on your model
        layer.trainable = False
    
    for layer in model.layers[-10:]:
        layer.trainable = True
    
    # Recompile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        CHECKPOINTS_DIR + '/soil_best_model.h5',
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
        SOIL_LABELED_DIR,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',
        validation_split=0.2,
        subset='training',
        seed=42
    ).map(build_data_pipeline(), num_parallel_calls=tf.data.AUTOTUNE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        SOIL_LABELED_DIR,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',
        validation_split=0.2,
        subset='validation',
        seed=42
    ).map(build_data_pipeline(), num_parallel_calls=tf.data.AUTOTUNE)
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save(FINAL_MODEL_PATH)
    
    # Export to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(TFLITE_MODELS_DIR + '/soil_classifier.tflite', 'wb') as f:
        f.write(tflite_model)