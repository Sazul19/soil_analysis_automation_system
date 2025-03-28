import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_model_optimization as tfmot

# ----------------------------
# 2. MobileNetV3-LAB Architecture
# ----------------------------
def mobile_soil_classifier(input_shape=(224, 224, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    
    # LAB Preprocessing
    x = LABPreprocessing()(inputs)
    
    # MobileNetV3-small backbone
    base = tf.keras.applications.MobileNetV3Small(
        input_tensor=x,
        weights=None,  # Train from scratch
        include_top=False
    )
    
    # Lightweight head
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ----------------------------
# 3. Quantization & Pruning
# ----------------------------
def optimize_for_mobile(model):
    # Apply quantization-aware training
    quantize_annotate = tfmot.quantization.keras.quantize_annotate_layer
    def _add_quant(layer):
        if isinstance(layer, layers.BatchNormalization):
            return quantize_annotate(layer)
        return layer
    
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=_add_quant,
    )
    
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    # Apply pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.3,
            final_sparsity=0.7,
            begin_step=0,
            end_step=2000
        )
    }
    
    return tfmot.sparsity.keras.prune_low_magnitude(qat_model, **pruning_params)