# Configuration settings
DATA_DIR = '/home/sala/data'
GENERAL_IMAGES_DIR = f'{DATA_DIR}/general'
SOIL_LABELED_DIR = f'{DATA_DIR}/soil/test/Black Soil'
SOIL_TEST_DIR = f'{DATA_DIR}/soil/test'
MUNSELL_DIR = f'{DATA_DIR}/munsell'
MUNSELL_CSV_PATH = f'{MUNSELL_DIR}/equivalent_munsell.csv'

MODELS_DIR = '/home/sala/iit/dsgp/soil_analysis_automation_system/color_features/models'
CHECKPOINTS_DIR = f'{MODELS_DIR}/checkpoints'
TFLITE_MODELS_DIR = f'{MODELS_DIR}/tflite_models'
FINAL_MODEL_PATH = f'{MODELS_DIR}/final_model.h5'

LOGS_DIR = '/home/sala/iit/dsgp/soil_analysis_automation_system/color_features/logs'
TRAINING_LOGS_DIR = f'{LOGS_DIR}/training_logs'
EVALUATION_LOGS_DIR = f'{LOGS_DIR}/evaluation_logs'

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 50

