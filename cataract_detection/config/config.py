class Config:
    # Data paths
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    MODEL_SAVE_PATH = "models"
    LOGS_PATH = "logs"

    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Training parameters
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Random seed for reproducibility
    RANDOM_SEED = 42