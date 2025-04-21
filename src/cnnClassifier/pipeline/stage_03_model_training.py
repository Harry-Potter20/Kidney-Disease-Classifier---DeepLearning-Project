from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        # Additional setup (if needed)
        pass

    def main(self):
        # Initialize configuration manager
        config = ConfigurationManager()
        
        # Load training configuration
        training_config = config.get_training_config()
        if not training_config:
            logger.error(f"Training configuration could not be loaded.")
            raise ValueError("Training configuration is invalid or missing.")
        
        # Initialize training component with the loaded configuration
        training = Training(config=training_config)
        
        # Start training
        logger.info("Starting model training...")
        training.train()
        logger.info("Model training completed successfully.")

if __name__ == '__main__':
    try:
        # Log the start of the pipeline
        logger.info(f"{STAGE_NAME} started")
        
        # Initialize and run the pipeline
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        
        # Log the successful completion of the pipeline
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx=====x")
        
    except Exception as e:
        # Log the exception with detailed information
        logger.exception(f"Error during {STAGE_NAME}: {e}")
        raise e
