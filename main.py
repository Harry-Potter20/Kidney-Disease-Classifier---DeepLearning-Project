from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_base_model import BaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=====x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Build Base Model"
try:
    logger.info(f"Build {STAGE_NAME} started")
    obj = BaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Build {STAGE_NAME} completed <<<<<<\n\nx=====x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"
try:
    logger.info(f"{STAGE_NAME} started")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx=====x")
except Exception as e:
    logger.exception(e)
    raise e
