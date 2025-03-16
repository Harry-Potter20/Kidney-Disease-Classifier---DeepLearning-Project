from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.base_model import BaseModel
from cnnClassifier import logger

STAGE_NAME = "Base Model"



class BaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"Build {STAGE_NAME} started")
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Build {STAGE_NAME} completed <<<<<<\n\nx=====x")
    except Exception as e:
        logger.exception(e)
        raise e
