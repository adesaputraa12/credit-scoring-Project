from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_trainer import ModelTrainer
from src.mlProject.logging import logger

STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        trainer_config = config.get_model_trainer_config()
        trainer = ModelTrainer(config=trainer_config)
        return trainer.initiate_model_training()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainerTrainingPipeline()
        metrics = pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e