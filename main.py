from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from src.mlProject.logging import logger
from src.mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.mlProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_validation_pipeline = DataValidationTrainingPipeline()
    data_validation_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
       logger.exception(e)
       raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
       logger.exception(e)
       raise e

STAGE_NAME = "Model Trainer Stage"
try:
     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
     model_trainer_pipeline = ModelTrainerTrainingPipeline()
     model_trainer_pipeline.main()
     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
       logger.exception(e)
       raise e

STAGE_NAME = "Model Evaluation Stage"
try:
     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
     model_evaluation_pipeline = ModelEvaluationTrainingPipeline()
     model_evaluation_pipeline.main()
     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
     logger.exception(e)
     raise e