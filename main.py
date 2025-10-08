from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Initializing training pipeline configuration")
        trainingpipelineconfig = TrainingPipelineConfig()

        logging.info("Creating data ingestion configuration")
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)

        logging.info("Instantiating DataIngestion component")
        data_ingestion = DataIngestion(dataingestionconfig)

        logging.info("Initiating data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")
        print(dataingestionartifact)

        logging.info("Creating data validation configuration")
        data_validation_config = DataValidationConfig(trainingpipelineconfig)

        logging.info("Instantiating DataValidation component")
        data_validation = DataValidation(dataingestionartifact, data_validation_config)

        logging.info("Initiating data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully")
        print(data_validation_artifact)

        logging.info("Creating data transformation configuration")
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)

        logging.info("Instantiating DataTransformation component")
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )

        logging.info("Initiating data transformation")
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        logging.info("Data transformation completed successfully")
        print(data_transformation_artifact)

        logging.info("Creating model trainer configuration")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)

        logging.info("Instantiating ModelTrainer component")
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )

        logging.info("Initiating model training")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully")
        print(model_trainer_artifact)

        logging.info("Model training artifact created successfully")

    except Exception as e:
        logging.exception("An error occurred during the training pipeline execution")
        raise NetworkSecurityException(e, sys)
