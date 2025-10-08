import os
import sys

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception.exception import NetworkSecurityException

load_dotenv()

MONGO_DB_URL = os.getenv("MONGODB_URI")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            if not MONGO_DB_URL:
                raise ValueError("MONGODB_URI is not set")
        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        client = None
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            client = pymongo.MongoClient(MONGO_DB_URL)
            collection = client[database_name][collection_name]

            count = collection.estimated_document_count()
            if count == 0:
                raise ValueError(
                    f"Collection '{database_name}.{collection_name}' is empty"
                )

            cursor = collection.find({}, {"_id": 0})
            df = pd.DataFrame(list(cursor))

            if df.empty:
                raise ValueError("Fetched DataFrame is empty")

            obj_cols = df.select_dtypes(include=["object"]).columns
            if len(obj_cols) > 0:
                df[obj_cols] = df[obj_cols].replace({"na": np.nan})

            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore
        finally:
            if client:
                client.close()

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            if dataframe is None or dataframe.empty:
                raise ValueError("Empty DataFrame cannot be exported")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            if dataframe is None or dataframe.empty:
                raise ValueError("Cannot split empty DataFrame")

            test_size = self.data_ingestion_config.train_test_split_ratio
            random_state = getattr(self.data_ingestion_config, "random_state", 42)
            target_col = getattr(self.data_ingestion_config, "target_column", None)

            stratify = None
            if target_col and target_col in dataframe.columns:
                if (
                    dataframe[target_col].nunique(dropna=True) >= 2
                    and len(dataframe) >= 5
                ):
                    stratify = dataframe[target_col]

            train_set, test_set = train_test_split(
                dataframe,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return dataingestionartifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore
